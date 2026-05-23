"""SAM 2-based segmentation refinement.

Runs Meta's SAM 2 auto-mask generator on rendered views, projects every
mesh-face centroid through the captured Blender camera matrix to find
which SAM mask each face lives in, then promotes promising masks to new
sub-components — typically the tongue, collar, or eyestay that the
spatial+k-means heuristic merges into the generic "upper" cluster.

Why this is worth a 2 GB ML dependency: the brief explicitly lists
"upper, sole, laces, tongue, etc." but the heuristic pipeline can't
reliably separate tongue from vamp on a worn shoe (the colors blend
and the spatial position overlaps). SAM 2 sees the visual seam and
gives us a clean boundary.

The model loads lazily and is cached for the lifetime of the Python
process (same pattern as material._load_clip). First run downloads
~155 MB of weights into the HuggingFace cache.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


_SAM_CACHE: dict = {}

DEFAULT_MODEL = "facebook/sam2-hiera-small"


def _load_sam(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Build (and cache) the SAM 2 auto-mask generator.

    `device="cpu"` for determinism — MPS gives different mask boundaries
    run-to-run due to non-deterministic op kernels."""
    if "gen" in _SAM_CACHE:
        return _SAM_CACHE["gen"]
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
    model = build_sam2_hf(model_name, device=device)
    gen = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=20,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        min_mask_region_area=400,
        crop_n_layers=0,
    )
    _SAM_CACHE["gen"] = gen
    return gen


# 3D → 2D projection via captured camera params

def project_world_to_pixel(world_pts: np.ndarray, cam_info: dict
                            ) -> Optional[np.ndarray]:
    """Project (N, 3) world-space Blender Unit points to (N, 2) pixel
    coordinates for the given view. Returns None for perspective cameras
    (we only segment ortho views).

    Image pixel convention: x = right, y = down, origin top-left."""
    M = cam_info["matrix_world"]            # 4x4 camera-to-world
    Minv = np.linalg.inv(M)                 # world-to-camera
    N = world_pts.shape[0]
    pts_h = np.concatenate([world_pts, np.ones((N, 1))], axis=1)
    pts_cam = pts_h @ Minv.T
    x_cam = pts_cam[:, 0]
    y_cam = pts_cam[:, 1]
    if not cam_info["is_ortho"]:
        return None
    S = cam_info["ortho_scale"]
    w = float(cam_info["width"])
    h = float(cam_info["height"])
    aspect = w / h
    # Blender's ortho_scale spans the larger sensor dimension. For our
    # 1100×800 renders (aspect>1), that's the width.
    if w >= h:
        half_w = S / 2.0
        half_h = S / (2.0 * aspect)
    else:
        half_h = S / 2.0
        half_w = S / 2.0 * aspect
    px = (x_cam + half_w) / (2.0 * half_w) * w
    py = h - (y_cam + half_h) / (2.0 * half_h) * h
    return np.stack([px, py], axis=1)


# View segmentation

def segment_view(view_path: str, model_name: str = DEFAULT_MODEL) -> list:
    """Run SAM 2 auto-mask on a single rendered PNG. Returns list of
    {"segmentation": HxW bool array, "area": int, "bbox": [x,y,w,h],
    "predicted_iou": float, "stability_score": float}."""
    gen = _load_sam(model_name)
    img = np.array(Image.open(view_path).convert("RGB"))
    masks = gen.generate(img)
    masks.sort(key=lambda m: -m["area"])
    return masks


# Canonical component vocabulary

# Heuristic-name → canonical-vocabulary mapping. Lets the downstream BOM
# table report industry-standard component names regardless of what the
# spatial+k-means heuristic happened to call them.
CANONICAL_NAME_MAP: Dict[str, str] = {
    "midsole":         "eva-midsole",
    "outsole":         "rubber-outsole",
    "laces":           "shoe-laces",
    "lateral-quarter": "quarter-lateral",
    "medial-quarter":  "quarter-medial",
    "heel-counter":    "heel-counter",
    "toe-cap":         "vamp",        # heuristic 'toe-cap' is the whole forefoot upper
    "tongue-vamp":     "vamp",        # collapse the catch-all into vamp
    "heel-tab":        "heel-tab",
    "collar":          "collar",
    "tongue":          "tongue",
    "eyestay":         "eyestay",
    "mudguard":        "mudguard",
}


def canonicalize_name(heuristic_name: str) -> str:
    """Map a heuristic component name to its canonical-vocabulary form.

    Falls back to the input string if no mapping is known."""
    n = heuristic_name.lower()
    if n in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[n]
    # Substring fallback for things like 'tongue-vamp-extra-suffix'.
    for k, v in CANONICAL_NAME_MAP.items():
        if k in n:
            return v
    return n


# Spatial classification of a SAM-discovered patch into one canonical slot.
# Each slot has a rule (cx, cy normalized to half-width, cz, area_frac of
# upper) → confidence in [0, 1]. The classifier returns the highest-scoring
# slot that is not already taken.

def _slot_scores(cx: float, cy: float, cz: float,
                  area_frac: float) -> Dict[str, float]:
    """Score each canonical slot for a patch with the given normalised
    position + relative area. cx ∈ [0, 1] heel→toe, cy ∈ [-1, 1] medial→
    lateral, cz ∈ [0, 1] sole→collar, area_frac is the patch's fraction
    of the *upper*'s total area."""
    abs_cy = abs(cy)
    s: Dict[str, float] = {}

    # toe-tip: very forward, low Z, small area
    if cx > 0.85 and cz < 0.55 and area_frac < 0.10:
        s["toe-tip"] = 0.55 + 0.40 * ((cx - 0.85) / 0.15) - 0.20 * cz

    # vamp: forefoot upper around the toe
    if 0.55 < cx < 0.92 and abs_cy < 0.65 and 0.25 < cz < 0.70:
        s["vamp"] = 0.85 - 0.30 * abs(cx - 0.72) - 0.20 * abs_cy

    # mudguard: low Z, off-center Y, mid-X overlay along the foxing line
    if 0.20 < cx < 0.85 and cz < 0.40 and abs_cy > 0.35:
        s["mudguard"] = 0.75 - 0.30 * cz + 0.10 * (abs_cy - 0.35)

    # eyestay: alongside the laces — mid-X, off-center Y, high-mid Z
    if 0.35 < cx < 0.72 and 0.25 < abs_cy < 0.85 and 0.45 < cz < 0.85:
        s["eyestay"] = 0.70 + 0.20 * (cz - 0.45) / 0.40

    # tongue: central Y, mid-front X, high Z
    if 0.38 < cx < 0.68 and abs_cy < 0.32 and cz > 0.50:
        s["tongue"] = 0.85 - 0.30 * abs(cx - 0.52) - 0.40 * abs_cy

    # collar: top of upper
    if cz > 0.75:
        s["collar"] = 0.85 + 0.15 * (cz - 0.75) / 0.25

    # heel-tab: rear-top, small area
    if cx < 0.22 and cz > 0.68 and area_frac < 0.06:
        s["heel-tab"] = 0.80 - 0.50 * area_frac

    # heel-counter: rear, mid-to-high Z (large area)
    if cx < 0.32 and 0.20 < cz < 0.78:
        s["heel-counter"] = 0.70 + 0.20 * (1.0 - cx / 0.32)

    # quarter-lateral: side, lateral Y > 0.3
    if 0.15 < cx < 0.75 and cy > 0.35 and 0.25 < cz < 0.75:
        s["quarter-lateral"] = 0.65 + 0.20 * (cy - 0.35)

    # quarter-medial: side, medial Y < -0.3
    if 0.15 < cx < 0.75 and cy < -0.35 and 0.25 < cz < 0.75:
        s["quarter-medial"] = 0.65 + 0.20 * (-cy - 0.35)

    return {k: max(0.0, min(1.0, v)) for k, v in s.items()}


def classify_canonical_slot(face_pos_mm: np.ndarray,
                             all_face_pos_mm: np.ndarray,
                             area_frac: float,
                             taken_slots: set,
                             ) -> Tuple[Optional[str], float]:
    """Classify a SAM-discovered patch into the best available canonical
    slot. `taken_slots` is the set of canonical names already assigned
    (so we don't promote a second 'collar', etc.). Returns (name, conf)
    or (None, 0.0) if no slot matches."""
    all_x = all_face_pos_mm[:, 0]
    all_y = all_face_pos_mm[:, 1]
    all_z = all_face_pos_mm[:, 2]
    L = max(float(all_x.max() - all_x.min()), 1e-6)
    W = max(float(all_y.max() - all_y.min()), 1e-6)
    H = max(float(all_z.max() - all_z.min()), 1e-6)
    x_min, z_min = float(all_x.min()), float(all_z.min())
    cx = (face_pos_mm[:, 0].mean() - x_min) / L
    cy = float(face_pos_mm[:, 1].mean()) / (W / 2.0)
    cz = (face_pos_mm[:, 2].mean() - z_min) / H

    scores = _slot_scores(cx, cy, cz, area_frac)
    # Drop slots already taken; pick best remaining.
    available = {k: v for k, v in scores.items() if k not in taken_slots}
    if not available:
        return None, 0.0
    best_name = max(available, key=available.get)
    return best_name, float(available[best_name])


# Refinement entry point

def refine_segmentation(
    face_component: np.ndarray,
    face_centroids_bu: np.ndarray,
    face_centroids_mm: np.ndarray,
    view_paths: Dict[str, str],
    view_cameras: Dict[str, dict],
    component_names: List[str],
    target_views: Tuple[str, ...] = ("side-lateral", "side-medial"),
    verbose: bool = True,
    min_slot_confidence: float = 0.50,
) -> Tuple[np.ndarray, List[str]]:
    """Refine `face_component` by promoting SAM masks that lie inside an
    existing upper component into new canonical sub-components.

    Returns (refined_face_component, new_component_names) where
    new_component_names lists the canonical names of newly-promoted
    components, appended after the existing ones. Names are from the
    fixed industry vocabulary (vamp, tongue, collar, eyestay, mudguard,
    heel-tab, toe-tip). Slots already filled by the heuristic (via
    canonicalize_name) are not re-promoted.

    Algorithm (per view):
      1. Run SAM 2 auto-mask → list of masks.
      2. Project every face centroid through the view's camera → pixel.
      3. For each mask, find faces whose pixel falls inside it.
      4. If a mask is "mostly inside" one existing upper component (≥55%)
         and covers 10–85% of that component, classify it into a
         canonical slot via spatial position. Reject anything with
         confidence < min_slot_confidence.
    """
    log = print if verbose else (lambda *a, **k: None)
    refined = face_component.copy()

    # Identify upper-region components (eligible for subdivision) and
    # collect which canonical slots are ALREADY taken by the heuristic
    # output. e.g. if the heuristic produced 'lateral-quarter', the
    # canonical 'quarter-lateral' slot is taken.
    upper_indices = set()
    taken_slots: set = set()
    for ci, name in enumerate(component_names):
        canonical = canonicalize_name(name)
        taken_slots.add(canonical)
        n = name.lower()
        if "outsole" in n or "midsole" in n:
            continue
        upper_indices.add(ci)

    if not upper_indices:
        log("  no upper components to refine")
        return refined, []

    next_ci = len(component_names)
    new_names: List[str] = []
    promoted_keys: set = set()

    for view_name in target_views:
        if view_name not in view_paths or view_name not in view_cameras:
            continue
        cam_info = view_cameras[view_name]
        if not cam_info.get("is_ortho", False):
            continue
        log(f"  SAM on {view_name}...")
        t0 = time.time()
        try:
            masks = segment_view(view_paths[view_name])
        except Exception as e:
            log(f"    failed: {e}")
            continue
        log(f"    {len(masks)} masks in {time.time() - t0:.1f}s")
        face_px = project_world_to_pixel(face_centroids_bu, cam_info)
        if face_px is None:
            continue
        h, w = int(cam_info["height"]), int(cam_info["width"])
        px_int = np.clip(face_px[:, 0].astype(np.int32), 0, w - 1)
        py_int = np.clip(face_px[:, 1].astype(np.int32), 0, h - 1)
        # Filter faces whose projected pixel is actually inside the image
        # AND visible from this camera (z_cam < 0 means behind us).
        Minv = np.linalg.inv(cam_info["matrix_world"])
        N = face_centroids_bu.shape[0]
        pts_h = np.concatenate([face_centroids_bu, np.ones((N, 1))], axis=1)
        pts_cam = pts_h @ Minv.T
        z_cam = pts_cam[:, 2]
        visible = (z_cam < 0)  # in front of camera
        for mi, m in enumerate(masks):
            area = int(m["area"])
            if area > 0.5 * h * w:
                continue  # background
            seg = m["segmentation"]
            # face indices that fall inside this mask AND are visible
            in_mask = np.zeros(N, dtype=bool)
            in_mask[visible] = seg[py_int[visible], px_int[visible]]
            face_idx_in_mask = np.where(in_mask)[0]
            if len(face_idx_in_mask) < 30:
                continue
            existing_comps = refined[face_idx_in_mask]
            unique, counts = np.unique(existing_comps, return_counts=True)
            # Be more permissive for upper sub-parts: SAM masks often overlap
            # 2-3 heuristic clusters at their boundaries. Require only
            # 55% dominance, and require that >=60% of the mask's faces lie
            # on the upper.
            if counts.max() < 0.55 * counts.sum():
                continue
            dom_comp = int(unique[counts.argmax()])
            if dom_comp not in upper_indices:
                continue
            upper_face_share = sum(
                c for u, c in zip(unique, counts) if int(u) in upper_indices
            ) / counts.sum()
            if upper_face_share < 0.60:
                continue
            comp_face_count = int((refined == dom_comp).sum())
            mask_face_count = int(counts.max())
            cov = mask_face_count / max(comp_face_count, 1)
            if cov < 0.10 or cov > 0.85:
                continue  # too small or essentially the whole component
            promote_idx = face_idx_in_mask[refined[face_idx_in_mask] == dom_comp]
            if len(promote_idx) < 30:
                continue
            upper_total_faces = sum((refined == ui).sum() for ui in upper_indices)
            area_frac = len(promote_idx) / max(upper_total_faces, 1)
            name, conf = classify_canonical_slot(
                face_centroids_mm[promote_idx],
                face_centroids_mm,
                area_frac,
                taken_slots,
            )
            if name is None or conf < min_slot_confidence:
                continue
            key = (dom_comp, name)
            if key in promoted_keys:
                continue
            promoted_keys.add(key)
            taken_slots.add(name)
            refined[promote_idx] = next_ci
            new_names.append(name)
            log(f"    promoted '{name}' (conf {conf:.2f}) from "
                f"{component_names[dom_comp]} "
                f"({len(promote_idx)} faces, {cov*100:.0f}% of parent)")
            next_ci += 1

    return refined, new_names
