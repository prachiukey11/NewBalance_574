"""Material inference per component.

Two independent signals, fused:

1. **CLIP zero-shot** on a per-component texture patch cropped from the
   diffuse map's UV bounding box. OpenCLIP ViT-B/32 against a 12-class
   footwear material taxonomy with prompt-ensembling per class.

2. **PBR statistics** (when roughness/metallic maps are present): mean
   roughness, mean metallic, normal-map gradient magnitude, alpha
   variance. A rule-based classifier produces an independent material
   guess + confidence.

Fusion: if both signals agree → high confidence. Disagreement → flag
for human review; CLIP wins by default since it sees real texture.

CLIP weights download once to the HF cache (~150 MB for ViT-B-32).
Subsequent runs are offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# Material taxonomy

# 12 classes covering what shows up on a real shoe. Each class has 2-3
# prompts that get encoded and averaged — a small prompt ensemble.
MATERIAL_TAXONOMY: Dict[str, List[str]] = {
    "smooth_leather": [
        "a close-up photo of smooth polished leather material",
        "finished leather hide texture",
        "shiny full-grain leather surface",
    ],
    "suede": [
        "a close-up photo of suede leather material",
        "brushed napped suede texture",
        "soft velvety suede surface",
    ],
    "nubuck": [
        "a close-up photo of nubuck leather",
        "sanded leather with a velvety finish",
        "fine-grain nubuck texture",
    ],
    "canvas": [
        "a close-up photo of canvas fabric",
        "cotton canvas weave material",
        "cotton duck textile",
    ],
    "mesh": [
        "a close-up photo of engineered mesh fabric",
        "breathable shoe mesh material",
        "open weave synthetic mesh",
    ],
    "knit": [
        "a close-up photo of knit textile material",
        "flyknit fabric texture",
        "knitted shoe upper material",
    ],
    "rubber": [
        "a close-up photo of rubber outsole material",
        "molded rubber tread texture",
        "vulcanized rubber surface",
    ],
    "eva_foam": [
        "a close-up photo of EVA foam midsole material",
        "white EVA foam cushioning",
        "expanded foam midsole texture",
    ],
    "tpu": [
        "a close-up photo of TPU thermoplastic plastic",
        "translucent TPU shoe overlay",
        "molded thermoplastic polyurethane",
    ],
    "plastic": [
        "a close-up photo of hard plastic material",
        "molded plastic shoe component",
        "rigid plastic finish",
    ],
    "metal": [
        "a close-up photo of metal hardware",
        "metallic eyelet or buckle",
        "polished metal finish on shoe",
    ],
    "synthetic_leather": [
        "a close-up photo of synthetic leather",
        "PU leather material",
        "faux leather upholstery",
    ],
}

# Region prior: which material classes are physically plausible for a
# given component name. Anything outside the mask gets zeroed out before
# the softmax — a rubber outsole is never going to be "suede", regardless
# of what CLIP thinks. Keys are matched as substrings of the hint name
# (case-insensitive), falling through to "_upper_" for everything else.
REGION_CLASS_MASK: Dict[str, List[str]] = {
    "outsole": ["rubber", "plastic", "tpu", "metal"],
    "midsole": ["eva_foam", "rubber", "tpu", "plastic"],
    "sole":    ["rubber", "eva_foam", "tpu", "plastic"],
    "lace":    ["synthetic_leather", "canvas", "mesh", "knit"],
    "tongue":  ["mesh", "knit", "canvas", "synthetic_leather", "smooth_leather"],
    "_upper_": ["smooth_leather", "suede", "nubuck", "canvas", "mesh",
                "knit", "synthetic_leather", "plastic", "metal"],
}


MATERIAL_LABELS: Dict[str, str] = {
    "smooth_leather":   "Smooth leather",
    "suede":            "Suede",
    "nubuck":           "Nubuck",
    "canvas":           "Canvas",
    "mesh":             "Engineered mesh",
    "knit":             "Knit textile",
    "rubber":           "Rubber",
    "eva_foam":         "EVA foam",
    "tpu":              "TPU",
    "plastic":          "Plastic",
    "metal":            "Metal",
    "synthetic_leather":"Synthetic leather",
}


# Data classes

@dataclass
class PBRFeatures:
    """Per-component statistics derived from PBR maps. All None if the
    GLB doesn't provide the corresponding map."""
    mean_roughness: Optional[float] = None
    mean_metallic: Optional[float] = None
    normal_gradient_mag: Optional[float] = None  # mean |∇normal|, 0..1
    alpha_variance: Optional[float] = None       # variance in alpha
    base_color_saturation: Optional[float] = None
    base_color_value: Optional[float] = None


@dataclass
class MaterialPrediction:
    """Final per-component material guess."""
    material_class: str             # taxonomy key
    label: str                      # human-readable
    confidence: float               # 0..1
    clip_top3: List[Tuple[str, float]] = field(default_factory=list)
    pbr_class: Optional[str] = None
    pbr_features: Optional[PBRFeatures] = None
    agreement: bool = False         # CLIP and PBR agreed
    needs_review: bool = False      # low confidence or disagreement


# CLIP loader (cached)

_CLIP_CACHE: dict = {}


def _load_clip(model_name: str = "ViT-B-32",
                pretrained: str = "openai"):
    """Load OpenCLIP model once; cache in module globals."""
    if "model" in _CLIP_CACHE:
        return (_CLIP_CACHE["model"], _CLIP_CACHE["preprocess"],
                _CLIP_CACHE["tokenizer"], _CLIP_CACHE["device"])
    import torch
    import open_clip
    device = "cpu"   # deterministic; MPS gives non-deterministic ops
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    # Pre-encode the prompt ensembles, normalized.
    class_features = {}
    with torch.no_grad():
        for cls, prompts in MATERIAL_TAXONOMY.items():
            toks = tokenizer(prompts).to(device)
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cf = feats.mean(dim=0, keepdim=True)
            cf = cf / cf.norm(dim=-1, keepdim=True)
            class_features[cls] = cf
    _CLIP_CACHE["model"] = model
    _CLIP_CACHE["preprocess"] = preprocess
    _CLIP_CACHE["tokenizer"] = tokenizer
    _CLIP_CACHE["device"] = device
    _CLIP_CACHE["class_features"] = class_features
    return model, preprocess, tokenizer, device


# Per-component texture patch extraction

def extract_component_patch(
    face_indices: np.ndarray,
    face_uvs: np.ndarray,
    diffuse: np.ndarray,
    fill_rgb: Optional[Tuple[int, int, int]] = None,
    target_size: int = 224,
) -> Optional[np.ndarray]:
    """Crop the diffuse texture to the UV bbox of the component, mask out
    pixels not covered by any face in the component, fill the masked
    background with `fill_rgb` (defaults to mean of kept pixels), and
    resize to target_size square.

    `face_uvs`: (Nf, 3, 2) per-face UV coordinates.

    Returns None if the component has no UV coverage.
    """
    if face_uvs is None or diffuse is None or len(face_indices) == 0:
        return None
    H, W = diffuse.shape[:2]
    # UVs of all faces in the component, shape (Nf_comp, 3, 2).
    comp_face_uvs = face_uvs[face_indices]
    if comp_face_uvs.size == 0:
        return None
    u = comp_face_uvs[..., 0]
    v = comp_face_uvs[..., 1]
    u_min, u_max = float(np.clip(u.min(), 0, 1)), float(np.clip(u.max(), 0, 1))
    v_min, v_max = float(np.clip(v.min(), 0, 1)), float(np.clip(v.max(), 0, 1))
    if u_max - u_min < 1e-4 or v_max - v_min < 1e-4:
        return None
    # PIL/numpy image: y=0 at top; UV v=0 at bottom in glTF convention.
    x0 = int(np.floor(u_min * (W - 1)))
    x1 = int(np.ceil(u_max * (W - 1)))
    y0 = int(np.floor((1 - v_max) * (H - 1)))
    y1 = int(np.ceil((1 - v_min) * (H - 1)))
    x0, x1 = max(0, x0), min(W, x1 + 1)
    y0, y1 = max(0, y0), min(H, y1 + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    # Build a mask within the crop window — which pixels are actually
    # covered by a face in this component? Rasterize each UV triangle.
    from PIL import ImageDraw
    cw, ch = x1 - x0, y1 - y0
    mask_im = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask_im)
    for tri in comp_face_uvs:
        pts = []
        for (uu, vv) in tri:
            px = uu * (W - 1) - x0
            py = (1 - vv) * (H - 1) - y0
            pts.append((px, py))
        draw.polygon(pts, fill=255)
    mask = np.array(mask_im) > 0  # (ch, cw) bool
    if not mask.any():
        return None

    crop = diffuse[y0:y1, x0:x1, :3].astype(np.uint8)
    # Find the tightest bbox within the crop where the mask is true, then
    # build a 224×224 patch by tiling kept pixels (no fill area diluting
    # the texture). This gives CLIP a uniformly-textured patch even when
    # the component's UV island is small or non-rectangular.
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    mx0, mx1 = xs.min(), xs.max() + 1
    my0, my1 = ys.min(), ys.max() + 1
    sub_crop = crop[my0:my1, mx0:mx1]
    sub_mask = mask[my0:my1, mx0:mx1]
    kept_pixels = sub_crop[sub_mask]
    if len(kept_pixels) < 16:
        return None
    # Strategy: if the sub-crop has at least 70% coverage, use it directly
    # (preserves real texture structure). Otherwise tile kept pixels into
    # a 224×224 grid (loses spatial structure but keeps the texture's
    # color statistics intact — better than filling with one color).
    coverage = float(sub_mask.mean())
    # Prefer direct spatial crop when coverage is moderate or better —
    # texture structure (weave pattern, brushed nap) is informative even
    # with some background fill. Use the mean of kept pixels as the fill
    # color so CLIP doesn't latch onto an arbitrary background colour.
    if coverage >= 0.4 and min(sub_crop.shape[:2]) >= 32:
        fill = tuple(int(x) for x in kept_pixels.mean(axis=0))
        filled = sub_crop.copy()
        filled[~sub_mask] = fill
        img = Image.fromarray(filled)
        img = img.resize((target_size, target_size), Image.LANCZOS)
        return np.array(img)
    # Tile kept pixels: arrange them in a square layout sampled from the
    # kept set. Seeded for determinism.
    rng = np.random.default_rng(0)
    n_needed = target_size * target_size
    idx = rng.integers(0, len(kept_pixels), size=n_needed)
    tiled = kept_pixels[idx].reshape(target_size, target_size, 3)
    return tiled.astype(np.uint8)


# CLIP zero-shot

def _allowed_classes(hint_name: str) -> List[str]:
    """Return the list of plausible material classes for a component
    based on its positional name. Defaults to the 'upper' region."""
    if not hint_name:
        return list(MATERIAL_TAXONOMY.keys())
    h = hint_name.lower()
    for key, allowed in REGION_CLASS_MASK.items():
        if key != "_upper_" and key in h:
            return allowed
    return REGION_CLASS_MASK["_upper_"]


def clip_classify(patch: np.ndarray, hint_name: str = "") -> Dict[str, float]:
    """Return softmax probabilities over MATERIAL_TAXONOMY for a single
    224×224 RGB patch, masked by the region prior implied by hint_name.
    Classes outside the region mask get zero probability."""
    import torch
    model, preprocess, _tok, device = _load_clip()
    class_features = _CLIP_CACHE["class_features"]
    img = Image.fromarray(patch.astype(np.uint8))
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        scores = {}
        for cls, cf in class_features.items():
            s = float((feats @ cf.T).item())
            scores[cls] = s
    # CLIP's standard logit scaling is exp(100 * sim).
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=np.float64) * 100.0
    # Apply region mask before softmax: classes outside the mask get -inf.
    allowed = set(_allowed_classes(hint_name))
    mask = np.array([1.0 if k in allowed else -1e9 for k in keys])
    vals = vals + (mask - mask.max() + (mask == -1e9).astype(np.float64) * -1e9)
    # The above is a safer way to write: "leave allowed scores untouched,
    # disallowed → -inf so softmax → 0". Simplify:
    vals = np.where(np.array([k in allowed for k in keys]), vals, -1e9)
    vals -= vals.max()
    probs = np.exp(vals)
    probs /= probs.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


# PBR-feature heuristic classifier

def pbr_features_for_component(
    face_indices: np.ndarray,
    face_uvs: Optional[np.ndarray],
    roughness_map: Optional[np.ndarray],
    metallic_map: Optional[np.ndarray],
    normal_map: Optional[np.ndarray],
    diffuse: Optional[np.ndarray],
) -> PBRFeatures:
    """Compute per-component PBR statistics by averaging the relevant
    texture map(s) over the component's UV footprint.

    `face_uvs`: (Nf, 3, 2) per-face UV coordinates."""
    f = PBRFeatures()
    if face_uvs is None or diffuse is None or len(face_indices) == 0:
        return f
    # Per-face UV centroid → one sample per face.
    face_uv = face_uvs[face_indices].mean(axis=1)
    if roughness_map is not None:
        f.mean_roughness = _sample_map(face_uv, roughness_map, channel=1) / 255.0
    if metallic_map is not None:
        f.mean_metallic = _sample_map(face_uv, metallic_map, channel=2) / 255.0
    if normal_map is not None:
        f.normal_gradient_mag = _normal_gradient(face_uv, normal_map)
    if diffuse is not None:
        rgb = _sample_map_rgb(face_uv, diffuse)
        f.base_color_saturation = _saturation(rgb)
        f.base_color_value = float(rgb.max(axis=-1).mean()) / 255.0
        if diffuse.shape[2] == 4:
            alphas = _sample_map(face_uv, diffuse, channel=3) / 255.0
            f.alpha_variance = float(np.var(alphas))
    return f


def _sample_map(face_uv: np.ndarray, tex: np.ndarray, channel: int = 0) -> float:
    H, W = tex.shape[:2]
    u = np.clip(face_uv[:, 0], 0, 1)
    v = np.clip(face_uv[:, 1], 0, 1)
    px = (u * (W - 1)).astype(np.int32)
    py = ((1 - v) * (H - 1)).astype(np.int32)
    return float(tex[py, px, channel].astype(np.float64).mean())


def _sample_map_rgb(face_uv: np.ndarray, tex: np.ndarray) -> np.ndarray:
    H, W = tex.shape[:2]
    u = np.clip(face_uv[:, 0], 0, 1)
    v = np.clip(face_uv[:, 1], 0, 1)
    px = (u * (W - 1)).astype(np.int32)
    py = ((1 - v) * (H - 1)).astype(np.int32)
    return tex[py, px, :3].astype(np.float64)


def _saturation(rgb: np.ndarray) -> float:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(sat.mean())


def _normal_gradient(face_uv: np.ndarray, tex: np.ndarray) -> float:
    """Mean |∇normal| over the component's UV footprint, normalized to 0..1.
    A proxy for surface texture frequency: low for smooth leather, high
    for mesh / knit / suede."""
    H, W = tex.shape[:2]
    # Sobel on each channel of the normal map.
    gx = np.gradient(tex[..., :3].astype(np.float32), axis=1)
    gy = np.gradient(tex[..., :3].astype(np.float32), axis=0)
    mag = np.sqrt((gx ** 2 + gy ** 2).sum(axis=-1))  # (H, W)
    u = np.clip(face_uv[:, 0], 0, 1)
    v = np.clip(face_uv[:, 1], 0, 1)
    px = (u * (W - 1)).astype(np.int32)
    py = ((1 - v) * (H - 1)).astype(np.int32)
    samples = mag[py, px]
    # Normalize to 0..1 by an empirical scale (max possible per-pixel diff = 255).
    return float(np.clip(samples.mean() / 64.0, 0, 1))


def pbr_heuristic_classify(f: PBRFeatures, hint_name: str) -> Tuple[Optional[str], float]:
    """Rule-based classifier on PBR features. Returns (class, confidence)
    or (None, 0) if there's not enough signal.

    `hint_name` is the component's positional name (e.g. "outsole",
    "midsole") used as a weak prior — the spatial segmenter already
    knows which region this is, so we lean on it where the PBR signal
    is ambiguous.
    """
    rough = f.mean_roughness
    metal = f.mean_metallic
    grad = f.normal_gradient_mag

    # If we have no PBR maps at all, fall back to spatial prior only.
    if rough is None and metal is None and grad is None:
        if hint_name == "outsole":
            return "rubber", 0.7
        if hint_name == "midsole":
            return "eva_foam", 0.7
        return None, 0.0

    # Hard cuts first.
    if metal is not None and metal > 0.55:
        return "metal", min(0.9, metal)

    # Rubber: very low metallic, mid-to-high roughness, low normal frequency.
    if (rough is not None and rough > 0.6 and (metal is None or metal < 0.1)
            and (grad is None or grad < 0.3)):
        if hint_name == "outsole":
            return "rubber", 0.85
        return "rubber", 0.65

    # EVA foam: high roughness, low metallic, fairly uniform normal.
    if (rough is not None and rough > 0.7 and (metal is None or metal < 0.1)
            and (grad is None or grad < 0.2)):
        if hint_name == "midsole":
            return "eva_foam", 0.85
        return "eva_foam", 0.6

    # Mesh / knit: moderate roughness, high normal-gradient (texture-y).
    if grad is not None and grad > 0.5 and (metal is None or metal < 0.2):
        # Mesh and knit are CLIP-distinguishable; here just signal "textile".
        return "mesh", 0.55

    # Smooth leather: low roughness, low metallic, low normal-grad.
    if (rough is not None and rough < 0.4 and (metal is None or metal < 0.1)
            and (grad is None or grad < 0.2)):
        return "smooth_leather", 0.6

    # Suede / nubuck: mid roughness, mid-to-high normal-grad.
    if (rough is not None and 0.4 <= rough <= 0.75
            and grad is not None and 0.25 <= grad <= 0.55):
        return "suede", 0.5

    # Plastic/TPU: low roughness, low normal-grad, but glossier than leather.
    if (rough is not None and rough < 0.3 and (metal is None or metal < 0.15)):
        return "plastic", 0.5

    return None, 0.0


# Fusion

def infer_material(
    face_indices: np.ndarray,
    face_uvs: Optional[np.ndarray],
    diffuse: Optional[np.ndarray],
    roughness_map: Optional[np.ndarray] = None,
    metallic_map: Optional[np.ndarray] = None,
    normal_map: Optional[np.ndarray] = None,
    fill_rgb: Optional[Tuple[int, int, int]] = None,
    hint_name: str = "",
    use_clip: bool = True,
) -> MaterialPrediction:
    """Run the full per-component pipeline and return one prediction.

    `face_uvs`: (Nf, 3, 2) per-face UV coordinates.
    `hint_name`: positional-segmentation name ("outsole", "midsole",
        "toe-cap", ...). Used as a weak prior for the PBR heuristic and
        a tie-breaker on disagreement.
    """
    # PBR
    pbr = pbr_features_for_component(
        face_indices, face_uvs, roughness_map, metallic_map, normal_map,
        diffuse,
    )
    pbr_cls, pbr_conf = pbr_heuristic_classify(pbr, hint_name)

    # CLIP
    clip_probs: Dict[str, float] = {}
    if use_clip:
        patch = extract_component_patch(face_indices, face_uvs, diffuse,
                                         fill_rgb=fill_rgb)
        if patch is not None:
            clip_probs = clip_classify(patch, hint_name=hint_name)

    # Fusion
    if clip_probs:
        ranked = sorted(clip_probs.items(), key=lambda kv: -kv[1])
        clip_top3 = ranked[:3]
        clip_cls, clip_conf = ranked[0]
        if pbr_cls is None:
            chosen, conf, agreement = clip_cls, clip_conf, False
        elif clip_cls == pbr_cls:
            chosen = clip_cls
            conf = min(1.0, 0.5 * clip_conf + 0.5 * pbr_conf + 0.1)  # boost on agreement
            agreement = True
        else:
            # Disagree → trust CLIP (sees real texture), but downweight.
            chosen = clip_cls
            conf = clip_conf * 0.75
            agreement = False
    else:
        clip_top3 = []
        if pbr_cls is not None:
            chosen, conf = pbr_cls, pbr_conf
        else:
            chosen, conf = "smooth_leather", 0.0  # neutral default
        agreement = False

    return MaterialPrediction(
        material_class=chosen,
        label=MATERIAL_LABELS.get(chosen, chosen),
        confidence=float(conf),
        clip_top3=[(MATERIAL_LABELS.get(k, k), float(v)) for k, v in clip_top3],
        pbr_class=MATERIAL_LABELS.get(pbr_cls) if pbr_cls else None,
        pbr_features=pbr,
        agreement=agreement,
        needs_review=(conf < 0.35) or (clip_probs and pbr_cls and not agreement),
    )
