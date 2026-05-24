"""End-to-end TechPack pipeline.

Usage:
    python -m src.pipeline --glb path/to/shoe.glb --out output/ \\
        --model-id NB574-001 --model-name "New Balance 574"

Writes 6 PDFs (cover, views, BOM, colorway, construction, techdrawings)
plus a _renders/ subfolder. Deterministic by default: same input + same
--seed + same --out path = byte-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from .colorway import (component_color_entries,
                        component_palette_from_face_uvs, extract_palette)
from .material import MaterialPrediction, infer_material
try:
    from .segmentation_sam import refine_segmentation as _sam_refine
except Exception:
    _sam_refine = None
try:
    from .anatomy_detect import (detect_anatomy as _anatomy_detect,
                                  detection_to_callout as _det_to_callout,
                                  display_for as _ml_display_for)
except Exception:
    _anatomy_detect = None
    _det_to_callout = None
    _ml_display_for = None
from .geometry import GeometryAnalysis, analyze_geometry
from .pdf import (Callout, TechPackHeader, _build_callouts_from_mesh,
                  write_bom, write_colorway, write_views)
from .render_blender import (BlenderRenderConfig, extract_face_data,
                              render_all_views, render_component_overlay,
                              render_exploded_overlay)
from .techdrawings import (TechDrawConfig, render_exploded_view,
                            render_lineart_views, render_section_view,
                            write_techdrawings)


def _distinct_palette(n: int) -> list:
    """Generate n visually distinct RGB colours (0..1 floats) via HSL
    hue-rotation. Used for the BOM diagram so every segmented component
    reads as a different colour even when their actual texture colours
    are all near-grey (as on a worn 574)."""
    import colorsys
    out = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1) + 0.05) % 1.0
        # Vary saturation slightly per index so adjacent hues are
        # distinguishable even when shrunken on the page.
        s = 0.55 + 0.10 * ((i % 3) - 1) / 1.0
        l = 0.55
        out.append(colorsys.hls_to_rgb(h, l, s))
    return out


# Canonical-component-name -> friendly display label used on the new BOM
# parts-anatomy infographic. Falls back to the canonical name if missing.
_DISPLAY_NAME_MAP = {
    "rubber-outsole":  "Outsole",
    "eva-midsole":     "Midsole",
    "shoe-laces":      "Laces",
    "vamp":            "Toe Box (Vamp)",
    "mudguard":        "Mudguard (Toe Cap)",
    "quarter-lateral": "Quarter Panel (Lateral)",
    "quarter-medial":  "Quarter Panel (Medial)",
    "heel-counter":    "Heel Counter",
    "heel-tab":        "Heel Tab",
    "collar":          "Heel Collar (Topline)",
    "tongue":          "Tongue",
    "eyestay":         "Eyestay",
}

_DESCRIPTION_MAP = {
    "rubber-outsole":  "Bottom layer that grips the ground.",
    "eva-midsole":     "Foam layer that provides cushioning.",
    "shoe-laces":      "Lacing that secures the upper.",
    "vamp":            "Front upper that covers the toes.",
    "mudguard":        "Overlay that protects the toe area.",
    "quarter-lateral": "Outboard side panel of the upper.",
    "quarter-medial":  "Inboard side panel of the upper.",
    "heel-counter":    "Stiff support that holds the heel in place.",
    "heel-tab":        "Pull tab at the back for easy wear.",
    "collar":          "Padded area around the ankle.",
    "tongue":          "The padded flap under the laces.",
    "eyestay":         "Reinforced panel around the eyelets.",
}


# Palette used to flat-color components in the overlay render.
COMPONENT_PALETTE = [
    (0.77, 0.35, 0.07),   # ochre
    (0.34, 0.60, 0.76),   # steel blue
    (0.42, 0.62, 0.28),   # green
    (0.75, 0.25, 0.25),   # red
    (0.54, 0.38, 0.69),   # purple
    (0.89, 0.67, 0.16),   # yellow
    (0.45, 0.45, 0.45),   # gray
    (0.24, 0.60, 0.60),   # teal
    (0.83, 0.43, 0.62),   # pink
]


# Adapter: Blender face-data dict -> trimesh.Trimesh proxy for analyze_geometry

@dataclass
class _MeshProxy:
    """Minimal Trimesh-shaped object used by `geometry.analyze_geometry`.

    `geometry.analyze_geometry` reads only:
      - mesh.vertices (Nv, 3)
      - mesh.faces (Nf, 3)        (vertex indices)
      - mesh.face_normals
      - mesh.area_faces
      - mesh.extents
      - mesh.face_normals
    """
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    area_faces: np.ndarray

    @property
    def extents(self) -> np.ndarray:
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)


def _build_mesh_proxy(face_data: dict) -> _MeshProxy:
    """Trimesh-shaped proxy backed by real Blender topology when available.

    Previous versions used face centroids as fake vertices with degenerate
    (i, i, i) faces; that broke any code path needing real edge topology
    (e.g., per-component perimeter). When `face_v_idx` is present we use
    real vertices + faces, which is correct for every existing consumer
    (face centroids are still derivable via mesh.vertices[mesh.faces].mean()).
    """
    if "face_v_idx" in face_data and "vertices_mm" in face_data:
        return _MeshProxy(
            vertices=face_data["vertices_mm"],
            faces=face_data["face_v_idx"],
            face_normals=face_data["face_normals"],
            area_faces=face_data["face_areas_mm2"],
        )
    # Legacy fallback (no per-face vertex indices exposed).
    centroids = face_data["face_centroids_mm"]
    n = len(centroids)
    faces = np.stack([np.arange(n), np.arange(n), np.arange(n)], axis=1)
    return _MeshProxy(
        vertices=centroids,
        faces=faces,
        face_normals=face_data["face_normals"],
        area_faces=face_data["face_areas_mm2"],
    )


def _analyze_via_blender(face_data: dict, n_upper_clusters: int, seed: int) -> GeometryAnalysis:
    """Run the geometry analyzer on Blender-extracted face data."""
    # geometry.analyze_geometry expects: mesh, uv, diffuse_image.
    # We've already sampled face_colors during extraction, so pass uv=None,
    # diffuse=None and patch the colors into the analyzer afterwards.

    # The analyzer relies on `sample_face_colors(mesh, uv, diffuse)` only when
    # both uv and diffuse are present. With None it falls back to gray.
    # We need to bypass that and inject our already-sampled colors. Do this by
    # calling the analyzer's internals directly.
    from .geometry import (Component, GeometryAnalysis, _build_component,
                            _classify_upper_subcluster, _kmeans_pos_color,
                            _measure, _spatial_labels, smooth_face_labels)
    mesh = _build_mesh_proxy(face_data)
    face_colors = face_data["face_colors_rgb"]
    areas = face_data["face_areas_mm2"]
    centroids = face_data["face_centroids_mm"]

    spatial = _spatial_labels(mesh)
    extents = tuple(mesh.extents.tolist())

    components: list = []
    face_component = np.full(len(mesh.faces), -1, dtype=np.int32)

    def _push(comp):
        if comp is None:
            return
        ci = len(components)
        components.append(comp)
        face_component[comp.face_indices] = ci

    _push(_build_component("outsole", "rubber outsole",
                            np.where(spatial == 0)[0], mesh, face_colors, areas))
    _push(_build_component("midsole", "EVA foam midsole",
                            np.where(spatial == 1)[0], mesh, face_colors, areas))

    upper_idx = np.where(spatial == 2)[0]
    if len(upper_idx) > 50:
        sub_centroids = centroids[upper_idx]
        sub_colors = face_colors[upper_idx]
        sub_areas = areas[upper_idx]
        upper_z_lo = float(sub_centroids[:, 2].min())
        upper_z_hi = float(sub_centroids[:, 2].max())
        labels, centers = _kmeans_pos_color(
            sub_centroids, sub_colors, sub_areas, n_upper_clusters, seed,
        )
        bucket: dict = {}
        bucket_material: dict = {}
        for ci in range(n_upper_clusters):
            sub = np.where(labels == ci)[0]
            if len(sub) == 0:
                continue
            global_idx = upper_idx[sub]
            c = centroids[global_idx]
            cent = tuple(np.average(c, axis=0, weights=areas[global_idx]).tolist())
            name, mat = _classify_upper_subcluster(
                tuple(int(x) for x in centers[ci]), cent, extents,
                upper_z_lo, upper_z_hi,
            )
            bucket.setdefault(name, []).append(global_idx)
            bucket_material.setdefault(name, mat)
        for name, idx_list in bucket.items():
            merged = np.concatenate(idx_list)
            _push(_build_component(
                name, bucket_material[name], merged, mesh, face_colors, areas,
                note=("merged from {} sub-clusters".format(len(idx_list))
                      if len(idx_list) > 1 else ""),
            ))

    # Smooth cluster boundaries with face-neighbour majority vote. Each face
    # is replaced by the modal label among its k spatially-nearest neighbours
    # for a few iterations, which turns the jagged k-means assignment into
    # clean curves that follow real geometric component edges.
    face_component = smooth_face_labels(centroids, face_component,
                                         k=10, iterations=4)

    # Some components may have lost all their faces during smoothing
    # (small clusters absorbed by neighbours). Rebuild Component objects
    # from the smoothed labels.
    rebuilt: list = []
    new_face_component = np.full_like(face_component, -1)
    for old_ci, comp in enumerate(components):
        idx = np.where(face_component == old_ci)[0]
        if len(idx) < 5:
            continue
        new_comp = _build_component(comp.name, comp.inferred_material, idx,
                                     mesh, face_colors, areas, note=comp.note)
        if new_comp is None:
            continue
        new_face_component[idx] = len(rebuilt)
        rebuilt.append(new_comp)
    components = rebuilt
    face_component = new_face_component

    # Sort by area, largest first; remap face_component.
    order = sorted(range(len(components)), key=lambda i: -components[i].area_mm2)
    components = [components[i] for i in order]
    remap = {old: new for new, old in enumerate(order)}
    face_component = np.array([remap.get(int(c), -1) for c in face_component], dtype=np.int32)

    measurements = _measure(mesh, components)
    return GeometryAnalysis(
        components=components,
        measurements=measurements,
        face_colors_rgb=face_colors,
        face_component=face_component,
    )


# Header

def _rebuild_components_from_face_component(
    face_component: np.ndarray, face_data: dict,
    original_components: list, new_names: list,
    new_materials: Optional[list] = None,
):
    """Rebuild Component objects from a refined face_component array (e.g.
    after SAM 2 adds sub-components on the upper).

    Returns (new_components, renumbered_face_component) where the face
    indices in the returned array correspond to the new component order
    (sorted by area descending).
    """
    from .geometry import _build_component
    proxy = _build_mesh_proxy(face_data)
    face_colors = face_data["face_colors_rgb"]
    face_areas = face_data["face_areas_mm2"]
    all_names = [c.name for c in original_components] + list(new_names)
    all_materials = [c.inferred_material for c in original_components]
    if new_materials is None:
        new_materials = ["TBD"] * len(new_names)
    all_materials += list(new_materials)
    rebuilt = []
    for ci in range(len(all_names)):
        face_idx = np.where(face_component == ci)[0]
        if len(face_idx) < 5:
            continue
        comp = _build_component(
            all_names[ci], all_materials[ci],
            face_idx, proxy, face_colors, face_areas,
        )
        if comp is not None:
            rebuilt.append(comp)
    rebuilt.sort(key=lambda c: -c.area_mm2)
    # Renumber face_component to match the new (post-sort) order.
    name_to_idx = {c.name: i for i, c in enumerate(rebuilt)}
    new_fc = np.full_like(face_component, -1)
    for old_ci, name in enumerate(all_names):
        ni = name_to_idx.get(name)
        if ni is None:
            continue
        new_fc[face_component == old_ci] = ni
    return rebuilt, new_fc


def _infer_material_for_new_component(name: str) -> str:
    """Quick material guess for SAM-discovered sub-components. Used as a
    seed for the BOM table; the CLIP-based material classifier runs
    afterwards and supersedes this if it disagrees."""
    n = name.lower()
    if "tongue" in n:
        return "padded synthetic mesh"
    if "collar" in n:
        return "foam-padded textile collar"
    if "eyestay" in n or "eyelet" in n:
        return "synthetic-leather overlay"
    if "mudguard" in n:
        return "rubber / synthetic overlay"
    if "heel-tab" in n or "pull" in n:
        return "synthetic-leather pull-tab"
    if "toe-overlay" in n:
        return "synthetic-leather toe overlay"
    return "TBD"


def _make_header(source_path: str, model_id: str, model_name: str,
                 designer: str, factory: str, season: str) -> TechPackHeader:
    import datetime as _dt
    h = hashlib.sha256()
    with open(source_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return TechPackHeader(
        model_id=model_id, model_name=model_name,
        date_iso=_dt.date.today().isoformat(),
        source_file=source_path, source_hash=h.hexdigest(),
        designer=designer, factory=factory, season=season,
    )


# Run

def run(
    glb_path: str,
    output_dir: str,
    model_id: str,
    model_name: str,
    designer: str = "—",
    factory: str = "—",
    season: str = "—",
    seed: int = 0,
    n_upper_clusters: int = 7,
    n_palette_colors: int = 6,
    render_width: int = 1100,
    render_height: int = 800,
    samples: int = 48,
    deterministic: bool = True,
    target_length_mm: float = 270.0,
    verbose: bool = True,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Fix reportlab's PDF creation date to a constant so the PDF bytes are
    # deterministic. The exact epoch doesn't matter — only that it's fixed.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    os.makedirs(output_dir, exist_ok=True)
    # Intermediate PNG renders go to rough/_renders/<shoe>/ (gitignored)
    # — they're build artifacts, not deliverables, so they don't belong
    # next to the PDFs the interviewer / factory consumer opens. The
    # project root is two levels up from this file (src/pipeline.py).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    renders_dir = os.path.join(project_root, "rough", "_renders",
                                os.path.basename(os.path.abspath(output_dir)))
    os.makedirs(renders_dir, exist_ok=True)
    # Delete any stale PDFs from a previous run before writing fresh ones.
    # The _renders/ cache uses deterministic filenames and is safe to
    # overwrite in-place, so we don't touch it.
    import glob as _glob
    _stale = sorted(_glob.glob(os.path.join(output_dir, "*.pdf")))
    if _stale:
        print(f"cleaning {len(_stale)} stale PDF(s) from {output_dir}")
        for _p in _stale:
            try:
                os.unlink(_p)
            except OSError as _e:
                print(f"  warning: could not delete {_p}: {_e}")

    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    log(f"[1/6] loading + extracting face data (Blender)")
    log(f"      target_length_mm = {target_length_mm} (user-provided reference)")
    face_data = extract_face_data(glb_path, target_length_mm=target_length_mm)
    log(f"      length={face_data['length_mm']:.0f}mm width={face_data['width_mm']:.0f}mm "
        f"height={face_data['height_mm']:.0f}mm (1 BU = {1/face_data['scale_mm_per_unit']:.4f} m)")

    log(f"[2/6] segmenting + measuring (heuristic baseline)")
    ga = _analyze_via_blender(face_data, n_upper_clusters=n_upper_clusters, seed=seed)
    log(f"      {len(ga.components)} components: " +
        ", ".join(c.name for c in ga.components))

    mode = "deterministic" if deterministic else "fast"
    log(f"[3/6] rendering views (Cycles CPU, {render_width}x{render_height}, {samples} samples, {mode})")
    cfg = BlenderRenderConfig(width=render_width, height=render_height,
                               samples=samples, deterministic=deterministic)
    view_paths, am, ext, view_cameras = render_all_views(
        glb_path, renders_dir, cfg, verbose=False)
    for v in view_paths:
        log(f"      {v}")

    # SAM 2 refinement on the upper. Loads weights once (~155 MB, cached),
    # then projects every face centroid through the captured camera matrix
    # for each rendered view and promotes any SAM mask that lies inside an
    # existing upper component to a new sub-component (tongue, collar, …).
    if _sam_refine is not None and face_data.get("face_centroids_world") is not None:
        log(f"      SAM 2 segmentation refinement (sam2-hiera-small)")
        try:
            refined_fc, new_names = _sam_refine(
                ga.face_component,
                face_data["face_centroids_world"],
                face_data["face_centroids_mm"],
                view_paths, view_cameras,
                [c.name for c in ga.components],
                target_views=("side-lateral", "side-medial"),
                verbose=True,
            )
            if new_names:
                new_mats = [_infer_material_for_new_component(n) for n in new_names]
                new_components, refined_fc = _rebuild_components_from_face_component(
                    refined_fc, face_data, ga.components, new_names, new_mats,
                )
                from .geometry import GeometryAnalysis as _GA
                from .geometry import _measure as _measure_dims
                proxy = _build_mesh_proxy(face_data)
                new_measurements = _measure_dims(proxy, new_components)
                ga = _GA(
                    components=new_components,
                    measurements=new_measurements,
                    face_colors_rgb=ga.face_colors_rgb,
                    face_component=refined_fc,
                )
                log(f"      refined: {len(ga.components)} components — added "
                    + ", ".join(new_names))
            else:
                log(f"      no new sub-components found")
        except Exception as e:
            log(f"      SAM refinement skipped: {e}")

    # Canonicalise every component name to the industry vocabulary, so the
    # BOM table reads "rubber-outsole / eva-midsole / shoe-laces / vamp /
    # quarter-lateral / quarter-medial / heel-counter …" instead of the
    # heuristic naming. SAM-discovered names are already canonical.
    try:
        from .segmentation_sam import canonicalize_name as _canonicalize_name
        for c in ga.components:
            c.name = _canonicalize_name(c.name)
        log(f"      canonical component names: "
            + ", ".join(c.name for c in ga.components))
    except Exception as e:
        log(f"      canonicalisation skipped: {e}")

    log(f"[4/6] extracting colorway palette + classifying materials")
    palette = extract_palette(face_data["diffuse_image"],
                              n_colors=n_palette_colors, seed=seed)
    component_swatches = list(zip([c.name for c in ga.components],
                                   component_color_entries(ga.components)))

    # Per-component mini-palette (3 colors each) — better than a single
    # area-weighted mean for components with mottled materials like suede.
    face_uvs = face_data.get("face_uvs")
    component_palettes = []
    for comp in ga.components:
        cp = component_palette_from_face_uvs(
            comp.face_indices, face_uvs, face_data["diffuse_image"],
            n_colors=3, seed=seed,
        )
        component_palettes.append(cp)

    # Material classification per component (CLIP zero-shot + PBR heuristic).
    log(f"      material inference (OpenCLIP ViT-B/32, CPU)")
    materials: list = []
    for comp in ga.components:
        pred = infer_material(
            face_indices=comp.face_indices,
            face_uvs=face_uvs,
            diffuse=face_data["diffuse_image"],
            fill_rgb=tuple(int(x) for x in comp.dominant_color_rgb),
            hint_name=comp.name,
            use_clip=True,
        )
        log(f"        {comp.name:20s} → {pred.label} (conf={pred.confidence:.2f}"
            + (f", review" if pred.needs_review else "") + ")")
        materials.append(pred)

    log(f"      component overlay (dominant colors — used by colorway page)")
    # Dominant-color overlay for the colorway-callout page (faithful to the
    # real texture colors of each component).
    palette_rgb = []
    for i, comp in enumerate(ga.components):
        rgb = getattr(comp, "dominant_color_rgb", None)
        if rgb is None:
            palette_rgb.append(COMPONENT_PALETTE[i % len(COMPONENT_PALETTE)])
        else:
            palette_rgb.append((rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0))
    overlay_path = os.path.join(renders_dir, "component_overlay.png")
    render_component_overlay(glb_path, overlay_path,
                              ga.face_component, palette_rgb, cfg)

    # Distinct categorical palette (HSL hue rotation) so the BOM diagram
    # *visually* segments the shoe — on a 574 every dominant colour is
    # near-grey and the components blend into a uniform white.
    bom_palette_rgb = _distinct_palette(len(ga.components))
    log(f"      component id-mask ({len(ga.components)} distinct hues, emissive)")
    id_mask_path = os.path.join(renders_dir, "component_id_mask.png")
    # Emissive shader → rendered pixel colours match the input palette
    # exactly, so the centroid detector can use a tight tolerance.
    render_component_overlay(glb_path, id_mask_path,
                              ga.face_component, bom_palette_rgb, cfg,
                              use_emissive=True, view="side-lateral")
    # Medial id-mask: needed so the new parts-anatomy infographic can
    # place callouts on the medial photo render too (same framing as
    # view_side-medial.png).
    id_mask_medial_path = os.path.join(renders_dir, "component_id_mask_medial.png")
    render_component_overlay(glb_path, id_mask_medial_path,
                              ga.face_component, bom_palette_rgb, cfg,
                              use_emissive=True, view="side-medial")

    log(f"      exploded overlay (BOM-diagram distinct palette)")
    component_names = [c.name for c in ga.components]
    exploded_overlay_path = os.path.join(renders_dir, "exploded_overlay.png")
    _, _, exploded_meta = render_exploded_overlay(
        glb_path, exploded_overlay_path,
        ga.face_component, component_names, bom_palette_rgb, cfg,
    )
    # Both id-masks are now the same distinct palette; just point at the
    # already-rendered exploded overlay rather than rendering again.
    exploded_id_mask_path = exploded_overlay_path

    log(f"[5/5] writing PDFs")
    header = _make_header(glb_path, model_id, model_name, designer, factory, season)
    views_path = write_views(os.path.join(output_dir, "01_views.pdf"),
                              header, view_paths)

    # write_bom expects a NormalizedScene-shaped object for legacy reasons; we
    # only use `mesh.vertices` for the spec drawing label so we can synthesise a
    # tiny stand-in.
    @dataclass
    class _SceneShim:
        length_mm: float
        width_mm: float
        height_mm: float
        scale_mm_per_unit: float
    scene_shim = _SceneShim(
        length_mm=face_data["length_mm"],
        width_mm=face_data["width_mm"],
        height_mm=face_data["height_mm"],
        scale_mm_per_unit=face_data["scale_mm_per_unit"],
    )
    # Render the lateral line-art BEFORE write_bom so the BOM page-1
    # annotation diagram can use it as the background. Match the id-mask
    # resolution (and therefore framing) exactly so centroid fractions
    # from the id-mask map directly to dot positions on the line-art.
    log(f"      line-art lateral view (for BOM diagram + tech drawings)")
    td_cfg = TechDrawConfig(width=cfg.width, height=cfg.height)
    lineart_paths, _, _ = render_lineart_views(glb_path, renders_dir, td_cfg)

    # ---- Build the unified anatomy/colorway callout payload ----
    # Mesh path always runs (uses existing per-component centroids from
    # the id-mask renders). ML path opportunistically adds fine parts
    # (eyelets, n-logo, tongue-label, foxing) when transformers+torch
    # are installed.
    lateral_render = view_paths.get("side-lateral")
    medial_render = view_paths.get("side-medial")
    have_renders = (lateral_render and medial_render
                    and os.path.exists(lateral_render)
                    and os.path.exists(medial_render))

    anatomy_payload = None
    colorway_anatomy_payload = None
    # Initialise so the tech-sheet write below has a defined name even
    # when have_renders is False.
    vibrant_palette: list = []
    if have_renders:
        log(f"      mesh callouts (id-mask centroids, "
            f"{len(ga.components)} components)")
        mesh_callouts = _build_callouts_from_mesh(
            ga,
            lateral_id_mask_path=id_mask_path,
            medial_id_mask_path=id_mask_medial_path,
            id_palette=bom_palette_rgb,
            materials=materials,
            display_name_overrides=_DISPLAY_NAME_MAP,
            description_overrides=_DESCRIPTION_MAP,
            component_palettes=component_palettes,
        )

        # Build the vibrant palette: most-saturated swatch per component,
        # deduplicated by ~3-bit quantised hex, sorted by saturation desc.
        # This shows the user the blues, oranges, and accents that
        # `extract_palette`'s area-weighted K-means lumps into the dominant
        # grays.
        try:
            from .colorway import rgb_to_hsv_s as _hsv_s
        except Exception:
            _hsv_s = None
        vibrant_palette = []
        seen_keys = set()
        if _hsv_s is not None and component_palettes:
            ranked = []
            for swatches in component_palettes:
                cands = [e for e in (swatches or [])
                         if getattr(e, "fraction", 0.0) >= 0.05]
                if not cands and swatches:
                    cands = list(swatches)
                if not cands:
                    continue
                best = max(cands, key=lambda e: _hsv_s(e.rgb))
                ranked.append(best)
            ranked.sort(key=lambda e: -_hsv_s(e.rgb))
            for e in ranked:
                # Quantise to 3-bit per channel for dedup so visually
                # identical colours collapse.
                key = tuple((c // 32) for c in e.rgb)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                vibrant_palette.append(e)
                if len(vibrant_palette) >= 8:
                    break
        log(f"      vibrant palette: {len(vibrant_palette)} swatches "
            + ", ".join(e.hex for e in vibrant_palette))
        n_lat = sum(1 for c in mesh_callouts if c.lateral_centroid is not None)
        n_med = sum(1 for c in mesh_callouts if c.medial_centroid is not None)
        log(f"      mesh callouts: {len(mesh_callouts)} parts "
            f"(lateral {n_lat}, medial {n_med})")

        extra_callouts: list = []
        if _anatomy_detect is not None and _det_to_callout is not None:
            log(f"      ML fine-callout enricher (GroundingDINO + SAM 2)")
            try:
                ml = _anatomy_detect(
                    lateral_path=lateral_render,
                    medial_path=medial_render,
                    verbose=True,
                )
                # Index ML detections by canonical label across both views.
                by_label: dict = {}
                for view_name, side_attr in (("side-lateral", "lateral_centroid"),
                                              ("side-medial", "medial_centroid")):
                    for det in ml.get(view_name, []):
                        co = by_label.setdefault(det.label, {
                            "label": det.label,
                            "lateral_centroid": None,
                            "medial_centroid": None,
                        })
                        co[side_attr] = _det_to_callout(det)["centroid_frac"]
                for label, info in by_label.items():
                    disp, desc = (_ml_display_for(label)
                                  if _ml_display_for else (label, ""))
                    extra_callouts.append(Callout(
                        label=label, display_name=disp, description=desc,
                        material=None, hex_color=None,
                        lateral_centroid=info["lateral_centroid"],
                        medial_centroid=info["medial_centroid"],
                    ))
                log(f"      ML enriched +{len(extra_callouts)} parts: "
                    + ", ".join(c.label for c in extra_callouts))
            except Exception as e:
                log(f"      ML fine-callout enricher skipped: {e}")
        else:
            log(f"      ML fine-callout enricher: skipped "
                f"(transformers/torch not installed)")

        all_callouts = list(mesh_callouts) + list(extra_callouts)
        total_visible = sum(1 for c in all_callouts
                            if c.lateral_centroid is not None
                            or c.medial_centroid is not None)
        anatomy_payload = {
            "lateral_image_path": lateral_render,
            "medial_image_path": medial_render,
            "callouts": all_callouts,
            "title_main": f"{model_name.upper()} — PARTS ANATOMY",
            "title_sub": f"Total Parts: {total_visible}",
        }
        # Same callouts power the colorway pages; titles differ. The
        # vibrant palette feeds the bottom-of-page swatch strip on the
        # color-anatomy page so the user sees real accents (blues,
        # oranges) rather than the area-weighted grays in the engineering
        # palette table.
        colorway_anatomy_payload = {
            "lateral_image_path": lateral_render,
            "medial_image_path": medial_render,
            "callouts": all_callouts,
            "vibrant_palette": vibrant_palette,
            "fabric_title_main":
                f"FABRIC (MATERIAL) ANATOMY OF THE {model_name.upper()}",
            "fabric_title_sub":
                "Inferred materials per visible component (CLIP zero-shot + PBR heuristics).",
            "color_title_main":
                f"COLOR ANATOMY OF THE {model_name.upper()}",
            "color_title_sub":
                "Dominant color per component + global palette.",
            "color_summary":
                "Per-component dominant colors are sampled from the diffuse "
                "texture map. Pantone TCX and RAL Classic matches use CIE "
                "ΔE2000 under the D50 illuminant.",
        }

    bom_path = write_bom(
        os.path.join(output_dir, "02_bom_measurements.pdf"),
        header, ga, scene_shim,
        lineart_path=lineart_paths.get("side-lateral"),
        id_mask_path=id_mask_path,
        id_palette=bom_palette_rgb,
        materials=materials,
        anatomy=anatomy_payload,
    )
    colorway_path = write_colorway(
        os.path.join(output_dir, "03_colorway.pdf"),
        header, palette, component_swatches,
        callout_image_path=overlay_path,
        callout_id_mask_path=id_mask_path,
        callout_id_palette=bom_palette_rgb,
        component_names=component_names,
        component_palettes=component_palettes,
        materials=materials,
        colorway_anatomy=colorway_anatomy_payload,
        # Page-3 reference: Fabric Swatches + Parts List + Color Palette.
        ga=ga,
        face_uvs=face_uvs,
        diffuse_image=face_data["diffuse_image"],
        vibrant_palette=vibrant_palette,
        swatch_cache_dir=renders_dir,
    )

    log(f"      exploded view (3 groups along +height)")
    exploded_path = os.path.join(renders_dir, "linework_exploded.png")
    _, _, exploded_meta = render_exploded_view(
        glb_path, exploded_path,
        ga.face_component, component_names, td_cfg,
    )
    log(f"      sagittal section view")
    section_path = os.path.join(renders_dir, "linework_section.png")
    render_section_view(glb_path, section_path, td_cfg)
    techdrawings_path = write_techdrawings(
        os.path.join(output_dir, "04_techdrawings.pdf"),
        header, lineart_paths, ga,
        exploded_path=exploded_path,
        exploded_meta=exploded_meta,
        section_path=section_path,
    )

    log(f"\nDone in {time.time() - t0:.1f}s. PDFs:")
    for p in (views_path, bom_path, colorway_path, techdrawings_path):
        log(f"  {p}")
    return {
        "views": views_path, "bom": bom_path,
        "colorway": colorway_path, "techdrawings": techdrawings_path,
        "renders_dir": renders_dir,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glb", required=True, help="path to input .glb")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model-id", default="MODEL-001")
    p.add_argument("--model-name", default="Untitled Shoe")
    p.add_argument("--designer", default="—")
    p.add_argument("--factory", default="—")
    p.add_argument("--season", default="—")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--upper-clusters", type=int, default=7)
    p.add_argument("--palette-colors", type=int, default=6)
    p.add_argument("--render-w", type=int, default=1100)
    p.add_argument("--render-h", type=int, default=800)
    p.add_argument("--samples", type=int, default=48,
                   help="Cycles samples per pixel (higher = cleaner, slower)")
    p.add_argument("--target-length-mm", type=float, default=270.0,
                   help="The shoe's actual length in mm. glTF doesn't carry "
                        "physical units, so this is needed to ground absolute "
                        "measurements. Default 270 mm = US men's 9; use 266 "
                        "for US 8.5, 263 for US 8, etc.")
    p.add_argument("--fast", action="store_true",
                   help="Multi-threaded Cycles. ~4x faster but PNGs are "
                        "perceptually identical, not bit-identical, across reruns.")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_argparser().parse_args(argv)
    run(
        glb_path=args.glb, output_dir=args.out,
        model_id=args.model_id, model_name=args.model_name,
        designer=args.designer, factory=args.factory, season=args.season,
        seed=args.seed,
        n_upper_clusters=args.upper_clusters,
        n_palette_colors=args.palette_colors,
        render_width=args.render_w, render_height=args.render_h,
        samples=args.samples,
        deterministic=(not args.fast),
        target_length_mm=args.target_length_mm,
        verbose=(not args.quiet),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
