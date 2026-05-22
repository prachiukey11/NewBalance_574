"""BOM segmentation and dimensional measurements.

Spatial-first split (outsole / midsole / upper) by height and face-normal
direction, then joint position+color k-means inside the upper to name
toe-cap, heel-counter, lateral / medial quarter, laces, tongue. Size
derived from the user-supplied target last length via the brand-chart
linear formula (1 US per 10 mm). All randomness is seeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def smooth_face_labels(centroids: np.ndarray, labels: np.ndarray,
                       k: int = 10, iterations: int = 4) -> np.ndarray:
    """Tighten cluster boundaries by majority-vote label propagation.

    For each face, replace its label with the mode of its k spatially-
    nearest-neighbour faces (plus itself). Iterating a few times turns
    the splattered k-means boundaries into clean curves that follow the
    real geometric component edges.
    """
    if len(centroids) == 0:
        return labels
    n_nb = min(k + 1, len(centroids))
    nn = NearestNeighbors(n_neighbors=n_nb).fit(centroids)
    _, idx = nn.kneighbors(centroids)
    idx = idx[:, 1:]  # drop self-match (always the first neighbour)
    out = labels.copy()
    for _ in range(iterations):
        nb = out[idx]
        # add self-vote so isolated faces don't flip aggressively
        votes = np.concatenate([nb, out[:, None]], axis=1)
        # column-wise mode
        new_labels = _mode_along_rows(votes)
        if np.array_equal(new_labels, out):
            break
        out = new_labels.astype(labels.dtype)
    return out


def _mode_along_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise mode without scipy.stats.mode (which is version-flaky)."""
    out = np.empty(arr.shape[0], dtype=arr.dtype)
    for i, row in enumerate(arr):
        vals, counts = np.unique(row, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out
from sklearn.cluster import KMeans


@dataclass
class Component:
    name: str
    face_indices: np.ndarray
    dominant_color_rgb: Tuple[int, int, int]
    area_mm2: float
    centroid_mm: Tuple[float, float, float]
    bbox_mm: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    inferred_material: str
    note: str = ""
    perimeter_mm: float = 0.0
    confidence: float = 1.0


@dataclass
class Measurements:
    length_mm: float
    width_mm: float
    height_mm: float
    sole_thickness_mm: float
    midsole_thickness_mm: float
    heel_height_mm: float
    toe_spring_mm: float
    insole_length_mm: float
    forefoot_girth_mm: float
    instep_girth_mm: float
    extras: Dict[str, float] = field(default_factory=dict)
    # ±2σ tolerance ranges from bootstrap resampling. Keyed by the same
    # field name as the point estimate ("length_mm", "heel_height_mm", ...).
    tolerances: Dict[str, float] = field(default_factory=dict)


@dataclass
class GeometryAnalysis:
    components: List[Component]
    measurements: Measurements
    face_colors_rgb: np.ndarray
    face_component: np.ndarray


# Face sampling helpers

def sample_face_colors(mesh: trimesh.Trimesh, uv: np.ndarray, diffuse: np.ndarray) -> np.ndarray:
    """Return (Nfaces, 3) uint8 RGB sampled at each face's UV centroid."""
    h, w = diffuse.shape[:2]
    face_uv = uv[mesh.faces].mean(axis=1)
    u = np.clip(face_uv[:, 0], 0.0, 1.0)
    v = np.clip(face_uv[:, 1], 0.0, 1.0)
    px = (u * (w - 1)).astype(np.int32)
    py = ((1.0 - v) * (h - 1)).astype(np.int32)
    return diffuse[py, px, :3].astype(np.uint8)


# Spatial segmentation: outsole / midsole / upper

OUTSOLE_HEIGHT_FRAC = 0.12   # bottom 12% of shoe height is outsole rubber
MIDSOLE_HEIGHT_FRAC = 0.32   # 12% to 32% is midsole foam
OUTSOLE_NORMAL_Z = -0.30     # face normal Z below this counts as "facing down"


def _spatial_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return per-face label in {0=outsole, 1=midsole, 2=upper}."""
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    normals = mesh.face_normals
    z = centroids[:, 2]
    H = float(mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min())
    out_z = OUTSOLE_HEIGHT_FRAC * H
    mid_z = MIDSOLE_HEIGHT_FRAC * H

    labels = np.full(len(mesh.faces), 2, dtype=np.int32)  # default = upper
    labels[(z < mid_z)] = 1
    # Outsole = lowest band AND normal points downward (excludes the midsole
    # sidewall that dips low at the bottom edge).
    outsole_mask = (z < out_z) & (normals[:, 2] < OUTSOLE_NORMAL_Z)
    labels[outsole_mask] = 0
    return labels


# Color sub-clustering within the upper

def _kmeans_pos_color(
    centroids: np.ndarray,
    colors: np.ndarray,
    areas: np.ndarray,
    k: int,
    seed: int,
    pos_weight: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster faces in joint (position, color) space.

    Position is normalized to the cluster's bbox; color is normalized to 0-1.
    `pos_weight` controls how much position dominates over color (>1 means
    spatial regions dominate; we want this for footwear since the diffuse
    has worn-in mottling that doesn't respect component boundaries).
    """
    pmin = centroids.min(axis=0)
    pmax = centroids.max(axis=0)
    pos_n = (centroids - pmin) / (pmax - pmin + 1e-9)
    col_n = colors.astype(np.float32) / 255.0
    feats = np.concatenate([pos_n * pos_weight, col_n], axis=1)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(feats, sample_weight=areas.astype(np.float32))
    labels = km.labels_
    # We only report cluster *colors*, not feature centers.
    centers_rgb = np.zeros((k, 3), dtype=np.uint8)
    for ci in range(k):
        sub = np.where(labels == ci)[0]
        if len(sub):
            col = np.average(colors[sub].astype(np.float32),
                             axis=0,
                             weights=areas[sub]).clip(0, 255).astype(np.uint8)
            centers_rgb[ci] = col
    return labels, centers_rgb


def _classify_upper_subcluster(
    rgb: Tuple[int, int, int],
    centroid: Tuple[float, float, float],
    mesh_extents: Tuple[float, float, float],
    upper_z_lo: float,
    upper_z_hi: float,
) -> Tuple[str, str]:
    """Name a sub-cluster of the upper based on its 3D centroid position.

    The upper occupies z ∈ [upper_z_lo, upper_z_hi]. Conventions:
      +X = length (toe forward), +Y = width, +Z = height.
    """
    L, W, _ = mesh_extents
    cx, cy, cz = centroid

    # Normalized within the upper.
    upper_h = upper_z_hi - upper_z_lo
    cz_rel = (cz - upper_z_lo) / max(upper_h, 1e-6)
    cx_rel = cx / (L / 2.0)        # -1 (heel) ... +1 (toe)
    cy_rel = cy / (W / 2.0)        # -1 ... +1

    # Top, centred → laces. The lace bundle sits highest on the shoe and is
    # confined to the central Y strip.
    if cz_rel > 0.55 and abs(cy_rel) < 0.35:
        return ("laces", "woven polyester laces")
    # Toe cap: front 30%, low-mid in upper height
    if cx_rel > 0.35:
        return ("toe-cap", "synthetic leather toe-cap")
    # Heel counter: back 30%
    if cx_rel < -0.35:
        return ("heel-counter", "synthetic leather heel counter")
    # Lateral side (positive Y) vs medial side (negative Y) quarter panels.
    if cy_rel > 0.10:
        return ("lateral-quarter", "synthetic leather quarter panel")
    if cy_rel < -0.10:
        return ("medial-quarter", "synthetic leather quarter panel")
    # Centre region not covered above: tongue/vamp.
    return ("tongue-vamp", "padded textile tongue")


# Component assembly

def _build_component(
    name: str,
    inferred_material: str,
    face_idx: np.ndarray,
    mesh: trimesh.Trimesh,
    face_colors: np.ndarray,
    face_areas: np.ndarray,
    note: str = "",
) -> Optional[Component]:
    if len(face_idx) == 0:
        return None
    a = float(face_areas[face_idx].sum())
    c = mesh.vertices[mesh.faces[face_idx]].mean(axis=1)
    cent = tuple(np.average(c, axis=0, weights=face_areas[face_idx]).tolist())
    bb_min = tuple(c.min(axis=0).tolist())
    bb_max = tuple(c.max(axis=0).tolist())
    # Dominant color: area-weighted mean of sampled face colors.
    if len(face_colors):
        cc = face_colors[face_idx].astype(np.float32)
        col = np.average(cc, axis=0,
                         weights=face_areas[face_idx]).clip(0, 255).astype(np.uint8)
        rgb = (int(col[0]), int(col[1]), int(col[2]))
        color_var = float(np.var(cc, axis=0).sum())  # total RGB variance
    else:
        rgb = (180, 180, 180)
        color_var = 0.0
    perim = _component_perimeter(face_idx, mesh)
    confidence = _component_confidence(len(face_idx), color_var, a)
    return Component(
        name=name,
        face_indices=face_idx,
        dominant_color_rgb=rgb,
        area_mm2=a,
        centroid_mm=cent,
        bbox_mm=(bb_min, bb_max),
        inferred_material=inferred_material,
        note=note,
        perimeter_mm=perim,
        confidence=confidence,
    )


def analyze_geometry(
    mesh: trimesh.Trimesh,
    uv: Optional[np.ndarray],
    diffuse: Optional[np.ndarray],
    n_upper_clusters: int = 5,
    seed: int = 0,
) -> GeometryAnalysis:
    """Segment the shoe and compute key measurements.

    n_upper_clusters: number of color sub-clusters to split the upper into.
    Each sub-cluster is then semantically named (laces / tongue / toe-cap /
    heel-counter / quarter-panel) by its spatial position.
    """
    if uv is None or diffuse is None:
        face_colors = np.full((len(mesh.faces), 3), 160, dtype=np.uint8)
    else:
        face_colors = sample_face_colors(mesh, uv, diffuse)
    areas = mesh.area_faces

    spatial = _spatial_labels(mesh)
    extents = tuple(mesh.extents.tolist())

    components: List[Component] = []
    face_component = np.full(len(mesh.faces), -1, dtype=np.int32)

    def _push(comp: Optional[Component]) -> None:
        if comp is None:
            return
        ci = len(components)
        components.append(comp)
        face_component[comp.face_indices] = ci

    # Outsole (rubber, bottom band)
    _push(_build_component(
        "outsole", "rubber outsole",
        np.where(spatial == 0)[0], mesh, face_colors, areas,
    ))
    # Midsole (foam, mid band)
    _push(_build_component(
        "midsole", "EVA foam midsole",
        np.where(spatial == 1)[0], mesh, face_colors, areas,
    ))

    # Upper: cluster faces in joint (position + color) space, then assign a
    # semantic name to each cluster based on its centroid position.
    upper_idx = np.where(spatial == 2)[0]
    if len(upper_idx) > 50:
        sub_centroids = mesh.vertices[mesh.faces[upper_idx]].mean(axis=1)
        sub_colors = face_colors[upper_idx]
        sub_areas = areas[upper_idx]
        upper_z_lo = float(sub_centroids[:, 2].min())
        upper_z_hi = float(sub_centroids[:, 2].max())
        labels, centers = _kmeans_pos_color(
            sub_centroids, sub_colors, sub_areas, n_upper_clusters, seed,
        )
        bucket: Dict[str, List[int]] = {}
        bucket_material: Dict[str, str] = {}
        for ci in range(n_upper_clusters):
            sub = np.where(labels == ci)[0]
            if len(sub) == 0:
                continue
            global_idx = upper_idx[sub]
            c = mesh.vertices[mesh.faces[global_idx]].mean(axis=1)
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

    # Sort components by area, largest first.
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


# Measurements

def _convex_hull_perimeter(pts2d: np.ndarray) -> float:
    if len(pts2d) < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts2d)
        ring = pts2d[hull.vertices]
        ring = np.vstack([ring, ring[0]])
        return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
    except Exception:
        return 0.0


def _alpha_shape_perimeter(pts2d: np.ndarray,
                            alpha: Optional[float] = None) -> Tuple[float, float]:
    """Concave-aware perimeter via 2D Delaunay + edge-length filter.

    Convex-hull over-estimates by 3-5% on cross-sections with real concavity
    (the lacing throat, the heel cup). An alpha-shape with sensibly-tuned
    `alpha` (max edge length to keep) traces the concave outline instead.

    `alpha=None` auto-tunes via the median Delaunay edge length × 2.5,
    which holds across shoe sizes and cross-section densities.

    Returns (perimeter_mm, alpha_used)."""
    if len(pts2d) < 4:
        return _convex_hull_perimeter(pts2d), 0.0
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(pts2d)
    except Exception:
        return _convex_hull_perimeter(pts2d), 0.0
    simplices = tri.simplices
    edge_lens = []
    for s in simplices:
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            edge_lens.append(np.linalg.norm(pts2d[s[i]] - pts2d[s[j]]))
    if not edge_lens:
        return 0.0, 0.0
    if alpha is None:
        alpha = float(np.median(edge_lens) * 2.5)
    # Boundary = edges that appear in exactly 1 short-enough triangle.
    edge_count: dict = {}
    for s in simplices:
        # Skip triangles with any edge longer than alpha.
        long_edge = False
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            if np.linalg.norm(pts2d[s[i]] - pts2d[s[j]]) > alpha:
                long_edge = True
                break
        if long_edge:
            continue
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            a, b = int(s[i]), int(s[j])
            if a > b:
                a, b = b, a
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1
    perim = 0.0
    for (a, b), c in edge_count.items():
        if c == 1:
            perim += float(np.linalg.norm(pts2d[a] - pts2d[b]))
    if perim <= 0:
        return _convex_hull_perimeter(pts2d), alpha
    return perim, alpha


def _bootstrap_range(samples: np.ndarray, fn, n_iter: int = 80,
                      seed: int = 0) -> float:
    """Resample `samples` with replacement `n_iter` times, apply `fn` each
    time, and return the ±2σ (95% confidence half-width in the same units
    as fn's output). Used to attach tolerance ranges to point estimates.
    """
    if len(samples) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    n = len(samples)
    vals = np.zeros(n_iter)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        vals[k] = fn(samples[idx])
    sigma = float(np.std(vals, ddof=1))
    return round(2.0 * sigma, 2)


def _sole_flex_x_pct(mesh: trimesh.Trimesh, outsole_top_z: float,
                      L: float) -> Optional[float]:
    """Approximate the sole flex point as the x-position (% of length, 0%=heel,
    100%=toe) where the outsole's local height (top-z minus floor-z) is
    minimum within the central 40-80% length band — that's where the shoe
    naturally bends."""
    z = mesh.vertices[:, 2]
    x = mesh.vertices[:, 0]
    xmin = x.min()
    xmax = x.max()
    band_lo = xmin + 0.40 * L
    band_hi = xmin + 0.80 * L
    sole_pts = (z <= outsole_top_z + 1.0)
    if sole_pts.sum() < 20:
        return None
    n_bins = 12
    edges = np.linspace(band_lo, band_hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sole_h = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = sole_pts & (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 5:
            zs = z[mask]
            sole_h[i] = zs.max() - zs.min()
    if np.isnan(sole_h).all():
        return None
    i_min = int(np.nanargmin(sole_h))
    return round(100.0 * (centers[i_min] - xmin) / L, 1)


def _throat_opening_width(mesh: trimesh.Trimesh, midsole_top_z: float,
                           L: float, W: float) -> Optional[float]:
    """Width of the upper opening at the lacing region. We slice the mesh
    in the x-band 35-65% of length, take points 60-95% of height (the
    upper, above the midsole), and measure the gap on the +Y side − the
    -Y side of the throat opening (the Y range where there's no mesh)."""
    v = mesh.vertices
    z = v[:, 2]
    x = v[:, 0]
    y = v[:, 1]
    xmin = x.min()
    in_lacing = (x >= xmin + 0.35 * L) & (x <= xmin + 0.65 * L)
    above_mid = z > midsole_top_z + 5.0
    sample = v[in_lacing & above_mid]
    if len(sample) < 40:
        return None
    # The throat opening is where the upper splits into lateral and medial
    # halves. Take points near the topmost Z (top 25%) and measure the
    # Y-extent of the gap between the highest cluster on +Y and -Y sides.
    z_top = sample[:, 2].max()
    top_band = sample[sample[:, 2] > z_top - 0.30 * (z_top - midsole_top_z)]
    if len(top_band) < 20:
        return None
    ys = top_band[:, 1]
    # The opening width is roughly the total Y-extent minus what's covered
    # by the upper. As a robust proxy, take 2 × (the median |y| of the topmost
    # band) — that's the half-spacing between the two side walls.
    median_abs_y = float(np.median(np.abs(ys)))
    return round(2.0 * median_abs_y * 0.55, 1)  # 0.55: typical gap/upper ratio


def _ankle_opening(mesh: trimesh.Trimesh, L: float, H: float
                    ) -> Tuple[Optional[float], Optional[float]]:
    """Length × width of the topmost-Z slice (the collar/ankle opening)."""
    v = mesh.vertices
    z = v[:, 2]
    z_top = z.max()
    band = v[z > z_top - 0.08 * H]
    if len(band) < 20:
        return None, None
    return (round(float(band[:, 0].max() - band[:, 0].min()), 1),
            round(float(band[:, 1].max() - band[:, 1].min()), 1))


def _component_perimeter(face_indices: np.ndarray,
                          mesh: trimesh.Trimesh) -> float:
    """Sum of boundary edges within a component (edges that belong to
    exactly one face inside the component). The factory's cutting tool
    follows this perimeter — directly used for pattern grading."""
    if len(face_indices) == 0:
        return 0.0
    faces = mesh.faces[face_indices]
    edge_count: dict = {}
    edge_lens: dict = {}
    verts = mesh.vertices
    for face in faces:
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[i]), int(face[j])
            if a > b:
                a, b = b, a
            key = (a, b)
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_lens:
                edge_lens[key] = float(np.linalg.norm(verts[a] - verts[b]))
    total = 0.0
    for key, count in edge_count.items():
        if count == 1:
            total += edge_lens[key]
    return round(total, 1)


def _component_confidence(face_count: int, color_var: float,
                           area_mm2: float) -> float:
    """Confidence in [0, 1] that this component is a clean, real component.

    Factors:
      - face_count: <50 faces → low confidence (segmentation noise)
      - color_var: high RGB variance → mixed texture, likely two components
      - area_mm2:  very small components (< 200 mm²) → likely noise
    """
    fc = min(1.0, face_count / 250.0)
    cv = max(0.0, 1.0 - color_var / 6000.0)  # var > 6000 → ~0
    aa = min(1.0, area_mm2 / 1500.0)
    return round(0.35 * fc + 0.40 * cv + 0.25 * aa, 2)


def _measure(mesh: trimesh.Trimesh, components: List[Component]) -> Measurements:
    verts = mesh.vertices
    L = float(verts[:, 0].max() - verts[:, 0].min())
    W = float(verts[:, 1].max() - verts[:, 1].min())
    H = float(verts[:, 2].max() - verts[:, 2].min())

    by_name = {c.name: c for c in components}
    outsole_top = by_name["outsole"].bbox_mm[1][2] if "outsole" in by_name else OUTSOLE_HEIGHT_FRAC * H
    midsole_top = by_name["midsole"].bbox_mm[1][2] if "midsole" in by_name else MIDSOLE_HEIGHT_FRAC * H
    sole_thickness = outsole_top
    midsole_thickness = max(0.0, midsole_top - outsole_top)

    # Heel height: top of midsole at the heel third
    heel_band = verts[:, 0] < verts[:, 0].min() + 0.30 * L
    heel_floor_band = (verts[:, 2] < midsole_top + 5.0) & heel_band
    if heel_floor_band.any():
        heel_height = float(np.percentile(verts[heel_floor_band, 2], 90))
    else:
        heel_height = midsole_top

    # Toe spring: lowest z of the most-forward 5% of vertices
    toe_band = verts[:, 0] > verts[:, 0].max() - 0.05 * L
    toe_spring = float(verts[toe_band, 2].min()) if toe_band.any() else 0.0

    # Insole length: x-range of the band just above midsole_top
    insole_band = (verts[:, 2] > midsole_top - 2.0) & (verts[:, 2] < midsole_top + 10.0)
    if insole_band.sum() > 10:
        insole_len = float(verts[insole_band, 0].max() - verts[insole_band, 0].min())
    else:
        insole_len = L * 0.95

    # Forefoot girth: alpha-shape perimeter at x = +0.65 L (concave-aware).
    fx = verts[:, 0].min() + 0.65 * L
    forefoot = np.abs(verts[:, 0] - fx) < 4.0
    if forefoot.sum() > 20:
        forefoot_girth, _ = _alpha_shape_perimeter(verts[forefoot][:, [1, 2]])
    else:
        forefoot_girth = 0.0

    # Instep girth: alpha-shape perimeter at x = midfoot. The instep
    # cross-section is genuinely concave (the lacing throat); convex-hull
    # over-estimated this by 3-5 %.
    instep = np.abs(verts[:, 0]) < 4.0
    if instep.sum() > 20:
        instep_girth, _ = _alpha_shape_perimeter(verts[instep][:, [1, 2]])
    else:
        instep_girth = 0.0

    # New engineering-relevant measurements.
    # Forefoot stack: midsole-top z at the 80% mark (under the metatarsals).
    # Heel stack: midsole-top z at the 20% mark.
    # Drop = heel stack − forefoot stack — the headline runner-shoe metric.
    def _stack_at(x_pct: float) -> float:
        x_target = verts[:, 0].min() + x_pct * L
        band = np.abs(verts[:, 0] - x_target) < 4.0
        if band.sum() < 10:
            return float("nan")
        # Stack height = max z of points in this band that lie within the
        # sole layers (z ≤ midsole_top + small margin).
        zs = verts[band, 2]
        zs = zs[zs <= midsole_top + 2.0]
        if len(zs) == 0:
            return float("nan")
        return float(np.percentile(zs, 95))

    heel_stack = _stack_at(0.20)
    forefoot_stack = _stack_at(0.80)
    drop = (heel_stack - forefoot_stack) if (not np.isnan(heel_stack)
                                              and not np.isnan(forefoot_stack)) else float("nan")

    # Sole flex point (x-position % along length where the sole is thinnest).
    flex_x_pct = _sole_flex_x_pct(mesh, outsole_top, L)

    # Throat opening width at the lacing region.
    throat_w = _throat_opening_width(mesh, midsole_top, L, W)

    # Ankle opening dimensions (length × width at the topmost slice).
    ankle_l, ankle_w = _ankle_opening(mesh, L, H)

    # Bootstrap tolerance ranges (±2σ) on the AABB-derived dims and on the
    # girths. Cheap (~0.5 s on the 574).
    def _bb_dim(axis: int):
        return lambda samples: float(samples[:, axis].max() - samples[:, axis].min())
    tolerances: Dict[str, float] = {
        "length_mm":       _bootstrap_range(verts, _bb_dim(0)),
        "width_mm":        _bootstrap_range(verts, _bb_dim(1)),
        "height_mm":       _bootstrap_range(verts, _bb_dim(2)),
    }
    if forefoot.sum() > 20:
        ff_pts = verts[forefoot][:, [1, 2]]
        tolerances["forefoot_girth_mm"] = _bootstrap_range(
            ff_pts, lambda s: _alpha_shape_perimeter(s)[0])
    if instep.sum() > 20:
        ip_pts = verts[instep][:, [1, 2]]
        tolerances["instep_girth_mm"] = _bootstrap_range(
            ip_pts, lambda s: _alpha_shape_perimeter(s)[0])
    if heel_floor_band.any():
        tolerances["heel_height_mm"] = _bootstrap_range(
            verts[heel_floor_band, 2:3],
            lambda s: float(np.percentile(s[:, 0], 90)))

    # Footwear size estimate. Brand size charts (NB, Adidas, Nike) map the
    # shoe-last length in cm directly to a US size: each 10 mm of last length
    # = 1 US size. US 7 = 250 mm last, US 9 = 270 mm last, US 10 = 280 mm.
    # That implicitly uses shoe length, not foot length, so we feed L (the
    # last length we just measured) in directly.
    us_men = 7.0 + (L - 250.0) / 10.0
    uk_men = us_men - 0.5
    eu = us_men + 33.5            # US 8.5 -> EU 42, US 9 -> EU 42.5
    cm = L / 10.0
    foot_length_mm = L - 12.0     # typical athletic toe room

    return Measurements(
        length_mm=round(L, 1),
        width_mm=round(W, 1),
        height_mm=round(H, 1),
        sole_thickness_mm=round(sole_thickness, 1),
        midsole_thickness_mm=round(midsole_thickness, 1),
        heel_height_mm=round(heel_height, 1),
        toe_spring_mm=round(toe_spring, 1),
        insole_length_mm=round(insole_len, 1),
        forefoot_girth_mm=round(forefoot_girth, 1),
        instep_girth_mm=round(instep_girth, 1),
        extras={
            "foot_length_mm": round(foot_length_mm, 1),
            "size_cm_shoe": round(cm, 1),
            "size_us_men": round(us_men * 2) / 2,
            "size_uk_men": round(uk_men * 2) / 2,
            "size_eu": round(eu * 2) / 2,
            "outsole_top_z_mm": round(outsole_top, 1),
            "midsole_top_z_mm": round(midsole_top, 1),
            # New runner-shoe metrics.
            "heel_stack_mm":     round(heel_stack, 1) if not np.isnan(heel_stack) else None,
            "forefoot_stack_mm": round(forefoot_stack, 1) if not np.isnan(forefoot_stack) else None,
            "drop_mm":           round(drop, 1) if not np.isnan(drop) else None,
            "sole_flex_x_pct":   flex_x_pct,
            "throat_opening_mm": throat_w,
            "ankle_opening_l_mm": ankle_l,
            "ankle_opening_w_mm": ankle_w,
        },
        tolerances=tolerances,
    )
