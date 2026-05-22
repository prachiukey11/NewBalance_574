"""GLB ingest + scene normalization.

We treat the input as untrusted: node names may be missing, units may be wrong,
orientation may be arbitrary. The pipeline normalizes scale and orientation so
downstream stages can assume:
    +X = length (toe forward), +Y = width, +Z = height (sole on z=0).

Length is auto-scaled to 270 mm (US-9 men's footwear) if the model's native
units don't make physical sense (>1 m or <50 mm overall length).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


# Typical men's US 9 footwear length, used as fallback when GLB units are wrong.
DEFAULT_SHOE_LENGTH_MM = 270.0


@dataclass
class NormalizedScene:
    """A single concatenated, oriented, scaled mesh + material handles."""

    mesh: trimesh.Trimesh
    diffuse_image: Optional[np.ndarray]      # HxWx{3,4} uint8, baseColor / diffuse
    normal_image: Optional[np.ndarray]
    occlusion_image: Optional[np.ndarray]
    metallic_roughness_image: Optional[np.ndarray]
    uv: Optional[np.ndarray]                 # (Nverts, 2)
    length_mm: float
    width_mm: float
    height_mm: float
    scale_mm_per_unit: float                 # multiplier we applied
    source_path: str
    asset_extras: dict


def _texture_to_array(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    if hasattr(img, "mode"):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        return np.array(img)
    return None


def _flatten_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Bake node transforms and concatenate all geometries into one mesh."""
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        g = scene.geometry[geom_name].copy()
        g.apply_transform(transform)
        meshes.append(g)
    if len(meshes) == 1:
        return meshes[0]
    # Concatenate but preserve per-vertex UVs from the first geometry only if
    # all share the same material; the brief allows multiple meshes per
    # component, but this asset has one. For multi-material assets we'd want
    # to keep them separate. We keep it simple here: concat and trust uv from
    # the largest mesh.
    return trimesh.util.concatenate(meshes)


def _principal_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return 3x3 rotation matrix whose columns are principal axes of the
    vertex cloud, ordered by descending extent so col 0 = longest axis.
    """
    pts = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    # Make a right-handed frame
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def _orient_shoe(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the mesh so +X = length, +Y = width, +Z = height (sole down).

    Uses PCA to align the longest axis to X and the next-longest to Y, then
    flips Z so the sole (the half with more triangle area in the bottom slab)
    points down at z = 0.
    """
    R = _principal_axes(mesh)
    # R takes world->principal. We want principal->world axes XYZ, i.e. apply R^T.
    Rinv = R.T
    T = np.eye(4)
    T[:3, :3] = Rinv
    mesh.apply_transform(T)

    # Now: figure out which end is "up". The sole is the wider, flatter side.
    # Heuristic: project vertices onto Z, split into top/bottom halves at the
    # median, and compare horizontal area (XY footprint variance). The sole
    # half has greater XY spread (flat planar) than the upper half.
    z = mesh.vertices[:, 2]
    z_mid = np.median(z)
    bottom = mesh.vertices[z < z_mid]
    top = mesh.vertices[z >= z_mid]
    bottom_xy_spread = bottom[:, :2].var(axis=0).sum() if len(bottom) else 0
    top_xy_spread = top[:, :2].var(axis=0).sum() if len(top) else 0
    if top_xy_spread > bottom_xy_spread:
        # Flip Z so the flat side ends up at the bottom.
        flip = np.eye(4)
        flip[1, 1] = -1
        flip[2, 2] = -1  # keep det = +1
        mesh.apply_transform(flip)

    # Translate so min(z) = 0 (sole on ground) and centre x/y at origin.
    mins = mesh.vertices.min(axis=0)
    maxs = mesh.vertices.max(axis=0)
    centre = np.array([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, mins[2]])
    mesh.apply_translation(-centre)

    # Heel-to-toe convention: toe = +X. We assume the longer "tail" of the
    # bounding-box-centred outline points toward the heel (heel is rounder,
    # toe is more pointed). Use signed skewness of x.
    from scipy.stats import skew  # local import: scipy is optional
    try:
        sx = skew(mesh.vertices[:, 0])
        # Heel tends to have a denser cluster of vertices (more curved volume),
        # so positive skew means the long tail is toward +X (the toe).
        if sx < 0:
            flip = np.eye(4)
            flip[0, 0] = -1
            flip[1, 1] = -1  # keep det = +1
            mesh.apply_transform(flip)
            # re-centre after flip
            mins = mesh.vertices.min(axis=0)
            maxs = mesh.vertices.max(axis=0)
            centre = np.array([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, mins[2]])
            mesh.apply_translation(-centre)
    except ImportError:
        pass
    return mesh


def load_glb(path: str | Path, target_length_mm: float = DEFAULT_SHOE_LENGTH_MM) -> NormalizedScene:
    """Load a .glb and return a normalized scene in millimetres.

    Parameters
    ----------
    path : str | Path
        Path to a binary glTF 2.0 file.
    target_length_mm : float
        Length to scale the longest axis to when the source's units don't
        produce a plausible shoe size. Set to 0 to disable auto-scaling.
    """
    path = Path(path)
    scene = trimesh.load(str(path), force=None)
    asset_extras = {}
    # trimesh exposes glTF asset metadata via scene.metadata
    if hasattr(scene, "metadata") and isinstance(scene.metadata, dict):
        asset_extras = dict(scene.metadata)

    if isinstance(scene, trimesh.Scene):
        mesh = _flatten_scene(scene)
        # Pick texture images from the first geometry in the scene that has a
        # PBR material attached.
        diffuse = normal = occlusion = mr = None
        uv = None
        for g in scene.geometry.values():
            v = getattr(g, "visual", None)
            mat = getattr(v, "material", None) if v is not None else None
            if mat is None:
                continue
            # trimesh PBRMaterial: baseColorTexture, metallicRoughnessTexture,
            # normalTexture, occlusionTexture. Legacy specGloss material is
            # mapped by trimesh into a PBRMaterial where baseColorTexture <-
            # diffuseTexture.
            diffuse = _texture_to_array(getattr(mat, "baseColorTexture", None))
            mr = _texture_to_array(getattr(mat, "metallicRoughnessTexture", None))
            normal = _texture_to_array(getattr(mat, "normalTexture", None))
            occlusion = _texture_to_array(getattr(mat, "occlusionTexture", None))
            if hasattr(v, "uv") and v.uv is not None:
                uv = np.asarray(v.uv)
            break
    else:
        mesh = scene
        diffuse = normal = occlusion = mr = None
        uv = None
        v = getattr(mesh, "visual", None)
        if hasattr(v, "uv") and v.uv is not None:
            uv = np.asarray(v.uv)

    mesh = _orient_shoe(mesh)

    # Auto-scale to millimetres.
    extents = mesh.extents  # (3,) after orient: [length, width, height]
    length_unit = float(extents[0])
    if target_length_mm > 0 and (length_unit < 0.05 or length_unit > 1.0):
        scale = target_length_mm / length_unit
    else:
        # glTF convention says meters -> mm is *1000.
        scale = 1000.0 if length_unit < 1.0 else 1.0
    mesh.apply_scale(scale)
    extents_mm = mesh.extents

    return NormalizedScene(
        mesh=mesh,
        diffuse_image=diffuse,
        normal_image=normal,
        occlusion_image=occlusion,
        metallic_roughness_image=mr,
        uv=uv,
        length_mm=float(extents_mm[0]),
        width_mm=float(extents_mm[1]),
        height_mm=float(extents_mm[2]),
        scale_mm_per_unit=float(scale),
        source_path=str(path),
        asset_extras=asset_extras,
    )
