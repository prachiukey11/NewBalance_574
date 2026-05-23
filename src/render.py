"""Multi-view rendering via a small CPU software rasterizer.

We can't rely on OpenGL on this Mac (no EGL/OSMesa), so we render the textured
shoe in pure numpy + PIL. The renderer supports:

  - orthographic projection (top, bottom, side-lateral, side-medial, front, back)
  - perspective projection (3/4 hero view)
  - per-face Lambert shading using a fixed key+fill light rig
  - texture sampling via per-face UV centroid (we already have face colors)
  - painter's-algorithm z-sort (over-draw is fine for shoe-shaped meshes)
  - optional component overlay (each component gets a tint stripe in a margin)

This is a TechPack render, not a marketing asset — clarity and reproducibility
matter more than photoreal quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


# Fixed light rig: key from upper-front-lateral, fill from opposite.
KEY_LIGHT_DIR = np.array([0.3, 0.5, 0.8], dtype=np.float32)
KEY_LIGHT_DIR /= np.linalg.norm(KEY_LIGHT_DIR)
FILL_LIGHT_DIR = np.array([-0.3, -0.5, 0.4], dtype=np.float32)
FILL_LIGHT_DIR /= np.linalg.norm(FILL_LIGHT_DIR)
AMBIENT = 0.22
KEY_STRENGTH = 0.75
FILL_STRENGTH = 0.25
BG_COLOR = (244, 244, 244)


# Camera helpers

VIEWS = {
    # (projection, eye direction, up vector)
    # Camera sits at +eye direction and looks back at origin. "Up" is the
    # world-space direction we want to point toward the top of the image.
    # Convention: world +X = toe (forward), +Y = lateral (left side of a
    # left shoe), +Z = up. The asset has been normalized to this convention.
    "top":            ("ortho",       (0, 0, 1),       (1, 0, 0)),
    "bottom":         ("ortho",       (0, 0, -1),      (1, 0, 0)),
    "side-lateral":   ("ortho",       (0, 1, 0),       (0, 0, 1)),
    "side-medial":    ("ortho",       (0, -1, 0),      (0, 0, 1)),
    "front":          ("ortho",       (1, 0, 0),       (0, 0, 1)),
    "back":           ("ortho",       (-1, 0, 0),      (0, 0, 1)),
    "three-quarter":  ("perspective", (0.55, 0.75, 0.45), (0, 0, 1)),
}


def _look_at_basis(eye_dir, up_vec):
    f = -np.asarray(eye_dir, dtype=np.float32)
    f /= np.linalg.norm(f)
    up = np.asarray(up_vec, dtype=np.float32)
    s = np.cross(up, -f)
    s /= np.linalg.norm(s) + 1e-9
    u = np.cross(-f, s)
    # Rows of R take world points -> camera frame (x=right, y=up, z=towards-cam).
    R = np.stack([s, u, -f], axis=0).astype(np.float32)
    return R


def _project_orthographic(verts, R, image_size, bbox_padding=1.08):
    cam = verts @ R.T
    minx, miny = cam[:, 0].min(), cam[:, 1].min()
    maxx, maxy = cam[:, 0].max(), cam[:, 1].max()
    w_world = (maxx - minx) * bbox_padding
    h_world = (maxy - miny) * bbox_padding
    aspect_img = image_size[0] / image_size[1]
    aspect_world = w_world / h_world
    if aspect_world > aspect_img:
        scale = image_size[0] / w_world
    else:
        scale = image_size[1] / h_world
    cx_w = (minx + maxx) / 2
    cy_w = (miny + maxy) / 2
    sx = (cam[:, 0] - cx_w) * scale + image_size[0] / 2
    sy = -(cam[:, 1] - cy_w) * scale + image_size[1] / 2
    # With our R basis, cam.z = camera_back · world. Larger cam.z = farther
    # behind the viewer, i.e., closer to the camera (camera sits at the +back
    # direction). For z-buffering we want SMALLER = CLOSER, so we negate.
    sz = -cam[:, 2]
    return np.stack([sx, sy, sz], axis=1), scale


def _project_perspective(verts, R, image_size, fov_deg=32.0, bbox_padding=1.10):
    cam = verts @ R.T
    bbox_radius = float(np.linalg.norm(cam.max(axis=0) - cam.min(axis=0)) / 2)
    centre = (cam.max(axis=0) + cam.min(axis=0)) / 2
    cam = cam - centre
    dist = bbox_radius / np.tan(np.deg2rad(fov_deg) / 2) * bbox_padding
    f_px = 0.5 * image_size[1] / np.tan(np.deg2rad(fov_deg) / 2)
    # cam.z larger = closer to camera (same convention as ortho). Distance
    # from the camera plane = (dist - cam.z); smaller distance = closer.
    z = dist - cam[:, 2]
    z = np.where(z < 0.01, 0.01, z)
    sx = (cam[:, 0]) * f_px / z + image_size[0] / 2
    sy = -(cam[:, 1]) * f_px / z + image_size[1] / 2
    return np.stack([sx, sy, z], axis=1), f_px


# Lambert shading

def _shade_faces(normals: np.ndarray, base_colors_rgb: np.ndarray) -> np.ndarray:
    n_key = np.clip(normals @ KEY_LIGHT_DIR, 0, 1)
    n_fill = np.clip(normals @ FILL_LIGHT_DIR, 0, 1)
    intensity = AMBIENT + KEY_STRENGTH * n_key + FILL_STRENGTH * n_fill
    intensity = np.clip(intensity, 0, 1.0)[:, None]
    shaded = base_colors_rgb.astype(np.float32) * intensity
    return np.clip(shaded, 0, 255).astype(np.uint8)


# Rasterizer (triangle scanline with z-buffer)

def _rasterize(
    proj_verts: np.ndarray,
    faces: np.ndarray,
    face_colors_rgb: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Render triangles with z-buffer. Returns (H, W, 3) uint8."""
    W, H = image_size
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    v = proj_verts
    # Painter's: draw farthest first. With our depth convention smaller z =
    # closer, so we sort descending (largest = farthest first).
    mean_z = v[faces, 2].mean(axis=1)
    order = np.argsort(-mean_z)

    for fi in order:
        f = faces[fi]
        a, b, c = v[f[0]], v[f[1]], v[f[2]]
        x0, x1, x2 = a[0], b[0], c[0]
        y0, y1, y2 = a[1], b[1], c[1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area2 == 0:
            continue
        # Backface cull. With sy flipped (image-y goes down) an outward-
        # facing CCW triangle becomes CW in image, giving area2 < 0. Swap
        # vertices b and c to flip winding so subsequent barycentric math
        # works with positive area.
        if area2 < 0:
            b, c = c, b
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            area2 = -area2
        else:
            # Original CCW: front-facing means it's culled (interior face).
            continue
        xmin = int(max(0, np.floor(min(x0, x1, x2))))
        xmax = int(min(W - 1, np.ceil(max(x0, x1, x2))))
        ymin = int(max(0, np.floor(min(y0, y1, y2))))
        ymax = int(min(H - 1, np.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue

        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1),
                             np.arange(ymin, ymax + 1))
        xs = xs.astype(np.float32) + 0.5
        ys = ys.astype(np.float32) + 0.5

        w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area2
        w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area2
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        sub = inside & (z < zbuf[ymin:ymax + 1, xmin:xmax + 1])
        zbuf[ymin:ymax + 1, xmin:xmax + 1][sub] = z[sub]
        img[ymin:ymax + 1, xmin:xmax + 1][sub] = face_colors_rgb[fi]

    return img


# Public renderer

@dataclass
class RenderConfig:
    width: int = 1100
    height: int = 850
    component_tint: Optional[np.ndarray] = None  # per-face RGB to overlay
    use_face_colors: bool = True


def render_view(
    mesh: trimesh.Trimesh,
    face_colors_rgb: np.ndarray,
    view: str,
    config: Optional[RenderConfig] = None,
) -> np.ndarray:
    """Render one named view. Returns (H, W, 3) uint8."""
    config = config or RenderConfig()
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {list(VIEWS)}")
    proj, eye_dir, up_vec = VIEWS[view]
    R = _look_at_basis(eye_dir, up_vec)
    if proj == "ortho":
        pv, _ = _project_orthographic(mesh.vertices, R, (config.width, config.height))
    else:
        pv, _ = _project_perspective(mesh.vertices, R, (config.width, config.height))

    base = config.component_tint if config.component_tint is not None else face_colors_rgb
    shaded = _shade_faces(mesh.face_normals, base)
    img = _rasterize(pv, mesh.faces, shaded, (config.width, config.height))
    return img


def render_all_views(
    mesh: trimesh.Trimesh,
    face_colors_rgb: np.ndarray,
    output_dir: str,
    config: Optional[RenderConfig] = None,
) -> dict:
    """Render every view; write PNGs; return {view_name: path}."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for view in VIEWS:
        img = render_view(mesh, face_colors_rgb, view, config)
        path = os.path.join(output_dir, f"view_{view}.png")
        Image.fromarray(img).save(path)
        paths[view] = path
    return paths


# Component overlay render (each component a flat color, for the BOM page)

# Distinct but printable hues for component visualisation.
COMPONENT_PALETTE = [
    (197,  90,  17),   # ochre
    ( 87, 154, 193),   # steel blue
    (108, 159,  72),   # green
    (191,  64,  64),   # red
    (138,  98, 175),   # purple
    (226, 172,  42),   # yellow
    (115, 115, 115),   # gray
    ( 60, 153, 153),   # teal
    (211, 109, 158),   # pink
]


def render_component_overlay(
    mesh: trimesh.Trimesh,
    face_component: np.ndarray,
    view: str,
    component_names: list,
    config: Optional[RenderConfig] = None,
) -> np.ndarray:
    """Render the mesh with each component flat-colored from the palette."""
    config = config or RenderConfig()
    n = len(component_names)
    palette = np.array(COMPONENT_PALETTE[:n], dtype=np.uint8)
    face_rgb = np.zeros((len(mesh.faces), 3), dtype=np.uint8)
    for ci in range(n):
        face_rgb[face_component == ci] = palette[ci]
    img = render_view(mesh, face_rgb, view, RenderConfig(
        width=config.width, height=config.height,
        component_tint=face_rgb, use_face_colors=False,
    ))
    return img


def write_image(arr: np.ndarray, path: str) -> str:
    Image.fromarray(arr).save(path)
    return path
