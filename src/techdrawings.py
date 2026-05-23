"""2D technical drawings with dimension callouts.

Blender Freestyle gives the black-on-white line-art renders (lateral
and plan views); ReportLab draws the dimension callouts directly in
page space so they stay sharp at any zoom.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Vector  # type: ignore
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer)

from .geometry import GeometryAnalysis
from .pdf import TechPackHeader
from .render_blender import (AxisMap, _clear_scene, _frame_camera, _import_glb,
                             _join_into_one, _make_camera, _setup_world,
                             _strip_png_metadata, detect_axis_map)


def _draw_header_footer(c: rl_canvas.Canvas, header: TechPackHeader,
                        page_label: str, page_num: int,
                        page_w: float, page_h: float) -> None:
    margin = 15 * mm
    c.saveState()
    c.setLineWidth(0.5)
    c.setStrokeColor(colors.HexColor("#222222"))
    c.line(margin, page_h - margin + 4, page_w - margin, page_h - margin + 4)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#202020"))
    c.drawString(margin, page_h - margin + 7,
                 f"{header.model_id} · {header.model_name}")
    c.drawRightString(page_w - margin, page_h - margin + 7,
                      f"{page_label}   {header.date_iso}")
    c.line(margin, margin - 4, page_w - margin, margin - 4)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(margin, margin - 12,
                 f"Source: {os.path.basename(header.source_file)}  ·  "
                 f"SHA256:{header.source_hash[:12]}")
    c.drawRightString(page_w - margin, margin - 12, f"Page {page_num}")
    c.restoreState()


# Freestyle renders

def _setup_freestyle_layered(
    scene,
    silhouette_px: float = 2.0,
    crease_px: float = 1.0,
    hidden_px: float = 0.6,
    enable_hidden: bool = False,
) -> None:
    """Multi-lineset Freestyle: silhouette thick (the outer outline a factory
    would cut to), crease thin (the panel seams), hidden very thin & dashed
    (suggestion lines). Replaces the uniform 1.6 px lineset used by
    _setup_freestyle()."""
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    scene.render.line_thickness = silhouette_px

    # Plain white world background.
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs[0])

    # Replace each mesh's material with flat-white emissive so only freestyle is visible.
    white_mat = bpy.data.materials.get("LineArtWhite") or bpy.data.materials.new("LineArtWhite")
    white_mat.use_nodes = True
    n = white_mat.node_tree
    for nd in list(n.nodes):
        n.nodes.remove(nd)
    em = n.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs[1].default_value = 1.0
    mo = n.nodes.new("ShaderNodeOutputMaterial")
    n.links.new(em.outputs[0], mo.inputs[0])
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for slot in o.material_slots:
            slot.material = white_mat
        if not o.material_slots:
            o.data.materials.append(white_mat)

    view_layer = bpy.context.view_layer
    fs = view_layer.freestyle_settings
    while fs.linesets:
        fs.linesets.remove(fs.linesets[0])

    def _add_lineset(name, thickness, *, silhouette=False, border=False,
                     crease=False, contour=False, external=False,
                     dashed=False):
        ls = fs.linesets.new(name=name)
        ls.select_silhouette = silhouette
        ls.select_border = border
        ls.select_crease = crease
        ls.select_contour = contour
        ls.select_external_contour = external
        ls.select_ridge_valley = False
        ls.select_edge_mark = False
        style = bpy.data.linestyles.new(name=name + "Style")
        ls.linestyle = style
        style.use_chaining = True
        style.chaining = "PLAIN"
        style.thickness = thickness
        style.color = (0.0, 0.0, 0.0)
        style.alpha = 1.0
        if dashed:
            style.use_dashed_line = True
            style.dash1 = 6
            style.gap1 = 4
        return ls

    # Layer 1: silhouette (the outer outline) — thickest
    _add_lineset("Silhouette", silhouette_px, silhouette=True, external=True)
    # Layer 2: creases + borders (panel seams) — medium
    _add_lineset("Crease", crease_px, crease=True, border=True, contour=True)
    if enable_hidden:
        _add_lineset("Hidden", hidden_px, silhouette=False, dashed=True)

    # Minimal Cycles render: 1 sample, freestyle composited on top.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.use_auto_tile = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.cycles.seed = 0


def _setup_freestyle(scene, line_thickness_px: float = 1.6) -> None:
    """Black silhouette + crease lines on a white background, no shading."""
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    scene.render.line_thickness = line_thickness_px

    # Plain white world background.
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs[0])

    # Replace each mesh's material with a flat-white emissive so it blends
    # into the background, leaving only the freestyle overlay visible.
    white_mat = bpy.data.materials.get("LineArtWhite") or bpy.data.materials.new("LineArtWhite")
    white_mat.use_nodes = True
    n = white_mat.node_tree
    for nd in list(n.nodes):
        n.nodes.remove(nd)
    em = n.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs[1].default_value = 1.0
    mo = n.nodes.new("ShaderNodeOutputMaterial")
    n.links.new(em.outputs[0], mo.inputs[0])
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for slot in o.material_slots:
            slot.material = white_mat
        if not o.material_slots:
            o.data.materials.append(white_mat)

    # Freestyle line set + line style. In Blender >=4 the active lineset's
    # linestyle must be created explicitly (it's None by default).
    view_layer = bpy.context.view_layer
    fs = view_layer.freestyle_settings
    # Wipe and recreate so we know exactly what's there.
    while fs.linesets:
        fs.linesets.remove(fs.linesets[0])
    lineset = fs.linesets.new(name="MainLineSet")
    lineset.select_silhouette = True
    lineset.select_border = True
    lineset.select_crease = True
    lineset.select_contour = True
    lineset.select_external_contour = True
    lineset.select_ridge_valley = False
    lineset.select_edge_mark = False
    # Ensure a linestyle is assigned.
    if lineset.linestyle is None:
        ls = bpy.data.linestyles.new(name="MainLineStyle")
        lineset.linestyle = ls
    ls = lineset.linestyle
    ls.use_chaining = True
    ls.chaining = "PLAIN"
    ls.thickness = line_thickness_px
    ls.color = (0.0, 0.0, 0.0)
    ls.alpha = 1.0

    # Minimal Cycles render: 1 sample, freestyle composited on top.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.use_auto_tile = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    # Make sure the film is opaque white (not transparent), so the freestyle
    # lines land on white when there's no geometry behind them.
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.cycles.seed = 0


@dataclass
class TechDrawConfig:
    width: int = 1200
    height: int = 900
    line_thickness: float = 1.4


def render_lineart_views(
    glb_path: str,
    output_dir: str,
    cfg: Optional[TechDrawConfig] = None,
) -> Tuple[Dict[str, str], AxisMap, np.ndarray]:
    """Render lateral + top line-art views. Returns (paths, axis_map, ext)."""
    cfg = cfg or TechDrawConfig()
    os.makedirs(output_dir, exist_ok=True)

    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)
    centre = am.centre()
    radius = float(np.linalg.norm(am.bbox_max - am.bbox_min) / 2)
    extents = am.bbox_max - am.bbox_min

    _setup_world()
    scene = bpy.context.scene
    _setup_freestyle_layered(scene,
                              silhouette_px=cfg.line_thickness * 1.4,
                              crease_px=cfg.line_thickness * 0.7)
    scene.render.resolution_x = cfg.width
    scene.render.resolution_y = cfg.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    cam = _make_camera()
    scene.camera = cam

    length_ext = am.extent("length")
    width_ext = am.extent("width")
    height_ext = am.extent("height")
    length_v = am.vec("+length")
    width_v = am.vec("+width")
    height_v = am.vec("+height")

    paths: Dict[str, str] = {}

    # lateral
    _frame_camera(cam, width_v, height_v, centre, radius,
                  (length_ext, height_ext), cfg.width, cfg.height, ortho=True)
    p = os.path.join(output_dir, "linework_side-lateral.png")
    scene.render.filepath = p
    bpy.ops.render.render(write_still=True)
    _strip_png_metadata(p)
    paths["side-lateral"] = p

    # top
    _frame_camera(cam, height_v, length_v, centre, radius,
                  (width_ext, length_ext), cfg.width, cfg.height, ortho=True)
    p = os.path.join(output_dir, "linework_top.png")
    scene.render.filepath = p
    bpy.ops.render.render(write_still=True)
    _strip_png_metadata(p)
    paths["top"] = p

    return paths, am, extents


# Exploded view — separate components along +height, render line-art

from .render_blender import (_group_for_component_name as _group_name,
                              split_mesh_by_group as _shared_split)


def _group_for_component_name(name: str) -> str:
    return _group_name(name)


def _split_mesh_by_group(obj, face_component, component_group, group_offsets_bu):
    """Thin wrapper kept for backwards-compat with the line-art exploded
    view path. Drops the face_remap; that path doesn't need it."""
    new_objs, _ = _shared_split(obj, face_component, component_group,
                                 group_offsets_bu)
    return new_objs


def render_exploded_view(
    glb_path: str,
    output_path: str,
    face_component: np.ndarray,
    component_names: list,
    cfg: Optional[TechDrawConfig] = None,
    upper_offset_mm: float = 80.0,
    midsole_offset_mm: float = 30.0,
) -> Tuple[str, AxisMap, dict]:
    """Render a 3-group exploded view (outsole / midsole / upper) along +height.

    Returns (path, axis_map, frame_meta) where frame_meta has the ortho-frame
    info needed to position labels on the PDF page.
    """
    cfg = cfg or TechDrawConfig()
    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)

    # Per-component → group mapping
    component_group = [_group_for_component_name(n) for n in component_names]

    # Offsets along +height in BU. scale_mm_per_unit is recovered from am's
    # height extent and the known length_mm at run time; the simpler proxy
    # is to use the height extent as the unit of separation. We translate
    # each group by a multiple of the height extent.
    height_bu = am.extent("height")  # 1 BU = ? mm; we'll use a fraction of height
    height_v = am.vec("+height")
    # midsole rises by midsole_offset_mm in mm — but we don't know mm/BU here.
    # Use the geometry's actual heights as a proxy: shift midsole by 0.6×height,
    # upper by 1.6×height. This produces a readable separation regardless of
    # the GLB's native unit.
    offset_mid = tuple(height_v * 0.7 * height_bu)
    offset_upper = tuple(height_v * 1.8 * height_bu)
    new_objs = _split_mesh_by_group(
        obj, face_component, component_group,
        {
            "outsole": (0.0, 0.0, 0.0),
            "midsole": offset_mid,
            "upper":   offset_upper,
        },
    )

    # Recompute bbox over the exploded objects.
    all_verts = np.concatenate([
        np.array([(v.co.x, v.co.y, v.co.z) for v in o.data.vertices])
        for o in new_objs
    ])
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    centre = (bbox_min + bbox_max) / 2
    radius = float(np.linalg.norm(bbox_max - bbox_min) / 2)

    _setup_world()
    scene = bpy.context.scene
    _setup_freestyle_layered(scene,
                              silhouette_px=cfg.line_thickness * 1.4,
                              crease_px=cfg.line_thickness * 0.7)
    scene.render.resolution_x = cfg.width
    scene.render.resolution_y = cfg.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Frame from the side-lateral view (look along +width axis).
    cam = _make_camera()
    scene.camera = cam
    # In-plane extents on the lateral view: length × (extended height).
    length_ext = am.extent("length")
    width_v = am.vec("+width")
    height_v_world = am.vec("+height")
    # The exploded bbox along +height is bigger than the original; compute it.
    proj_height = float(
        (bbox_max - bbox_min) @ np.abs(height_v_world)
    )
    _frame_camera(cam, width_v, height_v_world, centre, radius,
                  (length_ext, proj_height), cfg.width, cfg.height, ortho=True)

    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    _strip_png_metadata(output_path)

    # Compute fraction-of-image positions for the three group centroids so the
    # PDF can label them.
    aspect = cfg.width / cfg.height
    ortho_scale = max(length_ext, proj_height * aspect) * 1.08
    # Project each group's centroid onto (length_axis, height_axis), relative to centre.
    def _img_xy(world_pt):
        rel = np.array(world_pt) - centre
        u = float(rel @ am.vec("+length"))   # horizontal in image
        v = float(rel @ height_v_world)      # vertical in image
        fx = 0.5 + u / ortho_scale
        # In image y, lower = larger pixel-y; reportlab convention used in
        # _draw_image_with_callouts treats fractions of page box where 0=bottom.
        fy = 0.5 + v / (ortho_scale / aspect)
        return fx, fy

    group_anchors = {}
    for o, group_id in zip(new_objs, [n.name.replace("explode_", "") for n in new_objs]):
        vs = np.array([(v.co.x, v.co.y, v.co.z) for v in o.data.vertices])
        group_anchors[group_id] = _img_xy(vs.mean(axis=0))

    return output_path, am, {
        "group_anchors": group_anchors,
        "ortho_scale": ortho_scale,
        "centre": centre.tolist(),
    }


# Sagittal section view — bisect mesh and render

def render_section_view(
    glb_path: str,
    output_path: str,
    cfg: Optional[TechDrawConfig] = None,
) -> Tuple[str, AxisMap]:
    """Render a sagittal section view: bisect the mesh on the centerline
    plane (perpendicular to +width) and clear the +width half so the cross-
    section through the sole stack is visible from the lateral camera.

    The camera uses the pre-cut centre and extents so the resulting image
    has the *same framing* as the lateral profile — this lets the PDF
    callouts (computed against the lateral framing) line up correctly on
    the section view too.
    """
    import bmesh  # type: ignore
    cfg = cfg or TechDrawConfig()
    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)
    pre_centre = am.centre()
    pre_radius = float(np.linalg.norm(am.bbox_max - am.bbox_min) / 2)
    pre_length = am.extent("length")
    pre_height = am.extent("height")
    width_v = am.vec("+width")

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        dist=1e-5,
        plane_co=Vector(tuple(float(x) for x in pre_centre)),
        plane_no=Vector(tuple(float(x) for x in width_v)),
        clear_outer=True,
        clear_inner=False,
    )
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    _setup_world()
    scene = bpy.context.scene
    _setup_freestyle_layered(scene,
                              silhouette_px=cfg.line_thickness * 1.4,
                              crease_px=cfg.line_thickness * 0.7)
    scene.render.resolution_x = cfg.width
    scene.render.resolution_y = cfg.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    cam = _make_camera()
    scene.camera = cam
    # Use the PRE-CUT centre + extents so the framing matches the lateral
    # profile exactly. The cut half is just hidden from view; the camera
    # doesn't move.
    _frame_camera(cam, am.vec("+width"), am.vec("+height"),
                  pre_centre, pre_radius,
                  (pre_length, pre_height), cfg.width, cfg.height, ortho=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    _strip_png_metadata(output_path)
    return output_path, am


# PDF assembly with dimension callouts

def _draw_dim_callout(c: rl_canvas.Canvas,
                      p1: Tuple[float, float],
                      p2: Tuple[float, float],
                      label: str,
                      offset: float = 6,
                      arrow_len: float = 2.4,
                      side: int = +1) -> None:
    """Draw an engineering-style dimension callout between two image points.

    `offset` and `arrow_len` are in mm of page space. `side` chooses which
    side of the segment to place the extension line on.
    """
    import math as _m
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = _m.hypot(dx, dy)
    if L < 1e-6:
        return
    nx, ny = -dy / L, dx / L
    nx *= side
    ny *= side
    o1 = (x1 + nx * offset, y1 + ny * offset)
    o2 = (x2 + nx * offset, y2 + ny * offset)
    # Extension lines
    c.setLineWidth(0.35)
    c.setStrokeColor(colors.HexColor("#202020"))
    c.line(x1, y1, o1[0], o1[1])
    c.line(x2, y2, o2[0], o2[1])
    # Dimension line + arrows
    c.line(o1[0], o1[1], o2[0], o2[1])
    # Arrowheads (small triangles)
    ang = _m.atan2(o2[1] - o1[1], o2[0] - o1[0])

    def _arrow(at, direction):
        ax, ay = at
        c.saveState()
        c.translate(ax, ay)
        c.rotate(_m.degrees(ang + (0 if direction > 0 else _m.pi)))
        path = c.beginPath()
        path.moveTo(0, 0)
        path.lineTo(arrow_len, arrow_len * 0.35)
        path.lineTo(arrow_len, -arrow_len * 0.35)
        path.close()
        c.setFillColor(colors.HexColor("#202020"))
        c.drawPath(path, fill=1, stroke=0)
        c.restoreState()
    _arrow(o1, -1)
    _arrow(o2, +1)

    # Label placement: inline (along the dim line) if the dim line is long
    # enough to hold the label, otherwise use a leader-line callout that
    # points from the dim line out to a horizontal label in clear space.
    label_font = "Helvetica"
    label_size = 7.5
    label_w = c.stringWidth(label, label_font, label_size)
    midx = (o1[0] + o2[0]) / 2
    midy = (o1[1] + o2[1]) / 2

    if L >= label_w + 4:
        # Inline along the dim line.
        lab_offset = 1.5
        c.saveState()
        c.translate(midx + nx * lab_offset, midy + ny * lab_offset)
        c.rotate(_m.degrees(ang))
        c.setFont(label_font, label_size)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawCentredString(0, 0.6, label)
        c.restoreState()
        return

    # Leader-line callout: dim line is too short for an inline label, so
    # extend a leader perpendicular to the dim line out to where the label
    # can sit horizontally. Keep the leader short so the label fits inside
    # the page margins on either side.
    leader_len_pt = 5 * mm
    elx = midx + nx * leader_len_pt
    ely = midy + ny * leader_len_pt
    c.setLineWidth(0.35)
    c.setStrokeColor(colors.HexColor("#202020"))
    c.line(midx, midy, elx, ely)

    # Arrowhead at the dim-line end of the leader, pointing into the dim line
    # so it reads as "this label refers to that dimension".
    c.saveState()
    leader_ang = _m.atan2(ely - midy, elx - midx)
    c.translate(midx, midy)
    c.rotate(_m.degrees(leader_ang + _m.pi))
    path = c.beginPath()
    path.moveTo(0, 0)
    path.lineTo(arrow_len, arrow_len * 0.35)
    path.lineTo(arrow_len, -arrow_len * 0.35)
    path.close()
    c.setFillColor(colors.HexColor("#202020"))
    c.drawPath(path, fill=1, stroke=0)
    c.restoreState()

    # Label sits at the leader's outer end, horizontally. The text-anchor
    # direction follows the leader so the label flows away from the drawing.
    c.setFont(label_font, label_size)
    c.setFillColor(colors.HexColor("#101010"))
    if nx >= 0:
        c.drawString(elx + 1.5, ely - 1.4, label)
    else:
        c.drawRightString(elx - 1.5, ely - 1.4, label)


def _fit_image_rect(page_x: float, page_y: float,
                    page_w: float, page_h: float,
                    image_aspect: float) -> Tuple[float, float, float, float]:
    """Return (x, y, w, h) of the rectangle the image will actually occupy
    inside the page box, after preserveAspectRatio centering."""
    box_aspect = page_w / page_h
    if box_aspect > image_aspect:
        # Page box is wider than image; image fits to height, horizontal margins.
        actual_h = page_h
        actual_w = actual_h * image_aspect
        actual_x = page_x + (page_w - actual_w) / 2
        actual_y = page_y
    else:
        # Page box is taller than image; image fits to width, vertical margins.
        actual_w = page_w
        actual_h = actual_w / image_aspect
        actual_x = page_x
        actual_y = page_y + (page_h - actual_h) / 2
    return actual_x, actual_y, actual_w, actual_h


def _draw_image_with_callouts(
    c: rl_canvas.Canvas,
    image_path: str,
    page_x: float, page_y: float,
    page_w: float, page_h: float,
    callouts: list,
    image_aspect: float = 1200 / 900,
) -> None:
    """Draw a PNG into a page box and overlay dimension callouts.

    Callouts are positioned in fractions (0..1) of the *displayed image*,
    not of the page box. We resolve the actual image rectangle (after
    preserveAspectRatio centring) and anchor everything to that rectangle
    so the dim lines land on the line-art edges they're meant to mark.
    """
    img_x, img_y, img_w, img_h = _fit_image_rect(page_x, page_y,
                                                  page_w, page_h, image_aspect)
    c.drawImage(image_path, img_x, img_y, img_w, img_h,
                preserveAspectRatio=False, anchor="sw", mask="auto")
    for entry in callouts:
        label, (rx1, ry1), (rx2, ry2), side = entry
        p1 = (img_x + rx1 * img_w, img_y + ry1 * img_h)
        p2 = (img_x + rx2 * img_w, img_y + ry2 * img_h)
        _draw_dim_callout(c, p1, p2, label, side=side)


# Title block + scale bar

def _draw_title_block(c: rl_canvas.Canvas, header: TechPackHeader,
                       sheet_label: str, sheet_num: int, sheets_total: int,
                       scale_label: str, page_w_mm: float, page_h_mm: float,
                       margin_lr_mm: float = 15, margin_tb_mm: float = 22) -> None:
    """Engineering-style title block in the lower-right corner."""
    tb_w_mm = 90
    tb_h_mm = 24
    x0 = (page_w_mm - margin_lr_mm - tb_w_mm) * mm
    y0 = (margin_tb_mm + 2) * mm
    w = tb_w_mm * mm
    h = tb_h_mm * mm

    c.saveState()
    c.setStrokeColor(colors.HexColor("#202020"))
    c.setLineWidth(0.5)
    c.rect(x0, y0, w, h)
    # Two columns, four rows
    c.line(x0 + w / 2, y0, x0 + w / 2, y0 + h)
    for i in range(1, 4):
        c.line(x0, y0 + i * h / 4, x0 + w, y0 + i * h / 4)

    cell_w = w / 2
    max_text_w = cell_w - 3 * mm  # 1.5 mm padding each side

    def _cell(col, row, label, value, bold_value=True):
        cx = x0 + col * cell_w + 1.5 * mm
        cy = y0 + h - (row + 1) * (h / 4) + 0.6 * mm
        c.setFont("Helvetica", 6)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(cx, cy + (h / 4) - 3 * mm, label)
        # Auto-shrink font size if the value would overflow the cell.
        font_name = "Helvetica-Bold" if bold_value else "Helvetica"
        size = 8.5
        while size >= 5.5 and c.stringWidth(value, font_name, size) > max_text_w:
            size -= 0.5
        c.setFont(font_name, size)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawString(cx, cy, value)

    _cell(0, 0, "MODEL",  header.model_id)
    _cell(1, 0, "NAME",   header.model_name)
    _cell(0, 1, "SHEET",  f"{sheet_label}  ({sheet_num}/{sheets_total})")
    _cell(1, 1, "DATE",   header.date_iso)
    _cell(0, 2, "SCALE",  scale_label)
    _cell(1, 2, "UNITS",  "mm")
    _cell(0, 3, "DRAWN",  "auto-pipeline")
    _cell(1, 3, "REV",    "A")
    c.restoreState()


def _draw_scale_bar(c: rl_canvas.Canvas,
                     x0_mm: float, y0_mm: float,
                     mm_per_page_mm: float,
                     bar_length_mm: float = 50.0) -> None:
    """Draw a black/white striped scale bar showing the real-world length
    that occupies `bar_length_mm` of actual model. The bar's page-length is
    `bar_length_mm * mm_per_page_mm` (page mm per model mm)."""
    page_len_mm = bar_length_mm * mm_per_page_mm
    x = x0_mm * mm
    y = y0_mm * mm
    height_mm = 1.6
    # Five 10 mm segments, alternating fill.
    seg_mm = 10
    n_seg = int(bar_length_mm / seg_mm)
    seg_page_mm = page_len_mm / n_seg
    c.saveState()
    c.setStrokeColor(colors.HexColor("#202020"))
    c.setLineWidth(0.4)
    for i in range(n_seg):
        sx = x + i * seg_page_mm * mm
        c.setFillColor(colors.HexColor("#202020") if i % 2 == 0
                        else colors.white)
        c.rect(sx, y, seg_page_mm * mm, height_mm * mm,
               fill=1, stroke=1)
    # Tick labels under the bar at 0 / mid / end
    c.setFont("Helvetica", 6)
    c.setFillColor(colors.HexColor("#202020"))
    for i, label in [(0, "0"),
                      (n_seg // 2, str(seg_mm * (n_seg // 2))),
                      (n_seg, f"{int(bar_length_mm)} mm")]:
        tx = x + i * seg_page_mm * mm
        c.drawCentredString(tx, y - 2.2 * mm, label)
    c.restoreState()



# Multi-page line-art tech drawings PDF

def write_techdrawings(
    output_path: str,
    header: TechPackHeader,
    lineart_paths: Dict[str, str],
    ga: GeometryAnalysis,
    exploded_path: Optional[str] = None,
    exploded_meta: Optional[dict] = None,
    section_path: Optional[str] = None,
) -> str:
    """Multi-page A4 landscape line-art tech drawings.

    One page per drawing:
      1. Lateral profile (line-art) with length / height / sole / midsole /
         heel-height / toe-spring dimension callouts.
      2. Top (plan) view with width dimension callout.
      3. Exploded view (when supplied).
      4. Sagittal section (when supplied).

    Each page has a title block and a simple scale bar.
    """
    page_w, page_h = landscape(A4)
    m_lr = 15 * mm
    m_tb = 22 * mm

    c = rl_canvas.Canvas(output_path, pagesize=landscape(A4))
    c.setTitle(f"{header.model_id} · Tech Drawings")
    c.setAuthor("techpack-pipeline")

    m = ga.measurements

    def _draw_image_fit(image_path, x, y, w, h):
        if not (image_path and os.path.exists(image_path)):
            return None
        with Image.open(image_path) as im:
            iw, ih = im.size
        aspect = iw / ih if ih else 1.0
        if w / aspect <= h:
            dw, dh = w, w / aspect
        else:
            dw, dh = h * aspect, h
        dx = x + (w - dw) / 2
        dy = y + (h - dh) / 2
        c.drawImage(image_path, dx, dy, dw, dh,
                    preserveAspectRatio=False, anchor="sw", mask="auto")
        return dx, dy, dw, dh

    def _draw_page_chrome(page_label, page_num, total):
        _draw_header_footer(c, header, page_label, page_num, page_w, page_h)
        # Title block (bottom-right corner).
        bx = page_w - m_lr - 70 * mm
        by = m_tb - 18
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.4)
        c.rect(bx, by, 70 * mm, 14 * mm, stroke=1, fill=0)
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawString(bx + 2 * mm, by + 10 * mm,
                     f"{header.model_id}  ·  {header.model_name}")
        c.setFont("Helvetica", 6.5)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(bx + 2 * mm, by + 6 * mm,
                     f"Drawn {header.date_iso}   Sheet {page_num} of {total}")
        c.drawString(bx + 2 * mm, by + 2 * mm,
                     "All dimensions in mm  ·  Tolerances ±1 mm")

    def _draw_dimension_label(x, y, text):
        """Small black text on a faint white background — used for the
        leader-line dimension callouts."""
        c.setFont("Helvetica", 7)
        tw = c.stringWidth(text, "Helvetica", 7) + 3
        c.setFillColor(colors.white)
        c.rect(x - 1.5, y - 1, tw, 9, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawString(x, y, text)

    def _draw_h_dim(x0, x1, y, text):
        """Horizontal dimension line with arrowheads + label centred above."""
        c.setStrokeColor(colors.HexColor("#222222"))
        c.setLineWidth(0.5)
        c.line(x0, y, x1, y)
        for tip, sign in ((x0, +1), (x1, -1)):
            c.line(tip, y, tip + sign * 4, y + 2)
            c.line(tip, y, tip + sign * 4, y - 2)
        _draw_dimension_label((x0 + x1) / 2 - 8, y + 3, text)

    def _draw_v_dim(x, y0, y1, text):
        """Vertical dimension line with arrowheads + label centred to the right."""
        c.setStrokeColor(colors.HexColor("#222222"))
        c.setLineWidth(0.5)
        c.line(x, y0, x, y1)
        for tip, sign in ((y0, +1), (y1, -1)):
            c.line(x, tip, x - 2, tip + sign * 4)
            c.line(x, tip, x + 2, tip + sign * 4)
        _draw_dimension_label(x + 4, (y0 + y1) / 2 - 3, text)

    # Total page count (we may skip exploded/section if their renders are missing).
    total_pages = 2 + (1 if exploded_path and os.path.exists(exploded_path) else 0) \
                    + (1 if section_path and os.path.exists(section_path) else 0)

    # ===== Page 1: Lateral profile =====
    page = 1
    _draw_page_chrome("TECH DRAWINGS · LATERAL", page, total_pages)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#101010"))
    c.drawString(m_lr, page_h - m_tb - 8, "Lateral Profile (Line Art)")

    img_y0 = m_tb + 18
    img_h = page_h - m_tb - 26 - img_y0
    img_x0 = m_lr + 14
    img_w = page_w - 2 * m_lr - 28
    rect = _draw_image_fit(lineart_paths.get("side-lateral"),
                            img_x0, img_y0, img_w, img_h)
    if rect:
        dx, dy, dw, dh = rect
        # The Freestyle render letterboxes the shoe — the camera's
        # ortho_scale is chosen to fit (length, height*aspect)*1.08 inside
        # the larger sensor dimension, so the shoe occupies only part of
        # the image rect. Compute the shoe's actual fractional bbox
        # within (dx,dy,dw,dh) so dimension callouts pin to the silhouette,
        # not the empty letterbox edges.
        img_aspect = dw / dh if dh else 1.0
        L = m.length_mm
        H = m.height_mm
        ortho = max(L, H * img_aspect) * 1.08
        fL = L / ortho                       # fraction of image width
        fH = H / (ortho / img_aspect)        # fraction of image height
        x_lo = 0.5 - fL / 2
        x_hi = 0.5 + fL / 2
        y_lo = 0.5 - fH / 2
        y_hi = 0.5 + fH / 2
        # Convert to page coords (origin bottom-left).
        sx_lo = dx + x_lo * dw
        sx_hi = dx + x_hi * dw
        sy_lo = dy + y_lo * dh
        sy_hi = dy + y_hi * dh

        # Length dimension under the actual shoe.
        _draw_h_dim(sx_lo, sx_hi, sy_lo - 12, f"L  {m.length_mm:.0f} mm")
        # Height dimension on the right edge of the actual shoe.
        _draw_v_dim(sx_hi + 12, sy_lo, sy_hi, f"H  {m.height_mm:.0f} mm")

        # Stack thicknesses anchored to the shoe's bottom-left (heel).
        # All anchored at the bottom of the shoe silhouette (sy_lo).
        sole_h = m.sole_thickness_mm / max(H, 1) * (sy_hi - sy_lo)
        mid_h = m.midsole_thickness_mm / max(H, 1) * (sy_hi - sy_lo)
        heel_h = m.heel_height_mm / max(H, 1) * (sy_hi - sy_lo)
        _draw_v_dim(sx_lo - 14, sy_lo, sy_lo + sole_h,
                     f"sole {m.sole_thickness_mm:.0f}")
        _draw_v_dim(sx_lo - 30, sy_lo + sole_h, sy_lo + sole_h + mid_h,
                     f"mid  {m.midsole_thickness_mm:.0f}")
        _draw_v_dim(sx_lo - 46, sy_lo, sy_lo + heel_h,
                     f"heel {m.heel_height_mm:.0f}")
    c.showPage()

    # ===== Page 2: Top (plan) view =====
    page += 1
    _draw_page_chrome("TECH DRAWINGS · PLAN", page, total_pages)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#101010"))
    c.drawString(m_lr, page_h - m_tb - 8, "Top (Plan) View (Line Art)")

    rect = _draw_image_fit(lineart_paths.get("top"),
                            img_x0, img_y0, img_w, img_h)
    if rect:
        dx, dy, dw, dh = rect
        # Top view: camera frames (width, length), image-up = +length.
        # So image horizontal extent = width, image vertical extent =
        # length. ortho_scale = max(width, length*aspect) * 1.08.
        img_aspect = dw / dh if dh else 1.0
        W = m.width_mm
        L = m.length_mm
        ortho = max(W, L * img_aspect) * 1.08
        fW = W / ortho
        fL = L / (ortho / img_aspect)
        x_lo = 0.5 - fW / 2
        x_hi = 0.5 + fW / 2
        y_lo = 0.5 - fL / 2
        y_hi = 0.5 + fL / 2
        sx_lo = dx + x_lo * dw
        sx_hi = dx + x_hi * dw
        sy_lo = dy + y_lo * dh
        sy_hi = dy + y_hi * dh
        _draw_h_dim(sx_lo, sx_hi, sy_lo - 12, f"W  {m.width_mm:.0f} mm")
        _draw_v_dim(sx_hi + 12, sy_lo, sy_hi, f"L  {m.length_mm:.0f} mm")
    c.showPage()

    # ===== Page 3: Exploded view =====
    if exploded_path and os.path.exists(exploded_path):
        page += 1
        _draw_page_chrome("TECH DRAWINGS · EXPLODED", page, total_pages)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawString(m_lr, page_h - m_tb - 8, "Exploded View")
        _draw_image_fit(exploded_path, img_x0, img_y0, img_w, img_h)
        # Footer note about explosion offsets, if available.
        if exploded_meta and "offsets_mm" in exploded_meta:
            c.setFont("Helvetica", 7.5)
            c.setFillColor(colors.HexColor("#444444"))
            c.drawString(m_lr, m_tb - 8,
                          "Components separated vertically along +height "
                          "for assembly clarity.")
        c.showPage()

    # ===== Page 4: Sagittal section =====
    if section_path and os.path.exists(section_path):
        page += 1
        _draw_page_chrome("TECH DRAWINGS · SECTION A-A", page, total_pages)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.HexColor("#101010"))
        c.drawString(m_lr, page_h - m_tb - 8, "Sagittal Section (Medial Cut)")
        _draw_image_fit(section_path, img_x0, img_y0, img_w, img_h)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(m_lr, m_tb - 8,
                      "Section taken on the YZ plane through the shoe's "
                      "centreline (perpendicular to the lateral-medial axis).")
        c.showPage()

    c.save()
    return output_path
