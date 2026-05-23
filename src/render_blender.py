"""Cycles rendering + axis detection.

Imports the GLB, detects which world axis is length/width/height from the
bounding box and sole flatness, then frames seven cameras through that
axis map. The mesh is never rotated; only cameras move. Default mode is
single-threaded deterministic so reruns are byte-identical.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Vector  # type: ignore
from PIL import Image, PngImagePlugin


def _strip_png_metadata(path: str) -> None:
    """Rewrite a PNG with no tEXt/tIME/iTXt chunks. Makes Cycles output
    byte-identical across runs (Blender embeds a creation timestamp by
    default)."""
    img = Image.open(path)
    img.load()
    # PIL keeps any PNG text in img.text/info; building a fresh PngInfo
    # without that data drops it on save.
    pnginfo = PngImagePlugin.PngInfo()
    img.save(path, "PNG", pnginfo=pnginfo, optimize=False)


# View specs in a **canonical** shoe frame (toe = +length, sole = -height,
# lateral = +width). The mapping to actual world axes is computed at runtime.

VIEWS: Dict[str, Dict] = {
    "top":           {"axis": "+height",            "ortho": True,  "label": "TOP"},
    "bottom":        {"axis": "-height",            "ortho": True,  "label": "BOTTOM"},
    # The detected +width direction happens to land on the medial side of
    # the input mesh (without a left/right-foot hint we can't reason about
    # it; pick the side that produced the lateral profile for this shoe).
    "side-lateral":  {"axis": "-width",             "ortho": True,  "label": "LATERAL"},
    "side-medial":   {"axis": "+width",             "ortho": True,  "label": "MEDIAL"},
    "front":         {"axis": "+length",            "ortho": True,  "label": "FRONT"},
    "back":          {"axis": "-length",            "ortho": True,  "label": "BACK"},
    # Four 3/4 perspectives — one from each corner. W sign matches the
    # side-lateral / side-medial mapping above.
    "three-quarter":          {"axis": "3q:+L-W+H", "ortho": False, "label": "3/4 LAT-FRONT"},
    "three-quarter-med":      {"axis": "3q:+L+W+H", "ortho": False, "label": "3/4 MED-FRONT"},
    "three-quarter-lat-back": {"axis": "3q:-L-W+H", "ortho": False, "label": "3/4 LAT-BACK"},
    "three-quarter-med-back": {"axis": "3q:-L+W+H", "ortho": False, "label": "3/4 MED-BACK"},
}


# Axis detection

@dataclass
class AxisMap:
    length: str     # "X" | "Y" | "Z"
    length_sign: int
    width: str
    width_sign: int
    height: str
    height_sign: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    def vec(self, role: str) -> np.ndarray:
        """role = '+length' | '-length' | '+height' | ... -> unit vector."""
        sign = +1 if role.startswith("+") else -1
        kind = role[1:]
        axis = getattr(self, kind)
        sgn_local = getattr(self, kind + "_sign")
        v = np.zeros(3, dtype=float)
        v["XYZ".index(axis)] = sign * sgn_local
        return v

    def extent(self, kind: str) -> float:
        axis = getattr(self, kind)
        i = "XYZ".index(axis)
        return float(self.bbox_max[i] - self.bbox_min[i])

    def centre(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2


def detect_axis_map(mesh_obj) -> AxisMap:
    """Identify which world axis is length / width / height for this object."""
    pts = []
    mw = mesh_obj.matrix_world
    for v in mesh_obj.data.vertices:
        w = mw @ v.co
        pts.append((w.x, w.y, w.z))
    pts = np.asarray(pts)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    extents = bbox_max - bbox_min
    axes = ["X", "Y", "Z"]
    order = np.argsort(-extents)
    length = axes[order[0]]
    remaining = [axes[order[1]], axes[order[2]]]

    # Height = axis whose extreme slab is FLATTEST in the perpendicular
    # plane (the sole is flat). Measure flatness as the determinant of the
    # 2D covariance of the slab projected onto the other two axes — large
    # = filling the plane like a sole, small = bunched.
    def slab_planar_area(values, slab_axis):
        other = [i for i in range(3) if i != slab_axis]
        return float(np.linalg.det(np.cov(values[:, other].T)) + 1e-12)

    height_score = {}
    height_sign_for = {}
    for ax in remaining:
        i = "XYZ".index(ax)
        v = pts[:, i]
        lo = pts[v <= np.percentile(v, 5)]
        hi = pts[v >= np.percentile(v, 95)]
        lo_planar = slab_planar_area(lo, i) if len(lo) >= 5 else 0.0
        hi_planar = slab_planar_area(hi, i) if len(hi) >= 5 else 0.0
        # Asymmetry score: only one side should be flat (the sole side).
        height_score[ax] = max(lo_planar, hi_planar) / (min(lo_planar, hi_planar) + 1e-9)
        # If lo side is flatter, sole is at -axis, so height grows in +axis.
        height_sign_for[ax] = +1 if lo_planar >= hi_planar else -1
    height = max(remaining, key=lambda a: height_score[a])
    height_sign = height_sign_for[height]
    width = [a for a in remaining if a != height][0]

    # Toe vs heel: the heel cup is the tallest part of the upper. Split the
    # vertex cloud at the median of the length axis, measure max(height) on
    # each half along the height axis. The +length direction should point
    # AWAY from the half with the higher heel cup.
    i_len = "XYZ".index(length)
    i_h = "XYZ".index(height)
    height_values = pts[:, i_h] * height_sign  # so larger = higher
    median_len = np.median(pts[:, i_len])
    high_side = float(height_values[pts[:, i_len] > median_len].max())
    low_side = float(height_values[pts[:, i_len] < median_len].max())
    # If high-side has the taller cluster, that's the heel; toe is the other.
    # +length points toward the toe, i.e., away from the heel.
    length_sign = -1 if high_side > low_side else +1
    width_sign = +1

    return AxisMap(
        length=length, length_sign=length_sign,
        width=width, width_sign=width_sign,
        height=height, height_sign=height_sign,
        bbox_min=bbox_min, bbox_max=bbox_max,
    )


# Scene helpers

def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def _import_glb(path: str):
    pre = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in pre]


def _all_meshes(objs):
    return [o for o in objs if o.type == "MESH"]


def _join_into_one(objs):
    meshes = _all_meshes(objs)
    if not meshes:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _setup_world(bg_value: float = 0.94) -> None:
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg.inputs[0].default_value = (bg_value, bg_value, bg_value, 1.0)
    bg.inputs[1].default_value = 1.2
    nt.links.new(bg.outputs[0], out.inputs[0])


def _add_studio_lights(centre: np.ndarray, radius: float,
                       am: AxisMap) -> None:
    """3-point rig oriented to the shoe's canonical axes."""
    height_v = am.vec("+height")
    width_v = am.vec("+width")
    length_v = am.vec("+length")
    d = radius * 4

    def add(name, offset_canonical, energy, size):
        loc = centre + offset_canonical[0] * length_v + offset_canonical[1] * width_v + offset_canonical[2] * height_v
        ld = bpy.data.lights.new(name=name, type="AREA")
        ld.energy = energy
        ld.size = size
        obj = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(obj)
        obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))
        direction = Vector(centre.tolist()) - Vector(obj.location)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj

    e = radius ** 2 * 0.18  # tuned for mm-scale meshes
    add("KeyLight",  (d * 0.4, d * 0.7, d * 0.9), e * 4.0, radius * 2.0)
    add("FillLight", (-d * 0.3, -d * 0.6, d * 0.5), e * 2.0, radius * 2.5)
    add("RimLight",  (-d * 0.6, d * 0.3, d * 1.1), e * 2.5, radius * 1.5)


def _setup_cycles(scene, width: int, height: int, samples: int = 48,
                  deterministic: bool = True) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.cycles.seed = 0
    if deterministic:
        # Force a single CPU thread + no auto-tiling. This makes Cycles output
        # byte-deterministic at the cost of ~4x render time on a 4-core box.
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 1
        scene.cycles.use_auto_tile = False


def _make_camera() -> bpy.types.Object:
    cd = bpy.data.cameras.new("Cam")
    obj = bpy.data.objects.new("Cam", cd)
    bpy.context.collection.objects.link(obj)
    return obj


def _frame_camera(cam, view_dir: np.ndarray, up_world: np.ndarray,
                  centre: np.ndarray, radius: float,
                  in_plane: Tuple[float, float],
                  width_px: int, height_px: int,
                  ortho: bool, fov_deg: float = 28.0) -> None:
    """Set up camera fully explicitly: position, basis, projection.

    view_dir : unit vector pointing FROM the model TO the camera.
    up_world : world-space direction that should map to image-up.
    in_plane : (horizontal_extent, vertical_extent) in world units to fit.
    """
    cam_loc = centre + view_dir * radius * 6
    cam.location = (float(cam_loc[0]), float(cam_loc[1]), float(cam_loc[2]))

    # Build the camera's world basis. In Blender's convention the camera's
    # -Z faces the target; its +Y is image-up; its +X is image-right.
    z_axis = view_dir.astype(float)  # camera +Z = away from target
    up = up_world.astype(float)
    up = up - np.dot(up, z_axis) * z_axis
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        up = up - np.dot(up, z_axis) * z_axis
    up /= np.linalg.norm(up)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = up
    cam.matrix_world = Matrix((
        (float(x_axis[0]), float(y_axis[0]), float(z_axis[0]), float(cam_loc[0])),
        (float(x_axis[1]), float(y_axis[1]), float(z_axis[1]), float(cam_loc[1])),
        (float(x_axis[2]), float(y_axis[2]), float(z_axis[2]), float(cam_loc[2])),
        (0.0, 0.0, 0.0, 1.0),
    ))

    if ortho:
        cam.data.type = "ORTHO"
        aspect = width_px / height_px
        cam.data.ortho_scale = max(in_plane[0], in_plane[1] * aspect) * 1.08
    else:
        cam.data.type = "PERSP"
        cam.data.lens_unit = "FOV"
        cam.data.angle = math.radians(fov_deg)


@dataclass
class BlenderRenderConfig:
    width: int = 1100
    height: int = 800
    samples: int = 48
    deterministic: bool = True  # single-thread + no auto-tile -> byte-identical reruns


def render_all_views(
    glb_path: str,
    output_dir: str,
    cfg: Optional[BlenderRenderConfig] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, str], AxisMap, np.ndarray, Dict[str, dict]]:
    """Render every named view.

    Returns (paths, axis_map, extents_mm, view_cameras) where
    `view_cameras[name]` is a dict with the camera parameters needed to
    project a 3D world point into the rendered image (used by SAM-based
    segmentation to lift 2D masks back onto mesh faces).
    """
    cfg = cfg or BlenderRenderConfig()
    os.makedirs(output_dir, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)

    am = detect_axis_map(obj)
    centre = am.centre()
    radius = float(np.linalg.norm(am.bbox_max - am.bbox_min) / 2)
    extents = am.bbox_max - am.bbox_min
    log(f"  axes: length={am.length}{'+' if am.length_sign>0 else '-'} "
        f"width={am.width}{'+' if am.width_sign>0 else '-'} "
        f"height={am.height}{'+' if am.height_sign>0 else '-'}")
    log(f"  extents (world units): {extents.round(2).tolist()}")
    log(f"  bbox radius: {radius:.2f}")

    _setup_world(bg_value=0.95)
    _add_studio_lights(centre, radius, am)
    scene = bpy.context.scene
    _setup_cycles(scene, cfg.width, cfg.height, samples=cfg.samples,
                  deterministic=cfg.deterministic)
    cam = _make_camera()
    scene.camera = cam

    length_ext = am.extent("length")
    width_ext = am.extent("width")
    height_ext = am.extent("height")
    length_v = am.vec("+length")
    width_v = am.vec("+width")
    height_v = am.vec("+height")

    paths: Dict[str, str] = {}
    view_cameras: Dict[str, dict] = {}
    for name, spec in VIEWS.items():
        axis_spec = spec["axis"]
        if axis_spec.startswith("3q"):
            # Parse direction code, e.g. "3q:+L+W+H" → sign of length/width/height.
            code = axis_spec.split(":", 1)[1] if ":" in axis_spec else "+L+W+H"
            sL = +1 if "+L" in code else -1
            sW = +1 if "+W" in code else -1
            sH = +1 if "+H" in code else -1
            view_dir = (sL * length_v + sW * width_v * 1.1
                        + sH * height_v * 0.55)
            view_dir = view_dir / np.linalg.norm(view_dir)
            up_world = height_v
            in_plane = (length_ext * 1.05, height_ext * 1.05)
        elif axis_spec == "+height":     # top
            view_dir = height_v
            up_world = length_v
            # image_up = length → length is vertical, width horizontal.
            in_plane = (width_ext, length_ext)
        elif axis_spec == "-height":     # bottom
            view_dir = -height_v
            up_world = length_v
            in_plane = (width_ext, length_ext)
        elif axis_spec == "+width":      # lateral
            view_dir = width_v
            up_world = height_v
            in_plane = (length_ext, height_ext)
        elif axis_spec == "-width":      # medial
            view_dir = -width_v
            up_world = height_v
            in_plane = (length_ext, height_ext)
        elif axis_spec == "+length":     # front (toe-on)
            view_dir = length_v
            up_world = height_v
            in_plane = (width_ext, height_ext)
        elif axis_spec == "-length":     # back (heel-on)
            view_dir = -length_v
            up_world = height_v
            in_plane = (width_ext, height_ext)
        else:
            raise ValueError(f"unknown axis spec {axis_spec}")

        _frame_camera(cam, view_dir, up_world, centre, radius, in_plane,
                      cfg.width, cfg.height, ortho=spec["ortho"])
        out = os.path.join(output_dir, f"view_{name}.png")
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        if cfg.deterministic:
            _strip_png_metadata(out)
        paths[name] = out
        view_cameras[name] = {
            "matrix_world": np.array(cam.matrix_world),
            "ortho_scale": float(cam.data.ortho_scale) if spec["ortho"] else 0.0,
            "is_ortho": bool(spec["ortho"]),
            "width": cfg.width,
            "height": cfg.height,
            "view_dir": view_dir.tolist(),
        }
        log(f"  rendered {name}")

    return paths, am, extents, view_cameras


# Component overlay

def _group_for_component_name(name: str) -> str:
    """Map a component name to one of three exploded-view groups."""
    n = name.lower()
    if "outsole" in n:
        return "outsole"
    if "midsole" in n:
        return "midsole"
    return "upper"


def split_mesh_by_group(
    obj,
    face_component: np.ndarray,
    component_group: list,
    group_offsets_bu: dict,
) -> Tuple[list, dict]:
    """Replace `obj` with one mesh per group, each translated by group_offsets_bu.

    Returns (new_objs, face_remap) where face_remap[group_id] is an
    np.ndarray mapping the new mesh's face index → original face index.
    Used by callers that need to carry per-face attributes (component
    colour, segment id) through the split."""
    me = obj.data
    verts = np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices], dtype=np.float64)
    faces_v = [tuple(p.vertices) for p in me.polygons]
    if len(faces_v) != len(face_component):
        return [obj], {}
    new_objs = []
    face_remap: dict = {}
    for group_id, offset in group_offsets_bu.items():
        face_mask = np.array(
            [(0 <= fc < len(component_group)) and component_group[fc] == group_id
             for fc in face_component], dtype=bool,
        )
        face_indices = np.where(face_mask)[0]
        if len(face_indices) == 0:
            continue
        used_verts = set()
        for fi in face_indices:
            for vi in faces_v[fi]:
                used_verts.add(int(vi))
        used_verts_sorted = sorted(used_verts)
        remap = {old: new for new, old in enumerate(used_verts_sorted)}
        new_verts = verts[used_verts_sorted].copy()
        new_verts += np.array(offset, dtype=np.float64)
        new_faces = [[remap[int(vi)] for vi in faces_v[fi]] for fi in face_indices]
        new_me = bpy.data.meshes.new(f"explode_{group_id}")
        new_me.from_pydata(new_verts.tolist(), [], new_faces)
        new_me.update()
        new_obj = bpy.data.objects.new(f"explode_{group_id}", new_me)
        bpy.context.scene.collection.objects.link(new_obj)
        new_objs.append(new_obj)
        face_remap[group_id] = face_indices  # new face i ↔ original face indices[i]
    bpy.data.objects.remove(obj)
    return new_objs, face_remap


def render_component_overlay(
    glb_path: str,
    output_path: str,
    face_component: np.ndarray,
    palette_rgb: list,
    cfg: Optional[BlenderRenderConfig] = None,
    use_emissive: bool = False,
    view: str = "side-lateral",
) -> str:
    """Render a single view with components flat-colored from palette.

    `view` selects the camera direction; supported values:
      "side-lateral" (default, +width), "side-medial" (-width),
      "top" (+height), "bottom" (-height),
      "front" (+length), "back" (-length).
    The camera framing matches `render_all_views` for the same view, so
    id-mask centroids project onto the same pixels as the photo render.

    `use_emissive=True` paints faces with an emissive shader and disables
    studio lighting, so the rendered pixels match the input palette
    *exactly* (modulo PNG quantisation). That's required when the render
    is used as an ID-mask for nearest-colour centroid detection — under
    the default Principled BSDF + lighting, pixel colours drift 20-60
    units from the input palette and centroids smear across components.
    """
    cfg = cfg or BlenderRenderConfig()
    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)
    centre = am.centre()
    radius = float(np.linalg.norm(am.bbox_max - am.bbox_min) / 2)

    me = obj.data
    # Match counts: Blender's triangulation may differ from trimesh's; if
    # mismatched we paint everything gray.
    if len(me.polygons) != len(face_component):
        face_component = np.zeros(len(me.polygons), dtype=np.int32)
        palette_rgb = [(0.5, 0.5, 0.5)]

    if "Component" in me.color_attributes:
        me.color_attributes.remove(me.color_attributes["Component"])
    col_attr = me.color_attributes.new(name="Component", type="FLOAT_COLOR",
                                       domain="FACE")
    for fi, face in enumerate(me.polygons):
        ci = int(face_component[fi])
        rgb = palette_rgb[ci] if 0 <= ci < len(palette_rgb) else (0.5, 0.5, 0.5)
        col_attr.data[fi].color = (rgb[0], rgb[1], rgb[2], 1.0)

    for slot in obj.material_slots:
        slot.material = None
    mat_name = "ComponentOverlayEmissive" if use_emissive else "ComponentOverlay"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Component"
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    if use_emissive:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs[1].default_value = 1.0
        nt.links.new(attr.outputs["Color"], em.inputs[0])
        # Backface culling: medial-side faces of the tongue/collar mesh
        # would otherwise emit color when viewed from the lateral camera
        # and scatter pixels above/around the shoe silhouette, dragging
        # `_compute_centroids_from_id_mask`'s arithmetic mean off the
        # visible part. Mix Emission with Transparent on backfacing
        # geometry so the id-mask only contains front-facing pixels.
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
        nt.links.new(em.outputs[0], mix.inputs[1])
        nt.links.new(transp.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], output.inputs[0])
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.55
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], output.inputs[0])
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    _setup_world(bg_value=0.96)
    if not use_emissive:
        _add_studio_lights(centre, radius, am)
    scene = bpy.context.scene
    _setup_cycles(scene, cfg.width, cfg.height,
                  samples=1 if use_emissive else cfg.samples,
                  deterministic=cfg.deterministic)
    if use_emissive:
        # Emissive shader: keep colour-management linear so input → output
        # is preserved exactly. Default Filmic would shift values.
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    cam = _make_camera()
    scene.camera = cam
    # Camera direction per view name. Mirrors the per-view spec in
    # render_all_views so the id-mask projects 1:1 onto the photo render.
    _view_to_dir = {
        "side-lateral": ("+width",  "+height", ("length", "height")),
        "side-medial":  ("-width",  "+height", ("length", "height")),
        "top":          ("+height", "+length", ("width",  "length")),
        "bottom":       ("-height", "+length", ("width",  "length")),
        "front":        ("+length", "+height", ("width",  "height")),
        "back":         ("-length", "+height", ("width",  "height")),
    }
    if view not in _view_to_dir:
        raise ValueError(f"render_component_overlay: unknown view {view!r}")
    view_axis, up_axis, in_plane_axes = _view_to_dir[view]
    view_dir = am.vec(view_axis)
    up_world = am.vec(up_axis)
    in_plane = (am.extent(in_plane_axes[0]), am.extent(in_plane_axes[1]))
    _frame_camera(cam, view_dir, up_world, centre, radius, in_plane,
                  cfg.width, cfg.height, ortho=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    if cfg.deterministic:
        _strip_png_metadata(output_path)
    return output_path


# Exploded view rendered with flat per-component colours (BOM diagram source)

def render_exploded_overlay(
    glb_path: str,
    output_path: str,
    face_component: np.ndarray,
    component_names: list,
    palette_rgb: list,
    cfg: Optional[BlenderRenderConfig] = None,
) -> Tuple[str, AxisMap, dict]:
    """Render an exploded 3-group view with per-face flat component colours.

    Mirror of render_component_overlay but with the mesh split into
    outsole / midsole / upper along +height first, so the BOM page can
    label each component on a designer-style exploded diagram.

    Returns (path, axis_map, frame_meta) where frame_meta has
    `group_anchors`, `component_anchors`, `ortho_scale`, `centre`.
    """
    cfg = cfg or BlenderRenderConfig()
    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)

    # Group each segmented component into outsole / midsole / upper.
    component_group = [_group_for_component_name(n) for n in component_names]
    height_bu = am.extent("height")
    height_v = am.vec("+height")
    offsets = {
        "outsole": (0.0, 0.0, 0.0),
        "midsole": tuple(height_v * 0.7 * height_bu),
        "upper":   tuple(height_v * 1.8 * height_bu),
    }
    new_objs, face_remap = split_mesh_by_group(
        obj, face_component, component_group, offsets,
    )
    if not new_objs:
        return output_path, am, {}

    # Build the materials and per-face colour attributes on every new mesh.
    mat = bpy.data.materials.new("ExplodedOverlay")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Component"
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.55
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], output.inputs[0])

    for new_obj in new_objs:
        nme = new_obj.data
        # Remove any auto-created colour attribute.
        if "Component" in nme.color_attributes:
            nme.color_attributes.remove(nme.color_attributes["Component"])
        col_attr = nme.color_attributes.new(
            name="Component", type="FLOAT_COLOR", domain="FACE")
        group_id = new_obj.name.replace("explode_", "")
        original_face_idx = face_remap.get(group_id)
        for fi in range(len(nme.polygons)):
            orig_fi = int(original_face_idx[fi]) if original_face_idx is not None else -1
            ci = int(face_component[orig_fi]) if 0 <= orig_fi < len(face_component) else -1
            rgb = palette_rgb[ci] if 0 <= ci < len(palette_rgb) else (0.5, 0.5, 0.5)
            col_attr.data[fi].color = (rgb[0], rgb[1], rgb[2], 1.0)
        for slot in new_obj.material_slots:
            slot.material = None
        if not new_obj.data.materials:
            new_obj.data.materials.append(mat)
        else:
            new_obj.data.materials[0] = mat

    # Frame camera from the lateral direction over the now-larger bbox.
    all_verts = np.concatenate([
        np.array([(v.co.x, v.co.y, v.co.z) for v in o.data.vertices])
        for o in new_objs
    ])
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    centre = (bbox_min + bbox_max) / 2
    radius = float(np.linalg.norm(bbox_max - bbox_min) / 2)

    _setup_world(bg_value=0.96)
    _add_studio_lights(centre, radius, am)
    scene = bpy.context.scene
    _setup_cycles(scene, cfg.width, cfg.height, samples=cfg.samples,
                  deterministic=cfg.deterministic)
    cam = _make_camera()
    scene.camera = cam
    length_ext = am.extent("length")
    width_v = am.vec("+width")
    height_v_world = am.vec("+height")
    proj_height = float((bbox_max - bbox_min) @ np.abs(height_v_world))
    _frame_camera(cam, width_v, height_v_world, centre, radius,
                  (length_ext, proj_height), cfg.width, cfg.height, ortho=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    if cfg.deterministic:
        _strip_png_metadata(output_path)

    # Compute per-component image anchors so the PDF can draw arrows.
    aspect = cfg.width / cfg.height
    ortho_scale = max(length_ext, proj_height * aspect) * 1.08
    length_axis = am.vec("+length")

    def _img_xy(world_pt):
        rel = np.array(world_pt) - centre
        u = float(rel @ length_axis)
        v = float(rel @ height_v_world)
        fx = 0.5 + u / ortho_scale
        fy = 0.5 + v / (ortho_scale / aspect)
        return fx, fy

    group_anchors = {}
    component_anchors: dict = {}
    for new_obj in new_objs:
        group_id = new_obj.name.replace("explode_", "")
        original_face_idx = face_remap.get(group_id)
        vs = np.array([(v.co.x, v.co.y, v.co.z) for v in new_obj.data.vertices])
        if len(vs):
            group_anchors[group_id] = _img_xy(vs.mean(axis=0))
        if original_face_idx is None:
            continue
        # Per-component centroid within this group: average face centroid
        # of all faces belonging to each component index.
        face_centroids = np.array([
            np.mean([new_obj.data.vertices[vi].co[:] for vi in p.vertices],
                    axis=0)
            for p in new_obj.data.polygons
        ])
        if len(face_centroids) == 0:
            continue
        for ci in np.unique([face_component[int(orig_fi)]
                              for orig_fi in original_face_idx
                              if 0 <= orig_fi < len(face_component)]):
            ci = int(ci)
            mask = np.array([face_component[int(orig_fi)] == ci
                              for orig_fi in original_face_idx], dtype=bool)
            if mask.sum() < 5:
                continue
            cent = face_centroids[mask].mean(axis=0)
            component_anchors[ci] = _img_xy(cent)

    return output_path, am, {
        "group_anchors": group_anchors,
        "component_anchors": component_anchors,
        "ortho_scale": ortho_scale,
        "centre": centre.tolist(),
    }


# Face data extraction (for downstream geometry analysis)

def extract_face_data(glb_path: str, target_length_mm: float = 270.0) -> Dict:
    """Load the GLB into Blender, return per-face arrays sampled from the
    real PBR diffuse, plus an axis map and bbox in *millimetres*.

    Returns dict with keys:
      face_colors_rgb: (Nfaces, 3) uint8
      face_areas: (Nfaces,) float64 (in mm^2)
      face_centroids: (Nfaces, 3) float64 (mm, length/width/height-aligned)
      face_normals: (Nfaces, 3) float64
      axis_map: AxisMap (world axes)
      length_mm, width_mm, height_mm: float
      faces_indices: (Nfaces, 3) int32 (vertex indices)
      vertices_mm: (Nverts, 3) float64 (in canonical frame: x=length, y=width, z=height)
    """
    _clear_scene()
    objs = _import_glb(glb_path)
    obj = _join_into_one(objs)
    am = detect_axis_map(obj)
    centre = am.centre()
    ext_world = am.bbox_max - am.bbox_min
    length_world = am.extent("length")

    # Scale factor. glTF carries no enforced physical units, so the caller
    # tells us what the longest axis *should* be in mm. The default 270 mm
    # corresponds to US-9 men's footwear; pass --target-length-mm to override
    # if the shoe's real-world size is known.
    scale_mm_per_unit = target_length_mm / length_world

    # World -> canonical-mm basis (length, width, height)
    L_v = am.vec("+length") * scale_mm_per_unit
    W_v = am.vec("+width") * scale_mm_per_unit
    H_v = am.vec("+height") * scale_mm_per_unit
    centre_world = centre

    def world_to_canon(p: np.ndarray) -> np.ndarray:
        # Translate to bbox centre at (x=0, y=0, z=floor)
        d = p - centre_world
        x = float(d @ am.vec("+length")) * scale_mm_per_unit
        y = float(d @ am.vec("+width")) * scale_mm_per_unit
        z = float((d @ am.vec("+height")) * scale_mm_per_unit) + am.extent("height") * scale_mm_per_unit / 2
        return np.array([x, y, z])

    # Find diffuse image array
    diffuse_arr = None
    mat = obj.material_slots[0].material if obj.material_slots else None
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and len(node.image.pixels):
                img = node.image
                px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)
                # Convert linear to sRGB for proper RGB sampling
                lin = px[:, :, :3]
                srgb = np.where(lin <= 0.0031308,
                                12.92 * lin,
                                1.055 * np.power(np.clip(lin, 1e-9, None), 1 / 2.4) - 0.055)
                diffuse_arr = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
                diffuse_arr = np.flipud(diffuse_arr)
                break

    me = obj.data
    me.calc_loop_triangles()
    n_faces = len(me.polygons)
    face_colors = np.full((n_faces, 3), 160, dtype=np.uint8)
    face_areas = np.zeros(n_faces, dtype=np.float64)
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_centroids_world = np.zeros((n_faces, 3), dtype=np.float64)
    face_normals = np.zeros((n_faces, 3), dtype=np.float64)

    uv_layer = me.uv_layers.active.data if me.uv_layers.active else None
    mw = obj.matrix_world

    h = w = 0
    if diffuse_arr is not None:
        h, w = diffuse_arr.shape[:2]

    for fi, face in enumerate(me.polygons):
        # Centroid in world coords (Blender Units) — used by SAM lifting
        c_world = mw @ face.center
        face_centroids_world[fi] = np.array((c_world.x, c_world.y, c_world.z))
        # Centroid in canonical mm
        face_centroids[fi] = world_to_canon(np.array((c_world.x, c_world.y, c_world.z)))
        # Area in canonical mm^2
        face_areas[fi] = float(face.area) * (scale_mm_per_unit ** 2)
        # Normal in canonical frame
        n_world = (mw.to_3x3() @ face.normal)
        n_canon = np.array([
            float(np.array((n_world.x, n_world.y, n_world.z)) @ am.vec("+length")),
            float(np.array((n_world.x, n_world.y, n_world.z)) @ am.vec("+width")),
            float(np.array((n_world.x, n_world.y, n_world.z)) @ am.vec("+height")),
        ])
        nn = np.linalg.norm(n_canon)
        if nn > 0:
            n_canon /= nn
        face_normals[fi] = n_canon
        # Color via UV centroid
        if uv_layer is not None and diffuse_arr is not None:
            u = v = 0.0
            for li in face.loop_indices:
                u += uv_layer[li].uv.x
                v += uv_layer[li].uv.y
            u /= face.loop_total
            v /= face.loop_total
            px = int(np.clip(u, 0, 1) * (w - 1))
            py = int((1.0 - np.clip(v, 0, 1)) * (h - 1))
            face_colors[fi] = diffuse_arr[py, px]

    # Also assemble vertices + face-vertex indices in canonical frame for any
    # downstream code (we don't strictly need them, but it's handy).
    vertices_world = np.array([(mw @ v.co)[:] for v in me.vertices])
    vertices_canon = np.array([world_to_canon(p) for p in vertices_world])

    # Per-face UV coordinates (up to 3 per face — first three loops). Used by
    # material.py to crop the diffuse texture per component for CLIP-based
    # material classification.
    face_uvs = None
    if uv_layer is not None:
        face_uvs = np.zeros((len(me.polygons), 3, 2), dtype=np.float32)
        for fi, face in enumerate(me.polygons):
            for ki, li in enumerate(face.loop_indices[:3]):
                face_uvs[fi, ki, 0] = uv_layer[li].uv.x
                face_uvs[fi, ki, 1] = uv_layer[li].uv.y

    # Per-face vertex indices (first 3 verts, assumes triangulated mesh). Used
    # by geometry.py to compute per-component perimeter from real topology.
    face_v_idx = np.zeros((len(me.polygons), 3), dtype=np.int32)
    for fi, face in enumerate(me.polygons):
        vs = list(face.vertices)
        for ki in range(3):
            face_v_idx[fi, ki] = vs[ki] if ki < len(vs) else vs[0]

    return {
        "face_colors_rgb": face_colors,
        "face_areas_mm2": face_areas,
        "face_centroids_mm": face_centroids,
        "face_centroids_world": face_centroids_world,
        "face_normals": face_normals,
        "face_uvs": face_uvs,
        "face_v_idx": face_v_idx,
        "vertices_mm": vertices_canon,
        "axis_map": am,
        "length_mm": am.extent("length") * scale_mm_per_unit,
        "width_mm": am.extent("width") * scale_mm_per_unit,
        "height_mm": am.extent("height") * scale_mm_per_unit,
        "diffuse_image": diffuse_arr,
        "scale_mm_per_unit": scale_mm_per_unit,
    }
