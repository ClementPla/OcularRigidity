"""Export a pulsation scene (see :mod:`.displacement3d`) to an animated .blend file."""

from __future__ import annotations

import sys

import bpy
import numpy as np
import pyvista as pv
from bpy_extras.anim_utils import action_get_channelbag_for_slot
from skimage.measure import marching_cubes

from ocularrigidity.viewer.displacement3d import _sample, _vessel_mesh

def _ramp_stops(cmap="RdBu_r", n=9):
    """``(position, linear RGBA)`` stops of a matplotlib colormap (ramps work in linear space)."""
    from matplotlib import colormaps

    x = np.linspace(0, 1, n)
    srgb = colormaps[cmap](x)[:, :3]
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [(float(p), (*map(float, c), 1.0)) for p, c in zip(x, lin)]


def _mesh_object(name, verts_mm, faces, collection):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts_mm.tolist(), [], faces.tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj


def _pulsation_nodes(lim_um):
    """Geometry nodes: z = rest_z + amplify * (z - rest_z)"""
    ng = bpy.data.node_groups.new("Pulsation", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    amp = ng.interface.new_socket("Amplify", in_out="INPUT", socket_type="NodeSocketFloat")
    amp.default_value, amp.min_value = 25.0, 0.0
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, link = ng.nodes, ng.links.new

    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    pos = n.new("GeometryNodeInputPosition")
    xyz = n.new("ShaderNodeSeparateXYZ")
    rest = n.new("GeometryNodeInputNamedAttribute")
    rest.data_type = "FLOAT"
    rest.inputs["Name"].default_value = "rest_z"
    disp = n.new("ShaderNodeMath")  # z - rest_z, mm (negative = thicker choroid)
    disp.operation = "SUBTRACT"
    to_um = n.new("ShaderNodeMath")
    to_um.operation = "MULTIPLY"
    to_um.inputs[1].default_value = -1000.0  # true thickening, um
    gain = n.new("ShaderNodeMath")  # amplify - 1
    gain.operation = "SUBTRACT"
    gain.inputs[1].default_value = 1.0
    extra = n.new("ShaderNodeMath")  # (amplify - 1) * (z - rest_z)
    extra.operation = "MULTIPLY"
    offset = n.new("ShaderNodeCombineXYZ")
    store = n.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT", "POINT"
    store.inputs["Name"].default_value = "disp_um"
    setpos = n.new("GeometryNodeSetPosition")

    link(pos.outputs["Position"], xyz.inputs[0])
    link(xyz.outputs["Z"], disp.inputs[0])
    link(rest.outputs["Attribute"], disp.inputs[1])
    link(disp.outputs[0], to_um.inputs[0])
    link(gin.outputs["Amplify"], gain.inputs[0])
    link(gain.outputs[0], extra.inputs[0])
    link(disp.outputs[0], extra.inputs[1])
    link(extra.outputs[0], offset.inputs["Z"])
    link(gin.outputs["Geometry"], store.inputs["Geometry"])
    link(to_um.outputs[0], store.inputs["Value"])
    link(store.outputs["Geometry"], setpos.inputs["Geometry"])
    link(offset.outputs[0], setpos.inputs["Offset"])
    link(setpos.outputs["Geometry"], gout.inputs["Geometry"])
    return ng, amp.identifier


def _displacement_material(name, lim_um, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    n, link = mat.node_tree.nodes, mat.node_tree.links.new
    bsdf = n["Principled BSDF"]
    attr = n.new("ShaderNodeAttribute")
    attr.attribute_type, attr.attribute_name = "GEOMETRY", "disp_um"
    rng = n.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = -lim_um
    rng.inputs["From Max"].default_value = lim_um
    ramp = n.new("ShaderNodeValToRGB")
    stops = _ramp_stops()
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = stops[0]
    els[1].position, els[1].color = stops[-1]
    for pos, color in stops[1:-1]:
        els.new(pos).color = color
    link(attr.outputs["Fac"], rng.inputs["Value"])
    link(rng.outputs["Result"], ramp.inputs["Fac"])
    link(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.surface_render_method = "BLENDED"
    return mat


def _flat_material(name, rgba, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.35
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.surface_render_method = "BLENDED"
    return mat


def _animate(obj, rest_mm, disp_mm, frames_per_phase, controls, ng, amp_id):
    """One shape key per phase + looping weights + the pulsation modifier."""
    T = disp_mm.shape[0]
    me = obj.data
    rest_z = me.attributes.new("rest_z", "FLOAT", "POINT")
    rest_z.data.foreach_set("value", rest_mm[:, 2].astype(np.float32))
    obj.shape_key_add(name="Basis", from_mix=False)
    keys = []
    for t in range(T):
        sk = obj.shape_key_add(name=f"phase_{t:02d}", from_mix=False)
        co = rest_mm.copy()
        co[:, 2] -= disp_mm[t]  # thicker choroid = CSI deeper
        sk.data.foreach_set("co", co.astype(np.float32).ravel())
        keys.append(sk)
    # Weight of phase t: 1 on its frame, 0 on the neighbours' (linear), period T phases.
    for t, sk in enumerate(keys):
        for k in range(T + 1):
            sk.value = 1.0 if (k % T) == t else 0.0
            sk.keyframe_insert("value", frame=1 + k * frames_per_phase)
    ad = me.shape_keys.animation_data
    for fc in action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.modifiers.new("CYCLES")

    mod = obj.modifiers.new("Pulsation", "NODES")
    mod.node_group = ng
    drv = mod.driver_add(f'["{amp_id}"]').driver
    drv.type = "AVERAGE"
    var = drv.variables.new()
    var.name = "amplify"
    var.targets[0].id = controls
    var.targets[0].data_path = '["amplify"]'
    obj.parent = controls


def export(
    scene: dict,
    out_path: str,
    amplify: float = 25.0,
    depth_scale: float = 2.0,
    frames_per_phase: int = 2,
    fps: int = 24,
    csi_step: int = 4,
):
    mask, dct, ct = scene["mask"], scene["dct"], np.nan_to_num(scene["ct"])
    px_um, dz_um = tuple(scene["px_um"]), float(scene["dz_um"])
    inside = np.isfinite(dct[0])
    dct0 = np.nan_to_num(dct)
    T = dct.shape[0]
    lim = float(np.nanpercentile(np.abs(dct), 99))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = str(scene.get("title", "pulsation"))
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, T * frames_per_phase
    col = sc.collection

    # --- vessels (um -> mm), centred on the field
    vmesh = _vessel_mesh(mask & inside[:, None, :], px_um, dz_um, 1.0)
    P = np.asarray(vmesh.points, dtype=np.float64)
    faces = vmesh.faces.reshape(-1, 4)[:, 1:]
    center = np.array([P[:, 0].mean(), P[:, 1].mean(), 0.0])
    ct_pts = _sample(ct, P[:, 0], P[:, 1], px_um)
    frac = np.clip(-P[:, 2] / np.maximum(ct_pts, 1), 0, 1)
    disp = np.stack([_sample(dct0[t], P[:, 0], P[:, 1], px_um) * frac for t in range(T)])

    controls = bpy.data.objects.new("Controls", None)
    controls.empty_display_type = "ARROWS"
    controls.scale = (1.0, 1.0, depth_scale)
    controls["amplify"] = float(amplify)
    controls.id_properties_ui("amplify").update(min=0.0, soft_max=100.0, description="Displacement exaggeration")
    col.objects.link(controls)

    ng, amp_id = _pulsation_nodes(lim)
    vessels = _mesh_object("Choroid vessels", (P - center) / 1000.0, faces, col)
    vessels.data.materials.append(_displacement_material("Displacement", lim))
    _animate(vessels, (P - center) / 1000.0, disp / 1000.0, frames_per_phase, controls, ng, amp_id)

    # --- CSI surface on a subsampled grid, quads fully inside the field
    s = csi_step
    H, W = ct.shape
    rr, cc = np.mgrid[0:H:s, 0:W:s]
    h, w = rr.shape
    G = np.c_[(cc * px_um[1]).ravel(), (-rr * px_um[0]).ravel(), -ct[::s, ::s].ravel()]
    ok = inside[::s, ::s].ravel()
    i = np.arange(h * w).reshape(h, w)
    quads = np.c_[i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()]
    quads = quads[ok[quads].all(axis=1)]
    used = np.unique(quads)
    remap = np.full(h * w, -1)
    remap[used] = np.arange(used.size)
    csi_disp = np.stack([dct0[t, ::s, ::s].ravel()[used] for t in range(T)])
    csi = _mesh_object("CSI", (G[used] - center) / 1000.0, remap[quads], col)
    csi.data.materials.append(_displacement_material("CSI", lim, alpha=0.3))
    _animate(csi, (G[used] - center) / 1000.0, csi_disp / 1000.0, frames_per_phase, controls, ng, amp_id)

    # --- retinal vessels, static
    if "retina" in scene:
        verts, rfaces, *_ = marching_cubes(
            np.pad(scene["retina"] & inside[:, None, :], 1).astype(np.float32),
            0.5,
            spacing=(px_um[0], dz_um, px_um[1]),
        )
        verts -= np.array([px_um[0], dz_um, px_um[1]])
        R = np.c_[verts[:, 2], -verts[:, 0], -(float(scene["retina_z0_um"]) + verts[:, 1])]
        rmesh = pv.PolyData(R, np.c_[np.full(len(rfaces), 3), rfaces].ravel()).smooth_taubin(
            n_iter=30, pass_band=0.1
        )
        R, rfaces = np.asarray(rmesh.points), rmesh.faces.reshape(-1, 4)[:, 1:]
        retina = _mesh_object("Retinal vessels", (R - center) / 1000.0, rfaces, col)
        retina.data.materials.append(_flat_material("Retina", (0.66, 0.37, 0.01, 1.0), alpha=0.7))
        retina.parent = controls

    # --- camera, light, world
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.lens = 35
    cam.data.clip_start, cam.data.clip_end = 0.01, 100
    cam.location = (-3.0, -6.5, 4.5)
    track = cam.constraints.new("TRACK_TO")
    track.target = controls
    col.objects.link(cam)
    sc.camera = cam
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.6, 0.2, 0.8)
    col.objects.link(sun)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    sc.world = world

    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)
    return out_path


if __name__ == "__main__":
    import json

    scene_path, out = sys.argv[1], sys.argv[2]
    kwargs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    with np.load(scene_path) as z:
        scene = {k: z[k] for k in z.files}
    scene["title"] = str(scene.get("title", ""))
    export(scene, out, **kwargs)
    print(out)
