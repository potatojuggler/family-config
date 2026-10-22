"""Grave Minstrel hero: models it in Blender, bakes textures, renders previews and exports FBX for Roblox.

Run with Blender as a Python module (bpy 4.5.4 on Python 3.11, plus numpy, scikit-image, Pillow):
    python build_minstrel.py            # everything
    python build_minstrel.py --shapes   # quick clay render of the detailed shapes only
Outputs land in ./out/minstrel/.

Blender units are meters, Z is up, the figure faces -Y, its left hand is +X.
Each moving piece is its own mesh so Roblox can drive it with Motor6Ds; J_* marker meshes sit on the joints.
"""
import math
import sys
import time
from functools import reduce
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from PIL import Image, ImageFilter
from skimage.measure import marching_cubes

OUT = Path(__file__).resolve().parent / "out" / "minstrel"
TEX = 1024  # Roblox's SurfaceAppearance maximum
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- signed distance fields (numpy, points are (N, 3))

def V(*a):
    return np.array(a, np.float32)


def length(v):
    return np.sqrt((v * v).sum(-1))


def sd_sphere(P, c, r):
    return length(P - V(*c)) - r


def sd_ellipsoid(P, c, r):
    r = V(*r)
    q = (P - V(*c)) / r
    k0, k1 = length(q), length(q / r)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def sd_cone(P, a, b, ra, rb):
    """Capsule whose radius goes from ra at a to rb at b (Inigo Quilez's round cone)."""
    a, b = V(*a), V(*b)
    ba = b - a
    l2 = float(ba @ ba)
    rr = ra - rb
    a2 = l2 - rr * rr
    pa = P - a
    y = pa @ ba
    z = y - l2
    x2 = length(pa * l2 - y[:, None] * ba) ** 2
    y2, z2 = y * y * l2, z * z * l2
    k = np.sign(rr) * rr * rr * x2
    return np.where(np.sign(z) * a2 * z2 > k, np.sqrt(x2 + z2) / l2 - rb,
                    np.where(np.sign(y) * a2 * y2 < k, np.sqrt(x2 + y2) / l2 - ra,
                             (np.sqrt(np.maximum(x2 * a2 / l2, 0)) + y * rr) / l2 - ra))


def sd_box(P, c, half, r=0.0, R=None):
    q = P - V(*c)
    if R is not None:
        q = q @ R.astype(np.float32)  # world -> box axes (columns of R)
    q = np.abs(q) - (V(*half) - r)
    return length(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(ds, k):
    return reduce(lambda a, b: smin(a, b, k), ds)


def wobble(P, amp, freq, seed, n=6):
    """Cheap smooth 3D noise: a few random plane waves."""
    r = np.random.default_rng(seed)
    dirs = r.normal(size=(n, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = np.zeros(len(P), np.float32)
    for i in range(n):
        out += np.sin(P @ dirs[i] * freq * (1 + 0.37 * i) + r.random() * 6.3) / (1 + 0.37 * i)
    return amp * out / 2


def rot(axis, deg):
    return np.array(Matrix.Rotation(math.radians(deg), 3, axis))


# ---------------------------------------------------------------- blender helpers

def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_obj(name, verts, faces, matrix=None, mat_index=None):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    me.loops.add(len(faces) * 3)
    me.loops.foreach_set("vertex_index", np.asarray(faces, np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32))
    if mat_index is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_index, np.int32))
    me.update(calc_edges=True)
    me.validate()
    me.shade_smooth()
    obj = link(bpy.data.objects.new(name, me))
    if matrix is not None:
        obj.matrix_world = Matrix(matrix)
    return obj


def sdf_obj(name, f, lo, hi, vox, matrix=None, materials=(), region=None):
    """Marching-cubes mesh of the SDF f inside box lo..hi (local coords); region(face centers) picks material slots."""
    xs = [np.arange(lo[i], hi[i] + vox, vox, dtype=np.float32) for i in range(3)]
    P = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    vol = f(P).reshape([len(x) for x in xs])
    vol[[0, -1]] = vol[:, [0, -1]] = vol[:, :, [0, -1]] = vox  # close anything the box cuts
    v, fc, _, _ = marching_cubes(vol, 0.0, spacing=(vox,) * 3)
    v = v.astype(np.float32) + V(*lo)
    tri = v[fc]
    if (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * tri.mean(1)).sum() < 0:  # want outward normals
        fc = fc[:, ::-1]
    mi = region(v[fc].mean(1)) if region else None
    obj = mesh_obj(name, v, fc, matrix, mi)
    for m in materials:
        obj.data.materials.append(m)
    log(f"  {name}: {len(fc)} tris")
    return obj


def decimated(src, name, tris):
    """Low-poly copy of src with about `tris` triangles (keeps src's transform and materials)."""
    obj = link(bpy.data.objects.new(name, src.data.copy()))
    obj.matrix_world = src.matrix_world
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = min(1.0, tris / max(1, len(src.data.polygons)))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    obj.modifiers.clear()
    old, obj.data = obj.data, me
    bpy.data.meshes.remove(old)
    me.name = name
    me.shade_smooth()
    return obj


def join(objs, name):
    """Copies of objs merged into one new object (originals untouched)."""
    copies = []
    for o in objs:
        c = link(bpy.data.objects.new(name, o.data.copy()))
        c.matrix_world = o.matrix_world
        copies.append(c)
    select(copies, active=copies[0])
    bpy.ops.object.join()
    out = bpy.context.view_layer.objects.active
    out.name = name
    return out


def select(objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


# ---------------------------------------------------------------- materials (procedural, baked into the atlas)

class Nodes:
    """Tiny node-graph builder: n(type, **inputs) where inputs are values, sockets, or _property settings."""

    def __init__(self, mat):
        self.nt = mat.node_tree

    def __call__(self, kind, *args, **kw):
        node = self.nt.nodes.new(kind)
        for i, v in list(enumerate(args)) + list(kw.items()):
            if isinstance(i, str) and i.startswith("_"):
                setattr(node, i[1:], v)
                continue
            sock = node.inputs[i]
            if isinstance(v, bpy.types.NodeSocket):
                self.nt.links.new(v, sock)
            else:
                sock.default_value = v
        return node

    def math(self, op, a, b=0.0, c=0.0):
        return self("ShaderNodeMath", a, b, c, _operation=op).outputs[0]

    def ramp(self, fac, stops):
        node = self("ShaderNodeValToRGB", Fac=fac)
        els = node.color_ramp.elements
        for i, (pos, col) in enumerate(stops):
            el = els[i] if i < 2 else els.new(pos)
            el.position, el.color = pos, (*col, 1)
        return node.outputs["Color"]

    def range(self, v, a, b, lo=0.0, hi=1.0):
        return self("ShaderNodeMapRange", v, a, b, lo, hi, _interpolation_type="SMOOTHSTEP").outputs[0]


def new_mat(name, color, rough, height=None, bump=0.5, dist=0.002):
    """color/rough/height are callables (n, co) -> socket, so each material only describes its pattern."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = Nodes(m)
    co = n("ShaderNodeTexCoord").outputs["Object"]
    bsdf = m.node_tree.nodes["Principled BSDF"]
    n.nt.links.new(color(n, co), bsdf.inputs["Base Color"])
    r = rough(n, co)
    if isinstance(r, float):
        bsdf.inputs["Roughness"].default_value = r
    else:
        n.nt.links.new(r, bsdf.inputs["Roughness"])
    if height:
        b = n("ShaderNodeBump", Strength=bump, Distance=dist, Height=height(n, co))
        n.nt.links.new(b.outputs[0], bsdf.inputs["Normal"])
    return m


def noise(n, co, scale, detail=4.0, distortion=0.0, rough=0.5):
    return n("ShaderNodeTexNoise", Vector=co, Scale=scale, Detail=detail, Distortion=distortion,
             Roughness=rough).outputs["Fac"]


def velvet_pattern(n, co):
    """Crushed velvet: dark mottling, swirly bright streaks running down the cloth, crumple and glitter, as 0..1."""
    tall = n("ShaderNodeMapping", Vector=co, Scale=(1.0, 1.0, 0.45)).outputs[0]
    swirl = n("ShaderNodeTexWave", Vector=tall, Scale=18.0, Distortion=10.0, Detail=4.0, _wave_type="BANDS",
              _bands_direction="DIAGONAL").outputs["Fac"]
    streak = n.range(swirl, 0.72, 0.95)
    crumple = noise(n, tall, 40.0, 10.0, 3.0, 0.65)
    mottle = noise(n, co, 2.5, 2.0)
    glitter = n.range(noise(n, co, 900.0, 0.0), 0.74, 0.8)
    v = n.math("MULTIPLY_ADD", mottle, 0.38, n.math("MULTIPLY_ADD", n.range(crumple, 0.35, 0.8), 0.32, 0.02))
    v = n.math("MULTIPLY_ADD", streak, 0.28, v)
    crinkle = n.math("MAXIMUM", streak, n.range(crumple, 0.6, 0.75))
    return n.math("ADD", v, n.math("MULTIPLY", glitter, 0.45)), crinkle, glitter


def mat_velvet():
    def color(n, co):
        v, _, _ = velvet_pattern(n, co)
        return n.ramp(v, [(0.0, (0.01, 0.0, 0.003)), (0.3, (0.08, 0.003, 0.009)), (0.55, (0.30, 0.014, 0.024)),
                          (0.8, (0.62, 0.07, 0.06)), (1.0, (0.9, 0.22, 0.17))])

    def rough(n, co):
        _, crinkle, glitter = velvet_pattern(n, co)
        return n.math("MULTIPLY_ADD", n.math("ADD", crinkle, glitter), -0.45, 0.75)

    return new_mat("Velvet", color, rough, lambda n, co: velvet_pattern(n, co)[0], bump=0.7, dist=0.003)


def mat_bone():
    def color(n, co):
        v = n.math("MULTIPLY_ADD", noise(n, co, 22.0, 6.0), 0.6, n.math("MULTIPLY", noise(n, co, 4.0), 0.4))
        return n.ramp(v, [(0.3, (0.26, 0.25, 0.24)), (0.5, (0.58, 0.56, 0.52)), (0.72, (0.80, 0.78, 0.72))])

    return new_mat("Bone", color, lambda n, co: 0.55, lambda n, co: noise(n, co, 70.0, 4.0), bump=0.25, dist=0.001)


def mat_stone():
    def color(n, co):
        v = noise(n, co, 7.0, 8.0, 0.4)
        stone = n.ramp(v, [(0.3, (0.05, 0.055, 0.07)), (0.55, (0.17, 0.18, 0.21)), (0.75, (0.36, 0.37, 0.40))])
        nz = n("ShaderNodeSeparateXYZ", n("ShaderNodeNewGeometry").outputs["Normal"]).outputs["Z"]
        moss = n.math("MULTIPLY", n.range(nz, 0.5, 0.9), n.range(noise(n, co, 5.0, 4.0), 0.5, 0.62))
        return mix(n, moss, stone, (0.06, 0.10, 0.03, 1))

    return new_mat("Stone", color, lambda n, co: 0.85, lambda n, co: noise(n, co, 18.0, 8.0), bump=0.8, dist=0.004)


def mat_wood(name, dark, light, scale=(1, 30, 30)):
    def grain(n, co):
        m = n("ShaderNodeMapping", Vector=co, Scale=scale).outputs[0]
        return noise(n, m, 2.0, 6.0, 1.5)

    return new_mat(name, lambda n, co: n.ramp(grain(n, co), [(0.3, dark), (0.7, light)]), lambda n, co: 0.6,
                   grain, bump=0.3, dist=0.001)


def mat_bowl():
    """Round lute back: alternating staves with dark seams (lute-local coords, neck along +X)."""
    def staves(n, co):
        xyz = n("ShaderNodeSeparateXYZ", co).outputs
        ang = n.math("ARCTAN2", xyz["Y"], n.math("MULTIPLY", xyz["Z"], -1.0))
        s = n.math("MULTIPLY", ang, 11 / math.pi)
        seam = n.range(n.math("ABSOLUTE", n.math("SUBTRACT", n.math("FRACT", s), 0.5)), 0.38, 0.5)
        tone = n.math("MODULO", n.math("FLOOR", n.math("ADD", s, 100.0)), 2.0)
        return seam, tone

    def color(n, co):
        seam, tone = staves(n, co)
        m = n("ShaderNodeMapping", Vector=co, Scale=(1, 40, 40)).outputs[0]
        v = n.math("MULTIPLY_ADD", tone, 0.35, n.math("MULTIPLY", noise(n, m, 2.0, 6.0, 1.5), 0.5))
        v = n.math("MULTIPLY_ADD", seam, -0.8, v)
        return n.ramp(v, [(0.0, (0.02, 0.008, 0.003)), (0.3, (0.10, 0.04, 0.015)), (0.75, (0.30, 0.13, 0.05))])

    return new_mat("LuteBowl", color, lambda n, co: 0.4, lambda n, co: staves(n, co)[0], bump=0.4, dist=0.002)


def mat_soundboard():
    """Dark top with a cream dotted border, a dotted ring round the sound hole and a black hole."""
    def pattern(n, co):
        xyz = n("ShaderNodeSeparateXYZ", co).outputs
        hx, y = n.math("SUBTRACT", xyz["X"], 0.06), xyz["Y"]
        r = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", hx, hx), n.math("MULTIPLY", y, y)))
        a = n.math("ARCTAN2", y, hx)
        ring = n.math("MULTIPLY", n.math("MULTIPLY", n.range(r, 0.05, 0.053), n.range(r, 0.072, 0.069)),
                      n.range(n.math("SINE", n.math("MULTIPLY", a, 18.0)), 0.0, 0.4))
        ex, ey = n.math("DIVIDE", n.math("ADD", xyz["X"], 0.04), 0.26), n.math("DIVIDE", y, 0.17)
        e = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", ex, ex), n.math("MULTIPLY", ey, ey)))
        ea = n.math("ARCTAN2", ey, ex)
        border = n.math("MULTIPLY", n.math("MULTIPLY", n.range(e, 0.84, 0.86), n.range(e, 0.94, 0.92)),
                        n.range(n.math("SINE", n.math("MULTIPLY", ea, 40.0)), -0.2, 0.3))
        hole = n.range(r, 0.049, 0.046)
        return n.math("MAXIMUM", ring, border), hole

    def color(n, co):
        deco, hole = pattern(n, co)
        v = n.math("MULTIPLY_ADD", deco, 0.6, n.math("MULTIPLY", noise(n, co, 30.0, 6.0), 0.3))
        v = n.math("MULTIPLY_ADD", hole, -1.0, n.math("ADD", v, 0.1))
        return n.ramp(v, [(0.0, (0.004, 0.003, 0.002)), (0.25, (0.07, 0.035, 0.015)), (0.45, (0.16, 0.08, 0.035)),
                          (0.7, (0.75, 0.62, 0.40))])

    return new_mat("Soundboard", color, lambda n, co: 0.5, lambda n, co: pattern(n, co)[0], bump=0.3, dist=0.001)


def mix(n, fac, a, b):
    """RGBA mix; the Mix node shares socket names across types, so pick sockets by identifier."""
    node = n.nt.nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    ins = {s.identifier: s for s in node.inputs}
    for key, v in (("Factor_Float", fac), ("A_Color", a), ("B_Color", b)):
        if isinstance(v, bpy.types.NodeSocket):
            n.nt.links.new(v, ins[key])
        else:
            ins[key].default_value = v
    return next(s for s in node.outputs if s.identifier == "Result_Color")


def _rgb(n, rgb):
    node = n("ShaderNodeRGB")
    node.outputs[0].default_value = (*rgb, 1)
    return node.outputs[0]


# ---------------------------------------------------------------- pose

HEAD = np.array(Matrix.Translation((0, -0.005, 1.215)) @ Matrix.Rotation(math.radians(5), 4, "Y")
                @ Matrix.Rotation(math.radians(10), 4, "X") @ Matrix.Scale(1.08, 4))
# lute-local: x along the neck, y across the body, z out of the soundboard
R0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # columns: local x->+X, y->+Z, z->-Y
LUTE_R = rot("Z", -10) @ rot("Y", -35) @ R0
LUTE_T = np.array([-0.04, -0.30, 0.80])
LUTE = np.eye(4)
LUTE[:3, :3], LUTE[:3, 3] = LUTE_R, LUTE_T


def lute_to_world(p):
    return LUTE_R @ np.asarray(p, float) + LUTE_T


SHOULDER = {"R": np.array([-0.2, 0.06, 0.98]), "L": np.array([0.2, 0.06, 0.98])}
ELBOW = {"R": np.array([-0.31, -0.06, 0.74]), "L": np.array([0.34, -0.10, 0.80])}


def strum_hand():
    """Right hand over the strings between rosette and bridge, in lute-local coords."""
    wrist = (-0.055, 0.10, 0.065)
    fingers = []
    for i, x in enumerate((-0.085, -0.063, -0.041, -0.02)):
        fan = (i - 1.5) * 0.004
        fingers.append([(x, 0.035, 0.05), (x + fan, 0.0, 0.036), (x + 2 * fan, -0.02, 0.022),
                        (x + 3 * fan, -0.044, 0.014)])
    thumb = [(-0.04, 0.085, 0.055), (-0.008, 0.062, 0.036), (0.02, 0.046, 0.02)]
    return wrist, fingers, thumb


def fret_hand():
    """Left hand round the neck: palm below it, thumb behind, fingers curling onto the fingerboard."""
    wrist = (0.43, -0.09, -0.045)
    fingers = []
    for i, x in enumerate((0.395, 0.418, 0.441, 0.464)):
        tip_y = (-0.012, 0.0, 0.01, 0.018)[i]
        fingers.append([(x, -0.04, -0.005), (x + 0.002, -0.044, 0.03), (x + 0.004, -0.018, 0.046),
                        (x + 0.006, tip_y, 0.022)])
    thumb = [(0.405, -0.085, -0.055), (0.43, -0.045, -0.062), (0.452, -0.008, -0.058)]
    return wrist, fingers, thumb


# ---------------------------------------------------------------- pieces

def body_sdf(P):
    d = union([
        sd_ellipsoid(P, (0, 0.06, 0.56), (0.24, 0.19, 0.14)),                       # hips
        sd_cone(P, (0, 0.07, 0.6), (0, 0.05, 0.94), 0.2, 0.2),                       # torso
        sd_ellipsoid(P, (0, 0.05, 1.0), (0.29, 0.17, 0.11)),                         # shoulders
        *[sd_cone(P, (s * 0.1, 0.05, 0.56), (s * 0.13, -0.36, 0.57), 0.12, 0.1) for s in (-1, 1)],  # thighs
        sd_box(P, (0, -0.33, 0.29), (0.26, 0.15, 0.29), 0.08),                       # skirt over the shins
        sd_ellipsoid(P, (0, -0.3, 0.02), (0.34, 0.25, 0.07)),                        # hem pooled on the ground
    ], 0.07)
    ang = np.arctan2(P[:, 0], P[:, 1] + 0.2)
    d = d + 0.013 * np.sin(ang * 11 + 2 * np.sin(P[:, 2] * 9)) * np.clip((0.58 - P[:, 2]) / 0.3, 0, 1)  # folds
    d = d + wobble(P, 0.004, 25, seed=1)
    h = V(0, 0.02, 1.235)
    hood = union([sd_ellipsoid(P, h + V(0, 0.015, 0), (0.18, 0.2, 0.21)),
                  sd_cone(P, h + V(0, 0.05, 0.06), h + V(0, 0.16, 0.21), 0.12, 0.025),       # peak
                  sd_ellipsoid(P, h + V(0, -0.08, 0.09), (0.13, 0.08, 0.06)),                  # brim
                  sd_ellipsoid(P, V(0, 0.04, 1.08), (0.25, 0.2, 0.1))], 0.05)                 # cowl
    d = smin(d, hood + wobble(P, 0.003, 30, seed=2), 0.04)
    d = smax(d, -sd_ellipsoid(P, h + V(0, 0, -0.01), (0.145, 0.165, 0.175)), 0.01)      # hollow
    return smax(d, -sd_ellipsoid(P, h + V(0, -0.165, -0.03), (0.1, 0.13, 0.15)), 0.015)  # face opening


def seat_sdf(P):
    return sd_box(P, (0, 0.14, 0.215), (0.36, 0.26, 0.215), 0.03) + wobble(P, 0.006, 22, seed=3)


def skull_sdf(P):
    d = union([
        sd_ellipsoid(P, (0, 0.012, 0.02), (0.071, 0.093, 0.082)),       # cranium
        sd_ellipsoid(P, (0, -0.052, -0.03), (0.06, 0.04, 0.055)),       # face
        sd_ellipsoid(P, (0, -0.066, -0.075), (0.036, 0.03, 0.026)),     # upper jaw
    ], 0.025)
    d = smin(d, np.minimum(*[sd_cone(P, (s * 0.045, -0.067, -0.03), (s * 0.066, -0.015, -0.034), 0.012, 0.008)
                              for s in (-1, 1)]), 0.012)                 # cheekbones
    d = smax(d, -np.minimum(*[sd_sphere(P, (s * 0.108, -0.025, -0.025), 0.04) for s in (-1, 1)]), 0.02)  # temples
    eyes = np.minimum(*[sd_ellipsoid(P, (s * 0.031, -0.085, -0.006), (0.023, 0.032, 0.021)) for s in (-1, 1)])
    d = smax(d, -eyes, 0.006)
    d = smax(d, -sd_cone(P, (0, -0.1, -0.033), (0, -0.1, -0.057), 0.004, 0.012), 0.003)  # nose
    d = d + wobble(P, 0.0012, 120, seed=4)
    return np.minimum(d, teeth(P, (0, -0.055, -0.093), 0.034, 10, 1))


def teeth(P, c, radius, count, up):
    d = np.full(len(P), 1e3, np.float32)
    for t in np.linspace(-0.95, 0.95, count):
        pos = (c[0] + math.sin(t) * radius, c[1] - math.cos(t) * radius, c[2])
        d = np.minimum(d, sd_box(P, pos, (0.0048, 0.0042, 0.0095), 0.003, rot("Z", -math.degrees(t))))
    return d


def jaw_sdf(P):
    d = union([
        sd_ellipsoid(P, (0, -0.072, -0.124), (0.022, 0.016, 0.017)),
        *[sd_cone(P, (0, -0.074, -0.12), (s * 0.05, -0.02, -0.114), 0.012, 0.011) for s in (-1, 1)],
        *[sd_cone(P, (s * 0.05, -0.02, -0.114), (s * 0.056, 0.004, -0.06), 0.011, 0.008) for s in (-1, 1)],
    ], 0.012)
    return np.minimum(d, teeth(P, (0, -0.053, -0.106), 0.031, 10, -1))


def lute_sdf(P):
    d = smin(sd_ellipsoid(P, (-0.04, 0, 0), (0.26, 0.17, 0.16)), sd_ellipsoid(P, (0.12, 0, 0), (0.14, 0.11, 0.1)), 0.06)
    d = np.maximum(d, P[:, 2])                                              # flat soundboard
    ang = np.arctan2(P[:, 1], -P[:, 2])
    d = d + 0.0015 * np.exp(-np.sin(ang * 11) ** 2 / 0.02) * (P[:, 2] < -0.004)  # stave seams
    r = np.hypot(P[:, 0] - 0.06, P[:, 1])
    d = np.maximum(d, -np.maximum(r - 0.048, -0.008 - P[:, 2]))           # sound hole
    d = smin(d, sd_box(P, (0.40, 0, -0.0115), (0.225, 0.028, 0.0235), 0.008), 0.02)  # neck
    d = np.minimum(d, sd_box(P, (-0.17, 0, 0.004), (0.011, 0.075, 0.006), 0.003))   # bridge
    hd = np.array([0.5, 0, -0.866])
    Rb = np.stack([hd, [0, 1, 0], np.cross(hd, [0, 1, 0])], 1)
    p0 = np.array([0.615, 0, -0.005])
    d = smin(d, sd_box(P, p0 + hd * 0.08, (0.085, 0.026, 0.017), 0.006, Rb), 0.01)   # pegbox, bent back
    for i, t in enumerate((0.03, 0.065, 0.1, 0.135)):
        s = 1 if i % 2 else -1
        a = p0 + hd * t
        d = np.minimum(d, sd_cone(P, a, a + (0, s * 0.06, 0), 0.0055, 0.0055))
        d = np.minimum(d, sd_sphere(P, a + (0, s * 0.062, 0), 0.011))
    return d


def lute_region(c):
    """Material slot per face center: 0 bowl, 1 soundboard, 2 dark wood (neck, pegbox, bridge, pegs)."""
    top = (c[:, 2] > -0.004) & (c[:, 0] < 0.2) & (np.abs(c[:, 1]) < 0.17)
    dark = (c[:, 0] > 0.19) | ((np.abs(c[:, 0] + 0.17) < 0.016) & (c[:, 2] > 0.0))
    return np.where(dark, 2, np.where(top, 1, 0))


def strings_mesh():
    """Six thin strings, built directly as 5-sided prisms (too thin for the voxel grid)."""
    verts, faces = [], []
    for yb, yn in zip(np.linspace(-0.045, 0.045, 6), np.linspace(-0.02, 0.02, 6)):
        a, b = np.array([-0.17, yb, 0.0105]), np.array([0.618, yn, 0.0145])
        ax = (b - a) / np.linalg.norm(b - a)
        u = np.cross(ax, [0, 0, 1.0])
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        base = len(verts)
        for end in (a, b):
            for k in range(5):
                t = k * 2 * math.pi / 5
                verts.append(end + 0.0013 * (math.cos(t) * u + math.sin(t) * w))
        for k in range(5):
            k2 = (k + 1) % 5
            faces += [(base + k, base + k2, base + 5 + k2), (base + k, base + 5 + k2, base + 5 + k)]
    return np.array(verts), np.array(faces)


def hand_bones(wrist, fingers, thumb, to_world):
    """Round cones (a, b, ra, rb) for a skeletal hand, mapped to world space."""
    W = np.array(wrist)
    bones = []
    for f in fingers:
        knuckle = np.array(f[0])
        bones.append((W + (knuckle - W) * 0.15, knuckle, 0.0072, 0.0068))      # metacarpal
        for j, (a, b) in enumerate(zip(f[:-1], f[1:])):
            bones.append((a, b, 0.0068 - j * 0.0007, 0.006 - j * 0.0008))       # phalanges
    th = [W + (np.array(thumb[0]) - W) * 0.3] + [np.array(t) for t in thumb]
    for j, (a, b) in enumerate(zip(th[:-1], th[1:])):
        bones.append((a, b, 0.0075 - j * 0.0006, 0.0068 - j * 0.0008))
    bones.append((W, W + (np.array(fingers[1][0]) - W) * 0.25, 0.016, 0.014))    # carpals
    return [(to_world(a), to_world(b), ra, rb) for a, b, ra, rb in bones]


def arm_sdfs(side, wrist_w):
    S, E = SHOULDER[side], ELBOW[side]
    back = (E - wrist_w) / np.linalg.norm(E - wrist_w)
    C = wrist_w + back * 0.085                                        # cuff opening
    upper = lambda P: sd_cone(P, S, E, 0.08, 0.075) + wobble(P, 0.003, 30, seed=5)
    def lower(P):
        d = smin(sd_cone(P, E, C, 0.075, 0.105), sd_ellipsoid(P, (E + C) / 2 + (0, 0, -0.055), (0.07, 0.07, 0.085)), 0.04)
        d = d + 0.005 * np.sin(np.arctan2(P[:, 0] - C[0], P[:, 2] - C[2]) * 9) * 0.5 + wobble(P, 0.003, 30, seed=6)
        d = smax(d, -sd_cone(P, E + back * 0.06, C - back * 0.05, 0.05, 0.088), 0.008)   # hollow
        return smax(d, (P - C) @ (-back).astype(np.float32), 0.01)                         # open end
    forearm = [(wrist_w + back * 0.2 + o, wrist_w + o * 0.6, 0.0105, 0.0095) for o in (V(0, 0, 0.012), V(0, 0, -0.012))]
    return S, E, C, upper, lower, forearm


def bones_sdf(bones):
    def f(P):
        d = [sd_cone(P, a, b, ra, rb) for a, b, ra, rb in bones]
        d += [sd_sphere(P, b, rb * 1.3) for a, b, ra, rb in bones]   # knobbly joints
        return union(d, 0.003)
    pts = np.array([p for a, b, *_ in bones for p in (a, b)])
    return f, pts.min(0) - 0.025, pts.max(0) + 0.025


def build(mats):
    """Returns {piece: (high objects, low object)} and joint positions in world space."""
    log("modeling")
    pieces = {}
    robe = sdf_obj("Body_hi", body_sdf, (-0.5, -0.62, -0.01), (0.5, 0.46, 1.52), 0.005, materials=[mats["velvet"]])
    seat = sdf_obj("Seat_hi", seat_sdf, (-0.42, -0.16, -0.01), (0.42, 0.44, 0.47), 0.006, materials=[mats["stone"]])
    pieces["Body"] = ([robe, seat], [(robe, 3600), (seat, 400)])

    skull = sdf_obj("Head_hi", skull_sdf, (-0.1, -0.13, -0.13), (0.1, 0.12, 0.12), 0.0018, HEAD, [mats["bone"]])
    jaw = sdf_obj("Jaw_hi", jaw_sdf, (-0.08, -0.11, -0.15), (0.08, 0.03, -0.04), 0.0015, HEAD, [mats["bone"]])
    pieces["Head"] = ([skull], [(skull, 1700)])
    pieces["Jaw"] = ([jaw], [(jaw, 500)])

    lute = sdf_obj("Lute_hi", lute_sdf, (-0.32, -0.2, -0.2), (0.75, 0.2, 0.06), 0.0025, LUTE,
                   [mats["bowl"], mats["board"], mats["darkwood"]], lute_region)
    sv, sf = strings_mesh()
    strings = mesh_obj("Strings_hi", sv, sf, LUTE)
    strings.data.materials.append(mats["string"])
    pieces["Lute"] = ([lute, strings], [(lute, 1900), (strings, None)])

    joints = {"Neck": HEAD[:3, 3] + (0, 0.01, -0.09), "Jaw": (HEAD @ np.array([0, 0.0, -0.06, 1]))[:3],
              "Lute": lute_to_world((0.0, 0, -0.08))}
    for side, hand in (("R", strum_hand()), ("L", fret_hand())):
        bones = hand_bones(*hand, lute_to_world)
        S, E, C, upper, lower, forearm = arm_sdfs(side, lute_to_world(hand[0]))
        f, lo, hi = bones_sdf(bones + forearm)
        hand_hi = sdf_obj(f"Hand{side}_hi", f, lo, hi, 0.0012, materials=[mats["bone"]])
        lo_u, hi_u = np.minimum(S, E) - 0.13, np.maximum(S, E) + 0.13
        up_hi = sdf_obj(f"ArmUpper{side}_hi", upper, lo_u, hi_u, 0.005, materials=[mats["velvet"]])
        lo_l, hi_l = np.minimum(E, C) - 0.17, np.maximum(E, C) + 0.17
        low_hi = sdf_obj(f"Sleeve{side}_hi", lower, lo_l, hi_l, 0.004, materials=[mats["velvet"]])
        pieces[f"ArmUpper{side}"] = ([up_hi], [(up_hi, 350)])
        pieces[f"ArmLower{side}"] = ([low_hi, hand_hi], [(low_hi, 700), (hand_hi, 1500)])
        joints[f"Shoulder{side}"], joints[f"Elbow{side}"] = S, E
    return pieces, joints


def make_lows(pieces):
    log("decimating")
    lows = {}
    for name, (_, parts) in pieces.items():
        objs = [decimated(src, f"{name}_part", t) if t else join([src], f"{name}_part") for src, t in parts]
        low = objs[0] if len(objs) == 1 else join(objs, name)
        for o in objs:
            if o is not low:
                bpy.data.objects.remove(o)
        low.name = low.data.name = name
        lows[name] = low
        log(f"  {name}: {len(low.data.polygons)} tris")
    return lows


# ---------------------------------------------------------------- UVs and baking

def unwrap(lows):
    log("unwrapping")
    objs = list(lows.values())
    for o in objs:
        o.data.materials.clear()
    select(objs)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")
    for name, k in (("Head", 1.8), ("Jaw", 1.5), ("ArmLowerR", 1.3), ("ArmLowerL", 1.3), ("Lute", 1.3)):
        uv = lows[name].data.uv_layers.active.data
        a = np.empty(len(uv) * 2, np.float32)
        uv.foreach_get("uv", a)
        uv.foreach_set("uv", a * k)   # more texels where the eye goes
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_all(pieces, lows):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    highs = [h for hs, _ in pieces.values() for h in hs]
    target = join(list(lows.values()), "BakeTarget")
    mat = bpy.data.materials.new("BakeTarget")
    mat.use_nodes = True
    target.data.materials.clear()
    target.data.materials.append(mat)
    img_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    mat.node_tree.nodes.active = img_node
    maps = {}
    for kind, color in (("color", True), ("normal", False), ("rough", False), ("ao", False)):
        img = bpy.data.images.new(kind, TEX, TEX, alpha=False)
        img.colorspace_settings.name = "sRGB" if color else "Non-Color"
        img_node.image = img
        log(f"baking {kind}")
        if kind == "ao":
            for h in highs:
                h.hide_render = True
            sc.cycles.samples = 96
            select([target])
            bpy.ops.object.bake(type="AO", margin=8, use_selected_to_active=False)
            for h in highs:
                h.hide_render = False
        else:
            sc.cycles.samples = 1
            select(highs + [target], active=target)
            kw = {"pass_filter": {"COLOR"}} if kind == "color" else {}
            bpy.ops.object.bake(type={"color": "DIFFUSE", "normal": "NORMAL", "rough": "ROUGHNESS"}[kind], margin=8,
                                use_selected_to_active=True, cage_extrusion=0.008, max_ray_distance=0.03, **kw)
        px = np.empty(TEX * TEX * 4, np.float32)
        img.pixels.foreach_get(px)
        maps[kind] = px.reshape(TEX, TEX, 4)[::-1, :, :3]
    bpy.data.objects.remove(target)
    return maps


def finish_textures(maps):
    """PS2-style: shadows baked into the color, over-sharpened and punchy. Writes the PNGs Roblox gets."""
    OUT.mkdir(parents=True, exist_ok=True)
    ao = maps["ao"][..., :1] ** 1.3
    color = np.clip(maps["color"] * (0.18 + 0.82 * ao), 0, 1)
    img = Image.fromarray((color * 255).astype(np.uint8))
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=1))
    files = {"color": img}
    files["normal"] = Image.fromarray((maps["normal"] * 255).astype(np.uint8))
    files["rough"] = Image.fromarray((maps["rough"][..., 0] * 255).astype(np.uint8))
    paths = {}
    for k, im in files.items():
        paths[k] = OUT / f"GraveMinstrel_{k}.png"
        im.save(paths[k], optimize=True)
    return paths


def apply_textures(lows, paths):
    mat = bpy.data.materials.new("GraveMinstrel")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    for k, sock in (("color", "Base Color"), ("rough", "Roughness"), ("normal", None)):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(paths[k]))
        if k != "color":
            tex.image.colorspace_settings.name = "Non-Color"
        if sock:
            nt.links.new(tex.outputs["Color"], bsdf.inputs[sock])
        else:
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.6
    for o in lows.values():
        o.data.materials.clear()
        o.data.materials.append(mat)


# ---------------------------------------------------------------- preview scene

def world_sky():
    w = bpy.data.worlds.new("Night")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    n = Nodes(w)
    d = n("ShaderNodeTexCoord").outputs["Generated"]
    z = n("ShaderNodeSeparateXYZ", d).outputs["Z"]
    sky = n.ramp(z, [(0.0, (0.06, 0.025, 0.11)), (0.25, (0.025, 0.015, 0.07)), (0.7, (0.004, 0.004, 0.02))])
    stars = n.range(n("ShaderNodeTexVoronoi", Vector=d, Scale=260.0).outputs["Distance"], 0.035, 0.0)
    clouds = n.math("MULTIPLY", n.range(noise(n, n("ShaderNodeMapping", Vector=d, Scale=(1, 1, 4)).outputs[0], 3.0, 5.0),
                                        0.55, 0.75), n.range(z, 0.05, 0.3))
    col = mix(n, clouds, sky, (0.10, 0.07, 0.20, 1))
    col = mix(n, n.math("MULTIPLY", stars, 0.9), col, (0.9, 0.9, 1.0, 1))
    bg = nt.nodes["Background"]
    nt.links.new(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0


def sun(name, rot_deg, energy, rgb):
    lamp = bpy.data.lights.new(name, "SUN")
    lamp.energy, lamp.color, lamp.angle = energy, rgb, math.radians(3)
    o = link(bpy.data.objects.new(name, lamp))
    o.rotation_euler = [math.radians(a) for a in rot_deg]
    return o


def look_at(obj, target):
    obj.rotation_euler = Vector(np.asarray(target) - np.array(obj.location)).to_track_quat("-Z", "Y").to_euler()


def preview_scene(head_target):
    world_sky()
    ground_mat = new_mat("Grass", lambda n, co: n.ramp(noise(n, co, 9.0, 6.0),
                                                      [(0.3, (0.01, 0.02, 0.015)), (0.7, (0.04, 0.07, 0.04))]),
                         lambda n, co: 0.9, lambda n, co: noise(n, co, 60.0, 4.0), bump=0.5)
    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.data.materials.append(ground_mat)
    stone = mat_stone()
    for x, y, h, tilt in ((-1.3, 1.6, 0.8, 6), (1.2, 2.4, 1.0, -8), (-0.4, 3.6, 0.7, 4), (2.2, 1.0, 0.6, 10)):
        bpy.ops.mesh.primitive_cube_add(location=(x, y, h / 2), scale=(0.28, 0.07, h / 2))
        o = bpy.context.object
        o.rotation_euler = (0, math.radians(tilt), math.radians(tilt * 2))
        o.modifiers.new("bev", "BEVEL").width = 0.03
        o.data.materials.append(stone)
    cam = link(bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")))
    bpy.context.scene.camera = cam
    moon_mat = bpy.data.materials.new("Moon")
    moon_mat.use_nodes = True
    n = Nodes(moon_mat)
    em = n("ShaderNodeEmission", Strength=1.4)
    co = n("ShaderNodeTexCoord").outputs["Object"]
    n.nt.links.new(n.ramp(noise(n, co, 2.2, 8.0, 0.0, 0.62), [(0.4, (0.42, 0.28, 0.13)), (0.55, (0.95, 0.78, 0.48)),
                                                              (0.7, (1.0, 0.93, 0.72))]),
                   em.inputs["Color"])
    out = moon_mat.node_tree.nodes["Material Output"]
    n.nt.links.new(em.outputs[0], out.inputs["Surface"])
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=1.0, fill_type="NGON")
    moon = bpy.context.object
    moon.data.materials.append(moon_mat)
    moon.visible_shadow = False
    sun("MoonRim", (-35, 0, 15), 3.5, (0.70, 0.78, 1.0))  # from behind
    sun("MoonKey", (55, 0, -25), 2.2, (0.62, 0.70, 1.0))  # front-left
    sun("Warm", (75, 0, 120), 0.25, (1.0, 0.6, 0.3))
    sc = bpy.context.scene
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - High Contrast"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        sc.cycles.use_denoising = False
    sc.render.film_transparent = False
    return cam, moon


def place_moon(cam, moon, head, dist=11.0, lift=1.5, radius=2.4):
    d = np.asarray(head) - np.array(cam.location)
    d /= np.linalg.norm(d)
    pos = np.array(cam.location) + d * dist + (0, 0, lift)
    moon.location = pos
    moon.scale = (radius,) * 3
    moon.rotation_euler = Vector(-d).to_track_quat("Z", "Y").to_euler()


def render(path, w, h, cam, loc, target, lens=40, moon=None):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = w, h
    cam.location = loc
    cam.data.lens = lens
    look_at(cam, target)
    if moon:
        place_moon(cam, moon, (0, 0, 1.25))
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    log(f"rendered {path.name}")


# ---------------------------------------------------------------- export

def add_joint_markers(joints):
    """Tiny octahedra centered on each joint: Roblox reads their positions to place Motor6Ds, then deletes them."""
    s = 0.01
    v = np.array([[s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]])
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return [mesh_obj(f"J_{name}", v + np.asarray(p), f) for name, p in joints.items()]


def export_fbx(lows, markers):
    for o in lows.values():
        select([o])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    select(list(lows.values()) + markers)
    path = OUT / "GraveMinstrel.fbx"
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types={"MESH"},
                             axis_forward="Z", axis_up="Y", apply_unit_scale=True, mesh_smooth_type="FACE",
                             use_mesh_modifiers=False, path_mode="COPY", embed_textures=True)
    log(f"exported {path.name} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- main

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    OUT.mkdir(parents=True, exist_ok=True)
    mats = {"velvet": mat_velvet(), "bone": mat_bone(), "stone": mat_stone(), "bowl": mat_bowl(),
            "board": mat_soundboard(), "darkwood": mat_wood("DarkWood", (0.012, 0.006, 0.003), (0.06, 0.03, 0.014)),
            "string": new_mat("String", lambda n, co: _rgb(n, (0.62, 0.58, 0.5)), lambda n, co: 0.35)}
    pieces, joints = build(mats)
    for k, p in joints.items():
        log(f"  joint {k}: {np.round(p, 3).tolist()}")
    cam, moon = preview_scene((0, 0, 1.2))
    if "--shapes" in sys.argv:
        bpy.context.scene.cycles.samples = 8
        render(OUT / "shapes_front.png", 480, 600, cam, (-0.35, -2.4, 0.8), (0, -0.05, 0.85), moon=moon)
        render(OUT / "shapes_side.png", 480, 600, cam, (2.4, -0.9, 0.9), (0, -0.05, 0.8))
        return
    lows = make_lows(pieces)
    unwrap(lows)
    maps = bake_all(pieces, lows)
    paths = finish_textures(maps)
    apply_textures(lows, paths)
    for hs, _ in pieces.values():
        for h in hs:
            h.hide_render = True
    total = sum(len(o.data.polygons) for o in lows.values())
    log(f"low-poly total: {total} tris")
    render(OUT / "preview_hero.png", 800, 1000, cam, (-0.35, -2.4, 0.8), (0, -0.05, 0.88), moon=moon)
    for i, (loc, tgt) in enumerate((((-1.9, -1.6, 1.0), (0, -0.05, 0.8)), ((2.3, -0.6, 0.95), (0, -0.05, 0.8)),
                                    ((-0.6, 0.6 + 1.6, 1.2), (0, 0, 0.8)), ((0.1, -0.85, 1.25), (0.0, -0.1, 1.15)))):
        render(OUT / f"view_{i}.png", 480, 600, cam, loc, tgt, lens=40 if i < 3 else 50)
    sheet = Image.new("RGB", (480 * 4, 600))
    for i in range(4):
        sheet.paste(Image.open(OUT / f"view_{i}.png"), (480 * i, 0))
        (OUT / f"view_{i}.png").unlink()
    sheet.save(OUT / "preview_turnaround.png")
    export_fbx(lows, add_joint_markers(joints))


if __name__ == "__main__":
    main()
