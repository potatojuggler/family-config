"""Builds reaper.glb: a low-poly, PS1-style hooded skeleton playing a lute, for Roblox Studio's 3D Importer.

Run: python3 build_reaper.py   (needs numpy and Pillow)
Units are meters, Y is up, the figure faces +Z. One mesh, one texture atlas, like a PS1 texture page.
"""
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).with_name("reaper.glb")
TEX = 128     # native atlas size, PS1-ish texel density
UPSCALE = 8   # baked to 1024 with hard pixel edges, since Roblox always filters textures smoothly
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- texture atlas

REG = {  # name: (x, y, w, h) in native texels
    "cloth": (0, 0, 64, 64), "skull": (64, 0, 64, 64),
    "board": (0, 64, 64, 32), "bowl": (64, 64, 32, 32), "bone": (96, 64, 32, 32),
    "stone": (0, 96, 32, 32), "dark": (32, 96, 32, 32), "shadow": (64, 96, 32, 32), "string": (96, 96, 32, 32),
}


def noise(h, w, octaves=(4, 8, 16)):
    out = np.zeros((h, w))
    for i, o in enumerate(octaves):
        g = Image.fromarray((rng.random((o, o)) * 255).astype(np.uint8))
        out += np.asarray(g.resize((w, h), Image.BILINEAR), float) / 255 / 2**i
    return out / sum(1 / 2**i for i in range(len(octaves)))


def mix(a, b, t):
    t = np.clip(t, 0, 1)[..., None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def paint_cloth(h, w):
    y, x = np.mgrid[0:h, 0:w]
    n = noise(h, w, (4, 8, 16, 32))
    swirl = np.sin(x * 0.55 + n * 7) * np.sin(y * 0.3 + n * 5)  # crushed-velvet swirls
    return mix((35, 2, 6), (205, 42, 36), np.clip(0.5 + 0.45 * swirl + 0.6 * (n - 0.5), 0, 1) ** 1.3)


def paint_skull(h, w):
    y, x = np.mgrid[0:h, 0:w]
    fx, fy = (x + 0.5) / w * 2 - 1, (y + 0.5) / h * 2 - 1  # front view, -1..1, fy points down
    c = mix((120, 110, 90), (228, 218, 188), noise(h, w) * 0.6 + 0.55 - 0.45 * fx**2 - 0.2 * np.clip(fy, 0, 1))
    eye = ((np.abs(fx) - 0.36) / 0.27) ** 2 + ((fy + 0.02) / 0.23) ** 2
    eye[fy < -0.05 - 0.35 * (np.abs(fx) - 0.15)] = 9  # brow slants down toward the nose
    c[eye < 1.7] *= 0.55
    c[eye < 1] = (12, 8, 8)
    cheek = ((np.abs(fx) - 0.55) / 0.2) ** 2 + ((fy - 0.36) / 0.14) ** 2
    c[cheek < 1] *= 0.7
    c[(fy > 0.2) & (fy < 0.44) & (np.abs(fx) < 0.03 + (fy - 0.2) * 0.45)] = (20, 12, 10)  # nose
    teeth = (fy > 0.54) & (fy < 0.74) & (np.abs(fx) < 0.42)
    c[teeth] = (232, 224, 196)
    c[teeth & ((((fx + 1) / 0.14) % 1 < 0.3) | (np.abs(fy - 0.64) < 0.035))] = (55, 45, 35)
    c[fy > 0.78] *= 0.7
    for t in np.linspace(0, 1, 40):  # forehead crack
        cx, cy = (0.15 - 0.1 * t, -0.85 + 0.25 * t) if t < 0.5 else (0.05 + 0.3 * (t - 0.5), -0.73 + 0.5 * (t - 0.5))
        c[int((cy + 1) / 2 * h), int((cx + 1) / 2 * w)] = (70, 60, 50)
    return c


def paint_board(h, w):
    y, x = np.mgrid[0:h, 0:w]
    lx, ly = -0.28 + (x + 0.5) / w * 0.5, 0.17 - (y + 0.5) / h * 0.34  # lute-local meters, see lute()
    c = mix((150, 105, 55), (215, 175, 105), 0.7 + 0.3 * np.sin(ly * 300 + noise(h, w) * 4) - 0.2 * noise(h, w))
    r = np.hypot(lx - 0.03, ly)
    rosette = (r > 0.045) & (r < 0.062)
    c[rosette] = (95, 60, 30)
    c[rosette & (np.sin(np.arctan2(ly, lx - 0.03) * 8) > 0.3)] = (225, 190, 120)
    c[r < 0.045] = (14, 9, 7)
    c[(np.abs(lx + 0.17) < 0.012) & (np.abs(ly) < 0.07)] = (60, 30, 15)  # bridge
    return c


def paint_bowl(h, w):
    x = np.mgrid[0:h, 0:w][1]
    c = mix((115, 58, 25), (160, 92, 42), (x // 4) % 2 * 0.8 + noise(h, w) * 0.3)
    c[x % 4 == 0] = (45, 22, 10)  # gaps between the staves
    return c


def paint_grainy(h, w, a, b, grain=0.0):
    y = np.mgrid[0:h, 0:w][0]
    return mix(a, b, noise(h, w, (4, 8, 16)) + grain * np.sin(y * 2.1))


def paint_stone(h, w):
    n = noise(h, w, (4, 8, 16, 32))
    c = mix((60, 62, 70), (140, 140, 146), n)
    c[noise(h, w, (4, 8)) > 0.68] = (58, 78, 40)  # moss
    c[np.abs(n - 0.5) < 0.012] = (35, 35, 40)  # cracks along the noise contour
    return c


def build_atlas():
    painters = {
        "cloth": paint_cloth, "skull": paint_skull, "board": paint_board, "bowl": paint_bowl,
        "bone": lambda h, w: paint_grainy(h, w, (150, 140, 112), (222, 212, 182)),
        "dark": lambda h, w: paint_grainy(h, w, (40, 22, 10), (90, 52, 26), 0.15),
        "stone": paint_stone,
        "shadow": lambda h, w: paint_grainy(h, w, (6, 2, 4), (30, 6, 9)),
        "string": lambda h, w: paint_grainy(h, w, (170, 165, 150), (215, 210, 195)),
    }
    img = np.zeros((TEX, TEX, 3))
    for name, (x, y, w, h) in REG.items():
        img[y:y + h, x:x + w] = painters[name](h, w)
    # PS1 look: 15-bit color with the console's 4x4 ordered dither
    bayer = np.array([[-4, 0, -3, 1], [2, -2, 3, -1], [-3, 1, -4, 0], [3, -1, 2, -2]])
    img = (np.clip(img + np.tile(bayer, (TEX // 4, TEX // 4))[..., None], 0, 255).astype(np.uint8) >> 3) << 3
    big = Image.fromarray(img).resize((TEX * UPSCALE,) * 2, Image.NEAREST)
    buf = io.BytesIO()
    big.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def ruv(name, u, v):
    """Region-local (u, v) in 0..1 -> atlas UV, inset one texel so mipmaps don't bleed neighbors in."""
    x, y, w, h = REG[name]
    u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
    return np.stack(np.broadcast_arrays((x + 1 + u * (w - 2)) / TEX, (y + 1 + v * (h - 2)) / TEX), -1)


# ---------------------------------------------------------------- geometry

parts = []                  # (triangles (T,3,3), uvs (T,3,2), smooth normals?)
xf = [np.eye(3), np.zeros(3)]  # current local->world transform, used to build the lute in its own frame


def add(tris, uvs, ref, smooth=True, inward=False):
    """Append triangles, flipping any whose front face doesn't point away from ref (toward it if inward)."""
    R, t = xf
    tris = np.asarray(tris, float) @ R.T + t
    ref = np.asarray(ref, float) @ R.T + t
    uvs = np.array(uvs, float)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ok = np.linalg.norm(n, axis=1) > 1e-10  # drop degenerate pole/cone-tip triangles
    flip = ((n * (tris.mean(1) - np.broadcast_to(ref, tris[:, 0].shape))).sum(1) < 0) != inward
    tris[flip], uvs[flip] = tris[flip][:, ::-1], uvs[flip][:, ::-1]
    parts.append((tris[ok], uvs[ok], smooth))


def grid(P, UV, refs, keep=None):
    """Quads between neighboring rows/cols of a point grid -> triangles, uvs and per-triangle refs."""
    q = lambda A: (A[:-1, :-1], A[:-1, 1:], A[1:, :-1], A[1:, 1:])
    keep = np.ones(P[:-1, :-1, 0].shape, bool) if keep is None else keep
    (a, b, c, d), (ua, ub, uc, ud) = q(P), q(UV)
    tris = np.concatenate([np.stack([a, c, b], 2)[keep], np.stack([b, c, d], 2)[keep]])
    uvs = np.concatenate([np.stack([ua, uc, ub], 2)[keep], np.stack([ub, uc, ud], 2)[keep]])
    refs = np.broadcast_to(refs, keep.shape + (3,))[keep]
    return tris, uvs, np.concatenate([refs, refs])


def loft(rings, axis, region, inner=None):
    """Surface through rings (R, n+1, 3) whose centers are axis (R, 3); inner adds a dark inside shell."""
    rings, axis = np.asarray(rings, float), np.asarray(axis, float)
    R, n1 = rings.shape[:2]
    UV = ruv(region, *np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, R)))
    refs = ((axis[:-1] + axis[1:]) / 2)[:, None]
    add(*grid(rings, UV, refs))
    if inner:
        shrunk = axis[:, None] + (rings - axis[:, None]) * 0.85
        add(*grid(shrunk, ruv(inner, *np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, R))), refs), inward=True)


def tube(path, radii, n=8, region="cloth", inner="cloth"):
    """Open sleeve along path, framed by parallel transport so it doesn't twist."""
    path = np.asarray(path, float)
    ang = np.linspace(0, 2 * math.pi, n + 1)[:, None]
    rings, e1 = [], None
    for i, (p, r) in enumerate(zip(path, radii)):
        d = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        d /= np.linalg.norm(d)
        e1 = np.cross(d, (1, 0, 0)) if e1 is None else e1 - d * (e1 @ d)
        e1 /= np.linalg.norm(e1)
        rings.append(p + r * (np.cos(ang) * e1 + np.sin(ang) * np.cross(e1, d)))
    loft(rings, path, region, inner)


def box(o, ex, ey, ez, region):
    """Box from corner o along edge vectors ex, ey, ez; every face shows the whole region."""
    o, ex, ey, ez = (np.asarray(v, float) for v in (o, ex, ey, ez))
    tris, uvs = [], []
    u0, u1, u2, u3 = ruv(region, 0, 0), ruv(region, 1, 0), ruv(region, 1, 1), ruv(region, 0, 1)
    for a, b, c in ((ex, ey, ez), (ey, ez, ex), (ez, ex, ey)):
        for base in (o, o + c):
            tris += [[base, base + a, base + a + b], [base, base + a + b, base + b]]
            uvs += [[u0, u1, u2], [u0, u2, u3]]
    add(tris, uvs, o + (ex + ey + ez) / 2, smooth=False)


def beam(p0, p1, w, region="bone", up=(0, 1, 0)):
    """Square-section bar from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    s = np.cross(d, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(d, (1, 0, 0))
    s = s / np.linalg.norm(s) * w
    t = np.cross(d / np.linalg.norm(d), s)
    box(p0 - s / 2 - t / 2, d, s, t, region)


def latlong(center, radii, nt, nphi, deform=lambda d, p: p):
    th = np.linspace(0, math.pi, nt + 1)[:, None]
    ph = np.linspace(0, 2 * math.pi, nphi + 1)[None, :]
    d = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)), -1)
    uv = np.stack(np.broadcast_arrays(ph / (2 * math.pi), th / math.pi), -1)
    return np.asarray(center) + deform(d, d * radii), d, uv


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def seat():
    box((-0.65, 0, -0.75), (1.3, 0, 0), (0, 0.4, 0), (0, 0, 0.9), "stone")
    box((-0.7, 0.4, -0.8), (1.4, 0, 0), (0, 0.08, 0), (0, 0, 1.0), "stone")


def robe():
    # (y, center z, radius x, radius z, pleat depth): hem on the ground, lap over the knees, up to the neck
    profile = [(0.00, 0.12, 0.52, 0.60, 0.07), (0.28, 0.14, 0.46, 0.56, 0.06), (0.50, 0.12, 0.45, 0.55, 0.04),
               (0.62, 0.08, 0.40, 0.50, 0.02), (0.72, -0.10, 0.34, 0.30, 0), (1.08, -0.12, 0.36, 0.24, 0),
               (1.32, -0.12, 0.42, 0.25, 0), (1.44, -0.10, 0.22, 0.18, 0)]
    n = 12
    ang = np.arange(n) * 2 * math.pi / n
    pleat = np.where(np.arange(n) % 2, -1, 1)
    rings = []
    for y, cz, rx, rz, pd in profile:
        k = 1 + pd * pleat + rng.normal(0, 0.015, n)
        ring = np.stack([rx * k * np.cos(ang), np.full(n, y), cz + rz * k * np.sin(ang)], 1)
        rings.append(np.vstack([ring, ring[:1]]))  # repeat the first point to close the UV seam
    loft(rings, [(0, y, cz) for y, cz, *_ in profile], "cloth")


def hood():
    def drape(d, p):
        p = p.copy()
        up, down = np.clip(d[..., 1], 0, 1), np.clip(-d[..., 1], 0, 1)
        p[..., 1] += 0.07 * up**2   # peak...
        p[..., 2] -= 0.07 * up**2   # ...swept back
        p[..., [0, 2]] *= 1 + 0.35 * down[..., None]  # flare onto the shoulders
        return p
    center = np.array([0, 1.6, -0.09])
    nt, nphi = 8, 16
    P, d, uv = latlong(center, (0.23, 0.27, 0.25), nt, nphi, drape)
    dc = (d[:-1, :-1] + d[:-1, 1:] + d[1:, :-1] + d[1:, 1:]) / 4
    keep = ~((dc[..., 2] > 0.55) & (np.abs(dc[..., 1]) < 0.6))  # face opening
    keep[-1] = False  # underside sits inside the robe
    add(*grid(P, ruv("cloth", uv[..., 0], uv[..., 1]), center, keep))
    P_in = center + (P - center) * 0.9
    add(*grid(P_in, ruv("shadow", uv[..., 0], uv[..., 1]), center, keep), inward=True)


def skull():
    center, radii = np.array([0, 1.575, 0.04]), np.array([0.115, 0.14, 0.13])

    def shape(d, p):
        p = p.copy()
        p[..., 0] *= np.where(d[..., 1] < -0.25, 0.8, 1)    # narrow jaw
        p[..., 2] *= np.where(d[..., 2] < 0, 1.25, np.where(d[..., 2] > 0.5, 0.92, 1))  # long cranium, flat face
        return p
    P, _, uv = latlong(center, radii, 8, 12, shape)
    tris, _, refs = grid(P, uv, center)
    # front-facing triangles get the face, projected straight on; the rest get plain bone
    out = tris.mean(1) - center
    front = out[:, 2] / np.linalg.norm(out, axis=1) > 0.25
    u = (tris[..., 0] - center[0]) / (2 * radii[0]) + 0.5
    v = (center[1] + radii[1] - tris[..., 1]) / (2 * radii[1])
    uvs = np.where(front[:, None, None], ruv("skull", u, v), ruv("bone", u, v))
    add(tris, uvs, refs)


LUTE_R = rot("y", -12) @ rot("z", 38)  # neck angled up toward the figure's left, slightly forward
LUTE_T = np.array([-0.05, 0.86, 0.36])
to_world = lambda p: np.asarray(p, float) @ LUTE_R.T + LUTE_T


def lute():
    """Built in its own frame: neck along +X, soundboard facing +Z, body centered on the origin."""
    th = np.arange(12) * 2 * math.pi / 12
    c = np.array([-0.03, 0, 0])
    outline = np.stack([-0.03 + 0.25 * np.cos(th), 0.17 * np.sin(th) * (1 - 0.35 * np.clip(np.cos(th), 0, 1)),
                        np.zeros(12)], 1)
    board_uv = lambda p: ruv("board", (p[..., 0] + 0.28) / 0.5, (0.17 - p[..., 1]) / 0.34)
    tris = np.stack([np.broadcast_to(c, outline.shape), outline, np.roll(outline, -1, 0)], 1)
    add(tris, board_uv(tris), c - (0, 0, 1), smooth=False)

    closed = np.vstack([outline, outline[:1]])
    rings = [c + s * (closed - c) + (0, 0, z) for s, z in ((1, 0), (0.9, -0.07), (0.6, -0.13), (0, -0.16))]
    UV = ruv("bowl", *np.meshgrid(np.linspace(0, 1, 13), np.linspace(0, 1, 4)))
    add(*grid(np.array(rings), UV, c))

    box((0.19, -0.03, -0.035), (0.43, 0, 0), (0, 0.06, 0), (0, 0, 0.05), "dark")  # neck
    box((-0.182, -0.07, 0), (0.024, 0, 0), (0, 0.14, 0), (0, 0, 0.012), "dark")    # bridge
    head0, head_dir = np.array([0.61, 0, -0.01]), np.array([0.5, 0, -0.866])        # pegbox bent back
    beam(head0, head0 + 0.15 * head_dir, 0.05, "dark", up=(0, 0, 1))
    for t in (0.3, 0.55, 0.8):
        for side in (-1, 1):
            p = head0 + 0.15 * t * head_dir
            beam(p + (0, side * 0.02, 0), p + (0, side * 0.065, 0), 0.012, "board", up=(0, 0, 1))
    for yb, yn in zip((-0.03, -0.01, 0.01, 0.03), (-0.02, -0.007, 0.007, 0.02)):
        beam((-0.17, yb, 0.013), (0.62, yn, 0.02), 0.004, "string", up=(0, 0, 1))


def hand(W, F, P, curls, thumb_dir, thumb_side):
    """Skeletal hand at wrist W, fingers along F, curling toward the palm direction P (cumulative degrees)."""
    W, F, P, thumb_dir = (np.asarray(v, float) for v in (W, F, P, thumb_dir))
    F = F / np.linalg.norm(F)
    P = P - F * (P @ F)
    P /= np.linalg.norm(P)
    S = np.cross(F, P)
    for off, scale in zip((-1.5, -0.5, 0.5, 1.5), (0.85, 1.0, 0.95, 0.8)):
        knuckle = W + F * 0.08 + S * off * 0.026
        beam(W + S * off * 0.011, knuckle, 0.015, up=P)
        p = knuckle
        for L, c in zip((0.045, 0.034, 0.026), curls):
            a = math.radians(c)
            q = p + (F * math.cos(a) + P * math.sin(a)) * L * scale
            beam(p, q, 0.014, up=P)
            p = q
    t0 = W + S * thumb_side * 0.02
    t1 = t0 + thumb_dir / np.linalg.norm(thumb_dir) * 0.055
    beam(t0, t1, 0.016, up=P)
    beam(t1, t1 + (thumb_dir / np.linalg.norm(thumb_dir) + P * 0.6) / 1.2 * 0.04, 0.014, up=P)


def arm(shoulder, elbow, wrist):
    """Bell sleeve from shoulder to just short of the wrist, with forearm bones showing past the cuff."""
    shoulder, elbow, wrist = (np.asarray(v, float) for v in (shoulder, elbow, wrist))
    back = elbow - wrist
    tube([shoulder, elbow, wrist + back * 0.3], [0.1, 0.09, 0.12])
    side = np.cross(back, (0, 1, 0))
    side = side / np.linalg.norm(side) * 0.012
    for s in (-1, 1):
        beam(wrist + back * 0.45 + s * side, wrist + s * side * 0.5, 0.014)


def build():
    seat()
    robe()
    hood()
    skull()
    xf[:] = [LUTE_R, LUTE_T]
    lute()
    strum = dict(W=(-0.10, 0.11, 0.11), F=(0.15, -1, -0.35), P=(0, 0, -1), curls=(30, 60, 85),
                 thumb_dir=(0.6, -0.6, -0.3), thumb_side=1)
    fret = dict(W=(0.44, -0.011, -0.055), F=(0, 0.868, 0.496), P=(0, -0.496, 0.868), curls=(50, 110, 175),
                thumb_dir=(-0.6, 0.5, 0), thumb_side=-1)
    hand(**strum)
    hand(**fret)
    xf[:] = [np.eye(3), np.zeros(3)]
    arm((-0.34, 1.27, -0.12), (-0.48, 0.92, 0.12), to_world(strum["W"]))
    arm((0.34, 1.27, -0.12), (0.50, 0.92, 0.10), to_world(fret["W"]))


# ---------------------------------------------------------------- export

def vertex_normals(tris, smooth):
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # area-weighted
    n = np.repeat(fn[:, None], 3, 1)
    if smooth:  # average over corners that share a position within the part
        _, inv = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        acc = np.zeros((inv.max() + 1, 3))
        np.add.at(acc, inv, n.reshape(-1, 3))
        n = acc[inv].reshape(-1, 3, 3)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def write_glb(path, pos, nrm, uv, idx, png):
    blobs = [pos.astype("<f4").tobytes(), nrm.astype("<f4").tobytes(), uv.astype("<f4").tobytes(),
             idx.astype("<u4").tobytes(), png]
    views, binbuf = [], b""
    for b in blobs:
        views.append({"buffer": 0, "byteOffset": len(binbuf), "byteLength": len(b)})
        binbuf += b + b"\0" * (-len(b) % 4)
    for v, target in zip(views, (34962, 34962, 34962, 34963)):
        v["target"] = target
    n = len(pos)
    gltf = {
        "asset": {"version": "2.0", "generator": "build_reaper.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "PS1Reaper"}],
        "meshes": [{"name": "PS1Reaper", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": "PS1Atlas", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9728, "minFilter": 9984}],  # nearest, for viewers that honor it
        "images": [{"bufferView": 4, "mimeType": "image/png", "name": "PS1Atlas"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binbuf)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                     + struct.pack("<I4s", len(binbuf), b"BIN\0") + binbuf)


def main():
    build()
    tris = np.concatenate([t for t, _, _ in parts])
    uvs = np.concatenate([u for _, u, _ in parts])
    nrms = np.concatenate([vertex_normals(t, s) for t, _, s in parts])
    corners = np.hstack([tris.reshape(-1, 3), nrms.reshape(-1, 3), uvs.reshape(-1, 2)])
    verts, idx = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    idx = idx.reshape(-1)

    assert np.isfinite(verts).all(), "NaN in geometry"
    assert ((uvs >= 0) & (uvs <= 1)).all(), "UV outside the atlas"
    assert len(tris) < 10_000, "keep well under Roblox's per-mesh triangle limit"
    write_glb(OUT, verts[:, :3], verts[:, 3:6], verts[:, 6:], idx, build_atlas())
    size = tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)
    print(f"{OUT.name}: {len(tris)} triangles, {len(verts)} vertices, "
          f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
