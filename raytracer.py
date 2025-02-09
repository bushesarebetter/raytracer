import argparse
import time

import numpy as np
from PIL import Image

EPS = 1e-4

# center, radius, color (linear RGB), reflectivity
SPHERES = [
    ((0.0, 1.0, 4.0), 1.0, (0.85, 0.85, 0.9), 0.6),
    ((-2.2, 0.7, 5.0), 0.7, (0.9, 0.15, 0.1), 0.15),
    ((2.0, 0.8, 3.2), 0.8, (0.1, 0.3, 0.9), 0.25),
    ((0.9, 0.35, 2.2), 0.35, (0.95, 0.75, 0.1), 0.1),
]
PLANE_Y = 0.0
PLANE_REFLECT = 0.2
CHECKER = (np.array([0.9, 0.9, 0.9]), np.array([0.08, 0.08, 0.08]))
LIGHTS = [
    (np.array([-6.0, 6.0, 1.5]), np.array([0.9, 0.85, 0.8])),
    (np.array([6.0, 5.0, 1.0]), np.array([0.3, 0.35, 0.45])),
]
AMBIENT = 0.06
SPECULAR = 0.6
SHININESS = 60


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sphere_hit(O, D, center, radius):
    """Distance along unit-length rays D from O to the sphere, inf on miss."""
    oc = O - np.asarray(center)
    b = dot(oc, D)
    c = dot(oc, oc) - radius * radius
    disc = b * b - c
    t = np.full(len(O), np.inf)
    ok = disc >= 0
    sq = np.sqrt(disc[ok])
    t0, t1 = -b[ok] - sq, -b[ok] + sq
    # near root unless it's behind us (ray starts inside the sphere)
    t[ok] = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))
    return t


def plane_hit(O, D):
    t = np.full(len(O), np.inf)
    ok = np.abs(D[:, 1]) > 1e-9
    tt = (PLANE_Y - O[ok, 1]) / D[ok, 1]
    t[ok] = np.where(tt > EPS, tt, np.inf)
    return t


def nearest(O, D):
    ts = np.stack([sphere_hit(O, D, c, r) for c, r, _, _ in SPHERES] + [plane_hit(O, D)])
    idx = ts.argmin(axis=0)
    t = ts[idx, np.arange(len(O))]
    idx[np.isinf(t)] = -1
    return t, idx


def sky(D):
    k = np.clip(D[:, 1], 0, 1)[:, None]
    return (1 - k) * np.array([0.75, 0.8, 0.9]) + k * np.array([0.2, 0.35, 0.7])


def trace(O, D, depth):
    color = sky(D)
    t, idx = nearest(O, D)
    hit = idx >= 0
    if not hit.any():
        return color
    O, D, t, idx = O[hit], D[hit], t[hit], idx[hit]
    P = O + D * t[:, None]
    N = np.empty_like(P)
    base = np.empty_like(P)
    refl = np.empty(len(P))

    for k, (c, r, col, rf) in enumerate(SPHERES):
        m = idx == k
        N[m] = (P[m] - np.asarray(c)) / r
        base[m] = col
        refl[m] = rf
    m = idx == len(SPHERES)
    N[m] = (0.0, 1.0, 0.0)
    odd = (np.floor(P[m, 0]) + np.floor(P[m, 2])) % 2 == 1
    base[m] = np.where(odd[:, None], CHECKER[1], CHECKER[0])
    refl[m] = PLANE_REFLECT

    out = AMBIENT * base
    Pn = P + N * 1e-3  # nudge off the surface to avoid self-shadowing
    V = -D
    for lpos, lcol in LIGHTS:
        Lv = lpos - P
        dist = np.linalg.norm(Lv, axis=1)
        L = Lv / dist[:, None]
        lit = nearest(Pn, L)[0] >= dist
        ndl = np.clip(dot(N, L), 0, None)
        R = 2 * ndl[:, None] * N - L
        spec = np.clip(dot(R, V), 0, None) ** SHININESS * (ndl > 0)
        out += lit[:, None] * lcol * (base * ndl[:, None] + SPECULAR * spec[:, None])

    if depth > 0:
        r = refl > 0
        Dr = D[r] - 2 * dot(D[r], N[r])[:, None] * N[r]
        rc = trace(Pn[r], Dr, depth - 1)
        out[r] = (1 - refl[r])[:, None] * out[r] + refl[r][:, None] * rc

    color[hit] = out
    return color


def render(width, height, ss=2, depth=3, fov=55.0):
    eye = np.array([0.0, 1.6, -1.5])
    fwd = normalize(np.array([0.0, 0.9, 4.0]) - eye)
    right = normalize(np.cross([0.0, 1.0, 0.0], fwd))
    up = np.cross(fwd, right)
    half = np.tan(np.radians(fov) / 2)
    aspect = width / height

    jj, ii = np.mgrid[0:height, 0:width]
    img = np.zeros((height * width, 3))
    O = np.broadcast_to(eye, (height * width, 3))
    # regular ss x ss grid of subpixel samples, averaged in linear space
    for oy in (np.arange(ss) + 0.5) / ss:
        for ox in (np.arange(ss) + 0.5) / ss:
            px = ((ii + ox) / width * 2 - 1) * half * aspect
            py = (1 - (jj + oy) / height * 2) * half
            D = normalize(fwd + px.reshape(-1, 1) * right + py.reshape(-1, 1) * up)
            img += trace(O, D, depth)
    img /= ss * ss
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    return (img.reshape(height, width, 3) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--ss", type=int, default=2, help="samples per pixel = ss*ss")
    ap.add_argument("--depth", type=int, default=3, help="reflection bounces")
    ap.add_argument("--out", default="render.png")
    a = ap.parse_args()
    t0 = time.perf_counter()
    Image.fromarray(render(a.width, a.height, a.ss, a.depth)).save(a.out)
    print(f"{a.width}x{a.height}, {a.ss * a.ss} spp, depth {a.depth}: {time.perf_counter() - t0:.1f}s -> {a.out}")
