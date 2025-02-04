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






