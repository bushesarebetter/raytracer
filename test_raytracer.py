import numpy as np

from raytracer import plane_hit, render, sphere_hit


def ray(o, d):
    d = np.asarray(d, float)
    return np.array([o], float), np.array([d / np.linalg.norm(d)])


def test_sphere_head_on():
    O, D = ray((0, 0, 0), (0, 0, 1))
    assert np.isclose(sphere_hit(O, D, (0, 0, 5), 1.0)[0], 4.0)


def test_sphere_off_center():
    # x = 0.6 on a unit sphere: depth sqrt(1 - 0.36) = 0.8 in front of the center
    O, D = ray((0.6, 0, 0), (0, 0, 1))
    assert np.isclose(sphere_hit(O, D, (0, 0, 5), 1.0)[0], 4.2)


def test_sphere_diagonal():
    # ray along (1,1,1) toward a radius-2 sphere at (4,4,4): hits at |c| - r
    O, D = ray((0, 0, 0), (1, 1, 1))
    assert np.isclose(sphere_hit(O, D, (4, 4, 4), 2.0)[0], 4 * np.sqrt(3) - 2)


def test_ray_inside_sphere_hits_far_side():
    O, D = ray((0, 0, 5), (0, 0, 1))
    assert np.isclose(sphere_hit(O, D, (0, 0, 5), 1.5)[0], 1.5)


def test_misses():
    O, D = ray((0, 0, 0), (0, 0, -1))  # pointing away
    assert np.isinf(sphere_hit(O, D, (0, 0, 5), 1.0)[0])
    O, D = ray((0, 0, 0), (0, 1, 0))  # sideways, passes well clear
    assert np.isinf(sphere_hit(O, D, (0, 0, 5), 1.0)[0])
    O, D = ray((1.01, 0, 0), (0, 0, 1))  # just outside the silhouette
    assert np.isinf(sphere_hit(O, D, (0, 0, 5), 1.0)[0])


def test_plane():
    O, D = ray((0, 2, 0), (0, -1, 1))
    assert np.isclose(plane_hit(O, D)[0], 2 * np.sqrt(2))
    O, D = ray((0, 2, 0), (0, 1, 0))
    assert np.isinf(plane_hit(O, D)[0])


def test_small_render():
    img = render(32, 24, ss=1, depth=1)
    assert img.shape == (24, 32, 3) and img.dtype == np.uint8
    assert img.std() > 10  # not a blank frame
