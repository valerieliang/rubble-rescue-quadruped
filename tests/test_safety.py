import numpy as np

from rescue.interfaces import Region
from rescue.safety import (CBFFilter, SafetyParams, box_sd, calibrate_tracking_error, chi_radius,
                           ellipse_barrier, solve_qp2)
from rescue.sim import Go2Params, step_unicycle


def test_chi_radius():
    assert abs(chi_radius(0.9) - 2.1460) < 1e-3
    assert chi_radius(0.99) > chi_radius(0.9) > chi_radius(0.5)


def test_ellipse_barrier_isotropic_is_exact_distance():
    mu, S = np.array([5.0, 5.0]), np.eye(2) * 4.0   # sigma = 2
    h, g = ellipse_barrier(np.array([15.0, 5.0]), mu, S, c=2.0, margin=1.0)
    assert abs(h - (10.0 - 2.0 * 2.0 - 1.0)) < 1e-9
    assert np.allclose(g, [1.0, 0.0])


def test_ellipse_barrier_anisotropic_lower_bounds_true_distance():
    mu, S = np.zeros(2), np.diag([9.0, 1.0])
    c = 2.0
    rng = np.random.default_rng(0)
    # boundary points of the ellipse
    th = np.linspace(0, 2 * np.pi, 720)
    boundary = np.stack([3 * c * np.cos(th), 1 * c * np.sin(th)], 1)
    for _ in range(50):
        p = rng.uniform(-15, 15, 2)
        h, _ = ellipse_barrier(p, mu, S, c, margin=0.0)
        true_d = np.min(np.linalg.norm(boundary - p, axis=1))
        inside = (p @ np.linalg.inv(S) @ p) < c**2
        if not inside:
            assert h <= true_d + 1e-2   # conservative: never overstates clearance


def test_box_sd():
    r = Region("hazard", 0, 0, 10, 10)
    d, g = box_sd(np.array([13.0, 5.0]), r)
    assert abs(d - 3.0) < 1e-9 and np.allclose(g, [1, 0])
    d, _ = box_sd(np.array([5.0, 5.0]), r)
    assert d < 0
    d, g = box_sd(np.array([13.0, 14.0]), r)
    assert abs(d - 5.0) < 1e-9 and np.allclose(g, [0.6, 0.8])


def test_qp_solver():
    u, ok = solve_qp2([1.0, 0.0], [1, 1], [], [], (-1, -1), (2, 1))
    assert ok and np.allclose(u, [1, 0])
    # constraint v <= 0.5  i.e.  -v >= -0.5
    u, ok = solve_qp2([1.0, 0.0], [1, 1], [[-1.0, 0.0]], [-0.5], (-1, -1), (2, 1))
    assert ok and np.allclose(u, [0.5, 0.0])
    # two constraints meeting at a vertex: v <= 0.5 and v + w <= 0.2
    u, ok = solve_qp2([1.0, 0.0], [1, 1], [[-1.0, 0.0], [-1.0, -1.0]], [-0.5, -0.2], (-1, -1), (2, 1))
    assert ok and -u[0] - u[1] >= -0.2 - 1e-9 and u[0] <= 0.5 + 1e-9
    # contradictory: v >= 1.5 but v <= 1
    u, ok = solve_qp2([1.0, 0.0], [1, 1], [[1.0, 0.0]], [1.5], (-1, -1), (1, 1))
    assert not ok


def _drive(goal, hazards=(), ellipses=(), steps=400, start=(0.0, 0.0), heading=0.0):
    """Closed loop: a naive go-straight-to-goal command through the CBF on the rate-limited plant."""
    p = Go2Params()
    cbf = CBFFilter(SafetyParams())
    pos, th, v = np.array(start, float), heading, 0.0
    min_h_body = np.inf
    for _ in range(steps):
        d = goal - pos
        err = (np.arctan2(d[1], d[0]) - th + np.pi) % (2 * np.pi) - np.pi
        u_nom = np.array([1.0 * max(0.0, np.cos(err)), np.clip(2 * err, -1.5, 1.5)])
        u, info = cbf.filter(pos, th, u_nom, 1.0, hazards, ellipses)
        pos, th, v = step_unicycle(pos, th, v, u[0], u[1], 0.1, p, (-100, -100, 100, 100))
        for r in hazards:
            min_h_body = min(min_h_body, box_sd(pos, r)[0])
        for mu, S in ellipses:
            min_h_body = min(min_h_body, np.linalg.norm(pos - mu) - chi_radius(0.9) * np.sqrt(S[0, 0]))
    return pos, min_h_body


def test_cbf_keeps_robot_out_of_hazard_even_when_commanded_through_it():
    hazard = Region("hazard", 10, -5, 20, 5)
    _, min_sd = _drive(np.array([40.0, 0.3]), hazards=[hazard])
    assert min_sd >= 0.0   # body never inside the rectangle


def test_cbf_keeps_clearance_from_survivor_region():
    mu, S = np.array([15.0, 0.2]), np.eye(2) * 1.0
    _, min_gap = _drive(mu, ellipses=[(mu, S)])
    assert min_gap >= SafetyParams().d_clear   # never within d_clear of the 90% region


def test_tracking_error_calibration_is_small_and_positive():
    e = calibrate_tracking_error(Go2Params())
    assert 0.05 < e < 0.6
