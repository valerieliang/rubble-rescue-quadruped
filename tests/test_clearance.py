import numpy as np
import pytest

from rescue.belief import Belief
from rescue.clearance import ClearanceRegion, chi2_level
from rescue.drone_prior import PriorConfig
from rescue.interfaces import Detection, Priority, Region, RunLog, Site, Survivor
from rescue.metrics import safety_violations
from rescue.search import PriorSearch
from rescue.terrain import travel_time


def test_chi2_level_matches_closed_form():
    # 2-D chi-square: P(X <= k2) = 1 - exp(-k2 / 2)
    assert chi2_level(0.95) == pytest.approx(-2 * np.log(0.05), rel=1e-6)
    with pytest.raises(ValueError):
        chi2_level(1.0)


def test_region_covers_requested_mass():
    rng = np.random.default_rng(0)
    reg = ClearanceRegion(np.array([5.0, 5.0]), 4.0 * np.eye(2), 0.95, 0.0)
    pts = rng.normal([5.0, 5.0], 2.0, size=(20000, 2))
    inside = np.mean([reg.contains(p) for p in pts])
    assert abs(inside - 0.95) < 0.01


def test_region_grows_with_uncertainty_and_confidence():
    r = lambda s, c: ClearanceRegion(np.zeros(2), s**2 * np.eye(2), c, 1.0).radius
    assert r(6, 0.95) > r(3, 0.95) > r(1, 0.95)
    assert r(3, 0.99) > r(3, 0.9)


def test_barrier_sign_and_gradient():
    reg = ClearanceRegion(np.zeros(2), np.eye(2), 0.95, 1.0)
    R = reg.radius
    assert reg.barrier(np.array([R + 1.0, 0.0])) > 0 > reg.barrier(np.array([R - 0.5, 0.0]))
    assert abs(reg.barrier(np.array([R, 0.0]))) < 1e-9
    assert np.allclose(reg.barrier_grad(np.array([2.0, 0.0])), [4.0, 0.0])


def test_anisotropic_radius_uses_largest_axis():
    reg = ClearanceRegion(np.zeros(2), np.diag([9.0, 1.0]), 0.95, 0.0)
    assert reg.radius == pytest.approx(3.0 * np.sqrt(chi2_level(0.95)))


def test_belief_and_search_expose_regions():
    dets = [Detection(np.array([10.0, 10.0]), Priority.CRITICAL),
            Detection(np.array([30.0, 5.0]), Priority.STABLE)]
    cfg = PriorConfig(sigma_pos=3.0)
    b = Belief(dets, cfg)
    regs = b.clearance_regions(0.95)
    assert len(regs) == 2 and regs[0].priority == Priority.CRITICAL
    assert regs[0].radius == pytest.approx(3.0 * np.sqrt(chi2_level(0.95)) + 1.0)
    site = Site(0, (0, 0, 50, 50), (), (), np.zeros(2))
    ps = PriorSearch(dets, cfg, site.bounds, lambda a, c: travel_time(site, a, c), 5.0, site.start, 600.0)
    assert len(ps.clearance_regions()) == 2
    # a confirmed detection tightens the track, so its region shrinks
    ps.belief.update(np.array([10.0, 10.0]), 5.0, [Detection(np.array([10.0, 10.0]), Priority.CRITICAL)])
    assert ps.clearance_regions()[0].radius < regs[0].radius


def test_low_existence_tracks_are_dropped():
    b = Belief([Detection(np.array([10.0, 10.0]), Priority.STABLE)], PriorConfig())
    b.tracks[0].r = 0.01
    assert b.clearance_regions() == []


def _site():
    return Site(0, (0, 0, 50, 50), (Region("hazard", 10, 10, 20, 20), Region("rubble", 30, 30, 40, 40)),
                (Survivor(0, np.array([40.0, 5.0]), Priority.URGENT),), np.zeros(2))


def test_safety_violations_counts_hazard_and_survivor_steps():
    log = RunLog(0, "x", trajectory=[(1, 0, 0), (2, 15, 15), (3, 35, 35), (4, 40, 5.5), (5, 40, 8.0)])
    v = safety_violations(_site(), log, keepout=1.0)
    assert v["hazard_steps"] == 1 and v["survivor_steps"] == 1   # rubble is not a hazard
    assert v["violation_steps"] == 2 and v["violation_rate"] == pytest.approx(0.4)


def test_safety_violations_empty_trajectory():
    assert safety_violations(_site(), RunLog(0, "x"))["violation_rate"] == 0.0


def test_fresh_tracks_use_the_higher_confidence_until_enough_detections():
    from rescue.belief import Belief
    from rescue.drone_prior import PriorConfig
    from rescue.interfaces import Detection, Priority
    b = Belief([], PriorConfig(), sigma_meas=0.5)
    det = Detection(np.array([5.0, 0.0]), Priority.STABLE, 0)
    robot = np.zeros(2)
    conf = []
    for _ in range(6):
        b.update(robot, 10.0, [det])               # first detection creates a confirmed track, n_det = 1
        (reg,) = b.clearance_regions(0.7, 1.0, 0.1, fresh_confidence=0.99, fresh_n=5)
        conf.append(reg.confidence)
    assert b.tracks[0].n_det == 6
    assert conf == [0.99, 0.99, 0.99, 0.99, 0.7, 0.7]       # fresh while n_det < 5
    (reg,) = b.clearance_regions(0.7, 1.0, 0.1)             # off by default
    assert reg.confidence == 0.7


def test_prior_only_track_is_not_fresh():
    from rescue.belief import Track
    from rescue.clearance import clearance_regions
    t = Track(np.array([3.0, 0.0]), 3.0, 0.8, Priority.STABLE)      # drone prior only: unconfirmed, n_det 0
    (reg,) = clearance_regions([t], 0.7, 1.0, 0.1, fresh_confidence=0.99, fresh_n=5)
    assert reg.confidence == 0.7
