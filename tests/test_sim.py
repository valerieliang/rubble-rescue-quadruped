"""Tests for the reduced-order Go2 simulator (rescue/sim.py)."""
import numpy as np
import pytest

from rescue import sim, toy_sim
from rescue.drone_prior import PriorConfig
from rescue.interfaces import Priority, Region, RunLog, Site, Survivor


class GoTo:
    """Policy that visits a fixed list of points in order (no belief)."""

    def __init__(self, points):
        self.points = [np.asarray(p, float) for p in points]

    def act(self, obs):
        while self.points and np.linalg.norm(self.points[0] - obs.robot_pos) < 1.0:
            self.points.pop(0)
        return self.points[0] if self.points else obs.robot_pos


class NearestTrue:
    """Visits survivors nearest-first using ground truth (sanity policy, not a baseline)."""

    def __init__(self, site):
        self.left = {s.id: s.pos for s in site.survivors}

    def act(self, obs):
        for r in obs.rescued:
            self.left = {i: p for i, p in self.left.items() if not np.allclose(p, r)}
        if not self.left:
            return obs.robot_pos
        return min(self.left.values(), key=lambda p: np.linalg.norm(p - obs.robot_pos))


def open_site(survivors, regions=()):
    return Site(0, (0.0, 0.0, 100.0, 100.0), tuple(regions), tuple(survivors), np.array([0.0, 0.0]))


def surv(i, x, y, pr=Priority.STABLE):
    return Survivor(i, np.array([x, y], float), pr)


def test_straight_line_time_open_ground():
    # 40 m in the open at 1 m/s from rest; accel 1 m/s^2 adds ~0.5 s; standoff 1.5 m saves 1.5 s.
    site = open_site([surv(0, 40, 0)])
    log = sim.run_episode(site, lambda s, d: GoTo([[40, 0]]), PriorConfig(), seed=0, heading0=0.0)
    assert 38.0 <= log.rescue_times[0] <= 39.5


def test_rubble_slows_robot():
    site = open_site([surv(0, 40, 0)])
    rubble = open_site([surv(0, 40, 0)], [Region("rubble", 10, -5, 30, 5)])
    t_open = sim.run_episode(site, lambda s, d: GoTo([[40, 0]]), PriorConfig(), 0, heading0=0.0)
    t_rub = sim.run_episode(rubble, lambda s, d: GoTo([[40, 0]]), PriorConfig(), 0, heading0=0.0)
    # 20 m at 0.5 m/s instead of 1 m/s costs ~20 s more.
    assert t_rub.rescue_times[0] - t_open.rescue_times[0] == pytest.approx(20, abs=1.5)


def test_turn_in_place_costs_time():
    site = open_site([surv(0, 20, 0)])
    ahead = sim.run_episode(site, lambda s, d: GoTo([[20, 0]]), PriorConfig(), 0, heading0=0.0)
    behind = sim.run_episode(site, lambda s, d: GoTo([[20, 0]]), PriorConfig(), 0,
                             heading0=np.pi)
    # The first 90 deg of a U-turn happens in place (>= ~1 s at 1.5 rad/s); the rest overlaps
    # with walking, so a U-turn costs roughly 1-2.5 s.
    extra = behind.rescue_times[0] - ahead.rescue_times[0]
    assert 1.0 <= extra <= 2.5


def test_turn_rate_respected():
    p = sim.Go2Params()
    _, omega = sim.track_waypoint(np.zeros(2), 0.0, np.array([-10.0, 0.1]), 1.0, p)
    assert abs(omega) <= p.max_turn_rate + 1e-12


def test_stays_in_bounds():
    site = open_site([surv(0, 50, 50)])
    log = sim.run_episode(site, lambda s, d: GoTo([[-50, -50]]), PriorConfig(), 0, T=30)
    xy = np.array([(x, y) for _, x, y in log.trajectory])
    assert xy.min() >= 0.0


def test_travel_time_matches_terrain():
    site = open_site([], [Region("rubble", 10, -5, 30, 5)])
    assert sim.travel_time(site, [0, 0], [40, 0]) == pytest.approx(20 + 40, abs=1.0)
    assert sim.travel_time(open_site([]), [0, 0], [30, 40]) == pytest.approx(50, abs=1e-6)
    tt = sim.make_travel_time(site)
    assert tt(np.array([0.0, 0.0]), np.array([40.0, 0.0])) == pytest.approx(60, abs=1.0)


@pytest.mark.parametrize("seed", range(10))
def test_same_contract_as_toy_sim(seed):
    site = sim.generate_site(seed)
    log = sim.run_episode(site, lambda s, d: NearestTrue(s), PriorConfig(), seed, method="x")
    assert isinstance(log, RunLog) and log.site_seed == site.seed and log.method == "x"
    assert set(log.rescue_times) <= {s.id for s in site.survivors}
    ts = [t for t, _, _ in log.trajectory]
    assert ts == sorted(ts) and ts[-1] <= 600.0 + 1e-9
    assert all(0 < t <= 600.0 for t in log.rescue_times.values())


def test_reproducible():
    site = sim.generate_site(3)
    a = sim.run_episode(site, lambda s, d: NearestTrue(s), PriorConfig(), 7)
    b = sim.run_episode(site, lambda s, d: NearestTrue(s), PriorConfig(), 7)
    assert a.rescue_times == b.rescue_times and a.trajectory == b.trajectory


def test_drone_prior_identical_to_toy_sim():
    # Same seed -> same detections handed to make_policy in both sims (paired comparisons).
    site = sim.generate_site(5)
    seen = {}

    def capture(name):
        def mk(s, dets):
            seen[name] = [(tuple(d.pos), d.priority, d.source_id) for d in dets]
            return NearestTrue(s)
        return mk

    toy_sim.run_episode(site, capture("toy"), PriorConfig(), 11)
    sim.run_episode(site, capture("go2"), PriorConfig(), 11)
    assert seen["toy"] == seen["go2"]


def test_close_to_toy_sim_on_average():
    # Unicycle should be a little slower than the point robot (turns, accel), not wildly off.
    diffs = []
    for seed in range(20):
        site = sim.generate_site(seed)
        a = toy_sim.run_episode(site, lambda s, d: NearestTrue(s), PriorConfig(), seed)
        b = sim.run_episode(site, lambda s, d: NearestTrue(s), PriorConfig(), seed)
        for i in set(a.rescue_times) & set(b.rescue_times):
            diffs.append(b.rescue_times[i] - a.rescue_times[i])
    diffs = np.array(diffs)
    assert diffs.mean() >= -1.0          # not faster than the point robot
    assert diffs.mean() <= 30.0          # but in the same ballpark