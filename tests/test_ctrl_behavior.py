import numpy as np

from rescue.ctrl_behavior import (BehaviorConfig, MotionTracker, Mode, StateBelief, assign_modes, make_site,
                             response_probs, sample_response, step_survivors)
from rescue.interfaces import Priority, Region, ResponseState, Site, Survivor

Z = ResponseState


def test_response_probs_are_distributions_and_silence_is_evidence():
    for z in Z:
        for q in (0.4, 0.8, 1.0):
            p = response_probs(z, q)
            assert abs(p.sum() - 1) < 1e-12 and (p >= 0).all()
    b = StateBelief()
    b.update_prompt(2, 0.8)            # "nothing"
    assert b.p_state()[Z.UNRESPONSIVE] > 0.6
    b = StateBelief()
    b.update_prompt(0, 0.8)            # "answer"
    assert b.map_state() == Z.MOBILE


def test_motion_update_separates_modes():
    for vel, mode in (([0.3, 0.0], Mode.APPROACHING), ([-0.3, 0.0], Mode.RETREATING), ([0.0, 0.0], Mode.STILL)):
        b = StateBelief()
        b.update_motion(vel, 0.06, 0.3)
        assert b.map_mode() == mode


def test_reveal_and_floor():
    b = StateBelief()
    b.reveal(Z.TRAPPED)
    assert b.map_state() == Z.TRAPPED and b.p_state()[Z.TRAPPED] > 0.99
    b.update_prompt(0, 1.0)            # contradicting cue can still move it (belief never absorbing)
    assert (b.p > 0).all()


def test_motion_tracker_recovers_velocity():
    mt = MotionTracker()
    rng = np.random.default_rng(0)
    for k in range(60):
        t = 0.1 * k
        mt.add(t, np.array([1.0 + 0.3 * t, 2.0 - 0.1 * t]) + rng.normal(0, 0.3, 2))
    assert mt.ready(5.0)
    v, sv = mt.velocity(0.3)
    assert np.allclose(v, [0.3, -0.1], atol=0.1) and sv < 0.1


def _site(state):
    return Site(0, (0, 0, 100, 100), (Region("hazard", 40, 0, 60, 100),),
                (Survivor(0, np.array([30.0, 50.0]), Priority.URGENT, state),), np.array([3.0, 3.0]))


def test_retreating_moves_away_and_approaching_stops():
    site = _site(Z.MOBILE)
    cfg = BehaviorConfig()
    robot = np.array([20.0, 50.0])
    for mode, sign in ((Mode.RETREATING, +1), (Mode.APPROACHING, -1)):
        pos, head = {0: np.array([30.0, 50.0])}, {0: 0.0}
        for _ in range(20):
            step_survivors(site, pos, {0: mode}, head, {0}, robot, 1.0, cfg, np.random.default_rng(0))
        assert (pos[0][0] - 30.0) * sign > 1.0   # moved ~0.3 m/s * 20 s = 6 m... capped by stop/hazard
    pos = {0: np.array([30.0, 50.0])}
    for _ in range(200):
        step_survivors(site, pos, {0: Mode.APPROACHING}, {0: 0.0}, {0}, robot, 1.0, cfg, np.random.default_rng(0))
    assert abs(np.linalg.norm(pos[0] - robot) - cfg.approach_stop) < 1e-6
    pos = {0: np.array([38.0, 50.0])}      # retreating east would enter the hazard rectangle: it must not
    for _ in range(200):
        step_survivors(site, pos, {0: Mode.RETREATING}, {0: 0.0}, {0}, np.array([30.0, 50.0]), 1.0, cfg,
                       np.random.default_rng(0))
    assert pos[0][0] < 40.0


def test_assign_modes_and_make_site():
    site = make_site(5, n_survivors=8, frac_mobile=0.5)
    assert len(site.survivors) == 8
    assert sum(s.state == Z.MOBILE for s in site.survivors) == 4
    modes = assign_modes(site, BehaviorConfig(p_oom=1.0), np.random.default_rng(0))
    assert all((modes[s.id] == Mode.ERRATIC) == (s.state == Z.MOBILE) for s in site.survivors)
    assert make_site(5, 8, 0.5).survivors[0].pos.tolist() == site.survivors[0].pos.tolist()  # seeded


def test_sample_response_follows_probs():
    rng = np.random.default_rng(0)
    draws = [sample_response(Z.UNRESPONSIVE, 0.8, rng) for _ in range(500)]
    assert np.mean(np.array(draws) == 2) > 0.9
