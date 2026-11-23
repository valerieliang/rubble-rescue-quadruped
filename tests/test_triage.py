"""Reward model, handling time, drone state cue, learned observer and heading legibility."""
import numpy as np

from rescue.ctrl_behavior import StateBelief, make_site
from rescue.drone_prior import PriorConfig, generate_detections
from rescue.interfaces import (BASE_REWARD, PRIORITY_DECAY, PRIORITY_WEIGHT, REWARDS, TRIAGE_REWARD, Priority,
                               ResponseState as Z, RewardModel)
from rescue.legibility import LearnedObserver
from rescue.metrics import oracle_score
from rescue.stack import R_RESCUE


def test_base_reward_is_state_scaled_class_reward():
    v = BASE_REWARD.value(Priority.CRITICAL, Z.UNRESPONSIVE, 50.0)
    assert np.isclose(v, PRIORITY_WEIGHT[Priority.CRITICAL] * 1.5 * np.exp(-PRIORITY_DECAY[Priority.CRITICAL] * 50.0))


def test_triage_reward_state_consequences():
    t = 100.0
    # unresponsive survivors decay faster than trapped ones (after removing the state factor)
    u = TRIAGE_REWARD.value(Priority.URGENT, Z.UNRESPONSIVE, t) / TRIAGE_REWARD.factor[Z.UNRESPONSIVE]
    tr = TRIAGE_REWARD.value(Priority.URGENT, Z.TRAPPED, t) / TRIAGE_REWARD.factor[Z.TRAPPED]
    assert u < tr
    # mobile survivors self-evacuate: no reward after evac_time
    assert TRIAGE_REWARD.value(Priority.URGENT, Z.MOBILE, TRIAGE_REWARD.evac_time + 1) == 0.0
    assert TRIAGE_REWARD.value(Priority.URGENT, Z.MOBILE, TRIAGE_REWARD.evac_time - 1) > 0.0


def test_expected_ratio_matches_reward_for_known_state():
    """With z known, ratio * w * exp(-lambda * arrive) equals the reward at completion time arrive + dwell."""
    m = TRIAGE_REWARD
    for z in Z:
        sb = StateBelief()
        sb.reveal(z)
        sb.p = np.array([1.0 if h == z else 1e-12 for h in (Z.UNRESPONSIVE, Z.TRAPPED, Z.MOBILE, Z.MOBILE)])
        sb.p /= sb.p.sum()
        arrive, pr = 40.0, Priority.URGENT
        got = sb.expected_ratio(pr, arrive, m) * PRIORITY_WEIGHT[pr] * np.exp(-PRIORITY_DECAY[pr] * arrive)
        want = m.value(pr, z, arrive + m.dwell[z])
        assert np.isclose(got, want, rtol=1e-3), z


def test_expected_dwell_is_posterior_mean():
    sb = StateBelief()
    sb.p = np.array([0.5, 0.5, 0.0, 0.0])
    sb.p = np.maximum(sb.p, 1e-12)
    sb.p /= sb.p.sum()
    assert np.isclose(sb.expected_dwell(TRIAGE_REWARD), 0.5 * 30 + 0.5 * 15, atol=0.01)


def test_oracle_includes_handling_time_and_knowing_state_never_hurts():
    site = make_site(3, 6, None)
    plain = RewardModel()
    slow = RewardModel(dwell={z: 20.0 for z in Z})
    assert oracle_score(site, 600.0, standoff=R_RESCUE, state_weighted=slow) < \
        oracle_score(site, 600.0, standoff=R_RESCUE, state_weighted=plain)
    blind = RewardModel(factor={z: 1.0 for z in Z}, dwell={z: 15.0 for z in Z})
    full = oracle_score(site, 600.0, standoff=R_RESCUE, state_weighted=TRIAGE_REWARD)
    blind_val = oracle_score(site, 600.0, standoff=R_RESCUE, state_weighted=TRIAGE_REWARD, blind_order=blind)
    assert blind_val <= full + 1e-9


def test_state_cue_off_by_default_and_unbiased_when_on():
    site = make_site(0)
    assert all(d.state_hint is None for d in generate_detections(site, PriorConfig(), np.random.default_rng(0)))
    hits = n = 0
    for s in range(200):
        for d in generate_detections(site, PriorConfig(p_state_cue=0.7, p_miss=0.0, n_false_pos=0.0),
                                     np.random.default_rng(s)):
            n += 1
            hits += d.state_hint == int(site.survivors[d.source_id].state)
    assert abs(hits / n - 0.7) < 0.05


def test_state_hint_update_moves_belief_toward_hint():
    sb = StateBelief()
    sb.update_hint(int(Z.UNRESPONSIVE), 0.8)
    assert sb.map_state() == Z.UNRESPONSIVE and sb.p_state()[0] > 0.5


def test_learned_observer_identifies_goal_and_is_deterministic():
    start = np.array([20.0, 50.0])
    goals = np.array([[42.0, 50.0], [40.0, 58.0]])
    path = np.linspace(start, goals[0], 150)
    o1, o2 = LearnedObserver(), LearnedObserver()
    assert np.allclose(o1.theta, o2.theta)
    post = o1.posterior(path, start, goals, np.random.default_rng(0))
    assert post.shape == (150, 2) and np.allclose(post.sum(1), 1.0)
    assert post[-1, 0] > 0.8


def test_heading_legibility_term_prefers_pointing_at_goal():
    from rescue.ctrl_mpc import ApproachMPC
    mpc = ApproachMPC()
    goals = np.array([[30.0, 0.0], [30.0, 6.0]])
    # a robot facing the true goal 0 turns away from distractor 1 with a heading term
    v0, w0 = mpc.solve(np.zeros(2), 0.3, 1.0, goals[0], leg=(goals, 0, np.zeros(2), 0.0, 0.7, 3.0))
    v1, w1 = mpc.solve(np.zeros(2), 0.3, 1.0, goals[0])
    assert w0 <= w1 + 1e-9
