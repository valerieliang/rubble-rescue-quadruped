import numpy as np

from rescue.approach import run_approach
from rescue.ctrl_behavior import make_site
from rescue.ctrl_sim import run_episode_ctrl
from rescue.drone_prior import PriorConfig
from rescue.interfaces import ResponseState
from rescue.legibility import CostObserver, identification_time, log_prob_goal
from rescue.ctrl_mpc import ApproachMPC, TerrainGrid, lookahead
from rescue.stack import VARIANTS, StackConfig

CFG = PriorConfig()


def test_mpc_drives_toward_goal_and_turns_in_place_when_goal_is_behind():
    m = ApproachMPC()
    v, w = m.solve(np.zeros(2), 0.0, 1.0, np.array([10.0, 0.0]))
    assert v > 0.5 and abs(w) < 0.5
    v, w = m.solve(np.zeros(2), 0.0, 1.0, np.array([-10.0, 0.5]))
    assert abs(w) > 0.5   # it turns (while moving or in place) instead of stalling


def test_terrain_grid_path_avoids_hazard():
    site = make_site(3)   # has a hazard rectangle
    grid = TerrainGrid(site)
    hz = [r for r in site.regions if r.kind == "hazard"][0]
    a = np.array([hz.xmin - 6, (hz.ymin + hz.ymax) / 2])
    b = np.array([hz.xmax + 6, (hz.ymin + hz.ymax) / 2])
    path = grid.path(a, b)
    assert np.allclose(path[0], a) and np.allclose(path[-1], b)
    mid = path[1:-1]
    inside = [(hz.xmin <= p[0] <= hz.xmax and hz.ymin <= p[1] <= hz.ymax) for p in mid]
    assert not any(inside)
    assert np.allclose(lookahead(path, a, 4.0), path[np.argmax(np.linalg.norm(path - a, axis=1) >= 4.0)])


def test_log_prob_goal_prefers_true_goal_when_heading_to_it():
    goals = np.array([[20.0, 0.0], [20.0, 10.0]])
    S = np.zeros(2)
    toward = log_prob_goal(np.array([5.0, 0.0]), S, goals, 0, 0.7)
    away = log_prob_goal(np.array([5.0, 3.0]), S, goals, 0, 0.7)
    assert toward > away
    post = CostObserver(0.7, 0.0).posterior(np.array([[0.0, 0.0], [10.0, 0.0], [18.0, 0.0]]), S, goals,
                                            np.random.default_rng(0))
    assert np.allclose(post.sum(1), 1) and post[-1, 0] > 0.9
    assert identification_time(post, 0, 1.0) <= 2.0


def test_ctrl_episode_perfect_prior_rescues_everyone_without_violations():
    site = make_site(1)
    cfg = PriorConfig(sigma_pos=0.5, p_miss=0.0, n_false_pos=0.0)
    log = run_episode_ctrl(site, cfg, VARIANTS["full"], 1, "full", T=600.0)
    assert len(log.rescue_times) == len(site.survivors)
    assert log.clear_viol_steps == 0 and log.haz_viol_steps == 0


def test_filter_removes_clearance_violations_that_the_unfiltered_stack_has():
    viol = {"full": 0, "no_cbf": 0}
    for seed in (1, 2, 4, 6):
        site = make_site(seed)
        for name in viol:
            log = run_episode_ctrl(site, CFG, VARIANTS[name], seed, name, T=400.0)
            viol[name] += log.clear_viol_steps
    assert viol["full"] == 0 and viol["no_cbf"] > 0


def test_hazard_blind_planner_still_never_enters_hazard_with_filter():
    cfg = StackConfig(hazard_aware_planner=False)
    for seed in (0, 3):
        site = make_site(seed)
        log = run_episode_ctrl(site, CFG, cfg, seed, "naive", T=300.0)
        assert log.haz_viol_steps == 0


def test_oracle_reveals_state_and_inference_is_mostly_right():
    ok = n = 0
    for seed in (1, 2, 3, 4, 5, 6):
        site = make_site(seed)
        log = run_episode_ctrl(site, CFG, VARIANTS["full"], seed, "full", T=400.0)
        for sid, tr in log.state_trace.items():
            p = tr[-1][1]
            ok += int(np.argmax([p[0], p[1], p[2] + p[3]])) == int(site.survivors[sid].state)
            n += 1
    assert n >= 15 and ok / n > 0.8
    site = make_site(2)
    log = run_episode_ctrl(site, CFG, VARIANTS["oracle"], 2, "oracle", T=400.0)
    for sid, tr in log.state_trace.items():
        p = tr[-1][1]
        assert int(np.argmax([p[0], p[1], p[2] + p[3]])) == int(site.survivors[sid].state)


def test_legibility_helps_cost_observer_and_costs_time_but_not_for_unresponsive():
    base = [run_approach(s, StackConfig(legibility=0.0)) for s in range(6)]
    leg = [run_approach(s, StackConfig(legibility=300.0), ResponseState.MOBILE) for s in range(6)]
    unresp = [run_approach(s, StackConfig(legibility=300.0), ResponseState.UNRESPONSIVE) for s in range(6)]
    assert np.mean([r.mean_p["cost"] for r in leg]) > np.mean([r.mean_p["cost"] for r in base]) + 0.05
    assert np.mean([r.t_total for r in leg]) > np.mean([r.t_total for r in base])
    assert abs(np.mean([r.t_total for r in unresp]) - np.mean([r.t_total for r in base])) < 1e-6
    assert all(r.reached for r in base + leg + unresp)


def test_components_controller_runs_in_the_closed_loop_and_is_safe():
    site = make_site(2)
    cfg = StackConfig(controller="components")
    log = run_episode_ctrl(site, CFG, cfg, 2, "components", T=300.0)
    assert len(log.rescue_times) >= 1
    assert log.clear_viol_steps == 0 and log.haz_viol_steps == 0
