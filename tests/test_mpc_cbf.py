"""MPC planner and CBF safety filter, in closed loop on a local unicycle integrator (no simulator)."""
import numpy as np
import pytest

from rescue.cbf import SafetyFilter, rect_sd, rect_sd_grad
from rescue.clearance import ClearanceRegion
from rescue.interfaces import Region, ResponseState
from rescue.mpc import ApproachController, MPCPlanner, legibility_weight, standoff_pose
from rescue.observer import goal_posterior

DT = 0.1


def step(pos, heading, v, v_cmd, omega, max_accel=1.0):
    v = v + float(np.clip(v_cmd - v, -max_accel * DT, max_accel * DT))
    heading = (heading + omega * DT + np.pi) % (2 * np.pi) - np.pi
    return pos + v * DT * np.array([np.cos(heading), np.sin(heading)]), heading, v


def go_straight(pos, heading, goal):
    """Naive nominal command: turn toward the goal and walk (ignores obstacles)."""
    d = goal - pos
    err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
    return max(0.0, np.cos(err)) * min(1.0, np.linalg.norm(d)), float(np.clip(2 * err, -1.5, 1.5))


# --- geometry ---------------------------------------------------------------------------

BOX = Region("hazard", 10.0, -5.0, 20.0, 5.0)


def test_rect_signed_distance_and_gradient():
    assert rect_sd(np.array([15.0, 0.0]), BOX) == pytest.approx(-5.0)
    assert rect_sd(np.array([25.0, 0.0]), BOX) == pytest.approx(5.0)
    assert rect_sd(np.array([23.0, 9.0]), BOX) == pytest.approx(5.0)       # corner: 3-4-5
    assert np.allclose(rect_sd_grad(np.array([25.0, 0.0]), BOX), [1.0, 0.0])
    assert np.allclose(rect_sd_grad(np.array([15.0, 4.0]), BOX), [0.0, 1.0])   # inside, nearest edge
    g = rect_sd_grad(np.array([23.0, 9.0]), BOX)
    assert np.allclose(g, [0.6, 0.8]) and np.linalg.norm(g) == pytest.approx(1.0)


# --- CBF filter -------------------------------------------------------------------------

def test_filter_leaves_safe_command_alone():
    f = SafetyFilter()
    v, om = f.filter(np.zeros(2), 0.0, 0.7, 0.3, hazards=[BOX])
    assert (v, om) == (pytest.approx(0.7), pytest.approx(0.3)) and f.n_interventions == 0


def test_filter_slows_toward_obstacle_and_stops_at_boundary():
    f = SafetyFilter(slide=False)
    far = f.filter(np.array([5.0, 0.0]), 0.0, 1.0, 0.0, hazards=[BOX])[0]
    near = f.filter(np.array([9.1, 0.0]), 0.0, 1.0, 0.0, hazards=[BOX])[0]
    at = f.filter(np.array([9.4, 0.0]), 0.0, 1.0, 0.0, hazards=[BOX])[0]
    assert far > near > at >= 0.0 and at == pytest.approx(0.0, abs=1e-6)


def test_filter_forces_exit_when_inside():
    f = SafetyFilter(slide=False)
    v, _ = f.filter(np.array([10.1, 0.0]), np.pi, 0.0, 0.0, hazards=[BOX])   # inside, facing out
    assert v > 0.0


def test_closed_loop_never_enters_hazard_even_with_naive_nominal():
    f = SafetyFilter()
    pos, heading, v = np.array([0.0, 0.0]), 0.0, 0.0
    goal = np.array([30.0, 0.0])                      # straight through the hazard
    worst = np.inf
    for _ in range(600):
        v_nom, om_nom = go_straight(pos, heading, goal)
        v_cmd, om = f.filter(pos, heading, v_nom, om_nom, hazards=[BOX], goal=goal)
        pos, heading, v = step(pos, heading, v, v_cmd, om)
        worst = min(worst, float(rect_sd(pos, BOX)))
    assert worst > 0.0 and f.n_interventions > 0


def test_region_barrier_keeps_robot_out_of_clearance_disc():
    reg = ClearanceRegion(np.array([15.0, 0.0]), 4.0 * np.eye(2), 0.95, 1.0)
    f = SafetyFilter()
    pos, heading, v = np.array([0.0, 0.0]), 0.0, 0.0
    goal = np.array([30.0, 0.0])
    dmin = np.inf
    for _ in range(500):
        v_nom, om_nom = go_straight(pos, heading, goal)
        v_cmd, om = f.filter(pos, heading, v_nom, om_nom, regions=[reg], goal=goal)
        pos, heading, v = step(pos, heading, v, v_cmd, om)
        dmin = min(dmin, float(np.linalg.norm(pos - reg.center)))
    assert dmin > reg.radius - 0.05


def test_target_region_is_relaxed_to_physical_keepout():
    reg = ClearanceRegion(np.array([15.0, 0.0]), 9.0 * np.eye(2), 0.95, 1.0)      # radius ~ 8.3 m
    f = SafetyFilter()
    blocked = f.filter(np.array([8.0, 0.0]), 0.0, 1.0, 0.0, regions=[reg])[0]
    ok = f.filter(np.array([8.0, 0.0]), 0.0, 1.0, 0.0, regions=[reg], target=reg.center)[0]
    assert blocked < 0.5 * ok


# --- MPC planner ------------------------------------------------------------------------

def run_mpc(target, ctrl, hazards=(), regions=(), others=(), t_max=90.0, start=(0.0, 0.0), heading=0.0):
    pos, v = np.array(start, float), 0.0
    traj = [pos.copy()]
    for _ in range(int(t_max / DT)):
        v_cmd, om = ctrl.command(pos, heading, v, target, None, hazards, regions, others)
        pos, heading, v = step(pos, heading, v, v_cmd, om)
        traj.append(pos.copy())
        if np.linalg.norm(pos - target) < 1.5:
            break
    return np.array(traj)


def test_standoff_pose_stops_short_of_target():
    t = np.array([10.0, 0.0])
    p = standoff_pose(np.zeros(2), t)
    assert np.linalg.norm(p - t) < 1.5 and p[0] < t[0]
    assert np.allclose(standoff_pose(np.array([10.5, 0.0]), t), t)   # inside the standoff: go to the target


def test_mpc_reaches_standoff_on_open_ground():
    traj = run_mpc(np.array([20.0, 10.0]), ApproachController(MPCPlanner(seed=1)))
    assert np.linalg.norm(traj[-1] - [20.0, 10.0]) < 1.5
    assert len(traj) * DT < 45.0                              # ~22 m at 1 m/s plus turning


def test_mpc_respects_speed_and_turn_limits():
    pl = MPCPlanner(seed=2)
    v, om = pl.plan(np.zeros(2), np.pi, 0.0, np.array([10.0, 0.0]))
    assert 0.0 <= v <= 1.0 and abs(om) <= 1.5


def test_mpc_routes_around_hazard_without_filter():
    traj = run_mpc(np.array([30.0, 0.0]), ApproachController(MPCPlanner(seed=3)), hazards=[BOX])
    assert np.linalg.norm(traj[-1] - [30.0, 0.0]) < 1.5
    assert rect_sd(traj, BOX).min() > -0.5                    # the cost alone keeps it (almost) out


def test_mpc_with_cbf_reaches_goal_and_stays_safe():
    ctrl = ApproachController(MPCPlanner(seed=4), SafetyFilter())
    traj = run_mpc(np.array([30.0, 0.0]), ctrl, hazards=[BOX])
    assert np.linalg.norm(traj[-1] - [30.0, 0.0]) < 1.5
    assert rect_sd(traj, BOX).min() > 0.0


def test_mpc_goes_around_other_survivors_clearance_region():
    reg = ClearanceRegion(np.array([15.0, 0.5]), 4.0 * np.eye(2), 0.95, 1.0)
    ctrl = ApproachController(MPCPlanner(seed=5), SafetyFilter())
    traj = run_mpc(np.array([30.0, 0.0]), ctrl, regions=[reg])
    assert np.linalg.norm(traj[-1] - [30.0, 0.0]) < 1.5
    assert np.linalg.norm(traj - reg.center, axis=1).min() > reg.radius - 0.05


# --- legibility -------------------------------------------------------------------------

def test_legibility_weight_scales_with_mobility():
    mobile = np.array([0.0, 0.0, 1.0])
    still = np.array([1.0, 0.0, 0.0])
    assert legibility_weight(mobile, 10.0) == pytest.approx(10.0)
    assert legibility_weight(still, 10.0) == pytest.approx(2.0)
    assert legibility_weight(None, 10.0) == 10.0


def test_legibility_term_makes_the_goal_identifiable_earlier():
    target, other = np.array([25.0, 6.0]), np.array([25.0, -6.0])
    goals = [target, other]
    travel = lambda a, b: float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

    def mid_posterior(w):
        ctrl = ApproachController(MPCPlanner(seed=7), legibility_base=w)
        traj = run_mpc(target, ctrl, others=[other], t_max=12.0)
        return goal_posterior(traj[::10], goals, travel, beta=0.5)[0]

    assert mid_posterior(20.0) > mid_posterior(0.0) + 0.1
