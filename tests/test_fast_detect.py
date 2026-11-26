"""Per-track confidence in the CBF filter, the 10 Hz first-detection path, and the disturbed plant."""
from dataclasses import replace

import numpy as np

from rescue.ctrl_behavior import make_site
from rescue.dist_plant import DisturbedUnicycle, measured_tracking_error
from rescue.drone_prior import PriorConfig, generate_detections
from rescue.interfaces import Detection, Priority
from rescue.safety import CBFFilter, SafetyParams
from rescue.sim import Go2Params
from rescue.stack import RescueStack, StackConfig


def _filter(ellipses):
    f = CBFFilter(SafetyParams(conf_p=0.7, w_max=1.0))
    u, info = f.filter(np.array([-3.0, 0.0]), 0.0, np.array([1.0, 0.0]), 1.0, [], ellipses)
    return u, info


def test_per_track_confidence_changes_the_barrier():
    S = np.eye(2) * 0.25
    base = _filter([(np.zeros(2), S)])
    same = _filter([(np.zeros(2), S, 0.7)])
    high = _filter([(np.zeros(2), S, 0.99)])
    assert np.allclose(base[0], same[0]) and base[1]["min_h"] == same[1]["min_h"]
    assert high[1]["min_h"] < base[1]["min_h"]        # a larger region leaves less slack
    assert high[0][0] <= base[0][0] + 1e-9            # and slows the robot at least as much


def _stack(**kw):
    site = make_site(0)
    rng = np.random.default_rng(0)
    return RescueStack(site, generate_detections(site, PriorConfig(), rng), PriorConfig(),
                       replace(StackConfig(), **kw), Go2Params(), 100.0)


def test_fast_detect_adds_provisional_track_and_drops_it_once_belief_holds_it():
    st = _stack(fast_detect=True, fast_gate=1)
    far = np.array([500.0, 500.0])
    st.perceive(1.0, np.zeros(2), (Detection(far, Priority.URGENT, None),))
    assert len(st.fast) == 1
    st.perceive(1.1, np.zeros(2), (Detection(far + 0.3, Priority.URGENT, None),))
    assert len(st.fast) == 1 and st.fast[0][1] <= 0.5 + 1e-9      # fused, not duplicated
    st.search.belief.update(far, 1.0, [Detection(far, Priority.URGENT, None)])
    st.perceive(2.0, np.zeros(2), ())
    assert st.fast == []                                           # the belief now holds it


def test_fast_detect_off_keeps_no_provisional_tracks_and_default_is_gate_3():
    assert StackConfig().fast_detect and StackConfig().fast_gate == 3
    st = _stack(fast_detect=False)
    st.perceive(1.0, np.zeros(2), (Detection(np.array([500.0, 500.0]), Priority.URGENT, None),))
    assert st.fast == []


def test_disturbed_plant_zero_noise_is_the_unicycle_and_noise_raises_the_gap():
    a, b = DisturbedUnicycle(0.0), DisturbedUnicycle(0.0)
    for _ in range(20):
        pa = a.step(0.8, 0.3)[0]
    assert np.all(np.isfinite(pa))
    assert measured_tracking_error(0.4, n=200) > measured_tracking_error(0.0, n=200)


def test_persistence_gate_needs_repeated_detections_within_the_window():
    far = np.array([500.0, 500.0])
    st = _stack(fast_detect=True, fast_gate=3)
    st.perceive(1.0, np.zeros(2), (Detection(far, Priority.URGENT, None),))
    assert len(st.fast) == 1 and st._fast_active(1.0) == []              # one detection: tracked, not yet used by the filter
    st.perceive(1.1, np.zeros(2), (Detection(far + 0.2, Priority.URGENT, None),))
    assert st._fast_active(1.1) == []
    st.perceive(1.2, np.zeros(2), (Detection(far - 0.2, Priority.URGENT, None),))
    assert len(st._fast_active(1.2)) == 1                                # third within 0.5 s: now a barrier
    assert st._fast_active(1.9) == []                                    # the detections age out of the window
    st1 = _stack(fast_detect=True, fast_gate=1)                          # gate 1: first detection counts at once
    st1.perceive(1.0, np.zeros(2), (Detection(far, Priority.URGENT, None),))
    assert len(st1._fast_active(1.0)) == 1


def test_mpc_fast_gives_the_mpc_a_disc_only_for_active_provisional_tracks():
    from rescue.interfaces import Observation
    far = np.array([500.0, 500.0])
    seen = []

    def run(**kw):
        st = _stack(fast_detect=True, fast_gate=1, apply_filter=False, **kw)
        st.plan(Observation(0.0, st.site.start.copy(), 10.0, (), ()))
        st.perceive(1.0, np.zeros(2), (Detection(far, Priority.URGENT, None),))
        orig = st.mpc.solve
        st.mpc.solve = lambda *a, **k: (seen.append(a[5]), orig(*a, **k))[1]
        st.control(1.0, st.site.start.astype(float), 0.0, 1.0)
        return seen[-1]

    base, with_fast = run(), run(mpc_fast=True)
    assert len(with_fast) == len(base) + 1 and np.allclose(with_fast[-1][0], far)
    assert with_fast[-1][1] > 0


def test_apply_filter_off_returns_the_mpc_command_unchanged():
    st = _stack(apply_filter=False)
    st.plan(__import__("rescue.interfaces", fromlist=["Observation"]).Observation(0.0, st.site.start.copy(), 10.0, (), ()))
    u = st.control(0.0, st.site.start.astype(float), 0.0, 1.0)
    assert st.last_info == {"modified": False, "feasible": True} and np.all(np.isfinite(u))
