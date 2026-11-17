import numpy as np
import pytest

from rescue.behavior import Behavior, BehaviorModel, ModeBelief, synthetic_track
from rescue.interfaces import ResponseState


def _run(mode, seed, n=8, sigma_obs=0.5, **kw):
    rng = np.random.default_rng(seed)
    t, z, r = synthetic_track(mode, rng, n=n, sigma_obs=sigma_obs, **kw)
    mb = ModeBelief(sigma_obs=sigma_obs)
    for ti, zi, ri in zip(t, z, r):
        mb.observe(ti, zi, ri)
    return mb


def test_prior_before_two_observations():
    mb = ModeBelief()
    assert np.allclose(mb.mode_posterior, BehaviorModel().prior_mode)
    assert np.allclose(mb.state_posterior, np.full(3, 1 / 3))
    mb.observe(0.0, np.zeros(2), np.array([8.0, 0.0]))
    assert np.allclose(mb.mode_posterior, BehaviorModel().prior_mode)


def test_posteriors_normalized():
    mb = _run(Behavior.APPROACHING, 0)
    assert mb.mode_posterior.sum() == pytest.approx(1.0)
    assert mb.state_posterior.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("mode", list(Behavior))
def test_low_noise_identifies_each_mode(mode):
    hits = np.mean([_run(mode, s, n=8, sigma_obs=0.2).map_mode() == mode for s in range(100)])
    assert hits > 0.95


def test_accuracy_degrades_with_noise():
    acc = lambda sig: np.mean([_run(m, s, sigma_obs=sig).map_mode() == m
                               for m in Behavior for s in range(60)])
    assert acc(0.2) > acc(1.5)


def test_more_observations_sharpen_the_correct_mode():
    p = lambda n: np.mean([_run(Behavior.APPROACHING, s, n=n).mode_posterior[Behavior.APPROACHING]
                           for s in range(100)])
    assert p(6) > p(3)


def test_mobile_state_favoured_when_survivor_moves():
    assert _run(Behavior.RETREATING, 1, sigma_obs=0.2).map_state() == ResponseState.MOBILE
    assert _run(Behavior.STILL, 1, sigma_obs=0.2).map_state() != ResponseState.MOBILE


def test_lateral_motion_is_surprising():
    in_model = np.mean([_run(Behavior.APPROACHING, s, sigma_obs=0.2).surprise for s in range(50)])
    lateral = np.mean([_run(Behavior.STILL, s, sigma_obs=0.2, lateral=True).surprise for s in range(50)])
    # surprise is a 2-dof chi-square-scaled residual: ~1 when the motion fits a mode
    assert in_model < 2.0 and lateral > 9.2


def test_window_forgets_old_motion():
    mb = ModeBelief(sigma_obs=0.05, window=4)
    r = np.array([8.0, 0.0])
    for k in range(5):                      # approaching, then stops
        mb.observe(float(k), np.array([0.5 * k, 0.0]), r)
    for k in range(5, 12):
        mb.observe(float(k), np.array([2.0, 0.0]), r)
    assert mb.map_mode() == Behavior.STILL


def test_window_validation():
    with pytest.raises(ValueError):
        ModeBelief(window=1)
