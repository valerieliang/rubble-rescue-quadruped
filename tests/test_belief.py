import numpy as np

from rescue.belief import Belief, disc_mass
from rescue.drone_prior import PriorConfig, generate_detections
from rescue.interfaces import Detection, Priority, Site, Survivor


def _site(seed=0):
    sv = tuple(Survivor(i, np.array([10.0 + 5 * i, 20.0]), Priority.CRITICAL) for i in range(4))
    return Site(seed, (0, 0, 50, 50), (), sv, np.zeros(2))


def test_generate_detections_seeded_and_noisy():
    cfg = PriorConfig(sigma_pos=2.0, p_miss=0.0, n_false_pos=0.0)
    a = generate_detections(_site(), cfg, np.random.default_rng(1))
    b = generate_detections(_site(), cfg, np.random.default_rng(1))
    assert len(a) == 4 and all(np.allclose(x.pos, y.pos) for x, y in zip(a, b))
    err = np.mean([np.linalg.norm(d.pos - s.pos) for d, s in zip(a, _site().survivors)])
    assert 0.5 < err < 6.0


def test_disc_mass_limits():
    c = np.array([0.0, 0.0])
    assert disc_mass(c, 1.0, c, 100.0) > 0.999
    assert disc_mass(c, 1.0, c + 50, 1.0) < 1e-6


def test_empty_footprint_on_track_lowers_existence():
    b = Belief([Detection(np.array([10.0, 10.0]), Priority.STABLE)], PriorConfig(n_false_pos=0.5))
    r0 = b.tracks[0].r
    b.update(np.array([10.0, 10.0]), 5.0, [])
    assert b.tracks[0].r < r0


def test_empty_footprint_far_away_changes_nothing():
    b = Belief([Detection(np.array([10.0, 10.0]), Priority.STABLE)], PriorConfig())
    r0 = b.tracks[0].r
    b.update(np.array([100.0, 100.0]), 5.0, [])
    assert abs(b.tracks[0].r - r0) < 1e-9


def test_detection_confirms_and_tightens():
    b = Belief([Detection(np.array([10.0, 10.0]), Priority.STABLE)], PriorConfig(sigma_pos=3.0))
    b.update(np.array([11.0, 10.0]), 5.0, [Detection(np.array([12.0, 10.0]), Priority.CRITICAL)])
    t = b.tracks[0]
    assert t.confirmed and t.r == 1.0 and t.sigma < 3.0 and t.priority == Priority.CRITICAL
    assert t.mean[0] > 11.5


def test_confirmed_track_fades_when_nobody_is_there_only_with_forget_confirmed():
    """Sites 16 and 18 (sigma_meas 1 m): the robot waited 500+ s at the standoff of a confirmed track whose
    survivor had walked away, because missed detections never lowered a confirmed track's existence."""
    from rescue.belief import Track
    for forget, expect_gone in ((False, False), (True, True)):
        b = Belief([], PriorConfig())
        b.forget_confirmed = forget
        b.tracks = [Track(np.array([3.0, 0.0]), 0.5, 1.0, Priority.URGENT, confirmed=True)]
        for _ in range(5):                                   # 5 s next to it, nothing detected
            b.update(np.zeros(2), 10.0, [])
        assert (b.tracks[0].r < 0.1) == expect_gone
    # a re-detection restores it
    b.update(np.zeros(2), 10.0, [Detection(np.array([3.1, 0.0]), Priority.URGENT, 0)])
    assert b.tracks[0].r == 1.0
