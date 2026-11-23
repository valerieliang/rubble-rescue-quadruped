import numpy as np

from rescue.drone_prior import PriorConfig
from rescue.interfaces import Detection, Observation, Priority
from rescue.search import ObservedMap, PriorSearch
from rescue.terrain import travel_time
from rescue.interfaces import Site

CFG = PriorConfig()
SITE = Site(0, (0, 0, 100, 100), (), (), np.array([0.0, 0.0]))


def _search(dets, retention, sigma=6.0):
    cfg = PriorConfig(sigma_pos=sigma)
    return PriorSearch(dets, cfg, SITE.bounds, lambda a, b: travel_time(SITE, a, b), 5.0, SITE.start, 600.0,
                       "priority", retention=retention)


def _obs(t, pos):
    return Observation(t, np.asarray(pos, float), 5.0)


def test_observed_map_marks_and_measures():
    m = ObservedMap((0, 0, 100, 100))
    assert m.frac_seen([50, 50], 5) == 0.0
    m.mark(np.array([50.0, 50.0]), 5.0)
    assert m.frac_seen([50, 50], 4) == 1.0
    assert 0.0 < m.frac_seen([50, 50], 10) < 1.0
    q = m.nearest_unseen([50, 50], 10.0, np.array([80.0, 50.0]))
    assert q is not None and np.linalg.norm(q - [50, 50]) > 4.5 and q[0] > 50


def test_default_has_no_observed_map():
    s = _search([Detection(np.array([40.0, 40.0]), Priority.URGENT)], "")
    assert s.obs_map is None and not s.flags


def test_track_dropped_without_keep_but_retained_with_keep():
    """A reported survivor 6 m from the report is missed by a robot that stops 6.3 m short: without 'keep' the
    track fades below r_min; with it the track stays until the 2-sigma region has been searched."""
    det = [Detection(np.array([40.0, 40.0]), Priority.URGENT)]
    pos = np.array([40.0, 33.7])      # 6.3 m from the reported mean, outside the 5 m sensor
    plain, keep = _search(det, "resume"), _search(det, "keep")
    for s in (plain, keep):
        for t in range(40):
            s.act(_obs(float(t), pos))
    assert plain.belief.tracks[0].r < plain.r_min
    assert keep.belief.tracks[0].r >= keep.r_min


def test_keep_searches_region_not_just_mean_and_then_lets_go():
    det = [Detection(np.array([40.0, 40.0]), Priority.URGENT)]
    s = _search(det, "keep", sigma=3.0)
    wp = s.act(_obs(0.0, [40.0, 40.0]))   # robot is at the mean: the mean is seen, the 6 m region is not
    assert np.linalg.norm(wp - [40, 40]) > 4.0      # heads for an unobserved part of the region
    for t, p in enumerate([[34, 40], [46, 40], [40, 34], [40, 46], [36, 36], [44, 44], [36, 44], [44, 36]]):
        s.act(_obs(float(t + 1), p))
    s.act(_obs(20.0, [40.0, 40.0]))
    assert s.belief.tracks[0].r < s.r_min            # region searched, nothing found: the track is released


def test_confirmed_track_is_never_held_by_keep():
    s = _search([], "keep")
    s.belief.update(np.array([50.0, 50.0]), 5.0, [Detection(np.array([50.0, 50.0]), Priority.URGENT)])
    assert s.belief.tracks[0].confirmed
    s._retain()
    assert s.belief.tracks[0].r == 1.0


def test_region_fallback_goes_to_unseen_area():
    s = _search([], "region")
    wp = s.act(_obs(0.0, [50.0, 50.0]))
    assert wp is not None
    assert not s.obs_map.seen[int(wp[0]), int(wp[1])]
    s2 = _search([], "resume")
    wp2 = s2.act(_obs(0.0, [50.0, 50.0]))
    assert wp2.shape == (2,)
