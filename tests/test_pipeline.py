import numpy as np

from rescue import baselines
from rescue.belief import Belief
from rescue.drone_prior import PriorConfig
from rescue.interfaces import Detection, Observation, Priority, RunLog, Site, Survivor
from rescue.metrics import oracle_score, rescue_rate, rescue_score, summarize, time_to_critical
from rescue.search import Coverage, PriorSearch
from rescue.terrain import travel_time
from rescue.toy_sim import generate_site, run_episode

CFG = PriorConfig()


def _open_site(survivors):
    return Site(0, (0, 0, 100, 100), (), tuple(survivors), np.array([0.0, 0.0]))


def test_generate_site_is_seeded():
    a, b = generate_site(3), generate_site(3)
    assert [s.priority for s in a.survivors] == [s.priority for s in b.survivors]
    assert 3 <= len(a.survivors) <= 8


def test_travel_time_open_ground_and_rubble():
    site = _open_site([])
    assert abs(travel_time(site, np.zeros(2), np.array([10.0, 0.0])) - 10.0) < 1e-6
    rubble = generate_site(0)   # has slow regions, so never faster than open ground
    assert travel_time(rubble, np.zeros(2), np.array([50.0, 50.0])) >= np.hypot(50, 50) - 1e-6


def test_unmatched_detection_creates_confirmed_track():
    b = Belief([], CFG)
    b.update(np.zeros(2), 5.0, [Detection(np.array([1.0, 1.0]), Priority.URGENT)])
    assert len(b.tracks) == 1 and b.tracks[0].confirmed


def test_remove_near():
    b = Belief([Detection(np.array([10.0, 10.0]), Priority.STABLE)], CFG)
    b.remove_near(np.array([10.5, 10.0]))
    assert not b.tracks


def test_priority_prefers_critical_when_roughly_equidistant():
    dets = [Detection(np.array([20.0, 0.0]), Priority.STABLE),
            Detection(np.array([0.0, 22.0]), Priority.CRITICAL)]
    site = _open_site([])
    mk = lambda mode: PriorSearch(dets, CFG, site.bounds, lambda a, b: travel_time(site, a, b),
                                  5.0, site.start, 600.0, mode)
    obs = Observation(0.0, np.zeros(2), 5.0)
    assert np.allclose(mk("priority").act(obs), [0.0, 22.0])
    assert np.allclose(mk("nearest").act(obs), [20.0, 0.0])


def test_coverage_waypoints_inside_bounds():
    c = Coverage((0, 0, 100, 100), 5.0, np.zeros(2))
    assert (c.wps >= 0).all() and (c.wps <= 100).all()


def test_oracle_and_metrics():
    site = _open_site([Survivor(0, np.array([10.0, 0.0]), Priority.CRITICAL),
                       Survivor(1, np.array([30.0, 0.0]), Priority.STABLE)])
    T = 600.0
    log = RunLog(0, "x", rescue_times={0: 10.0})
    assert rescue_rate(site, log, T) == 0.5
    assert time_to_critical(site, log, T) == 10.0
    assert rescue_score(site, log, T) < oracle_score(site, T)
    assert summarize(site, log, T)["norm_score"] < 1.0


def test_episode_with_perfect_prior_rescues_everyone():
    site = generate_site(1)
    cfg = PriorConfig(sigma_pos=0.5, p_miss=0.0, n_false_pos=0.0)
    log = run_episode(site, lambda s, d: baselines.priority_policy(s, d, cfg), cfg, 1, T=900.0)
    assert len(log.rescue_times) == len(site.survivors)


def test_prior_beats_no_prior_on_average():
    scores = {"cov": [], "nn": []}
    for seed in range(6):
        site = generate_site(seed)
        for k, f in (("cov", baselines.coverage_policy), ("nn", baselines.nearest_first_policy)):
            log = run_episode(site, lambda s, d, f=f: f(s, d, CFG), CFG, seed)
            scores[k].append(rescue_score(site, log, 600.0))
    assert np.mean(scores["nn"]) > np.mean(scores["cov"])
