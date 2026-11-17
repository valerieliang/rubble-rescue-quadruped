import numpy as np

from rescue.observer import goal_posterior, legibility_cost, path_cost, time_to_identify

travel = lambda a, b: float(np.linalg.norm(np.asarray(b) - np.asarray(a)))   # open ground, 1 m/s
GOALS = [np.array([20.0, 10.0]), np.array([20.0, -10.0])]


def test_uniform_before_any_motion():
    assert np.allclose(goal_posterior(np.zeros((1, 2)), GOALS, travel), [0.5, 0.5])


def test_symmetric_straight_path_is_ambiguous():
    prefix = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    assert np.allclose(goal_posterior(prefix, GOALS, travel), [0.5, 0.5], atol=1e-9)


def test_exaggerated_path_is_legible():
    prefix = np.array([[0.0, 0.0], [5.0, 3.0], [10.0, 6.0]])
    assert goal_posterior(prefix, GOALS, travel)[0] > 0.9


def test_higher_rationality_sharpens_posterior():
    prefix = np.array([[0.0, 0.0], [4.0, 2.0]])
    assert (goal_posterior(prefix, GOALS, travel, beta=2.0)[0]
            > goal_posterior(prefix, GOALS, travel, beta=0.1)[0])


def test_time_to_identify_and_cost():
    straight = np.array([[t, t * 1.0, 0.0] for t in range(0, 11)], float)   # t, x, y=0
    assert time_to_identify(straight, GOALS, 0, travel, threshold=0.8) == float("inf")
    bent = np.array([[k, 2.0 * k, 1.2 * k] for k in range(0, 8)], float)
    t = time_to_identify(bent, GOALS, 0, travel, threshold=0.8)
    assert 0 < t < 8
    assert legibility_cost(bent, travel) == 1.0
    detour = np.array([[0, 0, 0], [1, 5, 5], [2, 10, 0]], float)
    assert legibility_cost(detour, travel) > 1.0
    assert path_cost(np.array([[0.0, 0.0], [3.0, 4.0]]), travel) == 5.0
