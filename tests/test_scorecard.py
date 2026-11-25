"""Metrics shared by the safety experiments."""
import numpy as np

from rescue.scorecard import cost_per_violation_avoided, paired_diff, speed_for_zero, summarize, wilson


def test_wilson_interval_contains_the_rate_and_zero_of_n_is_not_zero_risk():
    lo, hi = wilson(0, 60)
    assert lo == 0.0 and 0.05 < hi < 0.07                    # 0 of 60: "below about 6%", not "zero"
    lo, hi = wilson(30, 60)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 0.01


def test_paired_diff_cancels_the_site_effect():
    base = np.array([0.2, 0.9, 0.5, 0.7])
    d, hw = paired_diff(base, base + 0.1)
    assert abs(d - 0.1) < 1e-12 and hw < 1e-9                # shared site spread does not widen the interval


def _row(viol_steps, clear, score, rescue=1.0):
    return {"any_clear_viol": int(viol_steps > 0), "clear_viol_steps": viol_steps, "min_clearance": clear,
            "norm_class": score, "rescue_rate": rescue}


def test_summarize_reports_rate_exposure_slack_and_score():
    s = summarize([_row(0, 2.0, 0.8), _row(5, 0.6, 0.6), _row(0, 1.5, 0.7), _row(0, 3.0, 0.9)])
    assert s["viol"] == 1 and s["n"] == 4 and abs(s["viol_rate"] - 0.25) < 1e-12
    assert abs(s["viol_s"] - 0.125) < 1e-12                  # 5 steps of 0.1 s over 4 runs
    assert s["clear_min"] == 0.6 and s["slack"] == s["clear_p5"] - 1.0
    assert abs(s["score"] - 0.75) < 1e-12


def test_cost_per_violation_and_speed_for_zero():
    a = {"viol_rate": 0.5, "score": 0.8}
    b = {"viol_rate": 0.1, "score": 0.7}
    assert abs(cost_per_violation_avoided(a, b) - 0.1 / 40) < 1e-12
    assert np.isnan(cost_per_violation_avoided(b, a))        # nothing removed going from the safer to the riskier
    assert speed_for_zero({0.5: 0, 0.75: 0, 1.0: 3}) == 0.75
    assert speed_for_zero({0.5: 2, 0.75: 0}) == 0.0
