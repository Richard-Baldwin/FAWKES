"""Gate arithmetic tests."""

from __future__ import annotations

import numpy as np

from fawkes.evaluate.gates import check_gates, summarize_metrics, time_to_adapt


def _metrics(endpoint, cross, heading=0.01, sat=0.0, boundary=0):
    return [
        {
            "endpoint": list(endpoint),
            "cross_rms": list(cross),
            "heading_max": [heading] * len(endpoint),
            "sat_frac": [sat] * len(endpoint),
            "leg_time": [1.0] * len(endpoint),
            "boundary": [boundary] * len(endpoint),
            "settle_s": [1.0] * len(endpoint),
        }
    ]


def test_summarize_and_check_gates():
    s = summarize_metrics(_metrics([0.01, 0.02, 0.03], [0.05, 0.06, 0.07]))
    g = check_gates(s)
    assert s["endpoint_p95_m"] <= 0.035
    assert g["endpoint_p95_m"]["pass"] is True
    assert g["all_pass"] is True
    s_bad = summarize_metrics(_metrics([0.05], [0.05], boundary=1))
    g_bad = check_gates(s_bad)
    assert g_bad["boundary_violations"]["pass"] is False
    assert g_bad["all_pass"] is False


def test_time_to_adapt_detects_convergence():
    err = np.array([[0.4] * 40, [0.4] * 40, [0.05] * 40, [0.05] * 40])
    err = np.column_stack([np.concatenate([np.full(50, 0.4), np.full(50, 0.05)]) for _ in range(6)])
    t = time_to_adapt(err)
    assert 0.9 < t < 1.3
    never = np.full((200, 6), 0.4)
    assert time_to_adapt(never) == -1.0
