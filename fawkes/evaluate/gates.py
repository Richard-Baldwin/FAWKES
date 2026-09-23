"""The quantitative gates (README section 5): what "perfect" means, operationally.

Every gate cites a limit (the house acceptance number) and a stretch target.
Evidence is a run directory, never a claim; simulation acceptance is never
physical authorisation.
"""

from __future__ import annotations

import numpy as np

# gate: (limit, stretch target) - limits are the house acceptance numbers
# (endpoint 35 mm / cross-track 120 mm / heading 0.08 rad from the 2026-07-23
# standard; saturation 8 % from rlearn's acceptance; coupled heading RMS 0.12
# rad from rlearn's acceptance).
GATES = {
    "endpoint_p95_m": (0.035, 0.010),
    "cross_track_p95_m": (0.120, 0.060),
    "heading_max_rad": (0.08, 0.05),
    "saturation_frac": (0.08, 0.04),
    "boundary_violations": (0, 0),
    "coupled_heading_rms_rad": (0.12, 0.08),
}


def summarize_metrics(env_metrics: list[dict]) -> dict:
    """Flatten per-row/per-leg metrics into a summary with percentiles."""
    def all_of(key):
        return np.array([v for row in env_metrics for v in row[key]], dtype=float)

    endpoint = all_of("endpoint")
    cross = all_of("cross_rms")
    heading = all_of("heading_max")
    sat = all_of("sat_frac")
    boundary = all_of("boundary")
    return {
        "legs": int(len(endpoint)),
        "endpoint_p95_m": round(float(np.percentile(endpoint, 95)) if len(endpoint) else float("inf"), 5),
        "endpoint_max_m": round(float(endpoint.max()) if len(endpoint) else float("inf"), 5),
        "cross_track_p95_m": round(float(np.percentile(cross, 95)) if len(cross) else float("inf"), 5),
        "cross_track_max_m": round(float(cross.max()) if len(cross) else float("inf"), 5),
        "heading_max_rad": round(float(heading.max()) if len(heading) else 0.0, 5),
        "saturation_frac": round(float(np.mean(sat)) if len(sat) else 0.0, 5),
        "boundary_violations": int(np.sum(boundary)) if len(boundary) else 0,
    }


def check_gates(summary: dict) -> dict:
    out = {}
    for gate, (limit, stretch) in GATES.items():
        value = summary.get(gate)
        if value is None:
            continue
        out[gate] = {
            "value": value,
            "limit": limit,
            "stretch": stretch,
            "pass": bool(value <= limit),
            "meets_stretch": bool(value <= stretch),
        }
    out["all_pass"] = all(g["pass"] for g in out.values() if isinstance(g, dict) and "pass" in g)
    return out


def time_to_adapt(z_err_history: np.ndarray, threshold: float = 0.15) -> float:
    """First time (s) after which the mean |z_hat - z_true| stays below
    threshold to the end of the episode (seconds; -1 = never).

    Accepts (ticks, dims) or an already-averaged (ticks,) series.
    """
    err = np.asarray(z_err_history)
    mean_err = err.mean(axis=1) if err.ndim == 2 else err
    T = len(mean_err)
    if T == 0:
        return -1.0
    ok_from = -1
    for t in range(T):
        if np.all(mean_err[t:] <= threshold):
            ok_from = t
            break
    return -1.0 if ok_from < 0 else ok_from * 0.02
