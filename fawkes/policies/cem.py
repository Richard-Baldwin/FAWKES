"""Cross-entropy method over bounded parameters (rlearn's search, generalised).

Common random numbers: the caller passes an objective that evaluates a vector
against a FIXED set of episode seeds, so every candidate in a generation sees
identical worlds (fair comparison, low-variance ranking).
"""

from __future__ import annotations

import numpy as np


def cem(
    objective,
    bounds: np.ndarray,
    seed: int = 0,
    population: int = 32,
    elites: int = 8,
    iterations: int = 12,
    init: np.ndarray | None = None,
    sigma_frac: float = 0.25,
    decay: float = 0.92,
    verbose: bool = False,
):
    """Maximise objective(vec) over box bounds (D, 2). Returns (best_vec, best_score, history)."""
    rng = np.random.default_rng(seed)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    mu = (lo + hi) / 2.0 if init is None else np.clip(np.asarray(init, dtype=float), lo, hi)
    sigma = (hi - lo) * sigma_frac
    best_vec, best = mu.copy(), -np.inf
    history = []
    for it in range(iterations):
        cand = rng.normal(mu, sigma, size=(population, D))
        cand = np.clip(cand, lo, hi)
        cand[0] = mu  # always evaluate the incumbent
        scores = np.array([objective(c) for c in cand])
        order = np.argsort(-scores)
        if scores[order[0]] > best:
            best = scores[order[0]]
            best_vec = cand[order[0]].copy()
        elite = cand[order[:elites]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), (hi - lo) * 1e-3) * decay + sigma * (1 - decay) * 0.5
        history.append({"iteration": it, "best": float(best), "mean_elite": float(elite.mean(axis=0).sum() * 0 + scores[order[:elites]].mean())})
        if verbose:
            print(f"cem it={it} best={best:.4f}")
    return np.clip(best_vec, lo, hi), float(best), history
