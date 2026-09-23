"""RMA training: Phase A (privileged CEM over the base policy) and Phase B
(closed-form adapter fit + probe). And the blind evaluation helpers."""

from __future__ import annotations

import numpy as np

from fawkes.envs.env import MovementEnv
from fawkes.policies.cem import cem
from fawkes.policies.rma import BasePolicy, LinearAdapter

TRAIN_SEEDS = (11, 22, 33, 44)


def phase_a(
    env: MovementEnv,
    base: BasePolicy,
    seeds=TRAIN_SEEDS,
    population: int = 24,
    iterations: int = 20,
    sigma_frac: float = 0.15,
    seed: int = 7,
):
    """CEM over base-policy weights, privileged z (the true conditioning)."""
    lo = np.full(base.net.n_params, -3.0)
    hi = np.full(base.net.n_params, 3.0)
    init = base.params

    def objective(vec):
        base.set_params(vec)
        total = 0.0
        for s in seeds:
            cost, _ = _run(env, 8, s, base=base, z_mode="oracle")
            total += float(np.mean(cost))
        return -total / len(seeds)

    best_vec, best_score, hist = cem(
        objective, np.column_stack([lo, hi]), seed=seed,
        population=population, iterations=iterations, init=init,
        sigma_frac=sigma_frac, decay=0.9,
    )
    base.set_params(best_vec)
    return {"best_score": float(best_score), "history": hist, "params": best_vec}


def phase_b(env: MovementEnv, base: BasePolicy, seeds=tuple(range(100, 132)), ridge: float = 1e-2):
    """Collect (history features, true z) pairs with the frozen base policy on
    fresh DR seeds, then fit the linear adapter in closed form. Returns the
    adapter and its probe report."""
    adapter = LinearAdapter()
    feats, zs = [], []
    for s in seeds:
        env.reset(8, s)
        for _ in range(600):
            obs = env.observe()
            z_true = env.z()
            # roll forward with the frozen base using ORACLE z (Phase A behaviour)
            acts = base.act(obs, z_true)
            f = env.history_features()
            if env.tick > 25:  # after the history window fills: the inference task
                feats.append(f.copy())
                zs.append(z_true.copy())
            _, _, done, _ = env.step(acts)
            if np.all(done):
                break
    F = np.concatenate(feats)
    Z = np.concatenate(zs)
    probe = adapter.fit(F, Z, ridge=ridge)
    return adapter, {"pairs": int(len(F)), "probe": probe}


def _run(env: MovementEnv, n: int, seed: int, base: BasePolicy, z_mode: str, adapter: LinearAdapter | None = None):
    env.reset(n, seed)
    for _ in range(600):
        obs = env.observe()
        if z_mode == "oracle":
            z = env.z()
        elif z_mode == "adapter":
            z = adapter.predict(env.history_features())
        else:
            z = np.zeros((n, 6))
        _, _, done, _ = env.step(base.act(obs, z))
        if np.all(done):
            break
    return env.cost, env.metrics
