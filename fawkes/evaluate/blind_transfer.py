"""The blind-transfer suite: train with the adapter, evaluate with the truth
hidden - and report the blind window honestly alongside adapted performance.

Conditions: oracle (privileged upper bound), adapter (the deployed mode - the
robot comes in blind), zero (the fallback when z carries no information),
none (pure cascade, no residual policy at all).
"""

from __future__ import annotations

import numpy as np

from fawkes.envs.env import MovementEnv
from fawkes.evaluate.gates import check_gates, summarize_metrics, time_to_adapt
from fawkes.policies.rma import BasePolicy, LinearAdapter

HELD_OUT_SEEDS = (9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008)


def _episode(env: MovementEnv, n: int, seed: int, base: BasePolicy | None,
             z_mode: str, adapter: LinearAdapter | None, knobs=None):
    obs = env.reset(n, seed)
    if knobs is not None:
        env.cascade.set_knobs_per_row(knobs)
    z_err = []
    for _ in range(600):
        actions = None
        if base is not None:
            if z_mode == "oracle":
                z = env.z()
            elif z_mode == "adapter":
                z = adapter.predict(env.history_features())
                z_err.append(np.abs(z - env.z()).mean())
            else:  # zero
                z = np.zeros((n, 6))
            actions = base.act(obs, z)
        _, _, done, _ = env.step(actions)
        if np.all(done):
            break
    return env.metrics, (np.array(z_err) if z_err else None)


def run_suite(env: MovementEnv, seeds=HELD_OUT_SEEDS, base: BasePolicy | None = None,
              adapter: LinearAdapter | None = None, knobs=None, n: int = 8,
              z_mode: str | None = None) -> dict:
    """Evaluate one policy configuration over held-out seeds; returns summary + gates."""
    if z_mode is None:
        z_mode = "adapter" if adapter is not None else ("zero" if base is not None else "none")
    per_seed = []
    z_errs = []
    for s in seeds:
        metrics, z_err = _episode(env, n, s, base, z_mode, adapter, knobs)
        per_seed.extend(metrics)
        if z_err is not None:
            z_errs.append(z_err)
    summary = summarize_metrics(per_seed)
    result = {"summary": summary, "gates": check_gates(summary), "z_mode": z_mode}
    if z_errs:
        m = min(len(z) for z in z_errs)
        stacked = np.stack([z[:m] for z in z_errs])
        result["time_to_adapt_s"] = round(float(np.mean([time_to_adapt(z) for z in stacked])), 2)
    return result


def blind_transfer(env: MovementEnv, base: BasePolicy, adapter: LinearAdapter,
                   seeds=HELD_OUT_SEEDS) -> dict:
    """The three conditions: oracle (upper bound), adapter (deployed), zero (fallback)."""
    return {
        "oracle": run_suite(env, seeds=seeds, base=base, z_mode="oracle"),
        "adapter": run_suite(env, seeds=seeds, base=base, adapter=adapter),
        "zero": run_suite(env, seeds=seeds, base=base, z_mode="zero"),
    }
