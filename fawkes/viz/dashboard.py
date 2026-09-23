"""The results dashboard: one PNG per training run (matplotlib, Agg).

Panels: practice-world and DR-world trajectories (baseline vs CEM champion),
the gate bars (endpoint / cross-track p95 vs the house limits), the RMA Phase-A
convergence, the adaptation curve (mean |z_hat - z_true| over an episode, with
the 0.15 threshold and the 2 s target), and the z-probe R^2 per dimension.

Everything it draws is recomputed from the artifacts in evidence/<tag>/ plus
fresh held-out episodes - no numbers are trusted from memory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fawkes.envs.cascade import BASELINE_KNOBS, knobs_to_vec
from fawkes.envs.env import MovementEnv
from fawkes.envs.surface import DRFamily, Z_NAMES


def record_rollout(env, knobs=None, base=None, adapter=None, z_mode="none",
                   seed=9001, rows=4, max_ticks=600):
    """Run one batch episode and record per-row xy trajectories + targets."""
    env.reset(rows, seed)
    if knobs is not None:
        env.cascade.set_knobs_per_row(np.tile(knobs, (rows, 1)))
    trajs = [[] for _ in range(rows)]
    targets: list[list] = [[] for _ in range(rows)]
    for _ in range(max_ticks):
        obs = env.observe()
        if z_mode == "oracle":
            action = base.act(obs, env.z())
        elif z_mode == "adapter":
            action = base.act(obs, adapter.predict(env.history_features()))
        else:
            action = None
        _, _, done, _ = env.step(action)
        for i in range(rows):
            trajs[i].append((float(env.pos[i, 0]), float(env.pos[i, 1])))
            targets[i].append((float(env.target[i, 0]), float(env.target[i, 1])))
        if np.all(done):
            break
    return trajs, targets
def z_error_curve(env, base, adapter, seed=9001, rows=8, max_ticks=400):
    """Mean |z_hat - z_true| per tick, averaged over rows (one held-out episode)."""
    env.reset(rows, seed)
    curves = []
    for _ in range(max_ticks):
        obs = env.observe()
        z_hat = adapter.predict(env.history_features())
        err = np.abs(z_hat - env.z()).mean(axis=1)
        curves.append(err)
        _, _, done, _ = env.step(base.act(obs, z_hat))
        if np.all(done):
            break
    return np.array(curves) if curves else np.zeros((0,))


def _load(evidence_dir, name):
    p = Path(evidence_dir) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _plot_traj(ax, trajs, targets, colour, label, style="-"):
    for i, (t, tg) in enumerate(zip(trajs, targets)):
        xs = [p[0] for p in t]
        ys = [p[1] for p in t]
        ax.plot(xs, ys, style, color=colour, linewidth=1.6, alpha=0.85,
                label=label if i == 0 else None)
        ax.plot(xs[0], ys[0], "o", color=colour, markersize=4)
        ax.plot(tg[-1][0], tg[-1][1], "x", color=colour, markersize=8, mew=2)
def build_dashboard(tag: str, out_png: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fawkes import paths as P
    from fawkes.policies.rma import BasePolicy, load_adapter

    out_dir = P.EVIDENCE / tag
    ev = _load(out_dir, "evaluation.json")
    champ = _load(out_dir, "cem_champion.json")
    rma = _load(out_dir, "rma_base.json")
    probe = _load(out_dir, "rma_probe.json")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"FAWKES run {tag}", fontsize=15, fontweight="bold")

    # --- panels A/B: trajectories, practice and DR worlds ------------------
    if champ:
        knobs_cem = knobs_to_vec({k: float(v) for k, v in champ["knobs"].items()})
    else:
        knobs_cem = knobs_to_vec(BASELINE_KNOBS)
    for ax, family, seed, title in (
        (axes[0][0], DRFamily.practice(), 9101, "practice world (held-out seed 9101)"),
        (axes[0][1], DRFamily(), 9003, "DR world (held-out seed 9003)"),
    ):
        env = MovementEnv(family=family, legs=2)
        tb, tgb = record_rollout(env, knobs=knobs_to_vec(BASELINE_KNOBS), seed=seed)
        tc, tgc = record_rollout(env, knobs=knobs_cem, seed=seed)
        _plot_traj(ax, tb, tgb, "#888888", "field-proven baseline", "--")
        _plot_traj(ax, tc, tgc, "#1f77b4", "FAWKES CEM champion")
        ax.set_title(f"trajectories - {title}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    # --- panel C: gate bars -------------------------------------------------
    ax = axes[0][2]
    suite = ev["practice"]
    conds = list(suite.keys())
    x = np.arange(len(conds))
    eps = [suite[c]["summary"]["endpoint_p95_m"] * 1000 for c in conds]
    crs = [suite[c]["summary"]["cross_track_p95_m"] * 1000 for c in conds]
    ax.bar(x - 0.19, eps, 0.38, label="endpoint p95 [mm]", color="#1f77b4")
    ax.bar(x + 0.19, crs, 0.38, label="cross-track p95 [mm]", color="#ff7f0e")
    ax.axhline(35, color="#1f77b4", ls="--", lw=1, label="endpoint limit 35 mm")
    ax.axhline(120, color="#ff7f0e", ls="--", lw=1, label="cross-track limit 120 mm")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, fontsize=8)
    ax.set_title("practice-suite gates (lower is better)")
    ax.legend(fontsize=7)

    # --- panel D: RMA Phase-A convergence -----------------------------------
    ax = axes[1][0]
    if rma:
        hist = rma["phase_a"]["history"]
        its = [h["iteration"] for h in hist]
        best = [h["best"] for h in hist]
        elite = [h["mean_elite"] for h in hist]
        ax.plot(its, best, "-o", ms=3, label="best score")
        ax.plot(its, elite, "--", lw=1, label="elite mean")
        ax.set_xlabel("CEM iteration"); ax.set_ylabel("episode score")
        ax.set_title(f"RMA Phase A (base MLP, pop {len(hist) and ''}")
        ax.set_title("RMA Phase A convergence")
        ax.legend(fontsize=8)
    # --- panel E: adaptation curve ------------------------------------------
    ax = axes[1][1]
    if rma:
        base = BasePolicy()
        base.set_params(np.array(rma["params"]))
        adapter = load_adapter(out_dir / "rma_adapter.json")
        env = MovementEnv(family=DRFamily(), legs=2)
        for seed in (9001, 9005, 9009):
            curve = z_error_curve(env, base, adapter, seed=seed)
            if len(curve):
                t = np.arange(len(curve)) * 0.02
                ax.plot(t, curve.mean(axis=1) if curve.ndim > 1 else curve,
                        label=f"seed {seed}")
        ax.axhline(0.15, color="r", ls="--", lw=1, label="adapted threshold 0.15")
        ax.axvline(2.0, color="k", ls=":", lw=1, label="2 s target")
        ax.set_xlabel("time in episode [s]")
        ax.set_ylabel("mean |z_hat - z_true|")
        ax.set_title("blind adaptation (the hidden vector converging)")
        ax.legend(fontsize=7)

    # --- panel F: z-probe per dimension -------------------------------------
    ax = axes[1][2]
    if probe:
        dims = sorted(probe["probe"].keys())
        r2 = [probe["probe"][d]["r2"] for d in dims]
        names = [Z_NAMES[int(d.split("_")[1])] for d in dims]
        ax.bar(names, r2, color="#2ca02c")
        ax.set_title("z-probe R^2 per world dimension\n(what the adapter infers - honest)")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.set_ylim(0, 1)

    out = Path(out_png) if out_png else out_dir / "dashboard.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
# END 
