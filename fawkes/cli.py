"""FAWKES command line. Every command writes its artifacts under evidence/<tag>/.

Nothing here ever contacts a robot, opens a socket, or deploys anything:
deployment_authorized stays false until a human flips it through the existing
ladder (README L5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fawkes import __version__
from fawkes.envs.cascade import BASELINE_KNOBS, SEARCH_SPACE, KNOB_NAMES, knobs_to_vec, vec_to_knobs
from fawkes.envs.env import MovementEnv, OBS_NAMES, ADAPTER_FEATURE_NAMES
from fawkes.envs.surface import DRFamily, Z_NAMES
from fawkes.policies.cem import cem
from fawkes.policies.rma import BasePolicy, LinearAdapter, save_adapter
from fawkes.policies.train_rma import phase_a, phase_b
from fawkes.evaluate.blind_transfer import run_suite, HELD_OUT_SEEDS
from fawkes.evaluate.gates import summarize_metrics, check_gates
from fawkes.evaluate.report import write_report, dump_json, gates_table, provenance_block
from fawkes.export.firmware_profile import registry_entry, write_registry_entry
from fawkes.export.residual_actor import build_actor, write_actor
from fawkes.export.phoenix_bundle import build_bundle, write_bundle, verify_bundle

TRAIN_SEEDS = (11, 22, 33, 44)
PRACTICE_SEEDS = tuple(range(9101, 9109))  # held-out, on the practice family


def _evidence(tag: str) -> Path:
    from fawkes import paths as p

    out = p.EVIDENCE / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------ commands
def cmd_audit(args) -> None:
    from fawkes.trace.audit import write_audit

    out = write_audit(tag="audit")
    print(f"audit -> {out}")


def cmd_identify(args) -> None:
    from fawkes.sysid.identify import run

    card = run(out_dir=_evidence(args.tag) / "sysid")
    (Path(_evidence(args.tag)) / "sysid_summary.json").write_text(
        json.dumps(card["validation"], indent=2), encoding="utf-8"
    )
    print(json.dumps(card["validation"], indent=2))


def _cem_objective(env: MovementEnv, seeds):
    def objective(vec):
        knobs = np.tile(vec, (8, 1))
        total = 0.0
        for s in seeds:
            env.reset(8, s)
            env.cascade.set_knobs_per_row(knobs)
            for _ in range(600):
                _, _, done, _ = env.step(None)
                if np.all(done):
                    break
            total += float(np.mean(env.cost))
        return -total / len(seeds)

    return objective


def cmd_cem(args) -> None:
    out = _evidence(args.tag)
    env = MovementEnv(family=DRFamily(), legs=2)
    bounds = np.array([SEARCH_SPACE[k] for k in KNOB_NAMES])
    best, score, hist = cem(
        _cem_objective(env, TRAIN_SEEDS),
        bounds,
        seed=args.seed,
        population=args.pop,
        iterations=args.iterations,
        init=knobs_to_vec(BASELINE_KNOBS),
    )
    knobs = vec_to_knobs(best)
    # held-out evaluation (seeds never trained on)
    env.reset(8, HELD_OUT_SEEDS[0])
    env.cascade.set_knobs_per_row(np.tile(best, (8, 1)))
    for _ in range(600):
        _, _, done, _ = env.step(None)
        if np.all(done):
            break
    summary = summarize_metrics(env.metrics)
    champion = {
        "kind": "fawkes-cem-champion-1",
        "knobs": knobs,
        "baseline": BASELINE_KNOBS,
        "training": {
            "algorithm": "cross-entropy method over the 11 interpretable gains",
            "seeds": list(TRAIN_SEEDS),
            "population": args.pop,
            "iterations": args.iterations,
            "seed": args.seed,
            "dr_family": "dr-family-1",
        },
        "held_out_summary": summary,
        "held_out_gates": check_gates(summary),
        "best_objective": score,
        "deployment_authorized": False,
    }
    dump_json(out / "cem_champion.json", champion)
    print(f"cem champion score={score:.3f} -> {out / 'cem_champion.json'}")


def cmd_rma(args) -> None:
    out = _evidence(args.tag)
    env = MovementEnv(family=DRFamily(), legs=2)
    base = BasePolicy()
    a = phase_a(env, base, seeds=TRAIN_SEEDS, population=args.pop, iterations=args.iterations, seed=args.seed)
    adapter, b = phase_b(env, base, seeds=tuple(range(100, 100 + args.b_seeds)))
    dump_json(out / "rma_base.json", {"params": [float(v) for v in base.params], "sizes": list(base.net.sizes), "phase_a": {"best_score": a["best_score"], "history": a["history"]}})
    save_adapter(out / "rma_adapter.json", adapter, Z_NAMES)
    dump_json(out / "rma_probe.json", b)
    print(f"rma phase_a={a['best_score']:.3f} probe_pairs={b['pairs']} -> {out}")


def _run_conditions(env, seeds, out):
    conditions = {}
    conditions["baseline_knobs"] = run_suite(env, seeds=seeds, knobs=np.tile(knobs_to_vec(BASELINE_KNOBS), (8, 1)), z_mode="none")
    cem_path = out / "cem_champion.json"
    if cem_path.exists():
        champ = json.loads(cem_path.read_text(encoding="utf-8"))
        conditions["cem_knobs"] = run_suite(env, seeds=seeds, knobs=np.tile(knobs_to_vec(champ["knobs"]), (8, 1)), z_mode="none")
    rma_path = out / "rma_base.json"
    if rma_path.exists():
        from fawkes.policies.rma import load_adapter

        base = BasePolicy()
        base.set_params(np.array(json.loads(rma_path.read_text(encoding="utf-8"))["params"]))
        adapter = load_adapter(out / "rma_adapter.json")
        conditions["rma_oracle"] = run_suite(env, seeds=seeds, base=base, z_mode="oracle")
        conditions["rma_adapter"] = run_suite(env, seeds=seeds, base=base, adapter=adapter)
        conditions["rma_zero"] = run_suite(env, seeds=seeds, base=base, z_mode="zero")
    return conditions


def cmd_evaluate(args) -> None:
    out = _evidence(args.tag)
    practice = _run_conditions(MovementEnv(family=DRFamily.practice(), legs=2), PRACTICE_SEEDS, out)
    dr = _run_conditions(MovementEnv(family=DRFamily(), legs=2), HELD_OUT_SEEDS, out)
    dump_json(out / "evaluation.json", {"practice": practice, "dr_family": dr})
    for suite, conds in (("practice", practice), ("dr-family", dr)):
        for name, res in conds.items():
            s = res["summary"]
            g = res["gates"]
            print(f"{suite:9s} {name:14s} endpoint_p95={s['endpoint_p95_m']:.4f} cross_p95={s['cross_track_p95_m']:.4f} pass={g['all_pass']}")


def cmd_export(args) -> None:
    out = _evidence(args.tag)
    env = MovementEnv(family=DRFamily(), legs=2)
    base = BasePolicy()
    rma_base = json.loads((out / "rma_base.json").read_text(encoding="utf-8"))
    base.set_params(np.array(rma_base["params"]))
    from fawkes.policies.rma import load_adapter

    adapter = load_adapter(out / "rma_adapter.json")
    provenance = {
        "framework": "fawkes",
        "version": __version__,
        "phase_a_score": rma_base["phase_a"]["best_score"],
        "dr_family": "dr-family-1",
    }
    champion = json.loads((out / "cem_champion.json").read_text(encoding="utf-8"))
    write_registry_entry(
        out / "firmware_registry_entry.json",
        registry_entry(champion["knobs"], champion["held_out_summary"], profile_id=f"fawkes-cem-{args.tag}", provenance=champion["training"]),
    )
    write_actor(out / "residual_actor.json", build_actor(base, adapter, OBS_NAMES, ADAPTER_FEATURE_NAMES, Z_NAMES, provenance))
    bundle = build_bundle(base, adapter, OBS_NAMES, ADAPTER_FEATURE_NAMES, Z_NAMES, provenance)
    write_bundle(out / "phoenix_bundle_movement-rma-1.json", bundle)
    verification = verify_bundle(out / "phoenix_bundle_movement-rma-1.json")
    print(f"export -> {out}")
    print(f"bundle verify: {verification}")


def cmd_report(args) -> None:
    out = _evidence(args.tag)
    sections = []
    ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    suites = ev if "dr_family" not in ev else {"practice": ev["practice"], "dr family (held-out)": ev["dr_family"]}
    for suite_name, conds in suites.items():
        rows = [
            "| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, res in conds.items():
            s = res["summary"]
            g = res["gates"]
            rows.append(f"| {name} | {s['endpoint_p95_m']:.4f} | {s['cross_track_p95_m']:.4f} | {s['heading_max_rad']:.4f} | {s['saturation_frac']:.4f} | {s['boundary_violations']} | {'PASS' if g['all_pass'] else 'FAIL'} |")
        sections.append((f"Suite: {suite_name}", "\n".join(rows)))
    first = next(iter(next(iter(suites.values())).values()))
    sections.append(("Gates (the FAILing condition shown for context)", gates_table(first["gates"])))
    sections.append(("Provenance", provenance_block({"tag": args.tag})))
    sections.append(("Honest caveats", (
        "- The champion is a hypothesis until the guarded A/B on the field; the\n"
        "  simulator-vs-real gap warning (203 vs 104 mm cross-track) is standing.\n"
        "- The v0.1 adapter is a closed-form linear head over engineered window\n"
        "  features; the GRU lands with the PPO gradient stack.\n"
        "- Pure-Python env backend at 50 Hz; the MJX/Brax backend is FK-3.\n"
        "- Nothing here touched a robot. deployment_authorized stays false."
    )))
    write_report(out / "LATEST.md", f"FAWKES run {args.tag}", sections)
    print(f"report -> {out / 'LATEST.md'}")


def cmd_verify(args) -> None:
    from fawkes.export.phoenix_bundle import verify_bundle

    print(json.dumps(verify_bundle(args.bundle), indent=2))


def cmd_dashboard(args) -> None:
    from fawkes.viz.dashboard import build_dashboard

    out = build_dashboard(args.tag)
    print(f"dashboard -> {out}")


def cmd_arena(args) -> None:
    from fawkes.viz.arena import build_arena

    out = build_arena(args.tag, practice_seed=args.practice_seed, dr_seed=args.dr_seed)
    print(f"arena -> {out}")


def cmd_run_smoke(args) -> None:
    cmd_audit(args)
    cmd_identify(args)
    args.pop, args.iterations = 12, 6
    cmd_cem(args)
    args.pop, args.iterations, args.b_seeds = 12, 5, 10
    cmd_rma(args)
    cmd_evaluate(args)
    cmd_export(args)
    cmd_report(args)


def cmd_flywheel(args) -> None:
    """One full loop: audit -> identify -> train -> evaluate -> export -> report,
    plus a promotion verdict. Fully offline; nothing touches a robot. The human
    decision is only ever "install the staged candidate or not"."""
    from datetime import datetime, timezone

    cmd_audit(args)
    cmd_identify(args)
    args.pop, args.iterations = args.cem_pop, args.cem_iters
    cmd_cem(args)
    args.pop, args.iterations, args.b_seeds = args.rma_pop, args.rma_iters, args.rma_b_seeds
    cmd_rma(args)
    cmd_evaluate(args)
    cmd_export(args)
    cmd_report(args)

    out = _evidence(args.tag)
    ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    practice = ev["practice"]
    base_s = practice["baseline_knobs"]["summary"]
    cem_s = practice["cem_knobs"]["summary"]
    cem_g = practice["cem_knobs"]["gates"]
    verdict = {
        "tag": args.tag,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": "cem_knobs",
        "practice": {
            "baseline_endpoint_p95_m": base_s["endpoint_p95_m"],
            "candidate_endpoint_p95_m": cem_s["endpoint_p95_m"],
            "baseline_cross_p95_m": base_s["cross_track_p95_m"],
            "candidate_cross_p95_m": cem_s["cross_track_p95_m"],
            "all_gates_pass": cem_g["all_pass"],
        },
        "promote": bool(
            cem_g["all_pass"]
            and cem_s["endpoint_p95_m"] <= base_s["endpoint_p95_m"]
            and cem_s["cross_track_p95_m"] <= base_s["cross_track_p95_m"]
        ),
        "staged_artifact": str(out / "firmware_registry_entry.json"),
        "rule": (
            "promote=true means the candidate beats the field-proven baseline on the "
            "practice suite and passes every gate. It is still SIMULATION ONLY and "
            "deployment_authorized=false: a human runs the guarded A/B (FIELD_DAY.md) "
            "before it touches a robot."
        ),
    }
    dump_json(out / "verdict.json", verdict)
    print(f"verdict: promote={verdict['promote']} -> {out / 'verdict.json'}")


def cmd_train(args) -> None:
    cmd_audit(args)
    args.pop, args.iterations = args.cem_pop, args.cem_iters
    cmd_cem(args)
    args.pop, args.iterations, args.b_seeds = args.rma_pop, args.rma_iters, args.rma_b_seeds
    cmd_rma(args)
    cmd_evaluate(args)
    cmd_export(args)
    cmd_report(args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fawkes", description="FAWKES movement RL framework")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(func=fn)
        return p

    add("audit", cmd_audit, help="audit the real data root against the evidence pack numbers")
    add("identify", cmd_identify, help="fit the wheel ensemble on real transitions")
    p = add("cem", cmd_cem, help="CEM search over the 11 interpretable gains")
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", type=str, default="runs")
    p = add("rma", cmd_rma, help="RMA phase A (privileged CEM) + phase B (adapter fit)")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--b-seeds", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", type=str, default="runs")
    p = add("evaluate", cmd_evaluate, help="gates over held-out seeds for every condition")
    p.add_argument("--tag", type=str, default="runs")
    p = add("export", cmd_export, help="write the export trio + verify the bundle")
    p.add_argument("--tag", type=str, default="runs")
    p = add("report", cmd_report, help="markdown report, house style")
    p.add_argument("--tag", type=str, default="runs")
    p = add("verify", cmd_verify, help="replay a bundle's verification pairs")
    p.add_argument("bundle")
    p = add("train", cmd_train, help="real-budget training: cem -> rma -> evaluate -> export -> report")
    p.add_argument("--tag", type=str, default="train")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cem-pop", type=int, default=20)
    p.add_argument("--cem-iters", type=int, default=24)
    p.add_argument("--rma-pop", type=int, default=16)
    p.add_argument("--rma-iters", type=int, default=16)
    p.add_argument("--rma-b-seeds", type=int, default=16)
    p = add("run-smoke", cmd_run_smoke, help="audit -> identify -> cem -> rma -> evaluate -> export -> report")
    p.add_argument("--tag", type=str, default="runs")
    p.add_argument("--seed", type=int, default=7)
    p = add("flywheel", cmd_flywheel, help="one full loop: audit -> identify -> train -> evaluate -> export -> report -> verdict")
    p.add_argument("--tag", type=str, default="flywheel")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cem-pop", type=int, default=20)
    p.add_argument("--cem-iters", type=int, default=24)
    p.add_argument("--rma-pop", type=int, default=16)
    p.add_argument("--rma-iters", type=int, default=16)
    p.add_argument("--rma-b-seeds", type=int, default=16)
    p = add("dashboard", cmd_dashboard, help="render evidence/<tag>/dashboard.png from a run's artifacts")
    p.add_argument("--tag", type=str, default="runs")
    p = add("arena", cmd_arena, help="animated side-by-side simulator: evidence/<tag>/arena.html")
    p.add_argument("--tag", type=str, default="runs")
    p.add_argument("--practice-seed", type=int, default=9101)
    p.add_argument("--dr-seed", type=int, default=9003)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
