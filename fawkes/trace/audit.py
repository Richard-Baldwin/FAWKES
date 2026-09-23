"""FK-0/FK-1: audit the real data root against the evidence pack's own numbers.

The evidence pack (2026-07-23) states: 162 CSV traces, 67,593 rows, 4 schemas,
12 onboard logs (17,061,241 bytes). This audit re-counts from the files
themselves, hashes the load-bearing artifacts, and records what is present vs
absent in this copy of the data.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from fawkes import paths
from fawkes.trace.converters import detect_csv_schema, load_rlearn_transitions
from fawkes.trace.manifest import sha256_file

# The evidence pack's own numbers (2026-07-23_TRACE_STATISTICS.md, "Totals").
EXPECTED = {
    "csv_traces": 162,
    "csv_rows": 67593,
    "csv_schemas": 4,
    "onboard_logs": 12,
    "onboard_log_bytes": 17061241,
}


def audit_data_root(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else paths.data_root()
    report: dict = {"schema": 1, "data_root": str(root), "expected": EXPECTED}

    # ---- CSV traces ----
    runs = paths.calibration_runs_dir() if root == paths.data_root() else root / "generalinformation" / "calibration_runs"
    csvs = sorted(runs.glob("*.csv")) if runs.exists() else []
    schema_hist: Counter[str] = Counter()
    rows = 0
    for p in csvs:
        try:
            with open(p, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            schema = detect_csv_schema(header)
            if schema is None:
                schema_hist["UNKNOWN"] += 1
                continue
            schema_hist[schema] += 1
            with open(p, encoding="utf-8") as f:
                f.readline()
                rows += sum(1 for _ in f)
        except OSError:
            schema_hist["UNREADABLE"] += 1
    report["csv"] = {
        "traces": len(csvs),
        "rows": rows,
        "schema_histogram": dict(schema_hist),
        "matches_expected": len(csvs) == EXPECTED["csv_traces"] and rows == EXPECTED["csv_rows"],
    }

    # ---- onboard logs: honest presence check ----
    logs = sorted(root.glob("**/motion_*.log")) if root.exists() else []
    report["onboard_logs"] = {
        "present": len(logs),
        "expected": EXPECTED["onboard_logs"],
        "note": (
            "motion_*.log files are absent from this copy; their prepared content "
            "lives in testing/rlearn/data/transitions.npz (61,087 transitions)"
        ),
    }

    # ---- the rlearn transition dataset ----
    try:
        data = load_rlearn_transitions()
        vals, counts = np.unique(data["split"], return_counts=True)
        report["rlearn_transitions"] = {
            "rows": int(data["X"].shape[0]),
            "features": int(data["X"].shape[1]),
            "split_counts": {int(v): int(c) for v, c in zip(vals, counts)},
            "sha256": data["provenance"]["raw_sha256"],
        }
    except (OSError, FileNotFoundError):
        report["rlearn_transitions"] = None

    # ---- load-bearing artifacts, hashed ----
    key_files = {
        "calibration_latest": paths.calibration_latest(),
        "practice_baseline_profile": paths.evidence_pack_dir() / "calibration" / "profiles" / "practice-baseline-2026-07-23.json",
        "offline_candidate_profile": paths.evidence_pack_dir() / "calibration" / "profiles" / "offline-learned-candidate-20260805.json",
        "rlearn_champion": paths.rlearn_dir() / "checkpoints" / "champion.json",
        "rlearn_manifest": paths.rlearn_manifest(),
        "firmware_motion_calibrations": paths.robotframework_dir() / "config" / "motion_calibrations.json",
        "firmware_motion_yaml": paths.robotframework_dir() / "config" / "Motion.yaml",
    }
    artifacts = {}
    for name, p in key_files.items():
        entry = {"path": str(p), "present": p.exists()}
        if p.exists():
            entry["sha256"] = sha256_file(p)
        artifacts[name] = entry
    report["artifacts"] = artifacts

    # ---- firmware snapshot revision ----
    fw = paths.robotframework_dir()
    if (fw / ".git").exists():
        try:
            head = subprocess.run(["git", "-C", str(fw), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            branch = subprocess.run(["git", "-C", str(fw), "branch", "--show-current"], capture_output=True, text=True, check=True).stdout.strip()
            report["firmware"] = {"branch": branch, "head": head}
        except (OSError, subprocess.CalledProcessError):
            report["firmware"] = None
    return report


def write_audit(out_dir: Path | None = None, tag: str = "audit") -> Path:
    from fawkes import paths as p

    out_dir = out_dir or p.EVIDENCE
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.json"
    out.write_text(json.dumps(audit_data_root(), indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(audit_data_root(), indent=2))
