"""House-style markdown reports (evidence tables, honest caveats, provenance)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def provenance_block(extra: dict | None = None) -> str:
    from fawkes import paths as p

    lines = [
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- fawkes repo: {_git_head(p.REPO)}",
        f"- data root: `{p.data_root()}`",
    ]
    if extra:
        lines += [f"- {k}: {v}" for k, v in extra.items()]
    return "\n".join(lines)


def gates_table(gates: dict) -> str:
    rows = ["| Gate | Value | Limit | Stretch | Pass |", "| --- | ---: | ---: | ---: | --- |"]
    for name, g in gates.items():
        if name == "all_pass" or not isinstance(g, dict):
            continue
        rows.append(
            f"| {name} | {g['value']} | {g['limit']} | {g['stretch']} |"
            f" {'PASS' if g['pass'] else 'FAIL'}{' (stretch)' if g['meets_stretch'] else ''} |"
        )
    rows.append(f"| **all gates** | | | | **{'PASS' if gates.get('all_pass') else 'FAIL'}** |")
    return "\n".join(rows)


def write_report(path: str | Path, title: str, sections: list[tuple[str, str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", "", f"> Evidence is a run directory, never a claim. Simulation acceptance is never physical authorisation.", ""]
    for heading, content in sections:
        body += [f"## {heading}", "", content, ""]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def dump_json(path: str | Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
