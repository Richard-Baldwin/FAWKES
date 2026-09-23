"""FTF-1 converters, our data first (README L0).

  phoenix_csv          the four documented CSV schemas of the 2026-07-23 pack
  onboard_motion_log   pipe-separated headerless logs (0- negative repair)
  rlearn_npz           the prepared transition dataset (47 features)
  phoenix_session      Measurement-page recordings - the flywheel's main food
                       (lands with FK-7; deliberately unimplemented)
  ssl_vision_log       any team's SSL-Vision log (deliberately unimplemented)

Every converted episode carries the SHA-256 of its raw source. Raw files are
never copied or changed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from fawkes.trace.manifest import sha256_file
from fawkes.trace.schema import Episode, SCHEMA

# The four CSV schemas documented in the 2026-07-23 evidence pack
# (2026-07-23_SESSION_CONTEXT_AND_UNKNOWNS.md section 9: 107 / 27 / 20 / 8 traces).
CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "A": (
        "t", "x", "y", "heading", "vx", "vy", "omega",
        "est_x", "est_y", "est_heading", "est_vx", "est_vy", "est_omega",
        "vision_age_ms", "error_m", "heading_error_rad",
    ),
    "B": (
        "t", "x", "y", "heading", "vx", "vy", "omega",
        "est_x", "est_y", "est_heading", "est_vx", "est_vy",
        "vision_age_ms", "error_m", "heading_error_rad",
    ),
    "C": (
        "t", "x", "y", "heading", "vx", "vy", "omega",
        "est_x", "est_y", "est_heading", "est_vx", "est_vy",
        "error_m", "heading_error_rad",
    ),
    "D": ("t", "x", "y", "heading", "vx", "vy", "omega", "error_m", "heading_error_rad"),
}

# CSV column -> FTF column.
_CSV_MAP = {
    "t": "t",
    "x": "vision_x",
    "y": "vision_y",
    "heading": "vision_heading",
    "vx": "vision_vx",
    "vy": "vision_vy",
    "omega": "vision_omega",
    "vision_age_ms": "vision_age_ms",
    "error_m": "error_m",
    "heading_error_rad": "heading_error_rad",
    "est_x": "est_x",
    "est_y": "est_y",
    "est_heading": "est_heading",
    "est_vx": "est_vx",
    "est_vy": "est_vy",
    "est_omega": "est_omega",
}

# Robot A, as recorded in the evidence pack (id 4 on the wire, hardware 0).
DEFAULT_ROBOT = {
    "id": 4,
    "hardware_id": 0,
    "chassis": "RobotA-mjbots",
    "kinematics": {
        "meters_per_motor_rev": 0.2171,
        "body_lateral_scale": 0.887,
        "wheel_order_canon": ["FR", "RR", "RL", "FL"],
        "per_wheel_command_scale": [1.006, 1.008, 1.004, 1.004],
    },
}


def detect_csv_schema(header_row: list[str]) -> str | None:
    for name, cols in CSV_SCHEMAS.items():
        if tuple(header_row) == cols:
            return name
    return None


def phoenix_csv(path: str | Path, team: str = "WSU-TurtleRabbit") -> Episode:
    """Convert one Phoenix calibration CSV into an FTF-1 episode."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_row = next(reader)
        schema = detect_csv_schema(header_row)
        if schema is None:
            raise ValueError(f"{path.name}: unknown CSV schema {header_row}")
        rows = [r for r in reader if r]

    # The pack documents duplicate timestamps in some traces: keep the first of
    # each duplicate and record the count as an event, never silently.
    seen_t: set[float] = set()
    kept: list[list[str]] = []
    duplicates = 0
    for row in rows:
        t = float(row[0])
        if t in seen_t:
            duplicates += 1
            continue
        seen_t.add(t)
        kept.append(row)

    columns: dict[str, np.ndarray] = {}
    for i, src in enumerate(header_row):
        dst = _CSV_MAP[src]
        columns[dst] = np.array([float(r[i]) for r in kept], dtype=np.float64)

    events = []
    if duplicates:
        events.append({"kind": "duplicate_timestamps_dropped", "count": duplicates})

    return Episode(
        header={
            "schema": SCHEMA,
            "team": team,
            "robot": dict(DEFAULT_ROBOT),
            "source": {
                "kind": "phoenix_csv",
                "schema": schema,
                "path": str(path),
                "recorded": "2026-07-23",
                "surface_id": "practice-field",
            },
            "session": {"camera_hz": 60.0, "control_hz": 250.0},
            "provenance": {
                "raw_sha256": sha256_file(path),
                "converter": "phoenix_csv@1",
                "rows": len(kept),
                "events": events,
            },
        },
        columns=columns,
    )


# The onboard logger's negative-number artifact: a padded "0-0.009499" (and
# the integer form "0-3") means a negative number (evidence pack section 11).
_NEGATIVE_PAD = re.compile(r"(^|\|)0-(?=\d)")


def repair_negative_padding(line: str) -> str:
    return _NEGATIVE_PAD.sub(lambda m: f"{m.group(1)}-", line)


def onboard_motion_log(path: str | Path, columns: list[str] | None = None) -> Episode:
    """Convert one pipe-separated onboard motion log.

    The logs are headerless; the column layout must come from the matching
    RobotFramework telemetry revision. Without one, the episode is parsed
    best-effort and its header records that the field semantics are
    UNVERIFIED - it stays excluded from training until a layout is pinned.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = repair_negative_padding(line)
        rows.append(line.split("|"))
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path.name}: ragged log rows {sorted(widths)}")
    width = widths.pop()
    pinned = columns is not None  # capture before the fallback rebind

    if columns is None:
        columns = [f"unverified_field_{i}" for i in range(width)]

    series: dict[str, np.ndarray] = {}
    for i, name in enumerate(columns):
        try:
            series[name] = np.array([float(r[i]) for r in rows], dtype=np.float64)
        except ValueError:
            continue  # non-numeric channel (states, flags): skip, do not invent
    series["t"] = np.arange(len(rows), dtype=np.float64) * 0.02  # 50 Hz logging

    return Episode(
        header={
            "schema": SCHEMA,
            "team": "WSU-TurtleRabbit",
            "robot": dict(DEFAULT_ROBOT),
            "source": {
                "kind": "onboard_motion_log",
                "path": str(path),
                "columns_pinned": pinned,
                "semantics": "pinned" if pinned else "UNVERIFIED",
            },
            "session": {"control_hz": 250.0, "log_hz": 50.0},
            "provenance": {
                "raw_sha256": sha256_file(path),
                "converter": "onboard_motion_log@1",
                "rows": len(rows),
                "events": [],
            },
        },
        columns=series,
    )


def load_rlearn_transitions(npz_path: str | Path | None = None):
    """Load the prepared rlearn transition dataset (61,087 rows, 47 features).

    Returns a dict with X, y, split, sample_weight, source, coupled,
    feature_names and provenance hashes. split: 0 = train, 1 = held-out
    (the rlearn chronological hold-out, already computed - do not reshuffle).
    """
    from fawkes import paths

    npz_path = Path(npz_path) if npz_path is not None else paths.rlearn_transitions()
    with np.load(npz_path) as z:
        data = {k: z[k] for k in z.files}
    data["provenance"] = {
        "raw_sha256": sha256_file(npz_path),
        "converter": "rlearn_npz@1",
        "rows": int(data["X"].shape[0]),
    }
    return data


def phoenix_session(path: str | Path) -> Episode:
    """Measurement-page session recordings: the flywheel's main food (FK-7)."""
    raise NotImplementedError(
        "phoenix_session converter lands with FK-7 (Measurement sessions are live "
        "captures; none exist in the 2026-07-23 pack). See README.md section L5."
    )


def ssl_vision_log(path: str | Path) -> Episode:
    """Official SSL vision logs: for teams without onboard logs (FK-1 follow-up)."""
    raise NotImplementedError("ssl_vision_log converter is a documented FK-1 follow-up.")
