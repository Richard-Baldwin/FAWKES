"""FTF-1: the Fawkes Trace Format.

One episode = a JSON header + columnar arrays. The header carries identity and
provenance; the arrays carry time series. Missing columns are absent, never
fabricated. Mirrors README.md section L0.

The header is intentionally small and universal: any team can produce one. The
robot block carries the kinematics constants the environment needs to replay
the episode; the provenance block carries the SHA-256 of the raw source file
(rlearn's rule: raw files are never copied or changed; hashes make every run
traceable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "ftf-1"

VISION_COLUMNS = (
    "vision_x",
    "vision_y",
    "vision_heading",
    "vision_vx",
    "vision_vy",
    "vision_omega",
    "vision_age_ms",
)
EST_COLUMNS = (
    "est_x",
    "est_y",
    "est_heading",
    "est_vx",
    "est_vy",
    "est_omega",
)
CMD_COLUMNS = (
    "cmd_skill",
    "cmd_x",
    "cmd_y",
    "cmd_heading",
    "cmd_vx",
    "cmd_vy",
    "cmd_omega",
    "cmd_vel_max",
    "cmd_acc_max",
    "cmd_primary_direction",
)
FB_COLUMNS = (
    "fb_pos_x",
    "fb_pos_y",
    "fb_pos_heading",
    "fb_vx",
    "fb_vy",
    "fb_omega",
    "fb_battery_v",
)
WHEEL_COLUMNS = tuple(
    f"wheel_measured_{i}" for i in range(1, 5)
) + tuple(f"wheel_commanded_{i}" for i in range(1, 5))
TRUTH_COLUMNS = ("truth_x", "truth_y", "truth_heading")
DIAGNOSTIC_COLUMNS = ("error_m", "heading_error_rad")

KNOWN_COLUMNS = frozenset(
    {"t", *VISION_COLUMNS, *EST_COLUMNS, *CMD_COLUMNS, *FB_COLUMNS, *WHEEL_COLUMNS, *TRUTH_COLUMNS, *DIAGNOSTIC_COLUMNS}
)

REQUIRED_HEADER_KEYS = ("schema", "team", "robot", "source", "provenance")

# The kinematics constants every episode's robot block should carry (README L0).
KINEMATICS_KEYS = (
    "meters_per_motor_rev",
    "body_lateral_scale",
    "wheel_order_canon",
    "per_wheel_command_scale",
)


@dataclass
class Episode:
    """One converted movement episode: header + columnar time series."""

    header: dict = field(default_factory=dict)
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        lengths = {len(v) for v in self.columns.values()}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise ValueError(f"ragged columns: lengths {sorted(lengths)}")
        return lengths.pop()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = dict(self.header)
        header["rows"] = self.rows
        np.savez(
            path,
            __header__=np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8),
            **self.columns,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        path = Path(path)
        with np.load(path) as z:
            header = json.loads(z["__header__"].tobytes().decode("utf-8"))
            columns = {k: z[k] for k in z.files if k != "__header__"}
        return cls(header=header, columns=columns)


def validate_episode(ep: Episode) -> list[str]:
    """Return a list of problems (empty list = valid FTF-1 episode)."""
    problems: list[str] = []
    header = ep.header
    for key in REQUIRED_HEADER_KEYS:
        if key not in header:
            problems.append(f"header missing required key {key!r}")
    if header.get("schema") != SCHEMA:
        problems.append(f"schema is {header.get('schema')!r}, expected {SCHEMA!r}")
    kin = header.get("robot", {}).get("kinematics", {})
    for key in KINEMATICS_KEYS:
        if key not in kin:
            problems.append(f"robot.kinematics missing {key!r}")
    unknown = sorted(set(ep.columns) - KNOWN_COLUMNS)
    if unknown:
        problems.append(f"unknown columns: {unknown}")
    if ep.columns:
        try:
            ep.rows
        except ValueError as exc:
            problems.append(str(exc))
    t = ep.columns.get("t")
    if t is not None and len(t) > 1:
        if not np.all(np.diff(t) > -1e-12):
            problems.append("t is not monotonically increasing")
    return problems
