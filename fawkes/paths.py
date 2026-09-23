"""Where the raw evidence lives and where derived artifacts go.

Raw data is never copied into this repository (the rlearn rule): converters
read it in place and carry SHA-256 hashes forward. Override the root with the
FAWKES_DATA_ROOT environment variable to point at another machine's copy.
"""

from __future__ import annotations

import os
from pathlib import Path

# The research folder holding the evidence pack, the rlearn lineage, the
# firmware snapshot and the digital twin (README Appendix A).
DEFAULT_DATA_ROOT = Path(r"C:\Users\rishi\Documents\GitHub\RoboCup-Research")

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "evidence"
DATA = REPO / "data"


def data_root() -> Path:
    return Path(os.environ.get("FAWKES_DATA_ROOT", str(DEFAULT_DATA_ROOT)))


def rlearn_dir() -> Path:
    return data_root() / "testing" / "rlearn"


def evidence_pack_dir() -> Path:
    return data_root() / "generalinformation"


def calibration_runs_dir() -> Path:
    return evidence_pack_dir() / "calibration_runs"


def robotframework_dir() -> Path:
    return data_root() / "RobotFramework"


def rlearn_transitions() -> Path:
    return rlearn_dir() / "data" / "transitions.npz"


def rlearn_manifest() -> Path:
    return rlearn_dir() / "data" / "manifest.json"


def calibration_latest() -> Path:
    return evidence_pack_dir() / "calibration" / "latest.json"
