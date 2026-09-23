"""FTF-1 and converter tests (hermetic: synthetic fixtures, no external data)."""

from __future__ import annotations

import numpy as np
import pytest

from fawkes.trace.converters import (
    CSV_SCHEMAS,
    detect_csv_schema,
    onboard_motion_log,
    phoenix_csv,
    repair_negative_padding,
)
from fawkes.trace.schema import Episode, validate_episode


def _write_csv(path, schema):
    cols = CSV_SCHEMAS[schema]
    rows = ["0.0,0.1,0.2,0.01,0.0,0.0,0.0," + ",".join("0.1" for _ in cols[7:])]
    # schema A has 16 cols; build full-length rows
    n = len(cols)
    rows = [",".join(str(i * 0.01) for i in range(n))]
    rows.append(rows[0])  # duplicate timestamp: converter must drop it
    path.write_text(",".join(cols) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_all_four_documented_schemas_detected(tmp_path):
    for name in ("A", "B", "C", "D"):
        p = _write_csv(tmp_path / f"trace_{name}.csv", name)
        with open(p, encoding="utf-8") as f:
            header = next(f).strip().split(",")
        assert detect_csv_schema(header) == name


def test_phoenix_csv_converts_schema_a(tmp_path):
    p = _write_csv(tmp_path / "a.csv", "A")
    ep = phoenix_csv(p)
    problems = validate_episode(ep)
    assert problems == [], problems
    assert ep.header["source"]["schema"] == "A"
    assert ep.rows == 1  # duplicate timestamp dropped
    assert "vision_x" in ep.columns and "est_omega" in ep.columns
    assert any(e["kind"] == "duplicate_timestamps_dropped" for e in ep.header["provenance"]["events"])


def test_phoenix_csv_rejects_unknown_schema(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        phoenix_csv(p)


def test_negative_padding_repaired():
    line = "12|0-0.009499|0.51|0-1.25"
    assert repair_negative_padding(line) == "12|-0.009499|0.51|-1.25"


def test_onboard_motion_log_parses(tmp_path):
    p = tmp_path / "motion_x.log"
    p.write_text("0|0|0|0\n0.02|1|2|0-3\n\n0.04|2|4|-6\n", encoding="utf-8")
    ep = onboard_motion_log(p, columns=["t", "v", "a", "w"])
    assert ep.rows == 3
    assert ep.columns["w"][1] == pytest.approx(-3.0)
    assert ep.header["source"]["columns_pinned"] is True
    unpinned = onboard_motion_log(p)
    assert unpinned.header["source"]["semantics"] == "UNVERIFIED"


def test_episode_save_load_round_trip(tmp_path):
    ep = phoenix_csv(_write_csv(tmp_path / "a.csv", "A"))
    ep.save(tmp_path / "ep.npz")
    back = Episode.load(tmp_path / "ep.npz")
    assert back.header["source"]["schema"] == "A"
    assert np.allclose(back.columns["vision_x"], ep.columns["vision_x"])


def test_validate_catches_missing_kinematics():
    ep = Episode(header={"schema": "ftf-1"}, columns={"t": np.zeros(3)})
    problems = validate_episode(ep)
    assert any("kinematics" in p for p in problems)
