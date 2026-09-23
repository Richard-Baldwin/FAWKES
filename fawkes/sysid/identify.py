"""Fit the wheel ensemble on the real transition dataset and emit a model card."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fawkes.sysid.wheel_model import WheelEnsemble, select_features
from fawkes.trace.converters import load_rlearn_transitions


def run(out_dir: Path | str | None = None, seed: int = 0) -> dict:
    """Identify the plant from real data. Writes model_card.json + ensemble.npz."""
    from fawkes import paths

    out_dir = Path(out_dir) if out_dir is not None else paths.EVIDENCE / "sysid"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_rlearn_transitions()
    X, y = select_features(data)
    split = data["split"]
    train, held = split == 0, split == 1

    ensemble = WheelEnsemble.fit(
        X[train], y[train], data["sample_weight"][train], seed=seed
    )
    heldout = ensemble.evaluate(X[held], y[held])
    card = ensemble.model_card(heldout)
    card["provenance"]["dataset_sha256"] = data["provenance"]["raw_sha256"]
    card["provenance"]["rows_total"] = int(len(X))
    card["provenance"]["rows_train"] = int(train.sum())
    card["provenance"]["rows_held_out"] = int(held.sum())

    ensemble.save(out_dir / "ensemble.npz")
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
