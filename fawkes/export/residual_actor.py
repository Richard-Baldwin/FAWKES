"""L4 export (b): the residual actor for the onboard bounded mode.

The bounded-mode clamps are copied verbatim from
Phoenix-Server/configs/motion/rl-bounded-dr-i900.toml - the artifact never
widens them. The runtime ABI (turtlerabbit-motion-v1) lives on the live
firmware experimental branch; this artifact declares itself v1-pending and must
be byte-checked against that ABI before any on-robot use (README FK-6).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Verbatim from configs/motion/rl-bounded-dr-i900.toml [onboard.overlay.rl].
BOUNDED_CLAMPS = {
    "residual_limit_fraction": 0.05,
    "residual_slew_linear_mps2": 0.5,
    "residual_slew_angular_radps2": 1.5,
    "confidence_threshold": 0.30,
    "max_consecutive_interventions": 5,
    "move_supervisor": "10 A / 300 ms",
    "jerk": "10 / 30",
    "combined_bound": "15 %",
}


def build_actor(base, adapter, obs_names, adapter_feature_names, z_names, provenance: dict) -> dict:
    rng = np.random.default_rng(99)
    self_tests = []
    for _ in range(8):
        obs = rng.normal(0, 1, (1, len(obs_names)))
        feats = rng.random((1, len(adapter_feature_names)))
        z = adapter.predict(feats)
        action = base.act(obs, z)
        self_tests.append(
            {
                "observation": [round(float(v), 6) for v in obs[0]],
                "adapter_features": [round(float(v), 6) for v in feats[0]],
                "expected_action": [round(float(v), 9) for v in action[0]],
            }
        )
    return {
        "kind": "fawkes-residual-actor-1",
        "abi": "turtlerabbit-motion-v1-pending-confirmation",
        "abi_note": (
            "Confirm against the live firmware experimental branch ABI before "
            "fielding (the RoboCup-Research snapshot has no adaptive runtime)."
        ),
        "observation_names": list(obs_names),
        "adapter_feature_names": list(adapter_feature_names),
        "z_names": list(z_names),
        "action_names": ["residual_vx", "residual_vy", "residual_omega"],
        "residual_scale": [0.125, 0.125, 0.3],
        "clamps": BOUNDED_CLAMPS,
        "base_mlp": {
            "sizes": list(base.net.sizes),
            "obs_scale": [float(s) for s in __import__("fawkes.policies.rma", fromlist=["OBS_SCALE"]).OBS_SCALE],
            "params": [float(v) for v in base.params],
        },
        "adapter": {
            "W": [[float(v) for v in row] for row in adapter.W.T],
            "b": [float(v) for v in adapter.b],
        },
        "provenance": provenance,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployment_authorized": False,
        "self_test": self_tests,
    }


def write_actor(path: str | Path, actor: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(actor, indent=2), encoding="utf-8")
    return path
