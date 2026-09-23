"""L4 export (c): the Phoenix bundle, spec movement-rma-1.

Extends the existing Phoenix bundle pattern (spec team-knobs-1, verified with
recorded observation->action round-trips) to a movement policy: the base MLP +
the linear adapter, observation/action name lists, provenance, hardware
profile, compatibility, and verification pairs so --verify can replay the
deployed implementation against the training one, exactly as
phoenix-rl-team --verify does today.

Requires one new pure-stdlib consumer module on the Phoenix side
(export/phoenix_consumer.py is that module; it goes through CODEOWNERS review).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fawkes.export.phoenix_consumer import consumer_forward


def build_bundle(base, adapter, obs_names, adapter_feature_names, z_names,
                 provenance: dict, hardware_profile: dict | None = None) -> dict:
    rng = np.random.default_rng(4242)
    verification = []
    for _ in range(12):
        obs = rng.normal(0, 1, (1, len(obs_names)))
        feats = rng.random((1, len(adapter_feature_names)))
        z = adapter.predict(feats)
        action = base.act(obs, z)
        verification.append(
            {
                "observation": [round(float(v), 6) for v in obs[0]],
                "adapter_features": [round(float(v), 6) for v in feats[0]],
                "z": [round(float(v), 6) for v in z[0]],
                "expected_action": [round(float(v), 9) for v in action[0]],
            }
        )
    from fawkes.policies.rma import OBS_SCALE

    bundle = {
        "kind": "movement-rma-bundle",
        "spec": "movement-rma-1",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": {
            "type": "rma-linear-adapter-1",
            "base_mlp": {
                "sizes": list(base.net.sizes),
                "obs_scale": [float(s) for s in OBS_SCALE],
                "params": [float(v) for v in base.params],
            },
            "adapter": {
                "W": [[float(v) for v in row] for row in adapter.W.T],
                "b": [float(v) for v in adapter.b],
                "feature_names": list(adapter_feature_names),
            },
            "z_names": list(z_names),
            "observation": list(obs_names),
            "action": ["residual_vx", "residual_vy", "residual_omega"],
            "residual_scale": [0.125, 0.125, 0.3],
        },
        "stage": {"name": "dr-flat-1", "surface": "dr-family-1", "opponents": "cascade+residual"},
        "training": provenance,
        "hardware_profile": hardware_profile
        or {
            "motion_active": "baseline-validated",
            "limits_sha256": "TBD-at-promotion",
            "server_sha256": "TBD-at-promotion",
            "note": "Re-pin against the target checkout and limits file at promotion.",
        },
        "compatibility": {
            "fawkes": "0.1.0",
            "consumer": "pure-stdlib (no numpy)",
            "observation_size": len(obs_names),
            "action_size": 3,
            "z_size": len(z_names),
        },
        "verification": verification,
    }
    import hashlib

    digest_src = json.dumps(
        {k: bundle[k] for k in ("policy", "compatibility", "verification")}, sort_keys=True
    )
    bundle["digest"] = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]
    return bundle


def write_bundle(path: str | Path, bundle: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def verify_bundle(path: str | Path) -> dict:
    """Replay the verification pairs through the pure-stdlib consumer."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    worst = 0.0
    checked = 0
    for pair in bundle["verification"]:
        action = consumer_forward(bundle, pair["observation"], pair["adapter_features"])
        worst = max(worst, max(abs(a - e) for a, e in zip(action, pair["expected_action"])))
        checked += 1
    return {
        "pairs": checked,
        "max_abs_error": worst,
        "digest": bundle.get("digest"),
        # not exact-zero on purpose: the numpy (BLAS) forward and the pure-Python
        # consumer sum in different orders, so ~1e-6 fp noise is expected on
        # actions in [-1, 1]. 1e-4 on a residual action = 12.5 um of authority.
        "ok": worst < 1e-4,
    }
