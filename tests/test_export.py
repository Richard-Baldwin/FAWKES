"""Export tests: the bundle round-trips through the pure-stdlib consumer,
the firmware entry carries the house safety rules, the actor's clamps are
verbatim, and the consumer never imports numpy."""

from __future__ import annotations

import json

import numpy as np

from fawkes.envs.env import ADAPTER_FEATURE_NAMES, OBS_NAMES
from fawkes.envs.surface import Z_NAMES
from fawkes.export.firmware_profile import registry_entry
from fawkes.export.phoenix_bundle import build_bundle, verify_bundle, write_bundle
from fawkes.export.residual_actor import BOUNDED_CLAMPS, build_actor
from fawkes.policies.rma import BasePolicy, LinearAdapter


def _fitted_adapter():
    rng = np.random.default_rng(9)
    W = rng.normal(0, 1.0, size=(30, 6))
    b = rng.normal(0, 0.1, size=6)
    feats = rng.random((2000, 30))
    z = np.clip(feats @ W + b, 0.0, 1.0)
    adapter = LinearAdapter(feature_names=ADAPTER_FEATURE_NAMES)
    adapter.fit(feats, z)
    return adapter


def test_bundle_verifies_through_the_pure_stdlib_consumer(tmp_path):
    base = BasePolicy()
    adapter = _fitted_adapter()
    bundle = build_bundle(base, adapter, OBS_NAMES, ADAPTER_FEATURE_NAMES, Z_NAMES, provenance={"seed": 1})
    path = write_bundle(tmp_path / "bundle.json", bundle)
    result = verify_bundle(path)
    assert result["ok"], result
    assert result["max_abs_error"] < 1e-9
    assert bundle["spec"] == "movement-rma-1"
    assert bundle["policy"]["observation"] == list(OBS_NAMES)


def test_firmware_registry_entry_keeps_the_house_rules():
    knobs = {
        "kp_pos": 5.0, "kp_vel": 0.7, "acc_ff_lead_s": 0.14,
        "target_brake_scale": 0.5, "target_brake_reaction_s": 0.2,
        "kp_heading": 6.0, "kp_yaw_rate": 0.3, "omega_corr_max": 3.0,
        "trajectory_brake_scale": 0.5, "orient_lag_tau_s": 0.05,
        "plant_speed_guard": 0.9,
    }
    entry = registry_entry(knobs, {"endpoint_p95_m": 0.01}, profile_id="fawkes-cem-test")
    assert entry["deployment_authorized"] is False
    assert entry["status"] == "SIMULATION ONLY"
    assert "controller.kp_pos" in entry["parameters"]
    assert "plant_speed_guard" in entry["simulation_only_parameter_not_applied_onboard"]
    assert "203 mm" in entry["source"]["warning"]


def test_actor_clamps_are_verbatim():
    actor = build_actor(BasePolicy(), _fitted_adapter(), OBS_NAMES, ADAPTER_FEATURE_NAMES, Z_NAMES, {})
    assert actor["clamps"] == BOUNDED_CLAMPS
    assert actor["clamps"]["residual_limit_fraction"] == 0.05
    assert actor["deployment_authorized"] is False
    assert len(actor["self_test"]) == 8


def test_the_consumer_is_pure_stdlib(tmp_path):
    """The deployment scenario exactly: the consumer module is copied into
    Phoenix ALONE and imported without the fawkes package (and without numpy)."""
    import inspect
    import subprocess
    import sys
    from pathlib import Path

    import fawkes.export.phoenix_consumer as consumer_mod

    src = inspect.getsource(consumer_mod)
    assert "import numpy" not in src and "from numpy" not in src
    consumer_path = Path(inspect.getfile(consumer_mod))

    bundle = build_bundle(BasePolicy(), _fitted_adapter(), OBS_NAMES, ADAPTER_FEATURE_NAMES, Z_NAMES, {})
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    code = (
        "import sys, json, importlib.util;"
        "sys.modules['numpy']=None;"  # simulate numpy absence
        "spec=importlib.util.spec_from_file_location('phoenix_consumer', sys.argv[1]);"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "b=json.load(open(sys.argv[2]));"
        "worst=max(abs(a-e) for p in b['verification'] "
        "for a,e in zip(m.consumer_forward(b,p['observation'],p['adapter_features']), p['expected_action']));"
        "print(f'{worst:.12f}')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(consumer_path), str(path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    worst = float(out.stdout)
    # BLAS vs pure-Python summation order: ~1e-6 fp noise is expected
    assert worst < 1e-4
