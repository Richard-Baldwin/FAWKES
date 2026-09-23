"""L4 export (a): the motion-calibration profile entry for RobotFramework.

Matches the structure of RobotFramework/config/motion_calibrations.json entries
(the rlearn champion's path): dotted firmware parameter names, status
SIMULATION ONLY, deployment_authorized false, simulation-only parameters kept
out of the robot block, and the field-proven rollback profile untouched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# fawkes knob name -> firmware dotted name (Motion.yaml / the registry).
# plant_speed_guard is simulation-only (rlearn's own rule) and is kept out.
FIRMWARE_MAP = {
    "trajectory_brake_scale": "trajectory.brake_scale",
    "orient_lag_tau_s": "trajectory.orient_lag_tau_s",
    "kp_pos": "controller.kp_pos",
    "kp_vel": "controller.kp_vel",
    "kp_heading": "controller.kp_heading",
    "kp_yaw_rate": "controller.kp_yaw_rate",
    "omega_corr_max": "controller.omega_corr_max",
    "acc_ff_lead_s": "controller.acc_ff_lead_s",
    "target_brake_scale": "controller.target_brake_scale",
    "target_brake_reaction_s": "controller.target_brake_reaction_s",
}
SIMULATION_ONLY = ("plant_speed_guard",)


def registry_entry(knobs: dict, metrics: dict, profile_id: str, provenance: dict | None = None) -> dict:
    params = {fw: float(knobs[k]) for k, fw in FIRMWARE_MAP.items()}
    return {
        "profile_id": profile_id,
        "name": "FAWKES offline champion (simulation)",
        "status": "SIMULATION ONLY",
        "deployment_authorized": False,
        "description": (
            "FAWKES CEM champion over the rlearn gain space, trained against the "
            "domain-randomised environment. It has NOT been validated on hardware. "
            "Install only for guarded field validation via the registry, never "
            "automatically. The field-proven rollback profile stays untouched."
        ),
        "parameters": params,
        "simulation_only_parameter_not_applied_onboard": {
            k: float(knobs[k]) for k in SIMULATION_ONLY if k in knobs
        },
        "source": {
            "framework": "fawkes",
            "metrics": metrics,
            "provenance": provenance or {},
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warning": (
                "Simulators overpredicted field cross-track before (rlearn: 203 mm "
                "sim vs 104 mm real). Replay-vs-real-traces and the shadow ladder "
                "are mandatory before any bounded run."
            ),
        },
    }


def write_registry_entry(path: str | Path, entry: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return path
