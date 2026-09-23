"""The canonical DR family (README section 2.1).

Every axis the lost residual_env.py randomised (reconstructed from
Phoenix-Server/docs/PHYSICS_SIMULATION.md section 2.3), rebuilt as explicit
distributions anchored to the identified/field values (README Appendix B).

The z vector is the compact conditioning code the RMA adapter must infer:
  (traction, lateral_scale, gain_asym, motor_tau, battery_sag, vision_noise)
normalized to ~[0, 1]. Held-out convention follows Phoenix: surface seeds
>= 9000 are never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HELD_OUT_SEED = 9000

# axis: (low, high, nominal)
AXES: dict[str, tuple[float, float, float]] = {
    "traction": (0.70, 1.05, 1.0),  # carpet traction scale on achievable accel
    "lateral_scale": (0.85, 0.92, 0.887),  # effective lateral roller scale
    "gain_asym": (0.0, 0.06, 0.0),  # per-wheel command-gain deviation
    "motor_tau_s": (0.015, 0.040, 0.020),  # first-order wheel response lag
    "battery_sag": (0.88, 1.0, 1.0),  # global torque/velocity scale
    "vision_noise_m": (0.002, 0.015, 0.005),  # vision position noise sigma
    "vision_latency_ticks": (0.0, 3.0, 1.0),
    "cmd_latency_ticks": (0.0, 2.0, 1.0),
    "disturbance_p": (0.0, 0.004, 0.0),  # per-tick push probability
    "mass_scale": (0.9, 1.3, 1.0),  # scales motor_tau by sqrt(mass)
}

Z_NAMES = ("traction", "lateral_scale", "gain_asym", "motor_tau", "battery_sag", "vision_noise")


def _norm(name: str, v: np.ndarray) -> np.ndarray:
    lo, hi, _ = AXES[name]
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class SurfaceDraw:
    """One sampled world per batch row (vectorised)."""

    traction: np.ndarray
    lateral_scale: np.ndarray
    gain_asym: np.ndarray
    motor_tau_s: np.ndarray
    battery_sag: np.ndarray
    vision_noise_m: np.ndarray
    vision_latency: np.ndarray  # ticks (float, applied as index offset)
    cmd_latency: np.ndarray
    disturbance_p: np.ndarray
    mass_scale: np.ndarray
    wheel_gain: np.ndarray  # (N, 4) per-wheel command gain
    z: np.ndarray  # (N, 6) normalized conditioning code
    seed: int

    @property
    def n(self) -> int:
        return len(self.traction)


class DRFamily:
    """Sampled around the identified values; every draw is reproducible."""

    def sample(self, n: int, seed: int) -> SurfaceDraw:
        rng = np.random.default_rng(seed)
        u = rng.random((n, len(AXES)))
        draw: dict[str, np.ndarray] = {}
        for i, (name, (lo, hi, _)) in enumerate(AXES.items()):
            draw[name] = lo + u[:, i] * (hi - lo)
        # integer-ish latencies
        draw["vision_latency_ticks"] = np.round(draw["vision_latency_ticks"])
        draw["cmd_latency_ticks"] = np.round(draw["cmd_latency_ticks"])
        # per-wheel asymmetric gains
        asym = draw["gain_asym"][:, None] * (rng.choice([-1.0, 1.0], size=(n, 4)))
        wheel_gain = 1.0 + asym
        # mass slows the response
        tau = draw["motor_tau_s"] * np.sqrt(draw["mass_scale"])
        z = np.column_stack(
            [
                _norm("traction", draw["traction"]),
                _norm("lateral_scale", draw["lateral_scale"]),
                _norm("gain_asym", draw["gain_asym"]),
                _norm("motor_tau_s", tau),
                _norm("battery_sag", draw["battery_sag"]),
                _norm("vision_noise_m", draw["vision_noise_m"]),
            ]
        )
        return SurfaceDraw(
            traction=draw["traction"],
            lateral_scale=draw["lateral_scale"],
            gain_asym=draw["gain_asym"],
            motor_tau_s=tau,
            battery_sag=draw["battery_sag"],
            vision_noise_m=draw["vision_noise_m"],
            vision_latency=draw["vision_latency_ticks"],
            cmd_latency=draw["cmd_latency_ticks"],
            disturbance_p=draw["disturbance_p"],
            mass_scale=draw["mass_scale"],
            wheel_gain=wheel_gain,
            z=z,
            seed=seed,
        )

    def nominal(self, n: int) -> SurfaceDraw:
        """The identified nominal world (no randomisation)."""
        draw = {}
        for name, (lo, hi, nom) in AXES.items():
            draw[name] = np.full(n, nom)
        z = np.column_stack(
            [
                _norm("traction", draw["traction"]),
                _norm("lateral_scale", draw["lateral_scale"]),
                _norm("gain_asym", draw["gain_asym"]),
                _norm("motor_tau_s", draw["motor_tau_s"]),
                _norm("battery_sag", draw["battery_sag"]),
                _norm("vision_noise_m", draw["vision_noise_m"]),
            ]
        )
        return SurfaceDraw(
            traction=draw["traction"],
            lateral_scale=draw["lateral_scale"],
            gain_asym=draw["gain_asym"],
            motor_tau_s=draw["motor_tau_s"],
            battery_sag=draw["battery_sag"],
            vision_noise_m=draw["vision_noise_m"],
            vision_latency=np.zeros(n),
            cmd_latency=np.zeros(n),
            disturbance_p=np.zeros(n),
            mass_scale=np.ones(n),
            wheel_gain=np.ones((n, 4)),
            z=z,
            seed=-1,
        )

    def family_id(self) -> str:
        return "dr-family-1"


DEFAULT_FAMILY = DRFamily()
