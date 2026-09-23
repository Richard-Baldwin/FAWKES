"""The deterministic cascade under test (Panthera-lite).

Mirrors the RobotFramework motion stack per docs/MOTION.md: per-tick bang-bang
trajectory regeneration from the reference state, velocity feed-forward with
accel lead, P on the reference-vs-estimate position error, P heading + P
yaw-rate, then inverse kinematics. The 11 knobs are exactly rlearn's search
space (configs/search_space.json); the baseline is the field-proven
2026-07-23 set (configs/baseline.json).

v0.1 simplifications (documented, deliberate - README section L2):
  - 50 Hz env tick, not the 250 Hz onboard tick;
  - the delayed-vision estimator is modelled as a noisy, delayed observation;
  - the 2D axis sync is a time-scaling approximation of TIGERs alpha-bisection.
The FK-3 upgrade path is the full simlink cascade; this module keeps the same
knob names so a champion transfers when that lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The 11 interpretable knobs (rlearn's search space; names verbatim).
KNOB_NAMES = (
    "kp_pos",
    "kp_vel",
    "acc_ff_lead_s",
    "target_brake_scale",
    "target_brake_reaction_s",
    "kp_heading",
    "kp_yaw_rate",
    "omega_corr_max",
    "trajectory_brake_scale",
    "orient_lag_tau_s",
    "plant_speed_guard",
)

# The field-proven baseline (rlearn configs/baseline.json controller block).
BASELINE_KNOBS = {
    "kp_pos": 5.0,
    "kp_vel": 0.7,
    "acc_ff_lead_s": 0.14,
    "target_brake_scale": 0.5,
    "target_brake_reaction_s": 0.2,
    "kp_heading": 6.0,
    "kp_yaw_rate": 0.3,
    "omega_corr_max": 3.0,
    "trajectory_brake_scale": 0.5,
    "orient_lag_tau_s": 0.05,
    "plant_speed_guard": 1.0,
}

# rlearn's search space bounds (configs/search_space.json), verbatim.
SEARCH_SPACE = {
    "kp_pos": (3.5, 7.0),
    "kp_vel": (0.4, 1.15),
    "acc_ff_lead_s": (0.06, 0.2),
    "target_brake_scale": (0.4, 0.8),
    "target_brake_reaction_s": (0.1, 0.28),
    "kp_heading": (4.0, 8.5),
    "kp_yaw_rate": (0.15, 0.65),
    "omega_corr_max": (2.2, 4.5),
    "trajectory_brake_scale": (0.4, 0.75),
    "orient_lag_tau_s": (0.025, 0.1),
    "plant_speed_guard": (0.82, 1.0),
}


def knobs_to_vec(d: dict) -> np.ndarray:
    return np.array([float(d[k]) for k in KNOB_NAMES])


def vec_to_knobs(v) -> dict:
    return {k: float(x) for k, x in zip(KNOB_NAMES, v)}


def clamp_knobs(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    for i, k in enumerate(KNOB_NAMES):
        lo, hi = SEARCH_SPACE[k]
        out[i] = min(max(v[i], lo), hi)
    return out


@dataclass(frozen=True)
class RobotLimits:
    """The fixed limits (rlearn baseline.json fixed_limits; field geometry)."""

    vel_max_xy_mps: float = 2.5
    acc_max_xy_mps2: float = 2.5
    vel_max_w_radps: float = 6.0
    acc_max_w_radps2: float = 15.0
    wheel_max_rev_s: float = 45.0
    wheel_slew_rev_s2: float = 120.0
    field_length_m: float = 4.5
    field_width_m: float = 3.0
    field_margin_m: float = 0.2
    endpoint_pass_m: float = 0.035
    cross_track_pass_m: float = 0.12


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Cascade:
    """Vectorised Panthera cascade for a batch of B robots (one knob set per row)."""

    def __init__(self, knobs: np.ndarray | None = None, limits: RobotLimits | None = None):
        self.knobs = (
            np.tile(knobs_to_vec(BASELINE_KNOBS), (1, 1)) if knobs is None else np.atleast_2d(knobs)
        )
        self.limits = limits or RobotLimits()

    def set_knobs_per_row(self, knobs: np.ndarray) -> None:
        self.knobs = np.atleast_2d(np.asarray(knobs, dtype=np.float64))

    def reset_reference(self, x, y, th) -> None:
        """Reference state starts at the given (estimated) states."""
        self.ref = np.column_stack([x, y, th, np.zeros_like(x), np.zeros_like(y), np.zeros_like(th)])

    def step(self, est: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """One control tick. est: (B, 6) [x, y, th, vx, vy, w] world-frame velocities.
        target: (B, 3) [x, y, th]. Returns world-frame body commands (B, 3)."""
        kn = self.knobs
        lim = self.limits
        B = est.shape[0]
        if kn.shape[0] == 1 and B > 1:
            kn = np.tile(kn, (B, 1))
        ref = self.ref  # (B, 6)

        # ---- trajectory regeneration (per tick, from the reference state) ----
        d = target[:, 0:2] - ref[:, 0:2]
        brake = kn[:, 8][:, None] * lim.acc_max_xy_mps2  # trajectory_brake_scale
        # 3 mm deadband: inside it the reference rests and the P term closes the
        # gap. Without it the braking law's sign flips every tick at the target
        # and the accel-lead feed-forward amplifies that chatter over the P term.
        d_eff = np.abs(d) - 0.003
        v_stop = np.sqrt(2.0 * brake * np.maximum(d_eff, 0.0))
        v_stop = np.minimum(v_stop, lim.vel_max_xy_mps)
        v_target = np.sign(d) * v_stop
        # final-approach damping from the ACTUAL state (rlearn target_brake_*):
        d_est = np.linalg.norm(target[:, 0:2] - est[:, 0:2], axis=1, keepdims=True)
        near = np.exp(-d_est / 0.15)
        v_target = v_target * (1.0 - kn[:, 3][:, None] * 0.5 * near)
        # axis sync: scale both axes so they finish together (straight-line path)
        t_ax = np.abs(d) / np.maximum(v_stop, 1e-6) + v_stop / (2.0 * brake)
        t_ax = np.where(v_stop > 1e-6, t_ax, 0.0)
        T = np.maximum(t_ax[:, 0:1], t_ax[:, 1:2])
        T = np.maximum(T, 1e-3)
        scale = np.where(t_ax > 1e-6, t_ax / np.repeat(T, 2, axis=1), 0.0)
        v_target = v_target * scale
        a_ref = np.clip((v_target - ref[:, 3:5]) / dt, -lim.acc_max_xy_mps2, lim.acc_max_xy_mps2)
        v_ref = ref[:, 3:5] + a_ref * dt
        ref_pos = ref[:, 0:2] + v_ref * dt

        # heading reference: smoothed orientation target (orient_lag_tau_s), braking law
        tau = np.maximum(kn[:, 9], 1e-3)[:, None]
        th_target = target[:, 2:3]
        th_smooth = ref[:, 2:3] + wrap_angle(th_target - ref[:, 2:3]) * (dt / tau)
        th_err = wrap_angle(th_smooth - ref[:, 2:3])
        # 0.01 rad deadband, same reasoning as the position deadband.
        th_eff = np.abs(th_err) - 0.01
        w_stop = np.sqrt(2.0 * kn[:, 8][:, None] * lim.acc_max_w_radps2 * np.maximum(th_eff, 0.0))
        w_target = np.sign(th_err) * np.minimum(w_stop, lim.vel_max_w_radps)
        a_w = np.clip((w_target - ref[:, 5:6]) / dt, -lim.acc_max_w_radps2, lim.acc_max_w_radps2)
        w_ref = ref[:, 5:6] + a_w * dt
        self.ref = np.column_stack(
            [ref_pos[:, 0], ref_pos[:, 1], th_smooth[:, 0], v_ref[:, 0], v_ref[:, 1], w_ref[:, 0]]
        )

        # ---- control law: FF + P velocity + P position (world frame) ----
        v_ff = v_ref + kn[:, 2][:, None] * a_ref  # acc_ff_lead_s
        pos_err = np.clip(ref_pos - est[:, 0:2], -0.5, 0.5)
        v_cmd = v_ff + kn[:, 1][:, None] * (v_ref - est[:, 3:5]) + kn[:, 0][:, None] * pos_err
        # plant_speed_guard: confidence cap on commanded speed
        guard = kn[:, 10][:, None] * lim.vel_max_xy_mps
        speed = np.linalg.norm(v_cmd, axis=1, keepdims=True)
        v_cmd = v_cmd * np.where(speed > guard, guard / np.maximum(speed, 1e-9), 1.0)

        # heading: P heading + P yaw rate, clamped
        th_cmd_err = wrap_angle(th_smooth[:, 0] - est[:, 2])
        w_cmd = kn[:, 5] * th_cmd_err + kn[:, 6] * (w_ref[:, 0] - est[:, 5])
        w_cmd = np.clip(w_cmd, -kn[:, 7], kn[:, 7])
        w_cmd = np.clip(w_cmd, -lim.vel_max_w_radps, lim.vel_max_w_radps)

        self.last_plan = {
            "v_ref": v_ref,
            "a_ref": a_ref,
            "ref_pos": ref_pos,
            "w_ref": w_ref[:, 0],
        }
        return np.column_stack([v_cmd[:, 0], v_cmd[:, 1], w_cmd])


# ---- omni-wheel kinematics (square chassis, wheels at 45 deg; canon FR,RR,RL,FL) ----
WHEEL_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)
WHEEL_RADIUS_M = 0.09  # chassis half-extent to wheel contact
MPR = 0.2171  # meters_per_motor_rev (identified, evidence pack)
LATERAL_NOMINAL = 0.887  # body_lateral_scale (identified)
PER_WHEEL_NOMINAL = np.array([1.006, 1.008, 1.004, 1.004])

_PHI = np.deg2rad(np.array(WHEEL_ANGLES_DEG))
# A maps [vx, vy, w] (m/s, rad/s) -> per-wheel contact speeds (m/s)
A = np.column_stack([np.cos(_PHI), np.sin(_PHI), np.full(4, WHEEL_RADIUS_M)])
A_INV = np.linalg.pinv(A)


def ik(v_body: np.ndarray, lateral_believed: float = LATERAL_NOMINAL) -> np.ndarray:
    """Body twist (B, 3) -> commanded wheel speeds (B, 4) in rev/s (controller view)."""
    v = v_body.copy()
    v[:, 1] = v[:, 1] / max(lateral_believed, 1e-3)
    w_mps = v @ A.T
    w_rev = w_mps / MPR
    return w_rev / PER_WHEEL_NOMINAL


def fk(wheels_rev: np.ndarray, lateral_true: float | np.ndarray) -> np.ndarray:
    """Wheel speeds (B, 4) rev/s -> body twist (B, 3). lateral_true: the plant's
    true lateral scale (a (B,) array or scalar)."""
    w_mps = wheels_rev * MPR
    body = w_mps @ A_INV.T
    if np.ndim(lateral_true) == 0:
        body[:, 1] = body[:, 1] * float(lateral_true)
    else:
        body[:, 1] = body[:, 1] * lateral_true
    return body
