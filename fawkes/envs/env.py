"""The batched, domain-randomised movement environment (L2, pure-Python backend).

One MovementEnv holds B independent robots as numpy rows. The controller under
test is the Cascade (fawkes.envs.cascade) - the learned policy is a bounded
residual on the cascade output, never a replacement (README L2 rule: the
simulator is the plant, never the controller).

The observation is what the robot can see (estimator view, never plant truth);
the true conditioning z is exposed separately for privileged (Phase A) training
and never leaks into observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fawkes.envs.cascade import (
    BASELINE_KNOBS,
    LATERAL_NOMINAL,
    RobotLimits,
    Cascade,
    ik,
    fk,
    knobs_to_vec,
    wrap_angle,
)
from fawkes.envs.surface import DRFamily, SurfaceDraw

DT = 0.02  # 50 Hz environment tick
HISTORY_K = 25  # adapter history window (0.5 s)
RESIDUAL_SCALE = np.array([0.125, 0.125, 0.30])  # the 5 % bounded authority
RESIDUAL_SLEW = np.array([0.5, 0.5, 1.5])  # m/s^2, m/s^2, rad/s^2 (verbatim clamps)

OBS_NAMES = (
    "target_dx_body", "target_dy_body", "target_dtheta",
    "est_vx_body", "est_vy_body", "est_omega",
    "wheel_1", "wheel_2", "wheel_3", "wheel_4",
    "last_cmd_vx", "last_cmd_vy", "last_cmd_omega",
    "speed_hint", "progress", "time_left",
)
HISTORY_FEATURES = (
    "wheel_1", "wheel_2", "wheel_3", "wheel_4",
    "cmd_vx", "cmd_vy", "cmd_omega",
    "est_vx", "est_vy", "est_omega",
)


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c, -s, s, c], axis=1).reshape(-1, 2, 2)


@dataclass
class TaskSpec:
    """A per-row list of waypoints [(x, y, theta), ...]."""

    targets: list  # (B, legs, 3) array


def sample_point_to_point(rng: np.random.Generator, n: int, legs: int = 1, coupled: bool = False) -> TaskSpec:
    lim = RobotLimits()
    xh = lim.field_length_m / 2 - lim.field_margin_m - 0.15
    yh = lim.field_width_m / 2 - lim.field_margin_m - 0.15
    out = np.zeros((n, legs, 3))
    for i in range(n):
        for j in range(legs):
            dist = rng.uniform(0.3, 1.8)
            ang = rng.choice(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            out[i, j, 0] = np.clip(rng.uniform(-0.3, 0.3) + dist * np.cos(ang), -xh, xh)
            out[i, j, 1] = np.clip(rng.uniform(-0.3, 0.3) + dist * np.sin(ang), -yh, yh)
            out[i, j, 2] = (
                rng.uniform(-2.5, 2.5) if coupled and rng.random() < 0.7 else 0.0
            )
    return TaskSpec(targets=out)


class MovementEnv:
    """B robots, one DR draw per row, cascade-in-the-loop, optional residual."""

    def __init__(
        self,
        family: DRFamily | None = None,
        limits: RobotLimits | None = None,
        dt: float = DT,
        legs: int = 2,
        coupled: bool = False,
    ):
        self.family = family or DRFamily()
        self.limits = limits or RobotLimits()
        self.dt = dt
        self.legs = legs
        self.coupled = coupled
        self.cascade = Cascade(limits=self.limits)
        self._action_rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------ reset
    def reset(self, n: int, seed: int) -> np.ndarray:
        self.n = n
        self.surface: SurfaceDraw = self.family.sample(n, seed)
        self.task = sample_point_to_point(
            np.random.default_rng(seed + 777), n, legs=self.legs, coupled=self.coupled
        )
        rng = np.random.default_rng(seed + 555)
        self.pos = np.column_stack([rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)])
        self.th = rng.uniform(-np.pi, np.pi, n) if self.coupled else np.zeros(n)
        self.vel_body = np.zeros((n, 3))
        self.wheels = np.zeros((n, 4))
        # estimator state (what the policy and cascade see)
        self.est_pos = self.pos.copy()
        self.est_th = self.th.copy()
        self.est_vel = np.zeros((n, 3))  # world frame
        self.vision_hist = np.repeat(
            np.column_stack([self.pos, self.th])[:, None, :], 8, axis=1
        )
        self.cmd_queue = np.zeros((n, 4, 3))
        self.last_cmd_body = np.zeros((n, 3))
        self.residual = np.zeros((n, 3))
        self.hist = np.zeros((n, HISTORY_K, len(HISTORY_FEATURES)))
        self.t = 0.0
        self.tick = 0
        self.done = np.zeros(n, dtype=bool)
        self.leg_idx = np.zeros(n, dtype=int)
        self.leg_start = self.pos.copy()
        self.leg_start_t = np.zeros(n)
        self.leg_init_dist = np.linalg.norm(
            self.task.targets[:, 0, 0:2] - self.pos, axis=1
        )
        self.metrics = [
            {"endpoint": [], "cross_rms": [], "heading_max": [], "sat_frac": [],
             "leg_time": [], "boundary": [], "settle_s": []}
            for _ in range(n)
        ]
        self.boundary = np.zeros(n, dtype=bool)
        self.sat_ticks = np.zeros(n)
        self.cross_sq = np.zeros(n)
        self.cross_n = np.zeros(n)
        self.heading_max = np.zeros(n)
        self.cost = np.zeros(n)
        self._new_leg_targets()
        self.cascade.reset_reference(self.est_pos[:, 0], self.est_pos[:, 1], self.est_th)
        self.cascade.set_knobs_per_row(knobs_to_vec(BASELINE_KNOBS)[None, :])
        return self.observe()

    def _new_leg_targets(self) -> None:
        idx = self.leg_idx
        B = self.n
        self.target = np.zeros((B, 3))
        for i in range(B):
            j = min(idx[i], self.task.targets.shape[1] - 1)
            self.target[i] = self.task.targets[i, j]

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        dt = self.dt
        lim = self.limits
        n = self.n
        sur = self.surface

        est = np.column_stack(
            [self.est_pos, self.est_th, self.est_vel[:, 0], self.est_vel[:, 1], self.est_vel[:, 2]]
        )
        v_cmd_world = self.cascade.step(est, self.target, dt)

        # bounded residual (the RMA policy path)
        if actions is not None:
            res = np.asarray(actions, dtype=np.float64) * RESIDUAL_SCALE
            dres = np.clip(res - self.residual, -RESIDUAL_SLEW * dt, RESIDUAL_SLEW * dt)
            self.residual = np.where(
                self.done[:, None], 0.0, self.residual + dres
            )
            R = _rot(self.est_th)
            res_world = np.column_stack(
                [
                    np.einsum("bij,bj->bi", R, self.residual[:, 0:2]),
                    self.residual[:, 2],
                ]
            )
            v_cmd_world = v_cmd_world + res_world

        # command latency (per-row queue, vectorised gather)
        self.cmd_queue = np.concatenate([v_cmd_world[:, None, :], self.cmd_queue[:, :-1, :]], axis=1)
        lat = np.clip(sur.cmd_latency.astype(int), 0, 3)
        rows = np.arange(n)
        v_cmd_world = self.cmd_queue[rows, lat]

        # inverse kinematics from the ESTIMATED heading (controller's view)
        R = _rot(self.est_th)
        R_T = np.transpose(R, (0, 2, 1))
        v_body_cmd = np.column_stack(
            [
                np.einsum("bij,bj->bi", R_T, v_cmd_world[:, 0:2]),
                v_cmd_world[:, 2],
            ]
        )
        wheels_cmd = ik(v_body_cmd, lateral_believed=LATERAL_NOMINAL)

        # plant: per-wheel gains, battery sag, first-order lag, slew, saturation
        alpha = np.clip(dt / sur.motor_tau_s, 0.0, 1.0)[:, None]
        w_target = wheels_cmd * sur.wheel_gain * sur.battery_sag[:, None]
        w_next = self.wheels + alpha * (w_target - self.wheels)
        w_next = np.clip(w_next, self.wheels - lim.wheel_slew_rev_s2 * dt, self.wheels + lim.wheel_slew_rev_s2 * dt)
        w_next = np.clip(w_next, -lim.wheel_max_rev_s, lim.wheel_max_rev_s)
        self.sat_ticks += np.any(np.abs(w_next) > 0.98 * lim.wheel_max_rev_s, axis=1)
        self.wheels = w_next

        # forward kinematics through the TRUE lateral scale, traction-limited body
        v_fk = fk(self.wheels, sur.lateral_scale)
        dv = np.clip(v_fk - self.vel_body, -(lim.acc_max_xy_mps2 * sur.traction)[:, None] * dt, (lim.acc_max_xy_mps2 * sur.traction)[:, None] * dt)
        self.vel_body[:, 0:2] += dv[:, 0:2]
        self.vel_body[:, 2] = v_fk[:, 2]

        # external disturbance
        p = sur.disturbance_p
        hit = self._action_rng.random(n) < p
        if np.any(hit):
            ang = self._action_rng.uniform(-np.pi, np.pi, n)
            imp = np.column_stack([0.3 * np.cos(ang), 0.3 * np.sin(ang), np.zeros(n)])
            self.vel_body[:, 0:2] += np.where(hit[:, None], imp[:, 0:2], 0.0)

        # integrate
        R = _rot(self.th)
        v_world = np.concatenate(
            [np.einsum("bij,bj->bi", R, self.vel_body[:, 0:2]), self.vel_body[:, 2:3]], axis=1
        )
        self.pos = self.pos + v_world[:, 0:2] * dt
        self.th = wrap_angle(self.th + self.vel_body[:, 2] * dt)
        self.t += dt
        self.tick += 1

        # vision: delayed, noisy pose -> estimator (vectorised gather)
        self.vision_hist = np.concatenate(
            [np.column_stack([self.pos, self.th])[:, None, :], self.vision_hist[:, :-1, :]], axis=1
        )
        vlat = np.clip(sur.vision_latency.astype(int), 0, 7)
        delayed = self.vision_hist[rows, vlat]  # (B, 3)
        self.est_pos = delayed[:, 0:2] + self._action_rng.normal(
            0.0, sur.vision_noise_m[:, None], (n, 2)
        )
        self.est_th = delayed[:, 2] + self._action_rng.normal(0.0, sur.vision_noise_m / 20.0)
        est_body_vel = np.concatenate(
            [np.einsum("bij,bj->bi", np.transpose(_rot(self.est_th), (0, 2, 1)), self.vel_body[:, 0:2]), self.vel_body[:, 2:3]], axis=1
        )
        self.est_vel = 0.5 * self.est_vel + 0.5 * np.concatenate(
            [np.einsum("bij,bj->bi", _rot(self.est_th), self.vel_body[:, 0:2]), self.vel_body[:, 2:3]], axis=1
        )
        self.last_cmd_body = v_body_cmd

        # history buffer for the adapter
        hrow = np.concatenate([self.wheels, v_body_cmd, est_body_vel], axis=1)
        self.hist = np.concatenate([self.hist[:, 1:, :], hrow[:, None, :]], axis=1)

        self._accumulate_leg_metrics(v_world)
        obs = self.observe()
        info = {"v_cmd_world": v_cmd_world, "est_body_vel": est_body_vel}
        return obs, -self.cost / 100.0, self.done.copy(), info

    # ------------------------------------------------------------- bookkeeping
    def _accumulate_leg_metrics(self, v_world) -> None:
        n = self.n
        lim = self.limits
        d_vec = self.target[:, 0:2] - self.pos
        dist = np.linalg.norm(d_vec, axis=1)
        speed = np.linalg.norm(v_world[:, 0:2], axis=1)
        # cross-track: distance to the start->target segment
        a = self.leg_start
        b = self.target[:, 0:2]
        ab = b - a
        L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
        tpar = np.clip(np.sum((self.pos - a) * ab, axis=1) / L2, 0.0, 1.0)
        proj = a + tpar[:, None] * ab
        cross = np.linalg.norm(self.pos - proj, axis=1)
        active = ~self.done
        self.cross_sq += np.where(active, cross**2, 0.0)
        self.cross_n += active
        h_err = np.abs(wrap_angle(self.target[:, 2] - self.th))
        self.heading_max = np.where(active & (speed > 0.05), np.maximum(self.heading_max, h_err), self.heading_max)

        # boundary
        xh = lim.field_length_m / 2 - lim.field_margin_m
        yh = lim.field_width_m / 2 - lim.field_margin_m
        out = (np.abs(self.pos[:, 0]) > xh) | (np.abs(self.pos[:, 1]) > yh)
        self.boundary |= out & active

        # settle / timeout / leg advance
        settled = (dist < 0.02) & (speed < 0.05)
        t_max = 2.2 * np.sqrt(np.maximum(self.leg_init_dist, 0.1) / lim.acc_max_xy_mps2) + 1.2
        leg_done = active & (settled | (self.t - self.leg_start_t > t_max) | out)
        if np.any(leg_done):
            for i in np.nonzero(leg_done)[0]:
                self.metrics[i]["endpoint"].append(float(dist[i]))
                self.metrics[i]["cross_rms"].append(float(np.sqrt(self.cross_sq[i] / max(self.cross_n[i], 1))))
                self.metrics[i]["heading_max"].append(float(self.heading_max[i]))
                self.metrics[i]["sat_frac"].append(float(self.sat_ticks[i] / max(self.tick, 1)))
                self.metrics[i]["leg_time"].append(float(self.t - self.leg_start_t[i]))
                self.metrics[i]["boundary"].append(bool(self.boundary[i]))
                self.metrics[i]["settle_s"].append(float(self.t - self.leg_start_t[i]))
                self.cost[i] += (
                    1000.0 * dist[i]
                    + 400.0 * np.sqrt(self.cross_sq[i] / max(self.cross_n[i], 1))
                    + 20.0 * self.heading_max[i]
                    + 50.0 * self.sat_ticks[i] / max(self.tick, 1)
                    + 2.0 * (self.t - self.leg_start_t[i])
                    + 500.0 * float(self.boundary[i])
                )
            self.leg_idx += leg_done
            done_all = self.leg_idx >= self.task.targets.shape[1]
            self.done |= done_all
            fresh = leg_done & ~self.done
            if np.any(fresh):
                self.leg_start[fresh] = self.pos[fresh]
                self.leg_start_t[fresh] = self.t
                self.leg_init_dist[fresh] = np.linalg.norm(
                    self.target[fresh, 0:2] - self.pos[fresh], axis=1
                )
                for i in np.nonzero(fresh)[0]:
                    self.sat_ticks[i] = 0
                self._new_leg_targets()
                self._reset_refs_per_row(fresh)
            # zero per-leg accumulators for continuing rows
            self.cross_sq[fresh] = 0.0
            self.cross_n[fresh] = 0.0
            self.heading_max[fresh] = 0.0

    def _reset_refs_per_row(self, rows) -> None:
        ref = self.cascade.ref
        ref[rows, 0] = self.est_pos[rows, 0]
        ref[rows, 1] = self.est_pos[rows, 1]
        ref[rows, 2] = self.est_th[rows]
        ref[rows, 3:6] = 0.0

    # ------------------------------------------------------------- observation
    def observe(self) -> np.ndarray:
        R = _rot(self.est_th)
        d_world = self.target[:, 0:2] - self.est_pos
        d_body = np.einsum("bij,bj->bi", np.transpose(R, (0, 2, 1)), d_world)
        dth = wrap_angle(self.target[:, 2] - self.est_th)
        est_body = np.concatenate(
            [np.einsum("bij,bj->bi", np.transpose(R, (0, 2, 1)), self.est_vel[:, 0:2]), self.est_vel[:, 2:3]], axis=1
        )
        plan = getattr(self.cascade, "last_plan", None)
        speed_hint = (
            np.linalg.norm(plan["v_ref"], axis=1) / self.limits.vel_max_xy_mps if plan is not None else np.zeros(self.n)
        )
        init = np.maximum(self.leg_init_dist, 1e-6)
        dist = np.linalg.norm(self.target[:, 0:2] - self.pos, axis=1)
        progress = np.clip(1.0 - dist / init, 0.0, 1.0)
        time_left = 1.0 - (self.t - self.leg_start_t) / 4.0
        return np.column_stack(
            [d_body, dth, est_body, self.wheels, self.last_cmd_body, speed_hint, progress, time_left]
        )

    def z(self) -> np.ndarray:
        return self.surface.z

    def history_features(self) -> np.ndarray:
        """Engineered adapter features (B, 30): window mean/std + recent mean."""
        h = self.hist  # (B, K, 10)
        mean = h.mean(axis=1)
        std = h.std(axis=1)
        recent = h[:, -5:, :].mean(axis=1)
        return np.concatenate([mean, std, recent], axis=1)


ADAPTER_FEATURE_NAMES = tuple(
    [f"{n}_mean" for n in HISTORY_FEATURES]
    + [f"{n}_std" for n in HISTORY_FEATURES]
    + [f"{n}_recent" for n in HISTORY_FEATURES]
)


def run_episode(
    env: MovementEnv,
    n: int,
    seed: int,
    knobs: np.ndarray | None = None,
    policy=None,
    z_mode: str = "zero",
    adapter=None,
    max_ticks: int = 600,
):
    """Run one episode; returns (cost, per-row metrics). z_mode: oracle|adapter|zero."""
    obs = env.reset(n, seed)
    if knobs is not None:
        env.cascade.set_knobs_per_row(knobs)
    if policy is not None:
        policy.reset()
    for _ in range(max_ticks):
        actions = None
        if policy is not None:
            if z_mode == "oracle":
                z = env.z()
            elif z_mode == "adapter":
                z = adapter.predict(env.history_features())
            else:
                z = np.zeros((n, adapter.z_dim if adapter else 6))
            actions = policy.act(obs, z)
        obs, _, done, _ = env.step(actions)
        if np.all(done):
            break
    return env.cost, env.metrics
