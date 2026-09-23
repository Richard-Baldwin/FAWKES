"""Environment tests: the cascade must hit the house gates on the nominal
world; the DR family must stay in bounds; the residual must stay bounded."""

from __future__ import annotations

import numpy as np

from fawkes.envs.cascade import BASELINE_KNOBS, knobs_to_vec
from fawkes.envs.env import MovementEnv
from fawkes.envs.surface import DRFamily
from fawkes.evaluate.gates import summarize_metrics


class NominalFamily:
    def sample(self, n, seed):
        return DRFamily().nominal(n)

    def family_id(self):
        return "nominal"


def _run(env, seed=1, legs=1):
    env.reset(8, seed)
    env.cascade.set_knobs_per_row(np.tile(knobs_to_vec(BASELINE_KNOBS), (8, 1)))
    for _ in range(600):
        _, _, done, _ = env.step(None)
        if np.all(done):
            break
    return summarize_metrics(env.metrics)


def test_cascade_meets_house_gates_on_nominal_world():
    env = MovementEnv(family=NominalFamily(), legs=1)
    s = _run(env)
    assert s["endpoint_p95_m"] <= 0.035, s
    assert s["cross_track_p95_m"] <= 0.120, s
    assert s["boundary_violations"] == 0
    assert s["saturation_frac"] <= 0.08


def test_dr_family_z_in_bounds_and_reproducible():
    fam = DRFamily()
    a = fam.sample(64, seed=5)
    b = fam.sample(64, seed=5)
    assert np.allclose(a.z, b.z)
    assert a.z.min() >= 0.0 and a.z.max() <= 1.0
    c = fam.sample(64, seed=6)
    assert not np.allclose(a.z, c.z)


def test_residual_stays_within_the_bounded_authority():
    env = MovementEnv(family=DRFamily(), legs=1)
    env.reset(4, seed=2)
    for _ in range(300):
        if np.all(env.done):
            break
        env.step(np.ones((4, 3)))  # full positive residual command
    # the residual ramps at the slew limit toward (and never past) 5 % authority
    assert np.all(env.residual[:, 0] <= 0.125 + 1e-9)
    assert np.all(env.residual[:, 0] >= 0.0)
    assert env.residual[:, 0].max() > 0.0
    assert np.all(env.residual[:, 2] <= 0.30 + 1e-9)


def test_observation_shape_and_names():
    from fawkes.envs.env import OBS_NAMES

    env = MovementEnv(family=DRFamily(), legs=1)
    obs = env.reset(8, seed=3)
    assert obs.shape == (8, 16)
    assert len(OBS_NAMES) == 16
    z = env.z()
    assert z.shape == (8, 6)
    feats = env.history_features()
    assert feats.shape[1] == 30


def test_cost_accumulates_and_legs_advance():
    env = MovementEnv(family=NominalFamily(), legs=2)
    env.reset(4, seed=4)
    for _ in range(600):
        _, _, done, _ = env.step(None)
        if np.all(done):
            break
    assert np.all(env.leg_idx >= 2)
    assert all(len(m["endpoint"]) == 2 for m in env.metrics)
