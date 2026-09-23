"""Learner tests: CEM converges, the adapter recovers a linear world,
the zero-init base policy is exactly the pure cascade."""

from __future__ import annotations

import numpy as np

from fawkes.policies.cem import cem
from fawkes.policies.rma import BasePolicy, GRUAdapter, LinearAdapter, OBS_SCALE


def test_cem_finds_a_quadratic_optimum():
    target = np.array([0.3, -0.2, 0.5])
    bounds = np.column_stack([np.full(3, -1.0), np.full(3, 1.0)])

    def objective(v):
        return -float(np.sum((v - target) ** 2))

    best, score, _ = cem(objective, bounds, seed=1, population=40, iterations=25)
    assert np.allclose(best, target, atol=0.02)


def test_linear_adapter_recovers_a_linear_world():
    # a world whose z is a bounded linear map of the history features (the
    # linear adapter's exact hypothesis class)
    rng = np.random.default_rng(5)
    W_true = rng.normal(0, 0.05, size=(30, 6))
    b_true = rng.normal(0.5, 0.05, size=6)
    feats = rng.random((4000, 30))
    z = feats @ W_true + b_true
    assert z.min() > 0.05 and z.max() < 0.95  # no clipping in the ground truth
    adapter = LinearAdapter()
    probe = adapter.fit(feats, z, ridge=1e-9)
    pred = adapter.predict(feats)
    assert float(np.max(np.abs(pred - z))) < 1e-6
    for dim, stats in probe.items():
        assert stats["r2"] > 0.999


def test_zero_init_base_policy_is_the_pure_cascade():
    base = BasePolicy()
    obs = np.ones((3, 16))
    z = np.full((3, 6), 0.5)
    action = base.act(obs, z)
    assert np.allclose(action, 0.0)  # tanh(0): no residual until training moves it


def test_gru_adapter_forward_shapes():
    gru = GRUAdapter(in_dim=10, hidden=16, z_dim=6)
    out = gru.forward(np.random.default_rng(0).normal(0, 1, size=(4, 30, 10)))
    assert out.shape == (4, 6)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_obs_scale_matches_observation_dim():
    assert len(OBS_SCALE) == 16
