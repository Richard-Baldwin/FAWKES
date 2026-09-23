"""The RMA stack: base policy + adaptation module (README section 2.2).

  BasePolicy   a small MLP: pi(u | obs, z) - the bounded residual command.
  Adapter      the hidden-vector module: watches (observation, command) history
               and writes z, the conditioning code the controller receives.
               v0.1 ships the closed-form LINEAR adapter (ridge over engineered
               window features); the GRU cell is implemented forward-only and is
               the FK-5 refinement once the gradient stack (PPO) lands.
  z_probe     the diagnostic readout: how well does z_hat predict the true
               world parameters (the console's "what the robot thinks the
               carpet is doing" - and the drift alarm).

Phase A trains the base policy with the TRUE z (privileged). Phase B freezes
the base and fits the adapter to reproduce that z from observable history only
(supervised, closed form). At deployment there is no privileged information:
the robot comes in blind and z converges within ~1 s of motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBS_SCALE = np.array([2.0, 2.0, 3.15, 2.5, 2.5, 6.0, 45.0, 45.0, 45.0, 45.0, 2.5, 2.5, 6.0, 1.0, 1.0, 4.0])


def _tanh(x):
    return np.tanh(x)


class MLP:
    """A tanh MLP, batched; parameter vector <-> weights for CEM."""

    def __init__(self, sizes: list[int], out_gain: np.ndarray | None = None):
        self.sizes = sizes
        self.out_gain = out_gain
        self.shapes = []
        self.n_params = 0
        for i in range(len(sizes) - 1):
            self.shapes.append((sizes[i], sizes[i + 1]))
            self.n_params += sizes[i] * sizes[i + 1] + sizes[i + 1]
        self.params = None

    def set_flat(self, vec: np.ndarray) -> None:
        self.params = np.asarray(vec, dtype=np.float64)

    def zero_init(self, rng: np.random.Generator) -> np.ndarray:
        """Random hidden layers, ZERO output layer: the policy starts as the
        pure cascade (residual zero) - a deliberate identity-start so CEM only
        accepts changes that beat the trusted controller."""
        vec = np.concatenate([rng.normal(0, 0.1, self.n_params - self._out_size()), np.zeros(self._out_size())])
        return vec

    def _out_size(self) -> int:
        f, b = self.shapes[-1]
        return f * b + b

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        p = self.params
        for li, (f, b) in enumerate(self.shapes):
            W = p[: f * b].reshape(f, b)
            p = p[f * b:]
            bias = p[:b]
            p = p[b:]
            h = _tanh(h @ W + bias) if li < len(self.shapes) - 1 else h @ W + bias
        return h


class BasePolicy:
    """pi(residual_command | obs, z): MLP -> tanh -> [-1, 1]^3."""

    def __init__(self, obs_dim: int = 16, z_dim: int = 6, hidden: int = 24):
        self.obs_dim, self.z_dim = obs_dim, z_dim
        self.net = MLP([obs_dim + z_dim, hidden, hidden, 3])
        self.params = self.net.zero_init(np.random.default_rng(0))
        self.net.set_flat(self.params)

    def act(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.concatenate([obs / OBS_SCALE[None, :], z], axis=1)
        return np.tanh(self.net.forward(x))

    def set_params(self, vec: np.ndarray) -> None:
        self.params = np.asarray(vec, dtype=np.float64)
        self.net.set_flat(self.params)

    def reset(self) -> None:
        pass


class LinearAdapter:
    """Closed-form ridge adapter: engineered history features -> z.

    Honest scope: linear in the engineered window statistics. This is enough to
    infer first-order properties (effective gains, lag, noise) and it trains
    from a handful of rollouts in milliseconds; the GRU lands with PPO (FK-4/5).
    """

    z_dim = 6

    def __init__(self, W: np.ndarray | None = None, b: np.ndarray | None = None, feature_names=None):
        self.W = W
        self.b = b
        self.feature_names = list(feature_names) if feature_names is not None else None

    def fit(self, feats: np.ndarray, z_true: np.ndarray, ridge: float = 1e-2) -> dict:
        A = np.column_stack([feats, np.ones(len(feats))])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        sol = np.linalg.solve(G, A.T @ z_true)
        self.W = sol[:-1]
        self.b = sol[-1]
        pred = self.predict(feats)
        return self.probe(pred, z_true)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("adapter not fitted")
        return np.clip(feats @ self.W + self.b, 0.0, 1.0)

    def probe(self, z_pred: np.ndarray, z_true: np.ndarray) -> dict:
        """Per-dimension R^2 and correlation: the diagnostic readout."""
        out = {}
        for i in range(z_true.shape[1]):
            e = z_true[:, i] - z_pred[:, i]
            var = np.var(z_true[:, i])
            r2 = 1.0 - np.mean(e**2) / var if var > 1e-9 else float("nan")
            if np.std(z_pred[:, i]) > 1e-9 and np.std(z_true[:, i]) > 1e-9:
                corr = float(np.corrcoef(z_pred[:, i], z_true[:, i])[0, 1])
            else:
                corr = float("nan")
            out[f"dim_{i}"] = {"r2": round(float(r2), 4), "corr": round(corr, 4)}
        return out


class GRUAdapter:
    """Forward-only GRU cell over (obs, cmd) history -> z (the FK-5 slot)."""

    def __init__(self, in_dim: int, hidden: int = 24, z_dim: int = 6):
        self.in_dim, self.hidden, self.z_dim = in_dim, hidden, z_dim
        self.Wz = np.random.default_rng(0).normal(0, 0.1, (in_dim + hidden, hidden))
        self.Wr = np.random.default_rng(1).normal(0, 0.1, (in_dim + hidden, hidden))
        self.Wh = np.random.default_rng(2).normal(0, 0.1, (in_dim + hidden, hidden))
        self.bz = np.zeros(hidden)
        self.br = np.zeros(hidden)
        self.bh = np.zeros(hidden)
        self.Wp = np.random.default_rng(3).normal(0, 0.1, (hidden, z_dim))
        self.bp = np.zeros(z_dim)

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        hx = np.concatenate([h, x], axis=1)
        z = _tanh(hx @ self.Wz + self.bz)
        r = _tanh(hx @ self.Wr + self.br)
        hh = np.tanh((np.concatenate([r * h, x], axis=1)) @ self.Wh + self.bh)
        return (1 - z) * h + z * hh

    def forward(self, seq: np.ndarray) -> np.ndarray:
        """seq: (B, T, in_dim) -> z (B, z_dim) from the final hidden state."""
        B = seq.shape[0]
        h = np.zeros((B, self.hidden))
        for t in range(seq.shape[1]):
            h = self.step(seq[:, t, :], h)
        return np.clip(h @ self.Wp + self.bp, 0.0, 1.0)


def save_adapter(path: str | Path, adapter: LinearAdapter, z_names) -> dict:
    doc = {
        "kind": "fawkes-linear-adapter-1",
        "W": adapter.W.tolist(),
        "b": adapter.b.tolist(),
        "feature_names": adapter.feature_names,
        "z_names": list(z_names),
    }
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return doc


def load_adapter(path: str | Path) -> LinearAdapter:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    a = LinearAdapter(W=np.array(doc["W"]), b=np.array(doc["b"]), feature_names=doc["feature_names"])
    a.z_names = doc["z_names"]
    return a
