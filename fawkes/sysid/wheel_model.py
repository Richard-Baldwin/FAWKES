"""The wheel model: physically anchored first-order response + ridge residual ensemble.

Heritage: testing/rlearn (README A.3) - a weighted first-order wheel response
fitted independently per wheel, plus a residual ensemble whose disagreement is
epistemic uncertainty, chronological hold-out, source weighting carried by the
dataset's sample_weight column. This port keeps the structure and reduces the
feature engineering to the physically transferable subset of rlearn's 47
features; the model card this module emits states the gap honestly.

Model per wheel i (rev/s, dt in seconds):
    w[t+1] = w[t] + alpha * (g_i * cmd - w[t]) + residual_i(x[t])
    alpha   = dt / tau      (the identified motor lag; Robot A: ~1 sample @ 50 Hz)
    g_i     = per-wheel command gain (asymmetry across wheels and direction)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The physically transferable subset of rlearn's 47 features, by name.
FEATURES = (
    [f"meas_w{i}" for i in range(1, 5)]
    + [f"cmd_w{i}" for i in range(1, 5)]
    + [f"prev_cmd_w{i}" for i in range(1, 5)]
    + ["cmd_body_vx", "cmd_body_vy", "cmd_body_w", "ref_vx", "ref_vy"]
    + [f"abs_cmd_w{i}" for i in range(1, 5)]
    + [f"sign_cmd_w{i}" for i in range(1, 5)]
    + [f"tracking_err_w{i}" for i in range(1, 5)]
    + ["dt"]
)

# rlearn's own held-out numbers (reports/LATEST.md), for the honest comparison.
RLEARN_REFERENCE_RMSE_ALL = [0.0798, 0.0494, 0.0526, 0.0445]
RLEARN_REFERENCE_RMSE_ACTIVE = [0.2113, 0.1296, 0.1356, 0.1151]


def select_features(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map the rlearn dataset onto the FAWKES feature set. X (N, F), y (N, 4)."""
    names = [str(n) for n in data["feature_names"]]
    index = {n: i for i, n in enumerate(names)}
    missing = [f for f in FEATURES if f not in index]
    if missing:
        raise KeyError(f"feature names missing from dataset: {missing}")
    X = data["X"][:, [index[f] for f in FEATURES]]
    y = data["y"]
    return X, y


@dataclass
class FirstOrderFit:
    """Per-wheel first-order response: alpha (response rate per tick) and gain."""

    alpha: np.ndarray  # (4,) 0..1 per 50 Hz tick
    gain: np.ndarray  # (4,) unitless command gain
    tau_s: np.ndarray  # (4,) seconds

    @classmethod
    def fit(cls, meas: np.ndarray, cmd: np.ndarray, dt: float) -> "FirstOrderFit":
        """Least squares on delta_t = a*cmd_t + b*w_t  =>  alpha = -b, gain = -a/b.

        Pairing matters: the delta w[t+1]-w[t] belongs with the command at t.
        """
        alpha = np.zeros(4)
        gain = np.ones(4)
        for i in range(4):
            w = meas[:, i]
            c = cmd[:, i]
            delta = w[1:] - w[:-1]
            A = np.column_stack([c[:-1], w[:-1]])
            try:
                coef, *_ = np.linalg.lstsq(A, delta, rcond=None)
            except np.linalg.LinAlgError:
                coef = np.zeros(2)
            a, b = float(coef[0]), float(coef[1])
            al = min(max(-b, 1e-3), 1.0)
            g = (a / al) if al > 0 else 1.0
            alpha[i] = al
            gain[i] = g if 0.5 < g < 2.0 else 1.0
        return cls(alpha=alpha, gain=gain, tau_s=alpha * dt)

    def apply(self, meas: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        """First-order prediction of the next-tick wheel DELTA (N, 4),
        matching the dataset target (next-tick measured-wheel velocity change)."""
        return self.alpha * (self.gain * cmd - meas)


class WheelEnsemble:
    """First-order response + ridge residual ensemble with uncertainty."""

    def __init__(
        self,
        first_order: FirstOrderFit,
        members: list[np.ndarray],  # ridge weight matrices (F+1, 4) each
        feature_names: list[str],
        dt_s: float,
        meta: dict | None = None,
    ):
        self.first_order = first_order
        self.members = members
        self.feature_names = list(feature_names)
        self.dt_s = dt_s
        self.meta = meta or {}

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        n_members: int = 9,
        ridge_lambda: float = 1e-3,
        seed: int = 0,
        dt_s: float = 0.02,
    ) -> "WheelEnsemble":
        """Fit the ensemble: first-order core on all rows, ridge members on
        weighted bootstrap resamples (source weighting rides sample_weight)."""
        first_order = FirstOrderFit.fit(X[:, 0:4], X[:, 4:8], dt_s)
        base = first_order.apply(X[:, 0:4], X[:, 4:8])
        residual = y - base

        A = np.column_stack([X, np.ones(len(X))])
        w = np.maximum(sample_weight, 1e-6)
        rng = np.random.default_rng(seed)
        members = []
        n = len(A)
        for _ in range(n_members):
            idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
            Ai, ri, wi = A[idx], residual[idx], w[idx]
            G = Ai.T @ (Ai * wi[:, None]) + ridge_lambda * np.eye(A.shape[1])
            Wm = np.linalg.solve(G, Ai.T @ (ri * wi[:, None]))
            members.append(Wm)
        return cls(
            first_order,
            members,
            FEATURES,
            dt_s,
            meta={"n_members": n_members, "ridge_lambda": ridge_lambda, "seed": seed},
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std (epistemic) of next-tick wheel speeds (N, 4)."""
        base = self.first_order.apply(X[:, 0:4], X[:, 4:8])
        A = np.column_stack([X, np.ones(len(X))])
        preds = np.stack([base + A @ Wm for Wm in self.members])
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray, active_cmd_rev_s: float = 1.0) -> dict:
        """Held-out delta RMSE per wheel, all rows and active rows (rlearn's split)."""
        mean, _ = self.predict(X)
        err = mean - y
        active = np.max(np.abs(X[:, 4:8]), axis=1) > active_cmd_rev_s

        def rmse(e):
            return [float(v) for v in np.sqrt(np.mean(e**2, axis=0))]

        return {
            "rmse_all": [round(v, 4) for v in rmse(err)],
            "rmse_active": [round(v, 4) for v in rmse(err[active])],
            "active_rows": int(active.sum()),
            "rows": int(len(X)),
        }

    def model_card(self, heldout: dict) -> dict:
        return {
            "schema": 1,
            "kind": "fawkes-wheel-ensemble-1",
            "intended_use": "Offline plant model for the DR environment; replay validation against held-out real traces.",
            "inputs": {
                "features": self.feature_names,
                "count": len(self.feature_names),
                "note": "The physically transferable subset of rlearn's 47 features.",
            },
            "outputs": ["next_wheel_1", "next_wheel_2", "next_wheel_3", "next_wheel_4"],
            "architecture": {
                "core": "first-order wheel response per wheel (alpha, gain)",
                "residual": f"{len(self.members)} ridge members on weighted bootstrap resamples",
                "uncertainty": "ensemble std (epistemic)",
                "first_order": {
                    "alpha_per_tick": [round(v, 4) for v in self.first_order.alpha],
                    "gain": [round(v, 4) for v in self.first_order.gain],
                    "tau_ms": [round(v * 1000, 1) for v in self.first_order.tau_s],
                },
            },
            "validation": {
                "held_out": heldout,
                "rlearn_reference": {
                    "rmse_all": RLEARN_REFERENCE_RMSE_ALL,
                    "rmse_active": RLEARN_REFERENCE_RMSE_ACTIVE,
                    "note": "rlearn used all 47 features; FAWKES v0.1 uses the "
                    "transferable subset, so a gap here is expected and reported, "
                    "not hidden (README FK-2).",
                },
            },
            "limitations": [
                "One robot, one carpet, one day (2026-07-23 session).",
                "Battery condition inferred, not measured per row.",
                "Simulator overpredicted field cross-track (rlearn's 203-vs-104 mm "
                "warning): replay-vs-real is a standing gate, never skipped.",
            ],
            "provenance": self.meta,
        }

    def save(self, path) -> None:
        np.savez(
            path,
            alpha=self.first_order.alpha,
            gain=self.first_order.gain,
            dt=self.dt_s,
            members=np.stack(self.members),
            feature_names=np.array(self.feature_names),
        )

    @classmethod
    def load(cls, path) -> "WheelEnsemble":
        with np.load(path) as z:
            fo = FirstOrderFit(alpha=z["alpha"], gain=z["gain"], tau_s=z["alpha"] * float(z["dt"]))
            return cls(fo, [m for m in z["members"]], [str(n) for n in z["feature_names"]], float(z["dt"]))
