"""System-identification tests: the fit must recover a known synthetic plant."""

from __future__ import annotations

import numpy as np

from fawkes.sysid.wheel_model import FEATURES, FirstOrderFit, WheelEnsemble

ALPHA_TRUE = 0.45
GAINS_TRUE = np.array([1.01, 0.99, 1.02, 0.98])


def _synthetic_transitions(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    cmd = rng.uniform(-30, 30, size=(n, 4))
    meas = np.zeros((n, 4))
    y = np.zeros((n, 4))
    w = np.zeros(4)
    for t in range(n):
        meas[t] = w
        delta = ALPHA_TRUE * (GAINS_TRUE * cmd[t] - w) + rng.normal(0, 0.002, 4)
        y[t] = delta
        w = w + delta
    X = np.zeros((n, len(FEATURES)))
    X[:, 0:4] = meas
    X[:, 4:8] = cmd
    X[:, 8:12] = np.vstack([np.zeros(4), cmd[:-1]])
    X[:, 12:15] = rng.normal(0, 0.3, size=(n, 3))
    X[:, 15:17] = rng.normal(0, 0.3, size=(n, 2))
    X[:, 17:21] = np.abs(cmd)
    X[:, 21:25] = np.sign(cmd)
    X[:, 25:29] = cmd - meas
    X[:, 29] = 0.02
    return X, y


def test_first_order_fit_recovers_alpha_and_gains():
    X, _ = _synthetic_transitions()
    fit = FirstOrderFit.fit(X[:, 0:4], X[:, 4:8], dt=0.02)
    assert np.allclose(fit.alpha, ALPHA_TRUE, atol=0.05)
    assert np.allclose(fit.gain, GAINS_TRUE, atol=0.03)


def test_ensemble_predicts_and_reports_uncertainty():
    X, y = _synthetic_transitions()
    ens = WheelEnsemble.fit(X, y, np.ones(len(X)), n_members=5, seed=1)
    mean, std = ens.predict(X)
    err = mean - y
    assert float(np.sqrt(np.mean(err**2))) < 0.02  # near-noise
    assert float(std.mean()) > 0.0  # epistemic uncertainty present
    ev = ens.evaluate(X, y)
    assert max(ev["rmse_all"]) < 0.02
    assert ev["active_rows"] > 0


def test_model_card_names_the_reference_honestly():
    X, y = _synthetic_transitions()
    ens = WheelEnsemble.fit(X, y, np.ones(len(X)), n_members=3, seed=2)
    card = ens.model_card(ens.evaluate(X, y))
    assert card["validation"]["rlearn_reference"]["rmse_all"]  # the comparison is stated
    assert any("203-vs-104" in lim for lim in card["limitations"])
