"""The pure-stdlib Phoenix-side consumer for movement-rma-1 bundles.

This module is the artifact Phoenix Server would host: NO third-party imports,
no numpy - plain Python 3.11+ (the Phoenix zero-dependency promise holds). It
implements the full forward pass: adapter (linear) -> z, then the base MLP
(obs, z) -> bounded residual action.

Used by fawkes.export.phoenix_bundle.verify_bundle to prove the deployed
implementation reproduces the training implementation exactly.
"""

from __future__ import annotations

import json
import math


def _tanh(x: float) -> float:
    return math.tanh(x)


def consumer_forward(bundle: dict, observation: list[float], adapter_features: list[float]) -> list[float]:
    """One tick: observation (16) + adapter features (30) -> action (3)."""
    policy = bundle["policy"]
    # adapter: z = clip(W^T f + b, 0, 1)  (W stored transposed: z_dim x feat_dim)
    W = policy["adapter"]["W"]
    b = policy["adapter"]["b"]
    z = [
        min(max(sum(w * f for w, f in zip(row, adapter_features)) + bias, 0.0), 1.0)
        for row, bias in zip(W, b)
    ]
    # base MLP: obs / obs_scale ++ z -> sizes stack, tanh everywhere, tanh output
    mlp = policy["base_mlp"]
    scale = mlp["obs_scale"]
    x = [o / s for o, s in zip(observation, scale)] + z
    params = mlp["params"]
    sizes = mlp["sizes"]
    pos = 0
    h = x
    for li in range(len(sizes) - 1):
        f_in, f_out = sizes[li], sizes[li + 1]
        # weights are stored row-major as (in, out), matching the numpy MLP
        Wl = [[params[pos + i * f_out + j] for j in range(f_out)] for i in range(f_in)]
        pos += f_in * f_out
        bias = params[pos : pos + f_out]
        pos += f_out
        if li < len(sizes) - 2:
            h = [
                _tanh(sum(Wl[i][j] * h[i] for i in range(f_in)) + bias[j])
                for j in range(f_out)
            ]
        else:
            h = [sum(Wl[i][j] * h[i] for i in range(f_in)) + bias[j] for j in range(f_out)]
    return [_tanh(v) for v in h]


def load_bundle(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify(bundle_path: str) -> dict:
    bundle = load_bundle(bundle_path)
    worst = 0.0
    for pair in bundle["verification"]:
        action = consumer_forward(bundle, pair["observation"], pair["adapter_features"])
        worst = max(worst, max(abs(a - e) for a, e in zip(action, pair["expected_action"])))
    return {"pairs": len(bundle["verification"]), "max_abs_error": worst, "ok": worst < 1e-9}


if __name__ == "__main__":
    import sys

    print(json.dumps(verify(sys.argv[1]), indent=2))
