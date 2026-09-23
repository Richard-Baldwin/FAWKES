# FAWKES smoke run smoke-2026-09-23

> Evidence is a run directory, never a claim. Simulation acceptance is never physical authorisation.

## Held-out conditions (seeds >= 9000, never trained on)

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.2328 | 0.0703 | 0.1892 | 0.0000 | 0 | FAIL |
| cem_knobs | 0.2009 | 0.0800 | 0.2469 | 0.0000 | 5 | FAIL |
| rma_oracle | 0.2708 | 0.0759 | 0.1906 | 0.0000 | 0 | FAIL |
| rma_adapter | 0.2470 | 0.0706 | 0.1898 | 0.0000 | 0 | FAIL |
| rma_zero | 0.2314 | 0.0705 | 0.1910 | 0.0000 | 0 | FAIL |

## Gates (the FAILing condition shown for context)

| Gate | Value | Limit | Stretch | Pass |
| --- | ---: | ---: | ---: | --- |
| endpoint_p95_m | 0.23275 | 0.035 | 0.01 | FAIL |
| cross_track_p95_m | 0.07031 | 0.12 | 0.06 | PASS |
| heading_max_rad | 0.18923 | 0.08 | 0.05 | FAIL |
| saturation_frac | 0.0 | 0.08 | 0.04 | PASS (stretch) |
| boundary_violations | 0 | 0 | 0 | PASS (stretch) |
| **all gates** | | | | **FAIL** |

## Provenance

- generated: 2026-09-23T07:30:39+00:00
- fawkes repo: 52a7a4e
- data root: `C:\Users\rishi\Documents\GitHub\RoboCup-Research`
- tag: smoke-2026-09-23

## Honest caveats

- v0.1 smoke budgets: small populations/iterations; a real run is FK-2..FK-5.
- The v0.1 adapter is a closed-form linear head over engineered window
  features; the GRU lands with the PPO gradient stack.
- Pure-Python env backend at 50 Hz; the MJX/Brax backend is FK-3.
- Nothing here touched a robot. deployment_authorized stays false.
