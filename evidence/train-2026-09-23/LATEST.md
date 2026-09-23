# FAWKES run train-2026-09-23

> Evidence is a run directory, never a claim. Simulation acceptance is never physical authorisation.

## Suite: practice

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0198 | 0.0625 | 0.0311 | 0.0000 | 0 | PASS |
| cem_knobs | 0.0192 | 0.0572 | 0.0284 | 0.0000 | 0 | PASS |
| rma_oracle | 0.0304 | 0.0591 | 0.0714 | 0.0000 | 0 | PASS |
| rma_adapter | 0.0306 | 0.0562 | 0.0782 | 0.0000 | 0 | PASS |
| rma_zero | 0.0332 | 0.0594 | 0.0747 | 0.0000 | 0 | PASS |

## Suite: dr family (held-out)

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0197 | 0.0673 | 0.1855 | 0.0000 | 1 | FAIL |
| cem_knobs | 0.0194 | 0.0638 | 0.1710 | 0.0000 | 0 | FAIL |
| rma_oracle | 0.0374 | 0.0623 | 0.1695 | 0.0000 | 3 | FAIL |
| rma_adapter | 0.0307 | 0.0709 | 0.2300 | 0.0000 | 4 | FAIL |
| rma_zero | 0.0328 | 0.0739 | 0.1970 | 0.0000 | 1 | FAIL |

## Gates (the FAILing condition shown for context)

| Gate | Value | Limit | Stretch | Pass |
| --- | ---: | ---: | ---: | --- |
| endpoint_p95_m | 0.01975 | 0.035 | 0.01 | PASS |
| cross_track_p95_m | 0.06251 | 0.12 | 0.06 | PASS |
| heading_max_rad | 0.03114 | 0.08 | 0.05 | PASS (stretch) |
| saturation_frac | 0.0 | 0.08 | 0.04 | PASS (stretch) |
| boundary_violations | 0 | 0 | 0 | PASS (stretch) |
| **all gates** | | | | **PASS** |

## Provenance

- generated: 2026-09-23T08:50:08+00:00
- fawkes repo: d2274ff
- data root: `C:\Users\rishi\Documents\GitHub\RoboCup-Research`
- tag: train-2026-09-23

## Honest caveats

- The champion is a hypothesis until the guarded A/B on the field; the
  simulator-vs-real gap warning (203 vs 104 mm cross-track) is standing.
- The v0.1 adapter is a closed-form linear head over engineered window
  features; the GRU lands with the PPO gradient stack.
- Pure-Python env backend at 50 Hz; the MJX/Brax backend is FK-3.
- Nothing here touched a robot. deployment_authorized stays false.
