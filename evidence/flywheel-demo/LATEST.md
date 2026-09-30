# FAWKES run flywheel-demo

> Evidence is a run directory, never a claim. Simulation acceptance is never physical authorisation.

## Suite: practice

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0198 | 0.0625 | 0.0311 | 0.0000 | 0 | PASS |
| cem_knobs | 0.0195 | 0.0664 | 0.0316 | 0.0000 | 0 | PASS |
| rma_oracle | 0.0198 | 0.0662 | 0.0325 | 0.0000 | 0 | PASS |
| rma_adapter | 0.0198 | 0.0658 | 0.0310 | 0.0000 | 0 | PASS |
| rma_zero | 0.0199 | 0.0613 | 0.0325 | 0.0000 | 0 | PASS |

## Suite: dr family (held-out)

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0197 | 0.0673 | 0.1855 | 0.0000 | 1 | FAIL |
| cem_knobs | 0.0198 | 0.0651 | 0.1828 | 0.0000 | 1 | FAIL |
| rma_oracle | 0.0198 | 0.0741 | 0.1887 | 0.0000 | 2 | FAIL |
| rma_adapter | 0.0198 | 0.0669 | 0.1892 | 0.0000 | 1 | FAIL |
| rma_zero | 0.0199 | 0.0778 | 0.1897 | 0.0000 | 2 | FAIL |

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

- generated: 2026-09-23T10:30:37+00:00
- fawkes repo: e1b365c
- data root: `C:\Users\rishi\Documents\GitHub\RoboCup-Research`
- tag: flywheel-demo

## Honest caveats

- The champion is a hypothesis until the guarded A/B on the field; the
  simulator-vs-real gap warning (203 vs 104 mm cross-track) is standing.
- The v0.1 adapter is a closed-form linear head over engineered window
  features; the GRU lands with the PPO gradient stack.
- Pure-Python env backend at 50 Hz; the MJX/Brax backend is FK-3.
- Nothing here touched a robot. deployment_authorized stays false.
