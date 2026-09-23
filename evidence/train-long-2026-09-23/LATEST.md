# FAWKES run train-long-2026-09-23

> Evidence is a run directory, never a claim. Simulation acceptance is never physical authorisation.

## Suite: practice

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0198 | 0.0625 | 0.0311 | 0.0000 | 0 | PASS |
| cem_knobs | 0.0181 | 0.0564 | 0.0257 | 0.0000 | 0 | PASS |
| rma_oracle | 0.0301 | 0.0611 | 0.0694 | 0.0000 | 0 | PASS |
| rma_adapter | 0.0315 | 0.0609 | 0.0713 | 0.0000 | 0 | PASS |
| rma_zero | 0.0294 | 0.0635 | 0.0747 | 0.0000 | 0 | PASS |

## Suite: dr family (held-out)

| Condition | Endpoint p95 (m) | Cross-track p95 (m) | Heading max (rad) | Sat frac | Boundary | All gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline_knobs | 0.0197 | 0.0673 | 0.1855 | 0.0000 | 1 | FAIL |
| cem_knobs | 0.0184 | 0.0619 | 0.1514 | 0.0000 | 0 | FAIL |
| rma_oracle | 0.0326 | 0.0678 | 0.2267 | 0.0000 | 3 | FAIL |
| rma_adapter | 0.0328 | 0.0727 | 0.2304 | 0.0000 | 2 | FAIL |
| rma_zero | 0.0326 | 0.0709 | 0.2345 | 0.0000 | 2 | FAIL |

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

- generated: 2026-09-23T09:47:21+00:00
- fawkes repo: 73c18fc
- data root: `C:\Users\rishi\Documents\GitHub\RoboCup-Research`
- tag: train-long-2026-09-23

## Honest caveats

- The champion is a hypothesis until the guarded A/B on the field; the
  simulator-vs-real gap warning (203 vs 104 mm cross-track) is standing.
- The v0.1 adapter is a closed-form linear head over engineered window
  features; the GRU lands with the PPO gradient stack.
- Pure-Python env backend at 50 Hz; the MJX/Brax backend is FK-3.
- Nothing here touched a robot. deployment_authorized stays false.
