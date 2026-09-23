# The FAWKES flywheel — continuous improvement, honestly

This is the answer to "how do we set it up and keep getting better". The short
version: **the machinery already exists and is tested. The loop closes when you
feed it data.** This document is the operating manual for that loop.

## The loop

```
        ┌─────────────────────────────────────────────────────────────┐
        │                    THE FLYWHEEL (offline)                   │
        │                                                             │
  field │   audit ──► identify ──► train ──► evaluate ──► export ──►  │
  data ─┼─► (hash &  (re-fit the  (CEM +    (gates on   (firmware    │
        │    pin)      wheel       RMA on    held-out    registry     │
        │              ensemble    the DR    seeds)      entry +      │
        │              on all      family)               bundle)      │
        │              traces)                                        │
        └──────────────────────────────┬──────────────────────────────┘
                                       │ verdict.json: promote=true/false
                                       ▼
                          ┌────────────────────────┐
                          │  THE HUMAN DECISION    │  ← the only manual step
                          │  install the staged    │     that must stay manual
                          │  candidate? (A/B it)   │
                          └───────────┬────────────┘
                                      │ yes, guarded A/B (FIELD_DAY.md)
                                      ▼
                              field run ──► new data ──► back to the top
```

## What already exists (and is tested)

Every box in the top half is a tested FAWKES command:

| Step | Command | What it does |
| --- | --- | --- |
| audit | `python -m fawkes.cli audit` | Hashes and pins every trace; the provenance lock |
| identify | `python -m fawkes.cli identify` | Re-fits the wheel ensemble on all accumulated transitions |
| train | `python -m fawkes.cli train` | CEM over the 11 gains + the RMA two-phase stack |
| evaluate | `python -m fawkes.cli evaluate` | Gates on held-out seeds, practice + DR families |
| export | `python -m fawkes.cli export` | Firmware registry entry + residual actor + Phoenix bundle |
| report | `python -m fawkes.cli report` | House-style markdown with the honest table |

And one command runs the whole thing end to end:

```powershell
python -m fawkes.cli flywheel --tag flywheel-<date>
```

It writes `evidence/flywheel-<date>/verdict.json` with a `promote` flag and the
staged artifact path. **That is the loop.** Everything above the human decision
is one command.

## What makes it *continuous* (the part you add)

The loop only gets better if the data keeps coming. Three rules:

1. **Every field run becomes data.** Before leaving the lab: export the console
   Measurement session and copy the robot's `logs/motion_v1` JSONL off the Pi.
   New Phoenix calibration CSVs land in `calibration_runs/`. FAWKES already reads
   all three formats (`phoenix_csv`, `onboard_motion_log`, `rlearn_npz`).

2. **Run the flywheel after every session** (or weekly). It re-fits the wheel
   ensemble on the *accumulated* traces — so the model of the robot is always
   current, and the drift report (`identify`) tells you when the chassis changed.

3. **The human decision stays manual, on purpose.** `verdict.json` says
   `promote=true` only when the candidate beats the field-proven baseline on the
   practice suite *and* passes every gate. Even then it's `SIMULATION ONLY` and
   `deployment_authorized: false`. A person runs the guarded A/B (FIELD_DAY.md)
   before it touches a robot. This is the rule that has kept every previous
   candidate honest — rlearn's `SAFETY.md`, unchanged.

## What "getting better and better" actually looks like

Not a single number going up forever — a sequence of honest, gated steps:

- **Week 1:** the CEM champion beats the baseline on the practice suite. You
  A/B it. If it wins on the field, it becomes the new baseline; the next
  flywheel run trains *against that*.
- **Week 2+:** the RMA adapter gets more (features, z) pairs every run. When
  its probe R² climbs and it starts beating the pure cascade on the blind-
  transfer suite, the residual actor becomes worth fielding — through the
  shadow → replay → bounded ladder, never before.
- **Always:** the DR family keeps the policy honest. A candidate that wins on
  the practice suite but collapses on the harsh family doesn't get promoted.

## What is NOT automatic (and why)

- **Deployment.** Nothing ever installs itself on a robot. The `ACTIVE` pointer
  is an operator act; the firmware profile is applied by a person.
- **The field A/B.** A simulator can rank candidates; only the field can
  validate them. rlearn's own simulator overpredicted cross-track 203 mm where
  reality gave 104 mm — that warning is why the last step is always human.
- **The gradient stack (PPO/GRU).** The current learners are derivative-free
  (CEM) and closed-form (the linear adapter). They are the right floor. The
  jump to "train the residual end-to-end" is the PPO/GRU work — FK-4/5 in the
  README — and it is what turns "a better tuning every week" into "a policy
  that adapts in real time". That is the next build, and the flywheel is the
  harness it drops into.

## The one-command summary

```powershell
# after every field session, from FAWKES:
python -m fawkes.cli flywheel --tag flywheel-<date>
# read evidence\flywheel-<date>\verdict.json
# if promote=true: FIELD_DAY.md section 3 (the guarded A/B)
```

That's the system. Set it up once, feed it data, and it keeps producing
candidates that are measurably better than the last accepted one — with a
human holding the only key that matters.
 
