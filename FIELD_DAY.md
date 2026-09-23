# FAWKES lab day — step-by-step

What this is: the exact sequence to take a FAWKES-trained champion to the field
today, test it against Phoenix, and bring the data home. Every command here
exists and has been checked against the repositories it references. Commands
run from: **FAWKES** = `C:\Users\rishi\Documents\GitHub\FAWKES`,
**Phoenix** = `C:\Users\rishi\Documents\GitHub\Phoenix-Server`.

## 0. What you are actually carrying today (read this first)

| Artifact | Installable today? | Path |
| --- | --- | --- |
| **CEM gain champion** (11 firmware gains) | **YES** — through the existing guarded A/B path (section 3) | `evidence/train-2026-09-23/firmware_registry_entry.json` |
| Residual actor (the RMA network) | **NO** — the live firmware's `turtlerabbit-motion-v1` ABI must first be confirmed against the robot's `experimental` branch, and Phoenix has no consumer module yet (FK-6 follow-up) | `evidence/train-2026-09-23/residual_actor.json` |
| Phoenix bundle `movement-rma-1` | **NO** — same reason; it verifies offline only | `evidence/train-2026-09-23/phoenix_bundle_movement-rma-1.json` |

Two rules that do not bend today:

1. **`deployment_authorized` is false until a human flips it** — by *passing*
   the guard protocol in section 3, not by editing a file.
2. **The champion is a hypothesis.** It was trained in FAWKES's pure-Python
   plant model. rlearn's own simulator overpredicted cross-track 203 mm where
   reality gave 104 mm — that warning is standing. The A/B exists to test the
   hypothesis, not to celebrate it.

## 1. Before leaving the desk (10 minutes)

In FAWKES:

```powershell
python -m pytest tests -q                       # 27 passed
python -m fawkes.cli verify evidence\train-2026-09-23\phoenix_bundle_movement-rma-1.json
type evidence\train-2026-09-23\LATEST.md
```

Read the `practice` suite table in `LATEST.md` and apply this decision rule:

- **cem_knobs beats baseline_knobs** on endpoint p95 and breaks no gate the
  baseline passed → carry the champion, section 3 applies.
- **cem_knobs does not beat baseline** → today is a **data-collection day**
  (sections 2 and 5 still run, baseline stays active). That is a good day too:
  the flywheel grows either way.

**Verdict for the 2026-09-23 champion (already computed — GO):** on the
practice suite the champion beats the field-proven baseline on both metrics
(endpoint p95 **19.2 vs 19.8 mm**, cross-track p95 **57.2 vs 62.5 mm**) and
passes every gate. Known gap, on the record before you leave: on the harsh DR
family the heading gate fails for *every* condition including the baseline
(0.17 rad vs the 0.08 limit) — start the ladder at 0.5 m/s and watch heading
on the diagonals.

## 2. Lab bring-up (15 minutes)

**One field server only.** Stop any previous server before starting a new one
(so only one process ever sends commands to robots).

From Phoenix:

```powershell
python main.py
```

This loads `configs/` and `state/fleet.toml` (real roster, feedback listener
on 50513, robot commands to 50514). The server starts **halted** — that is
correct. Open the console it prints.

1. **Vision on:** console → **Vision / simulator** → stop the simulator if one
   is running → **Connect field cameras / halt robots** (joins `224.5.23.2`
   on 10006 / 10005 / 10010). This halts motion and clears old tracks — expected.
2. **Check the robot** is registered, **online, visible in vision, and reporting
   motion-ready** (MOVE bit). An online robot without vision or with its motion
   bit clear is *not* ready. Confirm the hardware identity matches the robot
   you think it is (camera marker id ≠ wire id — check both).
3. **E-STOP ready before any movement:** a second person within reach, plus this
   typed and ready in its own terminal (do not run it unless stopping is
   required — it deliberately overrides motion):

   ```powershell
   python -m phoenix.tools.estop --configs configs
   ```

4. **Baseline sanity leg:** with the default `baseline-validated` profile active,
   do one gentle manual move (or the section 3A command at 0.5 m/s). If the
   field-proven baseline looks wrong, **stop** — fix that first; a champion A/B
   on top of a broken baseline proves nothing.

## 3. The guarded A/B — installing the champion (the only install path today)

The champion is **11 gain values** in a registry entry. The firmware already has
the whole mechanism (`config/motion_calibrations.json` + `tools/
select_motion_calibration.py`); you are only adding one more entry.

### 3A. Record the baseline leg (gates + fresh heading)

```powershell
python -m phoenix.tools.field_tune preflight
```

(no `--execute` — this is the no-motion check; it prints the robot's current
fused heading; copy it as `FRESH_HEADING` below). Then the baseline leg —
robot N near field centre, ball removed, kicker/dribbler inactive:

```powershell
python -m phoenix.tools.motion_benchmark real-straight 5 --real-robot --armed --estop-ready --speed 0.5 --accel 0.5 --length 0.5 --directions +x --repeats 1 --heading FRESH_HEADING --mode baseline --output artifacts\physical\r5_baseline_0p5.json
```

The command refuses to run until you type `ARM ROBOT 5`. A fully passing run
writes a content-hashed `.gate.json` — keep it, later speeds require `--gate`
and may rise at most 25% per stage.

### 3B. Install the champion gains on the robot

On the robot's Pi (`/home/pi/RobotFramework`, the **experimental** branch —
verify with `git log --oneline -1` before touching anything; the
RoboCup-Research copy of RobotFramework is a *stale* `main` snapshot and is
only the format reference):

1. Back up: `cp config/motion_calibrations.json config/motion_calibrations.json.bak-<date>`
2. Paste the `fawkes-cem-<tag>` entry from `evidence\train-2026-09-23\
   firmware_registry_entry.json` into the `profiles` map, **next to** (never
   over) `field-proven-20260723`.
3. Apply and verify:

   ```bash
   python3 tools/select_motion_calibration.py fawkes-cem-train-2026-09-23 --check   # dry check
   python3 tools/select_motion_calibration.py fawkes-cem-train-2026-09-23           # patches Motion.yaml
   ```

4. Restart RobotFramework through the normal operator workflow (it runs as
   `./RobotFramework -safe`).

### 3C. The A/B — same leg, new gains

Repeat the exact 3A command with `--output artifacts\physical\r5_fawkescem_0p5.json`
(mode stays `baseline`; the gains now come from the patched `Motion.yaml`).

Pass rule (from rlearn's `SAFETY.md`, unchanged): **every endpoint, heading and
cross-track metric at least as good as the baseline leg.** Any worse → rollback
(3E) and the day becomes data collection.

### 3D. The ladder (only while every stage passes repeatedly)

0.5 m/s → 1.0 → 1.5 → 2.0, each with `--gate` from the previous stage's
`.gate.json`; add translation+yaw and the other directions; 2.5 m/s only after
everything below it passes repeatedly. Do not skip stages.

### 3E. Rollback (one command, know it before you need it)

```bash
python3 tools/select_motion_calibration.py field-proven-20260723
```

then restart the service. Phoenix side: switch the motion profile back in the
console **Motion** tab (or `POST /api/settings {"section": "profile:baseline-validated"}`);
`configs/motion/ACTIVE` sets the startup default.

## 4. Testing with Phoenix, no robot required

If the robot isn't available (or while someone else drives it), these all run
safely:

**ER-Force simulator through the console:** Vision/simulator → **Start
ER-Force** (the `simulator-cli` instance, shared by everything). Scenarios run
against real physics with our controller — the FAWKES gains can be eyeballed
on the console's motion-profile preview before any robot sees them. Headless:

```powershell
uv run python scripts/erforce_duel.py --preset realistic --turnovers 4 --duration 150 --out .test-runs\fawkes-day
```

**Motion benchmark, fully offline** (baseline vs shadow vs adaptive modes in the
built-in simulator; nothing touches a robot):

```powershell
python -m phoenix.tools.motion_benchmark simulate all --seed 20260923 --output-dir artifacts\motion
python -m phoenix.tools.motion_benchmark report artifacts\motion\straight_baseline.json artifacts\motion\straight_rl-shadow.json --output artifacts\motion\report.md
```

**Shadow mode server** (`--shadow` runs the identical live process minus the
final `sendto` — a real dry run of the whole stack; SHADOW badge shows in the
console, nothing reaches the robots):

```powershell
python main.py --shadow
```

**Replay an onboard log** (after the lab day, against a recorded `motion_v1`
JSONL pulled off the robot) — evaluates a policy without applying it:

```powershell
python -m phoenix.tools.motion_benchmark replay C:\logs\motion_v1_robot5_RUN.jsonl --output artifacts\motion\replay.json
```

## 5. Bringing the data home (the flywheel — this is the point of the day)

1. Before leaving: **export the console Measurement session** (Testing → drag
   calibration → Export, or the session's own export), and copy the robot's
   `logs/motion_v1` JSONL files off the Pi.
2. Any new Phoenix calibration CSVs land in `calibration_runs/`; the onboard
   logs carry the wheel telemetry. Both are already readable by FAWKES
   (`phoenix_csv`, `onboard_motion_log` converters).
3. At home, re-fit and see what changed:

   ```powershell
   python -m fawkes.cli audit                     # re-pins provenance + hashes
   python -m fawkes.cli identify                   # re-fits the wheel ensemble incl. today's rows
   ```

   A drift report between the old and new model cards is exactly the
   "did the robot change?" alarm the README's flywheel promises.

## Abort rules (stop immediately, then mark it in the session)

Abnormal sound or motion · heading growth · collision · wheel slip · motor
fault · current/temperature/voltage warning · stale vision · any safety
intervention · a 0.2 m path deviation · unexpected robot identity. On any of
these: E-STOP, rollback (3E) if the champion was installed, and record what was
seen. **A failed trial is still data — record it, do not delete it.**

---

*This runbook follows the house rules: rlearn's `SAFETY.md` gate protocol,
Phoenix's `MOTION_BENCHMARK_RUNBOOK.md` physical section, and `FIELD_TESTING.md`'s
bring-up. Nothing in FAWKES ever deploys itself.*



