# FAWKES

**F**ield-**A**daptive **W**heel-control & **K**inetics **E**xecution **S**ystem —
a universal reinforcement-learning framework for RoboCup SSL robot movement,
built for every team, tailored first for **Phoenix Server** and the
TurtleRabbit fleet.

> Fawkes is a phoenix. This is the branch of the Phoenix family that learns to
> move.

---

**Status: v0.1 BUILT (2026-09-23).** The framework skeleton is implemented and
tested — all six layers, the CLI, and the export trio — with a smoke run on the
real data. See [§6.1 Build status](#61-build-status--v01-2026-09-23) for what
landed, what is a documented slot, and where the evidence lives. **Going to the
field? [FIELD_DAY.md](FIELD_DAY.md) is the step-by-step.** The plan below
is unchanged; it remains the plan of record.

**Status: PLAN — the sections below are the original plan of record.** Every
fact about existing systems is sourced and cited to the repositories it came
from; everything proposed is marked as such. It follows the house evidence
rules of Phoenix-Server (`docs/ERFORCE_2V2.md`): *evidence is a test or a run
directory, never a claim.*

**Written:** 2026-09-23, against Phoenix-Server branch `burn-in` (off
`pheonix-server-v2-experimental`, tip `3d24e7c`), RobotFramework `main` @
`bab9d03` (snapshot in `RoboCup-Research`), and the evidence pack in
`RoboCup-Research/generalinformation` (2026-07-23 session).

---

## 0. The mission, in one paragraph

Train robot movement from **real recorded data**, in a **domain-randomised**
reinforcement-learning environment, so that the deployed controller can **come
in blind** — unknown carpet, sagging battery, drifted motors, noisy camera —
and still move **as well as the best hand-tuned run, every time**, by watching
its own motion for a second and adapting. The framework is **universal**: any
SSL team can bring their traces, their robot description, and their simulator,
and get a deployable artifact. It is **tailored for us**: the first-class
consumers are Phoenix Server's motion-profile ladder and the RobotFramework
onboard runtime, and the first-class inputs are the traces this team has
already recorded.

"Perfect" is defined by gates, not adjectives (§7). The claim this plan aims
to earn, eventually, with evidence:

> On an unseen surface and battery, within 2 seconds of movement, the adaptive
> controller holds point-to-point endpoint error ≤ 10 mm p95 and cross-track
> ≤ 60 mm p95 — beating the field-proven hand-tuned baseline's worst leg
> (19.5 mm endpoint, 104 mm cross-track) — with zero boundary or
> speed-limit violations, inside the existing 5 % bounded-authority safety
> envelope.

---

## 1. Why this exists — the situation found on the ground

A full audit of the workspace, the research folders, and the repos produced
one big finding and several supporting ones. (Dig inventory and reliability
verdicts: Appendix A.)

### 1.1 The team already runs a learned controller on a real robot — and its training environment is lost

`Phoenix-Server/configs/motion/rl-bounded-dr-i900.toml` is a **fielded RL
champion**: `rgtap-ppo-dr-i900`, "wide domain-randomisation PPO, 1024 envs ×
horizon 128, seed 1, iter 900", applied for real on 2026-08-26 on the `.102`
chassis in **bounded mode** (5 % residual authority): 182/186 ticks healthy and
applied, residual mean 8.7 / max 29 mm/s, move settled to 4.8 mm. Shadow mode
before it: 25,000+ ticks. The deployment ladder (conservative → shadow ≥ 1000
samples → replay → bounded at 0.5 m/s → ladder) is disciplined and good.

But `Phoenix-Server/docs/PHYSICS_SIMULATION.md` §2.3 states it plainly:

> The firmware branch ships the runtime half of an RL motion system whose
> training half is not in either repo. … **we are running a bounded RL residual
> on a real chassis whose training environment we cannot currently rebuild,
> retrain, or audit.**

Its M-1 milestone asks: *where is `testing/rlearn/`?*

**Answer from this dig:** `RoboCup-Research/testing/rlearn` exists — but it is
the **offline system-identification + cross-entropy controller-search
lineage** (61,087 real motor transitions, wheel-response ensemble, CEM over 11
interpretable gains; its champion `rlearn-champion-20260805` is installed in the
firmware calibration registry for guarded field validation). It is **not** the
missing `residual_env.py` — there is no PPO trainer and no domain-randomised
physics environment anywhere in these folders. The 1024-environment DR rig
that produced `rgtap-ppo-dr-i900` is still missing.

**So FAWKES is that rig, rebuilt properly — and generalised** so it serves any
team, produces every artifact type this fleet already consumes, and never
again leaves a deployed controller without a reproducible trainer.

### 1.2 The assets that already exist (and their condition)

| Asset | Where | State | Verdict |
| --- | --- | --- | --- |
| Field-proven calibration (Robot A, 2026-07-23) | `generalinformation/` evidence pack | 8/8 legs passed: worst endpoint 19.45 mm (limit 35), worst cross-track 104.11 mm (limit 120), peak ~1.79 m/s, ceiling 2.5 m/s; Phoenix @ `77cd99c`, RobotFramework @ `905868c` | **Trustworthy** — hashed, provenance-pinned, honestly caveated |
| 162 CSV traces + 12 onboard logs (32.0 MB, SHA-256 per file) | `generalinformation/calibration_runs`, `…/motion_*.log` | 67,593 CSV rows, 721.7 s of motion; 4 CSV schemas; pipe-separated logs with a `0-0.009499` negative-number formatting artifact | **Usable after conversion** — schemas and quirks fully documented in the pack |
| Offline learner (`rlearn`) | `RoboCup-Research/testing/rlearn` | Wheel ensemble (47 features → 4 wheel deltas, 9 members), held-out RMSE 0.044–0.081 rev/s, CEM champion with `deployment_authorized: false` | **Solid, reproducible, honest** — but simulator-only lineage; overpredicted field cross-track (203 mm sim vs 104 mm real — its own warning) |
| Digital-twin replay validator | `RoboCup-Research/replaytest` | Commit-pinned measured baselines, uncertainty corridors, pre/post-IMU-fix comparison | **Good validation UX to inherit** |
| Onboard adaptive runtime | live firmware, `experimental` branch on the robot (per `Phoenix-Server/docs/TELEMETRY_SOURCE_INVENTORY.md`; *not* in the `RoboCup-Research` snapshot, which is `main` @ `bab9d03`) | off / collect / shadow / bounded ladder, `turtlerabbit-motion-v1` actor ABI, adaptive surface estimator, 5 % authority clamps | **Deployed and real — the runtime FAWKES targets** |
| Calibration registry + patcher | `RobotFramework/config/motion_calibrations.json`, `tools/select_motion_calibration.py` | Profiles `field-proven-20260723` (rollback) and `rlearn-champion-20260805` (guarded validation); patches `Motion.yaml` | **This is the firmware-side plug-in point** |
| Phoenix ES knob RL | `phoenix/rl/`, bundles under `docs/evidence/` | Linear knob policy over 7 tactical knobs, spec `team-knobs-1`, bundle + ACTIVE pointer + verify/activate/rollback; flat-stage result honestly reported "not evidence of a better policy" | **Pattern to extend, not the thing itself** (tactics, not movement) |
| Phoenix surface DR | `phoenix/sim/world.py` `Surface` | traction/rolling spreads, ridges, snag, traps; curriculum flat→traction→seams→traps→immobilised; held-out surface seeds ≥ 9000 | **The DR vocabulary, already in the house style** |
| ER-Force bridge | `phoenix/simlink/` | our controller, ER-Force physics, console-integrated | **Cross-validation plant** |

### 1.3 Reliability of the research data (the user's question, answered)

The data is **more reliable than feared, with named caveats**. The
`generalinformation` pack is exemplary in provenance: per-file SHA-256, exact
commit pins, four documented CSV schemas, explicit "facts not established"
list. The specific caveats that matter for learning from it:

1. **One robot, one carpet, one day** (2026-07-23, Robot A, command ID 4,
   feedback ID 0). No other surface, camera, payload, or battery chemistry.
2. **Battery drifted through the day** — fresh pack ~24.8 V, late-session
   ~23.0–23.5 V, 6S window 19.8–25.2 V. Battery is *inferred*, not measured
   per-trace (MatchFeedback `batteryLevel` exists — use it when replaying).
3. **Camera artifacts are real and documented**: vision-age quantised at
   ~16 ms, marker heading flips up to ~1.2 rad, vision-derived velocity
   filtered over 4 frames, phantom robot detections observed, vision speeds up
   to 3.38 m/s and omega spikes to 69.7 rad/s that are camera artifacts, not
   physics.
4. **The onboard logs are headerless, pipe-separated**, with the `0-0.009499`
   negative-padding artifact; columns must be recovered from the matching
   RobotFramework telemetry revision. (rlearn already does this repair.)
5. **Kicker was off; no ball work is evidenced.** No kick data exists in these
   traces at all.
6. **The honest gaps the pack itself lists**: no sustained 2.5 m/s (field too
   short; 2.32 m/s onboard peak), no thermal drift (27–30 °C all day), no
   stall currents, no long-term drift, per-robot IMU axis/sign differences
   (Robot A: axis 1, sign −1).

**Consequence for design:** this data is excellent for *system identification*
and *validation*, thin for *covering the real-world variation* the policy must
survive — which is precisely why the training environment must be
domain-randomised around the identified model rather than fitted to these
traces alone. The traces anchor the DR distributions; they do not bound them.

---

## 2. The idea, explained plainly

### 2.1 Domain randomisation: train on every world the robot might meet

Train in simulation, but never the *same* simulation twice. Each episode (or
each environment in the vectorised batch) samples the physics the robot might
actually face, around the values identified from real data:

| Axis (sampled per episode/env) | Anchor from real data | Range rationale |
| --- | --- | --- |
| Wheel radius / `meters_per_motor_rev` | 0.2171 m/rev | ± a few % (wear, load) |
| Wheel angle / mounting error | calibrated geometry | small errors, per-robot |
| Per-motor command gain (±, per wheel) | [1.006, 1.008, 1.004, 1.004] and sign-specific scales | asymmetry grows with wear |
| Anisotropic / directional friction | forward vs reverse legs differed on field | omni-roller reality |
| Rolling resistance / carpet traction | Phoenix `Surface` spreads | different venues |
| Carpet transitions, seams, ridges, traps | Phoenix surface model | Phoenix `Surface` vocabulary |
| Motor time constant / lag | 1 sample @ 50 Hz identified (~20 ms) | temperature, load |
| Battery voltage + sag | 24.8 → 23.0 V observed; 6S window 19.8–25.2 | the biggest real drift axis |
| Command delay / jitter / dropout | posDelay Q6.2 ms, 16 ms quantisation | Wi-Fi reality |
| Vision noise / delay / dropout | meas_std 5 mm, vision-age p95 ~32 ms+, marker flips | camera reality |
| IMU bias / noise / axis-sign | Robot A: axis 1, sign −1 | per-robot mounting |
| Encoder scale error | per-wheel scales | calibration drift |
| Payload / mass / inertia | measured chassis | ball-handling variants |
| External disturbance (pushes) | — | contact with other robots |

This is the same axis list the lost `residual_env.py` randomised (recorded in
`Phoenix-Server/docs/PHYSICS_SIMULATION.md` §2.3, quoting
`RobotFramework/docs/ADAPTIVE_MOTION.md`): *wheel radius and angle,
mass/inertia, anisotropic/directional friction, rolling resistance, carpet
transitions, motor gain, battery, command delay/jitter, vision
noise/delay/dropout, IMU bias/noise, encoder scale, thermal response and
external disturbances.* FAWKES reconstructs that list as its canonical DR
family — the old rig's ghost is the spec.

### 2.2 The hidden vector: a network that watches the robot move

The user's description — *"a network watches how it moves and control inputs,
it writes a hidden vector, that vector is given to the controller"* — is a
real, published technique. The reference is **RMA — Rapid Motor Adaptation**
(Kumar et al., 2021), used to make quadrupeds walk blind on ice and mud. FAWKES
adopts it, in the SSL/omni-wheel setting:

```
                     ┌──────────────────────────────────────────────┐
                     │            ADAPTATION MODULE  a(·)            │
   last K ticks of   │  a small GRU (or TCN) over the recent        │
   observation o_t   │  (observation, command) history:             │
   + command u_t  ──►│      z_t = a(o_{t-K..t}, u_{t-K..t})        │
   (what the robot    │  z_t ∈ R^8: "what world am I in right       │
    can SEE)          │  now?" — traction, sag, drift, asymmetry    │
                     └──────────────┬───────────────────────────────┘
                                    │  the hidden vector (latent state)
                                    ▼
   goal, current ──►  ┌──────────────────────────────────────────────┐
   estimator state    │         BASE POLICY  π(u | o_t, z_t, goal)   │
                     │  bounded residual / target adjustment         │
                     └──────────────┬───────────────────────────────┘
                                    ▼
                          controller output (within limits;
                          the governor and envelope stay in charge)
```

Training has two phases:

- **Phase A — privileged learning.** The DR environment knows the true sampled
  parameters (traction, sag, …). The base policy is trained with them handed to
  it, so it can learn the *best possible* behaviour per world. (PPO; the same
  family as the lost champion, which its own header names.)
- **Phase B — learn to infer.** Freeze the base policy. Train the adaptation
  module to produce, from *only* the observable history (estimator state,
  wheel feedback, commanded inputs — never the true parameters), a `z_t` that
  makes the frozen policy perform as well as it did with the truth. This is
  pure supervised/distillation-style training on rollouts — cheap and stable.

At deployment there is no privileged information at all. The robot starts
blind, moves, and within roughly a second the GRU's hidden vector has inferred
the world it is in; the policy is then effectively re-tuned for that world.
A linear probe on `z_t` (predicting traction / battery / per-motor gain) is
trained for free as a **diagnostic readout** — the console can display "what
the robot thinks the carpet is doing to it", which is also a drift alarm.

This is exactly the "come in blind, still perfect" behaviour: adaptation in
context, at 50 Hz, with no retraining on the robot.

### 2.3 Why not just tune the gains harder?

Because a fixed gain set is a compromise across all worlds. The evidence is on
the field: the same CEM champion that beat the baseline hugely in simulation
(§1.2) came with the simulator's own warning that it over-predicted
cross-track error 203 mm where reality gave 104 mm — a fixed-parameter search
against a fixed model overfits that model. The adaptation approach instead
buys *per-condition* optimality: one policy that behaves like the best-tuned
controller for whatever condition it finds itself in. It also degrades
gracefully: with `z` zeroed, the base policy is trained (via DR including the
zero-information corner) to fall back to a robust average-world behaviour —
the bounded-residual envelope guarantees the downside stays small.

---

## 3. Architecture

Six layers. Universal interfaces at every boundary; Phoenix/TurtleRabbit
implementations as first-class adapters on each side.

```
┌────────────────────────────────────────────────────────────────────────┐
│ L0  TRACE — universal movement-trace format (FTF-1) + converters       │
│     our CSVs, onboard logs, rlearn transitions, Phoenix sessions,       │
│     other teams' logs  →  one hashed, provenance-pinned format          │
├────────────────────────────────────────────────────────────────────────┤
│ L1  SYSID — wheel-response ensemble (rlearn heritage) + DR families     │
│     identified model per robot; DR distributions anchored to real data   │
├────────────────────────────────────────────────────────────────────────┤
│ L2  ENVS — vectorised, domain-randomised single-chassis tasks           │
│     GPU-batched (MJX/Brax class) to ~1024 envs; pure-Python fallback     │
│     for CI; ER-Force cross-check plant                                   │
├────────────────────────────────────────────────────────────────────────┤
│ L3  LEARNERS — CEM over interpretable gains (baseline) · PPO (DR) ·     │
│     RMA two-phase adaptation (the hidden vector)                         │
├────────────────────────────────────────────────────────────────────────┤
│ L4  EXPORT — three artifacts:                                            │
│     (a) motion-calibration profile JSON  → firmware registry/Motion.yaml│
│     (b) residual actor JSON               → turtlerabbit-motion-v1 ABI  │
│     (c) Phoenix bundle (spec movement-rma-1) → ACTIVE pointer + verify │
├────────────────────────────────────────────────────────────────────────┤
│ L5  DEPLOY & FLYWHEEL — shadow → replay → bounded 5 % → ladder;         │
│     every run becomes FTF traces; weekly re-identify/re-adapt/re-gate    │
└────────────────────────────────────────────────────────────────────────┘
```

### L0 — the universal trace format (FTF-1)

One episode = **JSON header + columnar arrays** (`.npz`). The header carries
identity and provenance; the arrays carry time series. Teams adopting FAWKES
only adopt this format (or ship a converter); nothing else about their stack
matters to the framework.

```jsonc
// ftf header (one per episode)
{
  "schema": "ftf-1",
  "team": "WSU-TurtleRabbit",
  "robot": {
    "id": 4, "hardware_id": 0, "chassis": "RobotA-mjbots",
    "kinematics": {
      "meters_per_motor_rev": 0.2171,
      "body_lateral_scale": 0.887,
      "wheel_order_canon": ["FR", "RR", "RL", "FL"],
      "per_wheel_command_scale": [1.006, 1.008, 1.004, 1.004]
    }
  },
  "source": {
    "kind": "phoenix_csv | onboard_motion_log | rlearn_npz | phoenix_session | ssl_vision_log | fawkes_sim",
    "repo": "RobotFramework@905868c", "phoenix": "77cd99c",
    "recorded": "2026-07-23T…", "surface_id": "practice-field"
  },
  "session": {
    "battery_start_v": 24.8, "battery_end_v": 23.2,
    "temperature_c": 29.5, "camera_hz": 60, "control_hz": 250
  },
  "standard": {
    "vel_max_xy_mps": 2.5, "acc_max_xy_mps2": 2.5,
    "endpoint_pass_m": 0.035, "cross_track_pass_m": 0.12,
    "deviation_abort_m": 0.2
  },
  "provenance": { "raw_sha256": "…", "converter": "ftf/convert/phoenix_csv@…", "rows": 371 }
}
```

```text
ftf columns (missing columns are absent, never fabricated):
  t                      s          monotonically increasing
  vision_x, vision_y, vision_heading            m, m, rad
  vision_vx, vision_vy, vision_omega            m/s, m/s, rad/s
  vision_age_ms           ms
  est_x, est_y, est_heading, est_vx, est_vy, est_omega   (estimator state)
  cmd_skill, cmd_x, cmd_y, cmd_heading, cmd_vx, cmd_vy, cmd_omega
  cmd_vel_max, cmd_acc_max, cmd_primary_direction          (what was sent)
  fb_pos_x/y/heading, fb_vx/vy/omega, fb_battery_v          (MatchFeedback)
  wheel_measured[4], wheel_commanded[4]                     rev/s
  truth_x, truth_y, truth_heading       (sim episodes only, NEVER mixed into
                                         evaluation of field episodes)
  events[]                              (abort, battery swap, vision dropout…)
```

Converters shipped with FAWKES, in priority order (all our data first):

| Converter | Input | Quirks it must handle |
| --- | --- | --- |
| `phoenix_csv` | the 4 documented CSV schemas (107/27/20/8 traces) | missing est columns per schema; duplicate timestamps; 16 ms vision-age steps |
| `onboard_motion_log` | pipe-separated headerless logs | `0-0.009499` negative-padding repair; column recovery from the pinned RobotFramework telemetry revision |
| `rlearn_npz` | `transitions.npz` (47 features) | already repaired; carry the manifest hashes through |
| `phoenix_session` | Measurement-page session recordings | live captures going forward — the flywheel's main food |
| `ssl_vision_log` | official SSL logs / any team's vision | for teams without onboard logs |

Every converted episode keeps the SHA-256 of its raw source (rlearn's rule:
raw files are never copied or changed; hashes make runs traceable).

### L1 — system identification and the DR families

Inherit rlearn's model wholesale and harden it: the physically anchored
first-order wheel response plus the 9-member ridge residual ensemble, source
weighting (final log 4×, corrected-IMU 2×, earlier 1×, inactive 0.15×), and
chronological 20 % hold-out per source. Extend it to:

- output an **uncertainty per prediction** (ensemble disagreement already
  exists) so DR sampling can cover model error, not just parameter ranges;
- attach the DR family (§2.1): identified value ± calibrated spread, spreads
  widened until held-out real traces fall inside the family's support —
  the honest fix for the 203-vs-104 mm sim-real gap warning.

### L2 — the domain-randomised environments

Vectorised single-chassis (then multi-chassis) task suite, gym-style, three
backends behind one interface:

| Backend | Use | Notes |
| --- | --- | --- |
| GPU-vectorised (MuJoCo MJX or Brax) | the real training (matches the lost rig's 1024-env scale) | training-time dependency only; never ships to the robot |
| Pure-Python (numpy, vectorised) | CI, smoke tests, determinism | keeps FAWKES runnable anywhere, mirroring Phoenix's zero-dependency ethos at the *core* |
| ER-Force via `phoenix/simlink` | cross-validation plant | our controller, ER-Force physics — no new seam |

Task suite v1 (all scored against the same gates): point-to-point (all 8
compass legs + random), trajectory tracking, spin-in-place, simultaneous
translation+yaw (the rotation-training lineage), fast-pos lanes, stop-on-a-mark
(precision), and — later, once ball traces exist — ball approach without a
dribbler. Each task parameterises targets inside the recorded envelope
(distances ≤ ~1.85 m as the field allows, speeds to 2.5 m/s, the field
geometry 4.5 × 3.0 m for us — configurable per team).

**The controller under test is ours:** the RobotFramework cascade
(trajectory regeneration → Panthera P/P/FF → wheel kinematics) runs *inside*
the environment, exactly as `phoenix/simlink/cascade.py` already does for
ER-Force. The learned policy is a residual on the cascade output (bounded) —
never a replacement for it. The simulator is the plant, never the controller
(the rule `PHYSICS_SIMULATION.md` §4 states and `simlink` already follows).

### L3 — the learners

1. **CEM over interpretable gains** (rlearn's method) — kept as the always-on
   baseline and the sanity floor. Any learned artifact must beat it to matter.
2. **PPO with domain randomisation** — the main line (and the family the lost
   champion's header names). Curriculum in the house style, mirroring
   `phoenix/rl/team/curriculum.py`: `flat → traction → seams → traps →
   immobilised → blind` (blind = vision dropout/delay extremes), with held-out
   surface seeds ≥ 9000 never trained on.
3. **RMA two-phase adaptation** (§2.2) — base policy + GRU adaptation module
   producing the hidden vector, plus the linear probe diagnostic. Phase B is
   where "come in blind" is actually manufactured, and it is retrainable
   *weekly from fresh field traces without touching the base policy* — the
   continual-learning flywheel's cheap step.

Also fixed in policy, from day one: every run records algorithm, seed,
hyperparameters, DR draws, env commit and data manifest hash in its artifact —
the TD3-vs-PPO documentation discrepancy (`ADAPTIVE_MOTION.md` vs the champion
header, still unresolved per `PHYSICS_SIMULATION.md` open question 2) is the
cautionary tale; FAWKES artifacts are self-describing so this cannot recur.

### L4 — export: what Phoenix (and others) actually load

Three artifacts from one champion, all versioned, hashed and reversible:

**(a) Motion-calibration profile** — a JSON entry for
`RobotFramework/config/motion_calibrations.json` (+ `Motion.yaml` patch via
`select_motion_calibration.py`): the interpretable gains, with
`status: "simulation_only"`, `deployment_authorized: false`, and the
field-proven rollback profile untouched. This is the path the CEM champion
already takes; a PPO/RMA champion whose gains-space projection beats CEM can
ride it too.

**(b) Residual actor** — the network for the onboard `turtlerabbit-motion-v1`
ABI in **bounded** mode: base policy + adaptation GRU, weights as plain JSON,
pure-stdlib forward pass (small MLP/GRU at 50 Hz is trivial in Python; the
robot-side runtime already consumes a champion JSON). Hard onboard clamps stay
untouched and are copied verbatim into the artifact header:
`residual_limit_fraction 0.05`, `residual_slew_linear 0.5 m/s²`,
`residual_slew_angular 1.5 rad/s²`, `confidence_threshold 0.30`,
`max_consecutive_interventions 5`, jerk 10/30, 10 A / 300 ms MOVE supervisor.

**(c) Phoenix bundle** — extends the existing bundle pattern
(`docs/evidence/rl-team-flat-2026-09-19/bundles/flat-18/bundle.json`) to a new
spec `movement-rma-1`: `kind: movement-rma-bundle`, policy (layers + weights +
the adapter), observation/action name lists, provenance (training seed,
iterations, data manifest hash, DR family id), `hardware_profile`
(`motion_active`, `limits_sha256`, `server_sha256`), `compatibility`, and
**verification pairs** — recorded (observation → action) round-trips so
`--verify` can replay the deployed implementation against the training
implementation, exactly as `phoenix-rl-team --verify` does today. Requires one
new pure-stdlib consumer module on the Phoenix side (goes through CODEOWNERS
review per house rules); everything downstream — ACTIVE pointer, activate /
rollback, the console's motion-profile ladder — is already built.

For **other teams**: (a) and (b) are generic JSON; (c) is Phoenix-specific; a
portable `policy.json` + schema is always emitted alongside.

### L5 — deployment ladder and the continual flywheel

The ladder already exists and is good; FAWKES feeds it, never replaces it:

```
conservative baseline ──► shadow (≥1000 clean samples, never applied)
      ──► replay validation (motion_benchmark replay --policy)
      ──► bounded 5 % authority @ 0.5 m/s
      ──► ladder: 1.0 → 1.25 → 1.44 m/s … (existing speed ladder)
```

The flywheel — "constantly learns and accounts every time":

1. Every shadow/bounded/field run logs to FTF (`phoenix_session` converter).
2. **Weekly (or per session) loop, fully offline, no robot contact:**
   re-fit the wheel ensemble on the accumulated traces → drift report
   (did the robot change?); re-evaluate the champion on the newest 20 %
   (chronological hold-out, rlearn's rule); if degraded → re-run Phase B
   adaptation training only (cheap); re-run gates; re-export artifacts.
3. A human promotes via the existing pointer/ladder mechanism. Champions are
   never auto-deployed — `deployment_authorized: false` until a person flips
   it, mirroring rlearn's rule that has been followed correctly so far.
4. Drift alarms come free: the `z`-probe (inferred traction/battery/motor
   asymmetry) trending away from the identified family is an early warning that
   something mechanical changed.

---

## 4. What is universal, what is tailored (the contract)

**Universal — required of any team using FAWKES:**

- movement traces in FTF-1 (or a converter they contribute);
- a robot description (kinematics constants, limits, wheel order);
- optionally: their own simulator behind the L2 env interface.

**Universal — provided to any team:**

- the FTF schema + converters; the DR family definition; the task suite;
- the learners (CEM/PPO/RMA) and evaluation gates;
- portable export: gain profile JSON + residual actor JSON + schema.

**Tailored — first-class for us:**

- converters for the 2026-07-23 evidence pack, rlearn `transitions.npz`,
  Phoenix measurement sessions;
- the RobotFramework cascade as the in-env controller; the firmware registry
  and `Motion.yaml` patch path; the `turtlerabbit-motion-v1` residual ABI;
- the Phoenix bundle spec `movement-rma-1` with ACTIVE pointer + verify;
- our field geometry (4.5 × 3.0 m), our limits (2.5 m/s, 45 rev/s, 120
  rev/s² slew), our ladder and gates.

No team is required to run Phoenix; Phoenix is simply the reference consumer.

---

## 5. Evaluation: what "perfect" means, operationally

Mirroring the house rules (`docs/ERFORCE_2V2.md`): evidence is a run directory,
never a claim; simulation acceptance is never physical authorisation.

**Per-episode gates (field-proven baseline numbers in parentheses):**

| Gate | Target | Baseline (2026-07-23, worst leg) |
| --- | --- | --- |
| Endpoint error, p95 | ≤ 10 mm | 19.45 mm (limit 35) |
| Cross-track, p95 | ≤ 60 mm | 104.1 mm (limit 120) |
| Settling: time inside 10 mm of target | ≤ 1.5× bang-bang optimal | not measured then |
| Heading error | ≤ 0.05 rad | 0.062 rad (estimator, worst) |
| Speed-limit / boundary violations | 0 | 0 |
| Wheel saturation fraction | ≤ 8 % | (rlearn acceptance already uses this) |
| Coupled translation+yaw heading RMS | ≤ 0.12 rad | (rlearn acceptance) |

**Robustness gates (the DR point of the whole exercise):**

- hold the above over **held-out DR draws** never trained on (seed ≥ 9000
  convention) — report win/draw/loss style tables with denominators, like
  Phoenix's M2 table;
- **blind-transfer suite**: train with the adapter, evaluate with the true
  parameters hidden; report *time-to-adapted-performance* (target ≤ 2 s) and
  p95 gates *after* adaptation, plus performance during the first 2 s (the
  blind window) — both numbers, honestly;
- **sim-vs-real honesty check**: replay each candidate against held-out real
  traces (the rlearn chronological hold-out) and publish the gap, including
  when the simulator is wrong (the 203-vs-104 mm warning becomes a standing
  table).

**Promotion evidence chain per candidate:** sim gates → replay-vs-real-traces
→ shadow ≥ 1000 samples → bounded ladder — the existing ladder, no exceptions.

---

## 6. Build plan — milestones FK-0 … FK-8

House style: each milestone has an entry, a deliverable, and an exit
criterion that is a command or a test. FK-0 through FK-2 are pure desk work,
no lab; FK-7+ need the room.

| # | Milestone | Deliverable | Exit criterion |
| --- | --- | --- | --- |
| **FK-0** | Provenance lockdown | Catalog of every asset (Appendix A) turned into `fawkes/trace/manifest.json`; resolve (or record as unresolvable) the TD3-vs-PPO question and the `rgtap-ppo-dr-i900` trainer's fate | The manifest regenerates and hashes clean; the M-1 question from `PHYSICS_SIMULATION.md` has a written answer |
| **FK-1** | FTF-1 + converters | Format spec + `phoenix_csv`, `onboard_motion_log`, `rlearn_npz` converters; all 162 CSVs + 12 logs + transitions convert | Converter round-trip tests; row/schema counts match the evidence pack's own numbers (67,593 rows, 4 schemas) |
| **FK-2** | SysID v2 | rlearn ensemble ported into FAWKES; champion reproduced (seed 20260805) then re-fit on the full FTF corpus; uncertainty output | Held-out RMSE ≤ rlearn's (0.044–0.081 rev/s); sim-vs-real gap table published |
| **FK-3** | DR environment | Pure-Python vectorised env (CI) + GPU backend (MJX/Brax); DR family anchored to FK-2; RobotFramework cascade in the loop (via the `simlink` cascade pattern) | ≥ 1000 envs stepping at PPO-consumable rate on the training GPU; flat-stage DR runs end-to-end overnight |
| **FK-4** | PPO champion | PPO + curriculum; CEM kept as floor | PPO candidate beats the CEM champion on the FK gates in sim, held-out seeds |
| **FK-5** | RMA — the hidden vector | Two-phase training; `z`-probe diagnostics; blind-transfer suite | Blind-transfer p95 gates met; time-to-adapted ≤ 2 s; degradation with `z` zeroed stays within baseline+ε |
| **FK-6** | Export trio | Registry profile JSON; residual actor JSON (bounded header verbatim); Phoenix bundle `movement-rma-1` + pure-stdlib consumer (PR to Phoenix, CODEOWNERS review) | `--verify` round-trip passes byte-exact; consumer passes Phoenix suite + `tests/test_module_boundaries.py` |
| **FK-7** | Shadow campaign | On-robot shadow runs via the existing `rl-shadow` profile path; ≥ 1000 clean samples per candidate | Shadow reports; replay gate passes; zero applied ticks on the robot (by construction) |
| **FK-8** | Bounded ladder + flywheel | Bounded 5 % @ 0.5 m/s → ladder; weekly offline re-identify/re-adapt loop running | Ladder legs pass with gates; flywheel produces a refreshed candidate from field data alone, human-promoted |

**Explicitly not claimed:** anything past FK-6 before the room is booked. FK-7
and FK-8 follow the M6 discipline of `docs/ERFORCE_2V2.md` — procedure written
now, run when robots and room exist.

### 6.1 Build status — v0.1 (2026-09-23)

The framework skeleton is built, tested, and has run once on the real data.
Evidence: `evidence/audit.json`, `evidence/smoke-2026-09-23/`; tests: 26 passed
(`python -m pytest tests -q`); smoke: `python -m fawkes.cli run-smoke --tag <date>`.

| Milestone | Landed in v0.1 | Evidence / honest reading |
| --- | --- | --- |
| FK-0 audit | **done** | `evidence/audit.json`: 162 CSV traces, 67,593 rows, schema histogram A:107/B:27/C:20/D:8 — **exactly the evidence pack's own numbers**; transitions 61,087 (48,942/12,145 split) hash-pinned; firmware snapshot confirmed `main@bab9d03` |
| FK-1 FTF-1 + converters | **done** (2 stubs documented) | `phoenix_csv` (all 4 schemas, duplicate-timestamp handling), `onboard_motion_log` (the `0-` negative repair), `rlearn_npz` round-trip tests. `phoenix_session` lands with FK-7; `ssl_vision_log` is the follow-up |
| FK-2 sys-ID | **done, structure** | Wheel ensemble fit on the real 61,087 transitions: held-out delta RMSE **[0.0835, 0.0522, 0.0593, 0.0465]** vs rlearn's own [0.0798, 0.0494, 0.0526, 0.0445] (all rows — matched within 4–12 % with the reduced 30-feature set); active-rows RMSE is worse [0.35, 0.18, 0.23, 0.18] vs [0.21, 0.13, 0.14, 0.12] — the gap is stated in the model card, not hidden (`evidence/smoke-2026-09-23/sysid/model_card.json`) |
| FK-3 DR env | **done** (pure-Python backend) | Batched env, 10 DR axes, z conditioning, cascade-in-the-loop, bounded residual with the verbatim clamps; ~400 batch-steps/s. The MJX/Brax GPU backend is the documented FK-3 slot |
| FK-4 learner | **CEM done** (real run), PPO slot open | CEM over the 11 gains (pop 20 × 24 iters): champion beats the baseline on **both** suites — practice 19.2 mm vs 19.8 mm endpoint p95 and 57.2 vs 62.5 mm cross-track p95; DR family 19.4 vs 19.7 / 63.8 vs 67.3. **The practice suite passes every gate** (the apply-today criterion). The DR family fails one gate for every condition: heading max 0.17 rad vs the 0.08 limit on asymmetric-wheel worlds — the known honest gap |
| FK-5 RMA | **two-phase structure done** (linear adapter v0.1) | Phase A CEM over the zero-init base MLP (identity-start at the pure cascade); Phase B closed-form ridge adapter, 31,480 (features, z) pairs + z-probe. The adapter recovers the oracle's behaviour (practice: zero-z 33.2 → adapter 30.6 → oracle 30.4 mm; DR: adapter 30.7 beats oracle 37.4) but does **not** beat the pure cascade at this budget — reported, not hidden |
| FK-6 export trio | **done** | Firmware registry entry (`SIMULATION ONLY`, `deployment_authorized: false`); residual actor (clamps verbatim, ABI pending-confirmation); Phoenix bundle `movement-rma-1` — verify: 12 pairs, max_abs_error 2.4e-6 (BLAS vs pure-Python summation order; tolerance 1e-4 = 12.5 µm of residual authority) |
| FK-7/8 | untouched — no robot, no room | procedure stays in this README |

**The honest training-run reading (2026-09-23, evidence/train-2026-09-23/).**
The first training run exposed a real environment bug, which is the system
working as designed: every second leg of a two-leg episode was being timed out
at ~1.6 s mid-flight (the leg timeout was computed against the *previous*
leg's already-reached target). Fixed with a regression test
(`tests/test_envs.py::test_two_leg_episodes_settle_the_second_leg_too`); the
numbers below are from the retrain on the fixed environment.

After the fix the picture is clean: on the **practice family** (tight ranges
around the field-proven 2026-07-23 world — the apply-today gate) **the CEM
champion passes every gate**: endpoint p95 19.2 mm (limit 35), cross-track p95
57.2 mm (limit 120), and it beats the field-proven baseline on both metrics.
On the harsh **DR family**, endpoint and cross-track hold (19.4 / 63.8 mm) but
the heading gate fails for *every* condition including the baseline
(0.17 rad max vs the 0.08 limit) — asymmetric-wheel worlds are the known gap;
that is FK-4/5 work, and it is why the field A/B starts at 0.5 m/s. The RMA
adapter at this budget recovers the oracle's behaviour (mechanism works) but
does not beat the pure cascade; the residual **actor is not fielded today** —
only the gain champion rides the existing registry path. Full tables:
`evidence/train-2026-09-23/LATEST.md`; the field procedure is
[FIELD_DAY.md](FIELD_DAY.md).

**Continued training, same day (`evidence/train-long-2026-09-23/`).** A
second, longer run (CEM 24 × 40, RMA 20 × 28) improved the champion again on
both suites — practice **18.1 mm endpoint / 56.4 mm cross-track p95** (baseline
19.8 / 62.5), DR family 18.4 / 61.9 — and this is the run FIELD_DAY.md
carries. Every run now renders a one-page picture of itself:
`python -m fawkes.cli dashboard --tag <tag>` → `evidence/<tag>/dashboard.png`
(trajectories baseline-vs-champion on practice and DR worlds, gate bars,
Phase-A convergence, the blind-adaptation curve, and the z-probe R² —
including its honest weakness).

---

## 7. Risks and unknowns, honestly

| Risk | Mitigation |
| --- | --- |
| The simulator lies (rlearn's 203-vs-104 mm warning is the precedent) | DR families cover model error, not just parameter spreads; replay-vs-real-traces is a standing gate; the ladder never trusts sim alone |
| `z` learns the wrong thing (adapts to noise, not physics) | Phase-B supervised probe; gate on blind-transfer suite; zero-`z` fallback trained explicitly; bounded authority caps damage while it's wrong |
| GPU availability for 1024-env-scale training | MJX/Brax run on one consumer GPU; cloud run acceptable (only the JSON artifact ships); pure-Python backend keeps development unblocked |
| Robot-side Python/JSON latency at 50 Hz | Small nets; measured budget on the Pi before FK-7; the runtime already fields a champion JSON actor, so the ABI is proven |
| Firmware drift: the `RoboCup-Research` snapshot is `main` @ `bab9d03` — **no adaptive runtime in it**; the live robot runs the `experimental` branch | FK-0 pins the exact live revision (see `docs/TELEMETRY_SOURCE_INVENTORY.md`); all firmware work targets the live branch, not the stale snapshot |
| One robot / one carpet / one day of real data | DR covers the family; flywheel grows coverage every session; per-robot profiles by design (Robot A's IMU axis 1 / sign −1 is baked into its description, not the framework) |
| Overclaiming | House evidence rules: every number cites a run; `deployment_authorized: false` until a human flips it; blind-window performance reported alongside adapted performance |
| Zero-dependency ethos of Phoenix | Training stack lives here, isolated; the only thing imported into Phoenix is a pure-stdlib consumer module; artifacts are JSON |

---

## 8. Prior art worth reading before building

- **RMA — Rapid Motor Adaptation** (Kumar et al., 2021): the hidden-vector
  idea FAWKES implements, proven on legged robots.
- **rSoccer / rSim** (RoboCIn + Mila, arXiv:2106.12895): published RL
  environments for SSL; benchmark tasks from the 2021 hardware challenges —
  read before writing task code; extending may beat starting over
  (`PHYSICS_SIMULATION.md` says the same).
- **TIGERs Mannheim** Sumatra + firmware: the architecture Phoenix follows;
  the delayed-vision estimator and per-tick trajectory regeneration come from
  there (`docs/TIGERS_MAPPING.md`, `RobotFramework/docs/MOTION.md`).
- **Dreamer / world models**: the heavier alternative to RMA (learn the
  dynamics, imagine rollouts). Not chosen first — the identified wheel model
  already gives us cheap, fast dynamics; revisit only if RMA stalls.
- **ER-Force simulator** (`simulator-cli`): already bridged in
  `phoenix/simlink` — the independent-physics cross-check.

---

## 9. Repository layout (as built in v0.1)

```
FAWKES/
  README.md            # this document — the plan and handoff
  fawkes/
    trace/             # FTF-1: format, converters, manifest, hashing
    sysid/             # wheel ensemble, DR families, model cards
    envs/              # vectorised DR envs, task suite, cascade-in-the-loop
    policies/          # CEM gains, PPO, RMA adapter, z-probe
    export/            # registry profile, residual actor, Phoenix bundle
    evaluate/          # gates, blind-transfer suite, reports (house style)
    viz/               # replaytest heritage: before/after, corridors
  data/                # converted FTF episodes (raw stays where it lives)
  evidence/            # run artifacts, per the Phoenix evidence convention
  tests/
  pyproject.toml       # training stack deps live HERE, not in Phoenix
```

---

## Appendix A — the dig: inventory and verdicts

Everything found while researching this plan (2026-09-23). Paths are under
`C:\Users\rishi\Documents\GitHub\` unless noted.

### A.1 `RoboCup-Research/` (root)

Five vision screenshots (`vision-0.blob.jpg`, `vision-field-latest.jpg`, …) and
five subfolders. Root images: field/vision captures, 2026-09-23 timestamps —
useful context only.

### A.2 `RoboCup-Research/generalinformation/` — the 2026-07-23 evidence pack

**Verdict: the most reliable artifact set in the workspace.** 187 files,
32,345,060 bytes, every one SHA-256 hashed, regenerated by
`generate_evidence_catalog.py`; commits pinned (Phoenix `77cd99c`,
RobotFramework `905868c`); explicit unknowns list ("facts not established").

Contents: the calibration dossier, final parameter reference, reproduction
runbook, evidence manifest, session context & unknowns (battery 24.8 →
23.0–23.5 V; camera artifacts; kicker off; qualitative observations kept
separate from quantitative acceptance), exhaustive trace statistics (162 CSVs /
67,593 rows / 721.7 s; dataset extremes honestly labelled as artifacts where
they are), CSV statistics, verbatim config snapshots, protocol/units doc, and
`calibration/`:

- `latest.json` — the 8-leg acceptance report, `passed: true`;
- `profiles/practice-baseline-2026-07-23.json` — `LOCKED ROLLBACK`,
  `deployment_authorized: true`, the field-proven reference;
- `profiles/offline-learned-candidate-20260805.json` — `SIMULATION ONLY`,
  `deployment_authorized: false`, with its own warning that the simulator
  overpredicted cross-track 203 vs 104 mm real.

Plus ~180 calibration-run CSVs (tune-run-*, kinfit-*, wheelnorm-*, brake-*,
yaw-*, imu-*, diag-*, strongyaw-*, stage*-*, center_from_*, x/y_pos/neg …),
12 runtime logs, and visual evidence. And `tests/` — a Phoenix test-suite
snapshot of that era (not current; the live repo is the authority).

### A.3 `RoboCup-Research/testing/rlearn/` — the offline learner

**Verdict: solid, reproducible, honest — and not the lost PPO rig.**

- Data: 61,087 one-step transitions (48,942 train / 12,145 held-out), 3,765
  coupled translation+yaw rows, 162 CSV traces hashed (32.0 MB), manifest
  SHA-256.
- Model: first-order wheel response + 9-member ridge ensemble, 47 features →
  4 next-tick wheel deltas; held-out RMSE 0.044–0.081 rev/s (all rows),
  0.115–0.215 (active rows); motor lag 1 sample @ 50 Hz.
- Search: cross-entropy method over 11 bounded controller gains
  (`configs/search_space.json`), acceptance gates incl. endpoint 35 mm,
  cross-track 120 mm, heading 0.08 rad, wheel saturation ≤ 8 %.
- Results: champion (seed 20260805) robust improvement 99.8 % vs baseline in
  sim (objective 2729 → 10.2), max endpoint 10.9 mm, cross-track 64.9 mm,
  zero boundary/speed violations — and still `simulation acceptance: False`
  under its own conservative gate, `deployment_authorized: false` everywhere.
- Ops: continuous offline search (`main.py continuous`, STOP file, atomic
  checkpoints), rotation-training handoff (generation 23/28 at stop, seed
  20260729), SAFETY.md gate protocol for any human-supervised field check.

**Not present:** `residual_env.py`, any PPO/DR trainer, or anything that could
have produced `rgtap-ppo-dr-i900`. The M-1 search continues elsewhere; FAWKES
builds the rig regardless (recovering the original, if ever found, becomes a
cross-check instead of a blocker).

### A.4 `RoboCup-Research/replaytest/` — digital twin validator

**Verdict: good validation UX; inherit as `fawkes/viz`.** Commit-pinned
(Phoenix `77cd99c`, RobotFramework `905868c`), measured baselines table
(2.0 m/s +X: endpoint 5.2 mm, cross RMS 20.9 mm; 2.5 m/s ceiling leg: 4.6 mm),
8-leg envelope (worst endpoint 19.45 mm / worst cross 104.11 mm), pre-IMU-fix
failure trace (830.6 mm endpoint) vs final (5.2 mm) as the standing argument
for per-chassis IMU verification, and an explicit "coverage is not a success
probability" honesty rule.

### A.5 `RoboCup-Research/RobotFramework/` — firmware snapshot

**Verdict: trustworthy but STALE — an era snapshot, not the live firmware.**
Branch `main`, HEAD `bab9d03` "Install RLearn motion calibration profile"
(atop `905868c` "Calibrate Robot A precision motion at 2.5 m/s"). It contains
the motion stack (250 Hz tick, delayed-vision estimator with 128-slot ring,
per-tick bang-bang regeneration, Panthera cascade, moteus velocity mode),
`config/motion_calibrations.json` (registry with the field-proven rollback and
the rlearn champion installed for guarded validation), `Motion.yaml` with the
RLEARN values and rollback comments, `tools/select_motion_calibration.py`,
host tests incl. `sim_robot.cpp` (perfect-tracking plant) and golden
MatchCtrl vectors — but **no `Motion/adaptive.cpp`, no off/collect/shadow/
bounded runtime**: the adaptive runtime lives on the robot's `experimental`
branch (per `Phoenix-Server/docs/TELEMETRY_SOURCE_INVENTORY.md`: branch
`experimental`, HEAD `86a7f05d` at audit time). All runtime-targeting work
must confirm against the live revision.

### A.6 `RoboCup-Research/Phoenix-Server/` — old server snapshot

A small, old server checkout (skills/get_ball era, no `rl/`, no `simlink/`).
**Ignore it; the live workspace repo is the authority.** Kept only as
provenance of the July era.

### A.7 The live Phoenix-Server workspace (branch `burn-in`, off experimental)

The authority for the server side. What matters to FAWKES: motion profiles
`configs/motion/` (`baseline-validated` ACTIVE, `rl-shadow`, `rl-bounded-dr-i900`
with the fielded champion header); `phoenix/sim/world.py` `Surface` DR
vocabulary; `phoenix/rl/team/` ES/bundle/curriculum patterns to extend;
`phoenix/simlink/` ER-Force bridge (our controller, their physics);
`docs/PHYSICS_SIMULATION.md` (the Option-C rationale FAWKES fulfils);
`docs/ERFORCE_2V2.md` (the evidence discipline to copy).

---

## Appendix B — key reference numbers (as found, unrounded where the source has them)

| Thing | Value | Source |
| --- | --- | --- |
| Field-proven worst endpoint | 0.01945406478910938 m | `generalinformation/calibration/latest.json` (DIAGONAL SE) |
| Field-proven worst cross-track | 0.10411076467542528 m | same (X NEGATIVE) |
| Acceptance limits | endpoint 0.035 m, cross-track 0.12 m, abort 0.2 m | `latest.json` standard |
| Speed ceiling / accel | 2.5 m/s / 2.5 m/s² (server) | baseline profile + `latest.json` |
| Baseline controller | kp_pos 5.0, kp_vel 0.7, kp_heading 6.0, kp_yaw_rate 0.3, acc_ff_lead 0.14 s, omega_corr_max 3.0, brake_scale 0.5, orient_lag_tau 0.05 s | `practice-baseline-2026-07-23.json` |
| Kinematics | meters_per_motor_rev 0.2171, lateral_scale 0.887, cmd scales [1.006, 1.008, 1.004, 1.004] | `rlearn/configs/baseline.json` |
| Identified motor lag | 1 sample @ 50 Hz (~20 ms) | rlearn manifest |
| Wheel limits | 45 rev/s, slew 120 rev/s² | baseline fixed limits |
| Wire | MatchCtrl 32 B → :50514, MatchFeedback 35 B → :50513, skills 0–7, LE ints | `RobotFramework/docs/PROTOCOL.md` |
| Rates | control 250 Hz, feedback 50 Hz, camera 60 Hz, vision age ~16 ms quantised | `MOTION.md` / PROTOCOL.md / trace stats |
| Battery | 6S, 19.8–25.2 V window, 19.0 V trip; session 24.8 → 23.0–23.5 V | session context |
| Bounded RL clamps | 5 % authority, slew 0.5 / 1.5, confidence 0.30, ≤ 5 interventions | `configs/motion/rl-bounded-dr-i900.toml` |
| Fielded champion | rgtap-ppo-dr-i900, 1024 envs × horizon 128, iter 900; 182/186 ticks applied, residual mean 8.7 / max 29 mm/s, settle 4.8 mm | same |
| rlearn held-out RMSE | 0.044–0.081 rev/s (all), 0.115–0.215 (active) | `reports/LATEST.md` |
| Phoenix ES bundle | spec `team-knobs-1`, 13 obs / 7 knobs, flat-18: 1W/23D/0L vs baseline | bundle.json + ERFORCE_2V2.md |

---

*This document is the plan of record for FAWKES. When the first milestone
lands, the status table in §6 gets dates and run directories — the Phoenix
way: evidence, not claims.*
