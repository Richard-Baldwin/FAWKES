# data/

Raw movement data is **never copied into this repository** (the rlearn rule):
converters read the raw evidence in place and carry SHA-256 hashes forward.

- Default raw-data root: `C:\Users\rishi\Documents\GitHub\RoboCup-Research`
  (override with the `FAWKES_DATA_ROOT` environment variable).
- What lives there: the 2026-07-23 evidence pack (162 CSV traces, calibration
  profiles), the `testing/rlearn` lineage (transitions.npz, 61,087 rows), the
  RobotFramework firmware snapshot, and the replaytest digital twin - see
  README.md Appendix A for the full inventory and reliability verdicts.
- `fawkes audit` re-counts and hashes everything and writes `evidence/audit.json`.
- Live Measurement-session recordings (the flywheel's main food) land here as
  FTF-1 episodes once the `phoenix_session` converter ships with FK-7.
