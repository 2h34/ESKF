# IMU + FAST-LIO ESKF groundwork (phase 1)

This directory currently implements only the data and rotation foundation for
the later 6D attitude ESKF:

- raw IMU and FAST-LIO CSV validation;
- accelerometer conversion from `g` to `m/s^2`;
- initial static-candidate statistics;
- monotonic one-to-one timestamp alignment;
- `xyzw` quaternion and rotation helpers;
- synthetic and real-data tests.

It intentionally contains no ESKF prediction, update, injection, reset, or 15D
position-state implementation.

## Conventions

- Quaternion storage: `[qx, qy, qz, qw]` (`xyzw`).
- Rotation matrix: `v_W = R_WB @ v_B`.
- Future error convention: right multiplicative,
  `R_true = R_hat Exp(delta_theta^)`.
- World gravity: `[0, 0, -9.81] m/s^2`.
- `dt[0]` is `NaN`; it is not a fabricated sampling interval.
- Alignment error is `pose_timestamp[j] - imu_timestamp[k]`.

## Run

Use a Python environment with NumPy installed:

```powershell
python scripts/run_preprocess.py
python -m unittest discover -s tests -v
```

The preprocessing command writes four requested `.npz` files and both JSON and
plain-text diagnostics under `data/processed/`.

## Invalid-row policy

The default policy drops a row only when it cannot be parsed as a complete,
finite numeric record. Every dropped source CSV row is listed in the diagnostic
report. Set `invalid_sample_policy="raise"` in a supplied `ProjectConfig` when a
strict fail-fast pipeline is required. Timestamp duplicates or reversals,
invalid quaternions, and negative covariance diagonals always raise errors.
