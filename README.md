# IMU + FAST-LIO ESKF groundwork (phase 1)

This directory currently implements only the data and rotation foundation for
the later 6D attitude ESKF:

- raw IMU and FAST-LIO CSV validation;
- accelerometer conversion from `g` to `m/s^2`;
- initial static-candidate statistics;
- monotonic one-to-one timestamp alignment;
- `xyzw` quaternion and rotation helpers;
- synthetic and real-data tests.
- validated processed-data loaders for later runtime code;
- a repeatable gravity sanity check for pose quaternion direction.

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

The phase-1 convention check uses only processed data and does not run a filter:

```powershell
python scripts/check_pose_convention.py
```

`tests/` and `scripts/check_pose_convention.py` are development-time validation
assets. They are deliberately isolated from runtime algorithm modules and can
be reviewed separately during the final submission cleanup.

## Invalid-row policy

The default policy drops a row only when it cannot be parsed as a complete,
finite numeric record. Every dropped source CSV row is listed in the diagnostic
report. Set `invalid_sample_policy="raise"` in a supplied `ProjectConfig` when a
strict fail-fast pipeline is required. Timestamp duplicates or reversals,
invalid quaternions, and negative covariance diagonals always raise errors.

## FAST-LIO quaternion direction evidence

The project continues to interpret the CSV quaternion as active `R_WB`, but the
evidence has two distinct scopes:

1. Standard FAST-LIO source supports `R_WB`. In
   [`pointBodyToWorld`](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L165-L169),
   `state_point.rot` multiplies an IMU/body-frame point before adding the global
   position. The process model similarly computes inertial acceleration as
   [`s.rot * (in.acc - s.ba)`](https://github.com/hku-mars/FAST_LIO/blob/main/include/use-ikfom.hpp#L42-L53).
   FAST-LIO copies [`state_point.rot` into `geoQuat`](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L887-L897),
   then publishes odometry with parent frame `camera_init` and child frame
   [`body`](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L545-L574).
2. The supplied CSV export code is not in this repository. Source evidence for
   standard FAST-LIO therefore cannot prove that the CSV exporter applied no
   additional inverse or frame transform.

The independent gravity check uses 347 matched static samples after excluding
the 20 poses with non-positive attitude covariance:

| Quaternion hypothesis | Mean angle | Median angle | Maximum angle | Mean residual |
| --- | ---: | ---: | ---: | ---: |
| `q = R_WB` | 0.736329 deg | 0.656430 deg | 1.699284 deg | 0.146254 m/s^2 |
| `q = R_BW` | 1.335364 deg | 1.293349 deg | 2.516289 deg | 0.240803 m/s^2 |

`R_WB` has the smaller per-sample angular error for 86.17% of the samples. The
representative mean-orientation errors are 0.400813 deg (`R_WB`) and 1.231878
deg (`R_BW`). Together with the source/frame evidence, the result **strongly
supports `R_WB`**. Gravity constrains roll and pitch but cannot independently
verify yaw semantics.

## Observation covariance assumptions

The first 20 pose frames have an all-zero covariance diagonal. Their timestamp
range is `1786181234.225484610` to `1786181234.319545984`; the selected static
interval ends at `1786181236.055275679`. A filter starting at the static end
therefore does not use these 20 observations in the current recording.

For the first ESKF version, an observation with any attitude variance `<= 0`
should be skipped. Zero variance would claim perfect certainty and can make an
update singular or overwhelmingly strong. Replacing it with an unspecified
floor would introduce a hidden tuning parameter and reinterpret uninitialized
metadata as a valid observation. A covariance floor is only appropriate later
if the source documents the meaning of zero and a defensible minimum variance
is calibrated; no `R_min` is chosen in this phase.

The exam labels `cov_33`, `cov_44`, and `cov_55` as roll, pitch, and yaw
variances. Euler-angle perturbations are not strictly the same coordinates as a
right-multiplicative local rotation vector: the mapping depends on the nominal
attitude, Euler convention, perturbation side/frame, and omitted cross terms.
The assignment nevertheless clearly expects these three values to be used.
For small attitude errors, this project will use their diagonal as an
observation-noise approximation, document that approximation, and avoid adding
an unrequested Euler-to-tangent covariance Jacobian.

## FAST-LIO and IMU correlation assumption

[FAST-LIO is a tightly coupled LiDAR-inertial estimator](https://github.com/hku-mars/FAST_LIO#fast-lio)
and uses IMU acceleration and angular velocity in its process model. Its
attitude output is therefore not statistically independent of the same IMU
gyro used by this assignment's prediction. Ignoring the cross-correlation can
double-count information, underestimate uncertainty, and make a nominal Kalman
filter inconsistent.

The exam does not request correlated-noise or cross-covariance modelling. This
assignment will therefore use the standard simplifying assumption that process
and observation noise are independent. This is an explicit coursework
approximation, not a claim of strict statistical independence.
