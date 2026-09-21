"""Raw/processed data I/O, SI conversion, static analysis, and time association."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

import config as cfg
from data_types import (
    AlignmentResult,
    IntArray,
    InitStats,
    ProcessedIMUData,
    ProcessedPoseData,
)


FloatArray = NDArray[np.float64]
IMU_COLUMNS = ("time", "ax", "ay", "az", "gx", "gy", "gz")
POSE_BASE_COLUMNS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")
COVARIANCE_COLUMNS = tuple(f"cov_{row}{column}" for row in range(6) for column in range(6))
POSE_COLUMNS = POSE_BASE_COLUMNS + COVARIANCE_COLUMNS
COVARIANCE_DIAGONAL_COLUMNS = tuple(f"cov_{index}{index}" for index in range(6))


def _read_numeric_csv(
    path: str | Path,
    expected_columns: Sequence[str],
    invalid_policy: str,
) -> tuple[FloatArray, tuple[int, ...], int]:
    """Read named numeric columns and either reject or explicitly drop bad rows."""

    csv_path = Path(path)
    valid_rows: list[list[float]] = []
    invalid_rows: list[int] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc
        if len(header) != len(set(header)):
            raise ValueError(f"{csv_path} contains duplicate column names")
        missing = [column for column in expected_columns if column not in header]
        unexpected = [column for column in header if column not in expected_columns]
        if missing or unexpected:
            raise ValueError(
                f"{csv_path} schema mismatch; missing={missing}, unexpected={unexpected}"
            )
        column_indices = [header.index(column) for column in expected_columns]
        for source_row, row in enumerate(reader, start=2):
            try:
                if len(row) != len(header):
                    raise ValueError("wrong column count")
                values = [float(row[index]) for index in column_indices]
                if not np.all(np.isfinite(values)):
                    raise ValueError("non-finite value")
            except (ValueError, IndexError):
                invalid_rows.append(source_row)
                continue
            valid_rows.append(values)
    raw_count = len(valid_rows) + len(invalid_rows)
    if invalid_rows and invalid_policy == "raise":
        raise ValueError(f"{csv_path} has invalid data rows {invalid_rows}")
    if invalid_policy not in {"raise", "drop"}:
        raise ValueError(f"unsupported invalid sample policy: {invalid_policy}")
    if not valid_rows:
        raise ValueError(f"{csv_path} has no valid data rows")
    return np.asarray(valid_rows, dtype=float), tuple(invalid_rows), raw_count


def _validate_timestamps(timestamp: FloatArray, label: str) -> FloatArray:
    if timestamp.ndim != 1 or timestamp.size < 2 or not np.all(np.isfinite(timestamp)):
        raise ValueError(f"{label} needs at least two timestamps")
    differences = np.diff(timestamp)
    duplicate = np.flatnonzero(differences == 0.0)
    decreasing = np.flatnonzero(differences < 0.0)
    if duplicate.size or decreasing.size:
        raise ValueError(
            f"{label} timestamps must strictly increase; duplicate transitions="
            f"{duplicate.tolist()}, decreasing transitions={decreasing.tolist()}"
        )
    return differences


def load_imu_csv(
    path: str | Path,
) -> tuple[ProcessedIMUData, dict]:
    """Load IMU CSV by column name and convert accelerometer g to m/s^2."""

    values, invalid_rows, raw_count = _read_numeric_csv(
        path, IMU_COLUMNS, cfg.INVALID_SAMPLE_POLICY
    )
    timestamp = values[:, 0]
    differences = _validate_timestamps(timestamp, "IMU")
    dt = np.empty(timestamp.shape, dtype=float)
    dt[0] = np.nan
    dt[1:] = differences
    median_dt = float(np.median(differences))
    small_limit = cfg.TIMESTAMP_SMALL_STEP_RATIO * median_dt
    large_limit = cfg.TIMESTAMP_LARGE_STEP_RATIO * median_dt
    small_step_indices = np.flatnonzero(differences < small_limit) + 1
    large_step_indices = np.flatnonzero(differences > large_limit) + 1
    processed = ProcessedIMUData(
        timestamp=timestamp,
        dt=dt,
        gyro_rad_s=values[:, [4, 5, 6]],
        # 原始加速度列以 g 为单位，统一换算成 SI 制 m/s^2；陀螺已是 rad/s。
        acc_mps2=values[:, [1, 2, 3]] * cfg.GRAVITY_MPS2,
    )
    diagnostics = {
        "sample_count_raw": raw_count, "sample_count_processed": int(timestamp.size),
        "invalid_sample_count": len(invalid_rows), "invalid_source_rows": invalid_rows,
        "start_timestamp": float(timestamp[0]), "end_timestamp": float(timestamp[-1]),
        "mean_dt": float(np.mean(differences)), "median_dt": median_dt,
        "min_dt": float(np.min(differences)), "max_dt": float(np.max(differences)),
        "small_step_count": int(small_step_indices.size),
        "large_step_count": int(large_step_indices.size),
    }
    return processed, diagnostics


def load_pose_csv(
    path: str | Path,
) -> tuple[ProcessedPoseData, dict]:
    """Load FAST-LIO pose/covariance CSV, validate, and normalize xyzw quaternions."""

    values, invalid_rows, raw_count = _read_numeric_csv(
        path, POSE_COLUMNS, cfg.INVALID_SAMPLE_POLICY
    )
    timestamp = values[:, 0]
    differences = _validate_timestamps(timestamp, "pose")
    quaternion_raw = values[:, 4:8]
    quaternion_norm = np.linalg.norm(quaternion_raw, axis=1)
    invalid_quaternion = (
        (quaternion_norm < cfg.MINIMUM_VALID_QUATERNION_NORM)
        | (np.abs(quaternion_norm - 1.0) > cfg.MAXIMUM_QUATERNION_NORMALIZATION_ERROR)
    )
    invalid_quaternion_count = int(np.count_nonzero(invalid_quaternion))
    if invalid_quaternion_count:
        indices = np.flatnonzero(invalid_quaternion).tolist()
        raise ValueError(f"pose has invalid quaternion frames {indices}")
    quaternion_xyzw = quaternion_raw / quaternion_norm[:, None]
    header_index = {column: index for index, column in enumerate(POSE_COLUMNS)}
    diagonal_indices = [header_index[column] for column in COVARIANCE_DIAGONAL_COLUMNS]
    covariance_diag = values[:, diagonal_indices]
    invalid_covariance = np.any(
        covariance_diag < -cfg.COVARIANCE_NEGATIVE_TOLERANCE, axis=1
    )
    invalid_covariance_count = int(np.count_nonzero(invalid_covariance))
    if invalid_covariance_count:
        indices = np.flatnonzero(invalid_covariance).tolist()
        raise ValueError(f"pose has negative covariance diagonal frames {indices}")
    zero_covariance = np.all(covariance_diag == 0.0, axis=1)
    processed = ProcessedPoseData(
        timestamp=timestamp,
        position=values[:, 1:4],
        quaternion_xyzw=quaternion_xyzw,
        cov_diag=covariance_diag,
    )
    diagnostics = {
        "sample_count_raw": raw_count, "sample_count_processed": int(timestamp.size),
        "invalid_sample_count": len(invalid_rows), "invalid_source_rows": invalid_rows,
        "start_timestamp": float(timestamp[0]), "end_timestamp": float(timestamp[-1]),
        "median_dt": float(np.median(differences)),
        "zero_covariance_frame_count": int(np.count_nonzero(zero_covariance)),
    }
    return processed, diagnostics


def analyze_initial_static_segment(
    imu: ProcessedIMUData,
) -> InitStats:
    """Validate and summarize the configured initial candidate static interval."""

    if cfg.CANDIDATE_DURATION_S <= 0.0:
        raise ValueError("candidate static duration must be positive")
    candidate_end = int(
        np.searchsorted(
            imu.timestamp,
            imu.timestamp[0] + cfg.CANDIDATE_DURATION_S,
            side="right",
        )
    )
    if candidate_end < 2:
        raise ValueError("candidate static interval contains fewer than two samples")
    start = 0
    stop = candidate_end
    gyro = imu.gyro_rad_s[start:stop]
    acc = imu.acc_mps2[start:stop]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)
    acc_mean = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    acc_norm_mean = float(np.mean(acc_norm))
    acc_norm_std = float(np.std(acc_norm))
    checks = (
        (
            float(np.linalg.norm(gyro_mean))
            <= cfg.MAX_GYRO_MEAN_NORM_RAD_S,
            "gyro mean norm within threshold",
            "gyro mean norm exceeds threshold",
        ),
        (
            float(np.max(gyro_std))
            <= cfg.MAX_GYRO_STD_COMPONENT_RAD_S,
            "gyro component standard deviations within threshold",
            "gyro component standard deviation exceeds threshold",
        ),
        (
            abs(acc_norm_mean - cfg.GRAVITY_MPS2)
            <= cfg.MAX_ACC_NORM_ERROR_MPS2,
            "accelerometer norm mean is close to 9.81 m/s^2",
            "accelerometer norm mean is not close to 9.81 m/s^2",
        ),
        (
            float(np.max(acc_std)) <= cfg.MAX_ACC_STD_COMPONENT_MPS2,
            "accelerometer component standard deviations within threshold",
            "accelerometer component standard deviation exceeds threshold",
        ),
        (
            acc_norm_std <= cfg.MAX_ACC_NORM_STD_MPS2,
            "accelerometer norm standard deviation within threshold",
            "accelerometer norm standard deviation exceeds threshold",
        ),
    )
    is_static = all(check[0] for check in checks)
    reasons = tuple(check[1] if check[0] else check[2] for check in checks)
    if not is_static:
        raise ValueError("initial candidate interval failed static checks: " + "; ".join(reasons))
    valid_dt = imu.dt[1:]
    return InitStats(
        static_start_idx=start,
        static_end_idx=stop,
        static_start_time=float(imu.timestamp[start]),
        static_end_time=float(imu.timestamp[stop - 1]),
        gyro_mean=gyro_mean,
        gyro_std=gyro_std,
        acc_mean=acc_mean,
        acc_std=acc_std,
        acc_norm_mean=acc_norm_mean,
        acc_norm_std=acc_norm_std,
        mean_dt=float(np.mean(valid_dt)),
        median_dt=float(np.median(valid_dt)),
        min_dt=float(np.min(valid_dt)),
        max_dt=float(np.max(valid_dt)),
        candidate_start_idx=start,
        candidate_end_idx=stop,
        candidate_start_time=float(imu.timestamp[start]),
        candidate_end_time=float(imu.timestamp[stop - 1]),
        is_static=is_static,
        decision_reasons=reasons,
    )



def align_timestamps(
    imu_timestamp: FloatArray,
    pose_timestamp: FloatArray,
) -> AlignmentResult:
    """Match nearest timestamps in O(N+M), with unique monotonic pose use."""

    imu_time = np.asarray(imu_timestamp, dtype=float)
    pose_time = np.asarray(pose_timestamp, dtype=float)
    imu_dt = _validate_timestamps(imu_time, "IMU")
    pose_dt = _validate_timestamps(pose_time, "pose")
    if cfg.ABSOLUTE_TOLERANCE_S is not None:
        tolerance = float(cfg.ABSOLUTE_TOLERANCE_S)
    else:
        tolerance = cfg.TOLERANCE_RATIO * max(float(np.median(imu_dt)), float(np.median(pose_dt)))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("timestamp tolerance must be finite and positive")
    pose_index = np.full(imu_time.shape, -1, dtype=np.int64)
    time_error = np.full(imu_time.shape, np.nan, dtype=float)
    pose_cursor = 0
    for imu_index, imu_value in enumerate(imu_time):
        while (
            pose_cursor < pose_time.size
            and pose_time[pose_cursor] < imu_value - tolerance
        ):
            pose_cursor += 1
        if pose_cursor >= pose_time.size:
            break
        candidate = pose_cursor
        if pose_cursor + 1 < pose_time.size:
            current_error = abs(pose_time[pose_cursor] - imu_value)
            next_error = abs(pose_time[pose_cursor + 1] - imu_value)
            if next_error < current_error:
                candidate = pose_cursor + 1
        error = float(pose_time[candidate] - imu_value)
        if abs(error) <= tolerance:
            pose_index[imu_index] = candidate
            time_error[imu_index] = error
            pose_cursor = candidate + 1
    return AlignmentResult(pose_index, time_error, tolerance)


def save_processed_data(
    directory: Path, imu: ProcessedIMUData, pose: ProcessedPoseData,
    stats: InitStats, alignment: AlignmentResult,
) -> None:
    """Save the four stable NPZ schemas; dataclass fields are their named keys."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("imu_processed.npz", imu), ("pose_processed.npz", pose),
        ("init_stats.npz", stats), ("match_table.npz", alignment),
    ):
        np.savez(directory / filename, **asdict(data))


def _read_npz(path: str | Path, required_keys: set[str]) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = required_keys.difference(archive.files)
            if missing:
                raise ValueError(f"{npz_path} is missing keys: {sorted(missing)}")
            return {key: archive[key] for key in required_keys}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(npz_path)):
            raise
        raise ValueError(f"failed to load {npz_path}: {exc}") from exc


def _float_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    try:
        value = np.asarray(arrays[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if value.shape != expected_shape:
        raise ValueError(f"{key} must have shape {expected_shape}, got {value.shape}")
    return value


def _finite_float_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    value = _float_array(arrays, key, expected_shape)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values")
    return value


def _scalar_float(arrays: Mapping[str, np.ndarray], key: str) -> float:
    value = _float_array(arrays, key, ())
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _scalar_int(arrays: Mapping[str, np.ndarray], key: str) -> int:
    value = np.asarray(arrays[key])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{key} must be an integer scalar")
    return int(value.item())


def _scalar_bool(arrays: Mapping[str, np.ndarray], key: str) -> bool:
    value = np.asarray(arrays[key])
    if value.shape != () or not np.issubdtype(value.dtype, np.bool_):
        raise ValueError(f"{key} must be a boolean scalar")
    return bool(value.item())


def load_processed_imu(path: str | Path) -> ProcessedIMUData:
    """Load and validate ``imu_processed.npz`` as ``ProcessedIMUData``."""

    arrays = _read_npz(path, {"timestamp", "dt", "gyro_rad_s", "acc_mps2"})
    timestamp_raw = np.asarray(arrays["timestamp"])
    if timestamp_raw.ndim != 1:
        raise ValueError(f"timestamp must have shape (N,), got {timestamp_raw.shape}")
    sample_count = timestamp_raw.size
    timestamp = _finite_float_array(arrays, "timestamp", (sample_count,))
    dt = _float_array(arrays, "dt", (sample_count,))
    gyro = _finite_float_array(arrays, "gyro_rad_s", (sample_count, 3))
    acc = _finite_float_array(arrays, "acc_mps2", (sample_count, 3))
    _validate_timestamps(timestamp, "IMU timestamp")
    if not np.isnan(dt[0]):
        raise ValueError("dt[0] must be NaN because the first sample has no predecessor")
    if not np.all(np.isfinite(dt[1:])) or np.any(dt[1:] <= 0.0):
        raise ValueError("dt[1:] must be finite and strictly positive")
    if not np.allclose(dt[1:], np.diff(timestamp), rtol=1e-10, atol=1e-12):
        raise ValueError("dt[1:] is inconsistent with timestamp differences")
    return ProcessedIMUData(timestamp, dt, gyro, acc)


def load_processed_pose(path: str | Path) -> ProcessedPoseData:
    """Load and validate ``pose_processed.npz`` as ``ProcessedPoseData``."""

    arrays = _read_npz(
        path, {"timestamp", "position", "quaternion_xyzw", "cov_diag"}
    )
    timestamp_raw = np.asarray(arrays["timestamp"])
    if timestamp_raw.ndim != 1:
        raise ValueError(f"timestamp must have shape (M,), got {timestamp_raw.shape}")
    sample_count = timestamp_raw.size
    timestamp = _finite_float_array(arrays, "timestamp", (sample_count,))
    position = _finite_float_array(arrays, "position", (sample_count, 3))
    quaternion = _finite_float_array(
        arrays, "quaternion_xyzw", (sample_count, 4)
    )
    covariance = _finite_float_array(arrays, "cov_diag", (sample_count, 6))
    _validate_timestamps(timestamp, "pose timestamp")
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    if np.any(quaternion_norm < cfg.MINIMUM_VALID_QUATERNION_NORM):
        raise ValueError("quaternion_xyzw contains a near-zero quaternion")
    if np.any(
        np.abs(quaternion_norm - 1.0)
        > cfg.MAXIMUM_QUATERNION_NORMALIZATION_ERROR
    ):
        raise ValueError("quaternion_xyzw contains a non-unit quaternion")
    if np.any(covariance < -cfg.COVARIANCE_NEGATIVE_TOLERANCE):
        raise ValueError("cov_diag contains a negative variance")
    return ProcessedPoseData(timestamp, position, quaternion, covariance)


_INIT_KEYS = {
    "static_start_idx",
    "static_end_idx",
    "static_start_time",
    "static_end_time",
    "candidate_start_idx",
    "candidate_end_idx",
    "candidate_start_time",
    "candidate_end_time",
    "gyro_mean",
    "gyro_std",
    "acc_mean",
    "acc_std",
    "acc_norm_mean",
    "acc_norm_std",
    "mean_dt",
    "median_dt",
    "min_dt",
    "max_dt",
    "is_static",
    "decision_reasons",
}


def load_init_stats(
    path: str | Path,
    *,
    imu_sample_count: int | None = None,
) -> InitStats:
    """Load and validate ``init_stats.npz`` as ``InitStats``."""

    arrays = _read_npz(path, _INIT_KEYS)
    static_start = _scalar_int(arrays, "static_start_idx")
    static_end = _scalar_int(arrays, "static_end_idx")
    candidate_start = _scalar_int(arrays, "candidate_start_idx")
    candidate_end = _scalar_int(arrays, "candidate_end_idx")
    if not (0 <= candidate_start <= static_start < static_end <= candidate_end):
        raise ValueError("static range must be non-empty and inside the candidate range")
    if imu_sample_count is not None and candidate_end > imu_sample_count:
        raise ValueError("candidate_end_idx exceeds the IMU sample count")

    static_start_time = _scalar_float(arrays, "static_start_time")
    static_end_time = _scalar_float(arrays, "static_end_time")
    candidate_start_time = _scalar_float(arrays, "candidate_start_time")
    candidate_end_time = _scalar_float(arrays, "candidate_end_time")
    if not (
        candidate_start_time
        <= static_start_time
        <= static_end_time
        <= candidate_end_time
    ):
        raise ValueError("static times must be ordered inside the candidate interval")

    gyro_mean = _finite_float_array(arrays, "gyro_mean", (3,))
    gyro_std = _finite_float_array(arrays, "gyro_std", (3,))
    acc_mean = _finite_float_array(arrays, "acc_mean", (3,))
    acc_std = _finite_float_array(arrays, "acc_std", (3,))
    if np.any(gyro_std < 0.0) or np.any(acc_std < 0.0):
        raise ValueError("standard deviations must be non-negative")

    acc_norm_mean = _scalar_float(arrays, "acc_norm_mean")
    acc_norm_std = _scalar_float(arrays, "acc_norm_std")
    mean_dt = _scalar_float(arrays, "mean_dt")
    median_dt = _scalar_float(arrays, "median_dt")
    min_dt = _scalar_float(arrays, "min_dt")
    max_dt = _scalar_float(arrays, "max_dt")
    if acc_norm_mean < 0.0 or acc_norm_std < 0.0:
        raise ValueError("accelerometer norm statistics must be non-negative")
    if min(mean_dt, median_dt, min_dt, max_dt) <= 0.0 or not (
        min_dt <= median_dt <= max_dt
    ):
        raise ValueError("dt statistics are invalid or inconsistent")

    reasons_array = np.asarray(arrays["decision_reasons"])
    if reasons_array.ndim != 1 or reasons_array.dtype.kind not in {"U", "S"}:
        raise ValueError("decision_reasons must be a one-dimensional string array")
    reasons = tuple(str(value) for value in reasons_array.tolist())
    return InitStats(
        static_start_idx=static_start,
        static_end_idx=static_end,
        static_start_time=static_start_time,
        static_end_time=static_end_time,
        gyro_mean=gyro_mean,
        gyro_std=gyro_std,
        acc_mean=acc_mean,
        acc_std=acc_std,
        acc_norm_mean=acc_norm_mean,
        acc_norm_std=acc_norm_std,
        mean_dt=mean_dt,
        median_dt=median_dt,
        min_dt=min_dt,
        max_dt=max_dt,
        candidate_start_idx=candidate_start,
        candidate_end_idx=candidate_end,
        candidate_start_time=candidate_start_time,
        candidate_end_time=candidate_end_time,
        is_static=_scalar_bool(arrays, "is_static"),
        decision_reasons=reasons,
    )


def load_alignment(
    path: str | Path,
    *,
    imu_sample_count: int,
    pose_sample_count: int,
) -> AlignmentResult:
    """Load and validate ``match_table.npz`` as ``AlignmentResult``."""

    arrays = _read_npz(
        path, {"pose_index_for_imu", "time_error", "tolerance_s"}
    )
    index_raw = np.asarray(arrays["pose_index_for_imu"])
    if index_raw.shape != (imu_sample_count,):
        raise ValueError(
            "pose_index_for_imu must have shape "
            f"({imu_sample_count},), got {index_raw.shape}"
        )
    if not np.issubdtype(index_raw.dtype, np.integer):
        raise ValueError("pose_index_for_imu must have an integer dtype")
    pose_index: IntArray = index_raw.astype(np.int64, copy=False)
    time_error = _float_array(arrays, "time_error", (imu_sample_count,))
    tolerance = _scalar_float(arrays, "tolerance_s")
    if tolerance <= 0.0:
        raise ValueError("tolerance_s must be positive")
    if np.any(pose_index < -1) or np.any(pose_index >= pose_sample_count):
        raise ValueError("pose_index_for_imu contains an out-of-range pose index")
    valid = pose_index >= 0
    if not np.all(np.isfinite(time_error[valid])):
        raise ValueError("matched time_error values must be finite")
    if not np.all(np.isnan(time_error[~valid])):
        raise ValueError("unmatched time_error values must be NaN")
    if np.any(np.abs(time_error[valid]) > tolerance + 1e-12):
        raise ValueError("a matched time_error exceeds tolerance_s")
    matched_pose = pose_index[valid]
    if matched_pose.size > 1 and np.any(np.diff(matched_pose) <= 0):
        raise ValueError("matched pose indices must be strictly increasing and unique")
    return AlignmentResult(pose_index, time_error, tolerance)
