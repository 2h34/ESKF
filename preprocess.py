"""Raw CSV loading, validation, SI conversion, and static-segment analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG, GRAVITY_MPS2, ProjectConfig
from data_types import (
    IMUDiagnostics,
    InitStats,
    PoseDiagnostics,
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
    if timestamp.ndim != 1 or timestamp.size < 2:
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
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[ProcessedIMUData, IMUDiagnostics]:
    """Load IMU CSV by column name and convert accelerometer g to m/s^2."""

    values, invalid_rows, raw_count = _read_numeric_csv(
        path, IMU_COLUMNS, config.preprocess.invalid_sample_policy
    )
    timestamp = values[:, 0]
    differences = _validate_timestamps(timestamp, "IMU")
    dt = np.empty(timestamp.shape, dtype=float)
    dt[0] = np.nan
    dt[1:] = differences
    median_dt = float(np.median(differences))
    small_limit = config.preprocess.timestamp_small_step_ratio * median_dt
    large_limit = config.preprocess.timestamp_large_step_ratio * median_dt
    small_step_indices = np.flatnonzero(differences < small_limit) + 1
    large_step_indices = np.flatnonzero(differences > large_limit) + 1
    processed = ProcessedIMUData(
        timestamp=timestamp,
        dt=dt,
        gyro_rad_s=values[:, [4, 5, 6]],
        acc_mps2=values[:, [1, 2, 3]] * GRAVITY_MPS2,
    )
    diagnostics = IMUDiagnostics(
        sample_count_raw=raw_count,
        sample_count_processed=timestamp.size,
        invalid_sample_count=len(invalid_rows),
        invalid_source_rows=invalid_rows,
        start_timestamp=float(timestamp[0]),
        end_timestamp=float(timestamp[-1]),
        duration_s=float(timestamp[-1] - timestamp[0]),
        mean_dt=float(np.mean(differences)),
        median_dt=median_dt,
        min_dt=float(np.min(differences)),
        max_dt=float(np.max(differences)),
        estimated_frequency_hz=1.0 / median_dt,
        small_step_count=int(small_step_indices.size),
        large_step_count=int(large_step_indices.size),
        small_step_destination_indices=tuple(int(index) for index in small_step_indices),
        large_step_destination_indices=tuple(int(index) for index in large_step_indices),
        large_step_dt_s=tuple(float(dt[index]) for index in large_step_indices),
    )
    return processed, diagnostics


def load_pose_csv(
    path: str | Path,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[ProcessedPoseData, PoseDiagnostics]:
    """Load FAST-LIO pose/covariance CSV, validate, and normalize xyzw quaternions."""

    values, invalid_rows, raw_count = _read_numeric_csv(
        path, POSE_COLUMNS, config.preprocess.invalid_sample_policy
    )
    timestamp = values[:, 0]
    differences = _validate_timestamps(timestamp, "pose")
    quaternion_raw = values[:, 4:8]
    quaternion_norm = np.linalg.norm(quaternion_raw, axis=1)
    numerical = config.numerical
    invalid_quaternion = (
        (quaternion_norm < numerical.minimum_valid_quaternion_norm)
        | (np.abs(quaternion_norm - 1.0) > numerical.maximum_quaternion_normalization_error)
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
        covariance_diag < -numerical.covariance_negative_tolerance, axis=1
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
    median_dt = float(np.median(differences))
    diagnostics = PoseDiagnostics(
        sample_count_raw=raw_count,
        sample_count_processed=timestamp.size,
        invalid_sample_count=len(invalid_rows),
        invalid_source_rows=invalid_rows,
        start_timestamp=float(timestamp[0]),
        end_timestamp=float(timestamp[-1]),
        duration_s=float(timestamp[-1] - timestamp[0]),
        median_dt=median_dt,
        estimated_frequency_hz=1.0 / median_dt,
        quaternion_norm_mean_raw=float(np.mean(quaternion_norm)),
        quaternion_norm_min_raw=float(np.min(quaternion_norm)),
        quaternion_norm_max_raw=float(np.max(quaternion_norm)),
        invalid_quaternion_count=invalid_quaternion_count,
        invalid_covariance_count=invalid_covariance_count,
        zero_covariance_frame_count=int(np.count_nonzero(zero_covariance)),
        covariance_diag_min=np.min(covariance_diag, axis=0),
        covariance_diag_mean=np.mean(covariance_diag, axis=0),
        covariance_diag_median=np.median(covariance_diag, axis=0),
        covariance_diag_max=np.max(covariance_diag, axis=0),
    )
    return processed, diagnostics


def analyze_initial_static_segment(
    imu: ProcessedIMUData,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> InitStats:
    """Validate and summarize the configured initial candidate static interval."""

    static_config = config.static
    if static_config.candidate_duration_s <= 0.0:
        raise ValueError("candidate static duration must be positive")
    candidate_end = int(
        np.searchsorted(
            imu.timestamp,
            imu.timestamp[0] + static_config.candidate_duration_s,
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
            <= static_config.max_gyro_mean_norm_rad_s,
            "gyro mean norm within threshold",
            "gyro mean norm exceeds threshold",
        ),
        (
            float(np.max(gyro_std))
            <= static_config.max_gyro_std_component_rad_s,
            "gyro component standard deviations within threshold",
            "gyro component standard deviation exceeds threshold",
        ),
        (
            abs(acc_norm_mean - GRAVITY_MPS2)
            <= static_config.max_acc_norm_error_mps2,
            "accelerometer norm mean is close to 9.81 m/s^2",
            "accelerometer norm mean is not close to 9.81 m/s^2",
        ),
        (
            float(np.max(acc_std)) <= static_config.max_acc_std_component_mps2,
            "accelerometer component standard deviations within threshold",
            "accelerometer component standard deviation exceeds threshold",
        ),
        (
            acc_norm_std <= static_config.max_acc_norm_std_mps2,
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


def save_processed_imu(path: str | Path, data: ProcessedIMUData) -> None:
    np.savez(
        Path(path),
        timestamp=data.timestamp,
        dt=data.dt,
        gyro_rad_s=data.gyro_rad_s,
        acc_mps2=data.acc_mps2,
    )


def save_processed_pose(path: str | Path, data: ProcessedPoseData) -> None:
    np.savez(
        Path(path),
        timestamp=data.timestamp,
        position=data.position,
        quaternion_xyzw=data.quaternion_xyzw,
        cov_diag=data.cov_diag,
    )


def save_init_stats(path: str | Path, stats: InitStats) -> None:
    np.savez(
        Path(path),
        static_start_idx=stats.static_start_idx,
        static_end_idx=stats.static_end_idx,
        static_start_time=stats.static_start_time,
        static_end_time=stats.static_end_time,
        candidate_start_idx=stats.candidate_start_idx,
        candidate_end_idx=stats.candidate_end_idx,
        candidate_start_time=stats.candidate_start_time,
        candidate_end_time=stats.candidate_end_time,
        gyro_mean=stats.gyro_mean,
        gyro_std=stats.gyro_std,
        acc_mean=stats.acc_mean,
        acc_std=stats.acc_std,
        acc_norm_mean=stats.acc_norm_mean,
        acc_norm_std=stats.acc_norm_std,
        mean_dt=stats.mean_dt,
        median_dt=stats.median_dt,
        min_dt=stats.min_dt,
        max_dt=stats.max_dt,
        is_static=stats.is_static,
        decision_reasons=np.asarray(stats.decision_reasons, dtype="U"),
    )
