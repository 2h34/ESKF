"""Validated loaders for the phase-1 processed ``.npz`` artifacts.

These functions are the only supported boundary between saved preprocessing
artifacts and later runtime code. Loading never changes quaternion direction,
covariance values, timestamps, or measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _read_npz(path: str | Path, required_keys: set[str]) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = required_keys.difference(archive.files)
            if missing:
                raise ValueError(f"{npz_path} is missing keys: {sorted(missing)}")
            return {key: np.array(archive[key], copy=True) for key in required_keys}
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


def _strictly_increasing(timestamp: FloatArray, label: str) -> None:
    if timestamp.size < 2:
        raise ValueError(f"{label} must contain at least two samples")
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")


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
    _strictly_increasing(timestamp, "IMU timestamp")
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
    _strictly_increasing(timestamp, "pose timestamp")
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    numerical = DEFAULT_CONFIG.numerical
    if np.any(quaternion_norm < numerical.minimum_valid_quaternion_norm):
        raise ValueError("quaternion_xyzw contains a near-zero quaternion")
    if np.any(
        np.abs(quaternion_norm - 1.0)
        > numerical.maximum_quaternion_normalization_error
    ):
        raise ValueError("quaternion_xyzw contains a non-unit quaternion")
    if np.any(covariance < -numerical.covariance_negative_tolerance):
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
