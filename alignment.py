"""Monotonic one-to-one timestamp association."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from config import DEFAULT_CONFIG, AlignmentConfig
from data_types import AlignmentDiagnostics, AlignmentResult


def _timestamps(value: ArrayLike, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{label} must be a one-dimensional sequence with at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")
    return array


def compute_alignment_tolerance(
    imu_timestamp: ArrayLike,
    pose_timestamp: ArrayLike,
    config: AlignmentConfig = DEFAULT_CONFIG.alignment,
) -> float:
    """Derive tolerance from measured median periods unless overridden.

    tolerance 表示两个传感器时间戳仍可视为同一时刻的最大误差；默认从实际
    采样周期推导，也允许配置显式秒数，避免把 CSV 行号当成时间对齐依据。
    """

    imu_time = _timestamps(imu_timestamp, "imu_timestamp")
    pose_time = _timestamps(pose_timestamp, "pose_timestamp")
    if config.absolute_tolerance_s is not None:
        if config.absolute_tolerance_s <= 0.0:
            raise ValueError("absolute timestamp tolerance must be positive")
        return float(config.absolute_tolerance_s)
    if config.tolerance_ratio <= 0.0:
        raise ValueError("timestamp tolerance ratio must be positive")
    imu_period = float(np.median(np.diff(imu_time)))
    pose_period = float(np.median(np.diff(pose_time)))
    return config.tolerance_ratio * max(imu_period, pose_period)


def align_timestamps(
    imu_timestamp: ArrayLike,
    pose_timestamp: ArrayLike,
    config: AlignmentConfig = DEFAULT_CONFIG.alignment,
) -> AlignmentResult:
    """Match nearest timestamps in O(N+M), with unique monotonic pose use.

    For each IMU sample, only the first two not-yet-consumed pose timestamps can
    be candidates. Selecting the second permanently skips the first, which is
    required to keep the association ordered and one-to-one.

    输出中的 ``-1`` 表示该 IMU 帧没有可用 observation，runner 应继续纯
    Prediction；pose cursor 只向前移动，保证同一条 FAST-LIO pose 不会复用。
    """

    imu_time = _timestamps(imu_timestamp, "imu_timestamp")
    pose_time = _timestamps(pose_timestamp, "pose_timestamp")
    tolerance = compute_alignment_tolerance(imu_time, pose_time, config)
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


def summarize_alignment(
    result: AlignmentResult,
    pose_sample_count: int,
) -> AlignmentDiagnostics:
    valid = result.pose_index_for_imu >= 0
    matched = result.pose_index_for_imu[valid]
    unmatched_indices = np.flatnonzero(~valid)
    unused_pose_indices = np.setdiff1d(
        np.arange(pose_sample_count, dtype=np.int64), matched, assume_unique=True
    )
    absolute_error = np.abs(result.time_error[valid])
    unique_count = np.unique(matched).size
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        mean_error = median_error = max_error = float("nan")
    else:
        mean_error = float(np.mean(absolute_error))
        median_error = float(np.median(absolute_error))
        max_error = float(np.max(absolute_error))
    return AlignmentDiagnostics(
        valid_match_count=valid_count,
        unmatched_imu_count=int(valid.size - valid_count),
        unused_pose_count=int(pose_sample_count - unique_count),
        match_ratio=float(valid_count / valid.size),
        mean_abs_time_error_s=mean_error,
        median_abs_time_error_s=median_error,
        max_abs_time_error_s=max_error,
        duplicate_pose_use_count=int(valid_count - unique_count),
        tolerance_s=result.tolerance_s,
        unmatched_imu_indices=tuple(int(index) for index in unmatched_indices),
        unused_pose_indices=tuple(int(index) for index in unused_pose_indices),
    )


def save_alignment(path: str | Path, result: AlignmentResult) -> None:
    np.savez(
        Path(path),
        pose_index_for_imu=result.pose_index_for_imu,
        time_error=result.time_error,
        tolerance_s=result.tolerance_s,
    )
