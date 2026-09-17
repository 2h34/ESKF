"""Typed runtime containers for phase-1 data and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ProcessedIMUData:
    """Validated IMU measurements in SI units.

    ``timestamp`` and ``dt`` have shape ``(N,)`` in seconds. ``dt[0]`` is NaN
    because no preceding measurement exists. ``gyro_rad_s`` and ``acc_mps2``
    have shape ``(N, 3)`` and remain raw measurements (no bias or gravity
    compensation is applied).
    """

    timestamp: FloatArray
    dt: FloatArray
    gyro_rad_s: FloatArray
    acc_mps2: FloatArray


@dataclass(frozen=True)
class ProcessedPoseData:
    """Validated FAST-LIO poses.

    ``quaternion_xyzw`` has shape ``(M, 4)``. Runtime code interprets it as
    ``R_WB`` under the documented project convention; standard FAST-LIO source
    semantics and the gravity sanity check support that interpretation, while
    the unavailable CSV export chain remains an explicit assumption. The
    covariance diagonal has order ``[x, y, z, roll, pitch, yaw]``.
    """

    timestamp: FloatArray
    position: FloatArray
    quaternion_xyzw: FloatArray
    cov_diag: FloatArray


@dataclass(frozen=True)
class InitStats:
    """Statistics for the accepted initial static interval."""

    static_start_idx: int
    static_end_idx: int
    static_start_time: float
    static_end_time: float
    gyro_mean: FloatArray
    gyro_std: FloatArray
    acc_mean: FloatArray
    acc_std: FloatArray
    acc_norm_mean: float
    acc_norm_std: float
    mean_dt: float
    median_dt: float
    min_dt: float
    max_dt: float
    candidate_start_idx: int
    candidate_end_idx: int
    candidate_start_time: float
    candidate_end_time: float
    is_static: bool
    decision_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    """Pose association for every IMU sample.

    ``pose_index_for_imu[k] == -1`` means no observation met the tolerance.
    For a valid match ``j``, ``time_error[k] = pose_time[j] - imu_time[k]``.
    Unmatched time errors are NaN.
    """

    pose_index_for_imu: IntArray
    time_error: FloatArray
    tolerance_s: float


@dataclass(frozen=True)
class IMUDiagnostics:
    sample_count_raw: int
    sample_count_processed: int
    invalid_sample_count: int
    invalid_source_rows: tuple[int, ...]
    start_timestamp: float
    end_timestamp: float
    duration_s: float
    mean_dt: float
    median_dt: float
    min_dt: float
    max_dt: float
    estimated_frequency_hz: float
    small_step_count: int
    large_step_count: int
    small_step_destination_indices: tuple[int, ...]
    large_step_destination_indices: tuple[int, ...]
    large_step_dt_s: tuple[float, ...]


@dataclass(frozen=True)
class PoseDiagnostics:
    sample_count_raw: int
    sample_count_processed: int
    invalid_sample_count: int
    invalid_source_rows: tuple[int, ...]
    start_timestamp: float
    end_timestamp: float
    duration_s: float
    median_dt: float
    estimated_frequency_hz: float
    quaternion_norm_mean_raw: float
    quaternion_norm_min_raw: float
    quaternion_norm_max_raw: float
    invalid_quaternion_count: int
    invalid_covariance_count: int
    zero_covariance_frame_count: int
    covariance_diag_min: FloatArray
    covariance_diag_mean: FloatArray
    covariance_diag_median: FloatArray
    covariance_diag_max: FloatArray


@dataclass(frozen=True)
class AlignmentDiagnostics:
    valid_match_count: int
    unmatched_imu_count: int
    unused_pose_count: int
    match_ratio: float
    mean_abs_time_error_s: float
    median_abs_time_error_s: float
    max_abs_time_error_s: float
    duplicate_pose_use_count: int
    tolerance_s: float
    unmatched_imu_indices: tuple[int, ...]
    unused_pose_indices: tuple[int, ...]
