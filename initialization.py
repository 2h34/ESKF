"""6D attitude ESKF initialization only.

This module freezes the nominal attitude and gyro-bias estimate at the final
sample of the accepted static interval, and constructs ``P0`` and the future
continuous-time process-noise covariance ``Qc``. It intentionally contains no
prediction, covariance propagation, observation update, injection, or reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG, ProjectConfig
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from rotation_utils import normalize_quaternion


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ESKF6DInitialization:
    """Initialization result at ``timestamp == imu.timestamp[imu_index]``.

    ``q0_xyzw`` is the nominal ``q_WB`` attitude and ``bg0_rad_s`` is the
    nominal gyro-bias estimate. ``P0`` is ordered as
    ``[delta_theta, delta_b_g]``. ``Qc`` is the continuous-time covariance of
    gyro measurement noise followed by gyro-bias random-walk driving noise.
    No persistent error-state mean is stored.
    """

    imu_index: int
    pose_index: int
    timestamp: float
    q0_xyzw: FloatArray
    bg0_rad_s: FloatArray
    P0: FloatArray
    Qc: FloatArray
    static_sample_count: int
    gyro_static_std_rad_s: FloatArray
    median_dt_s: float
    gyro_noise_density: FloatArray
    gyro_bias_random_walk_density: float


def _finite_vector(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _validate_positive_symmetric_matrix(matrix: FloatArray, name: str) -> None:
    if matrix.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError(f"{name} diagonal must be strictly positive")


def initialize_eskf6d(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> ESKF6DInitialization:
    """Construct ``q0``, ``bg0``, ``P0``, and ``Qc`` at the frozen index ``k0``."""

    imu_timestamp = np.asarray(imu.timestamp, dtype=float)
    if imu_timestamp.ndim != 1 or imu_timestamp.size == 0:
        raise ValueError("imu.timestamp must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(imu_timestamp)):
        raise ValueError("imu.timestamp must contain only finite values")

    static_start = int(init_stats.static_start_idx)
    static_end = int(init_stats.static_end_idx)
    if not (0 <= static_start < static_end <= imu_timestamp.size):
        raise ValueError("static interval must be non-empty and inside the IMU data")
    if not init_stats.is_static:
        raise ValueError("initialization requires an accepted static interval")

    static_sample_count = static_end - static_start
    imu_index = static_end - 1
    timestamp = float(imu_timestamp[imu_index])
    if not np.isclose(
        float(init_stats.static_end_time), timestamp, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "init_stats.static_end_time must equal the last static IMU timestamp"
        )

    pose_index_for_imu = np.asarray(alignment.pose_index_for_imu)
    if pose_index_for_imu.shape != (imu_timestamp.size,):
        raise ValueError(
            "alignment.pose_index_for_imu must have one entry per IMU sample"
        )
    pose_index = int(pose_index_for_imu[imu_index])
    if pose_index < 0:
        raise ValueError(
            f"initialization IMU index {imu_index} has no matched FAST-LIO pose"
        )

    pose_quaternion = np.asarray(pose.quaternion_xyzw, dtype=float)
    pose_covariance = np.asarray(pose.cov_diag, dtype=float)
    if pose_quaternion.ndim != 2 or pose_quaternion.shape[1] != 4:
        raise ValueError("pose.quaternion_xyzw must have shape (M, 4)")
    if pose_covariance.shape != (pose_quaternion.shape[0], 6):
        raise ValueError("pose.cov_diag must have shape (M, 6)")
    if pose_index >= pose_quaternion.shape[0]:
        raise ValueError(f"matched pose index {pose_index} is out of range")

    q0 = normalize_quaternion(pose_quaternion[pose_index], config.numerical)
    bg0 = _finite_vector(init_stats.gyro_mean, "init_stats.gyro_mean")
    gyro_static_std = _finite_vector(
        init_stats.gyro_std, "init_stats.gyro_std"
    )
    if np.any(gyro_static_std <= 0.0):
        raise ValueError("init_stats.gyro_std must be strictly positive")

    attitude_covariance = np.asarray(
        pose_covariance[pose_index, 3:6], dtype=float
    )
    if not np.all(np.isfinite(attitude_covariance)):
        raise ValueError("initialization pose attitude covariance must be finite")
    if np.any(attitude_covariance <= 0.0):
        raise ValueError(
            "initialization pose attitude covariance must be strictly positive"
        )

    bias_covariance = gyro_static_std**2 / static_sample_count
    P0 = np.diag(np.concatenate((attitude_covariance, bias_covariance)))

    median_dt = float(init_stats.median_dt)
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError("init_stats.median_dt must be finite and positive")
    gyro_noise_density = gyro_static_std * np.sqrt(median_dt)
    gyro_measurement_noise = gyro_noise_density**2

    bias_random_walk_density = float(
        config.filter_noise.gyro_bias_random_walk_density
    )
    if not np.isfinite(bias_random_walk_density) or bias_random_walk_density <= 0.0:
        raise ValueError(
            "gyro_bias_random_walk_density must be finite and strictly positive"
        )
    bias_random_walk_noise = np.full(
        3, bias_random_walk_density**2, dtype=float
    )
    Qc = np.diag(
        np.concatenate((gyro_measurement_noise, bias_random_walk_noise))
    )

    _validate_positive_symmetric_matrix(P0, "P0")
    _validate_positive_symmetric_matrix(Qc, "Qc")

    return ESKF6DInitialization(
        imu_index=imu_index,
        pose_index=pose_index,
        timestamp=timestamp,
        q0_xyzw=q0,
        bg0_rad_s=bg0,
        P0=P0,
        Qc=Qc,
        static_sample_count=static_sample_count,
        gyro_static_std_rad_s=gyro_static_std,
        median_dt_s=median_dt,
        gyro_noise_density=gyro_noise_density,
        gyro_bias_random_walk_density=bias_random_walk_density,
    )
