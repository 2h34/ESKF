"""Named arrays passed between preprocessing and the filter."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ProcessedIMUData:
    timestamp: FloatArray
    dt: FloatArray
    gyro_rad_s: FloatArray
    acc_mps2: FloatArray


@dataclass(frozen=True)
class ProcessedPoseData:
    timestamp: FloatArray
    position: FloatArray
    quaternion_xyzw: FloatArray
    cov_diag: FloatArray


@dataclass(frozen=True)
class InitStats:
    static_start_idx: int
    static_end_idx: int
    static_start_time: float
    static_end_time: float
    gyro_mean: FloatArray
    gyro_std: FloatArray
    acc_mean: FloatArray
    acc_std: FloatArray
    median_dt: float
    is_static: bool
    decision_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    # -1 denotes an IMU sample with no matching pose observation.
    pose_index_for_imu: IntArray
