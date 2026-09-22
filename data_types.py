"""Named array groups shared by preprocessing and the filter."""

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
    median_dt: float
    is_static: bool
    decision_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    """Pose association for every IMU sample.

    ``pose_index_for_imu[k] == -1`` means no observation met the tolerance.
    """

    pose_index_for_imu: IntArray
