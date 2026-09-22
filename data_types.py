"""预处理与滤波器共用的具名数组分组。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ProcessedIMUData:
    """经过校验的 IMU 测量，使用 SI 单位。

    ``timestamp`` 与 ``dt`` 形状为 ``(N,)``，单位秒。``dt[0]`` 为 NaN，
    因为第一个样本没有前一样本。``gyro_rad_s`` 与 ``acc_mps2`` 形状为
    ``(N, 3)``，且保持原始测量值（不做任何零偏补偿或重力补偿）。
    """

    timestamp: FloatArray
    dt: FloatArray
    gyro_rad_s: FloatArray
    acc_mps2: FloatArray


@dataclass(frozen=True)
class ProcessedPoseData:
    """经过校验的 FAST-LIO 位姿。

    ``quaternion_xyzw`` 形状为 ``(M, 4)``。运行期代码按项目文档约定的
    ``R_WB`` 解释它；标准 FAST-LIO 源码语义与重力合理性检查都支持这一解释，
    而无法获取的 CSV 导出链路仍是一个明确的假设。协方差对角项的顺序为
    ``[x, y, z, roll, pitch, yaw]``。
    """

    timestamp: FloatArray
    position: FloatArray
    quaternion_xyzw: FloatArray
    cov_diag: FloatArray


@dataclass(frozen=True)
class InitStats:
    """被接受的初始静止区间的统计量。"""

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
    """每个 IMU 样本对应的位姿关联结果。

    ``pose_index_for_imu[k] == -1`` 表示没有观测落在容差范围内。
    """

    pose_index_for_imu: IntArray
