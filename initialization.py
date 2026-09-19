"""6D and 15D ESKF initialization only.

This module freezes nominal states at the final sample of the accepted static
interval and constructs the corresponding ``P0`` and continuous-time ``Qc``.
It intentionally contains no prediction, covariance propagation, observation
update, injection, reset, or scheduling loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG, GRAVITY_WORLD_MPS2, ProjectConfig
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from rotation_utils import hat, normalize_quaternion, quaternion_to_rotation_matrix


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


@dataclass(frozen=True)
class ESKF15DInitialization:
    """Frozen 15D initialization at one matched IMU/FAST-LIO timestamp.

    ``p0_W_m`` and ``v0_W_mps`` are expressed in World frame, ``q0_xyzw`` is
    ``q_WB``, and ``bg0_rad_s`` / ``ba0_mps2`` are expressed in Body/IMU frame.
    ``ba0_mps2`` is accelerometer specific-force measurement bias, not a
    World-frame acceleration bias.

    ``P0`` ordering is ``[delta_p, delta_v, delta_theta, delta_b_g, delta_b_a]``.
    Its first version models the physically derived ``delta_theta`` /
    ``delta_b_a`` cross covariance caused by gravity compensation with an
    uncertain initial attitude; other unsupported initial cross blocks remain
    zero.
    Continuous-time ``Qc`` ordering is ``[n_a, n_g, n_bg, n_ba]``.
    """

    imu_index: int
    pose_index: int
    timestamp: float
    p0_W_m: FloatArray
    v0_W_mps: FloatArray
    q0_xyzw: FloatArray
    bg0_rad_s: FloatArray
    ba0_mps2: FloatArray
    P0: FloatArray
    Qc: FloatArray
    static_sample_count: int
    gyro_static_std_rad_s: FloatArray
    acc_static_std_mps2: FloatArray
    median_dt_s: float
    gyro_noise_density: FloatArray
    acc_noise_density: FloatArray
    gyro_bias_random_walk_density: float
    accelerometer_bias_random_walk_density: float


@dataclass(frozen=True)
class _CommonInitializationContext:
    """Quantities shared without changing the established 6D definitions."""

    imu_index: int
    pose_index: int
    timestamp: float
    q0_xyzw: FloatArray
    bg0_rad_s: FloatArray
    pose_covariance_diag: FloatArray
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


def _validate_positive_symmetric_matrix(
    matrix: FloatArray,
    name: str,
    expected_shape: tuple[int, int],
) -> None:
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError(f"{name} diagonal must be strictly positive")


def _common_initialization_context(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    config: ProjectConfig,
) -> _CommonInitializationContext:
    """Validate and compute quantities shared by 6D and 15D initialization."""

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

    # static_end_idx 采用 Python 切片的右开边界，因此 k0=static_end-1 才是
    # 已确认静止区间内最后一个 IMU 样本，也是正式传播开始前的名义状态时刻。
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

    # 6D 和 15D 使用同一个 j0，不在初始化中重新做 timestamp matching。
    q0 = normalize_quaternion(pose_quaternion[pose_index], config.numerical)
    bg0 = _finite_vector(init_stats.gyro_mean, "init_stats.gyro_mean")
    gyro_static_std = _finite_vector(
        init_stats.gyro_std, "init_stats.gyro_std"
    )
    if np.any(gyro_static_std <= 0.0):
        raise ValueError("init_stats.gyro_std must be strictly positive")

    pose_covariance_diag = np.array(
        pose_covariance[pose_index], dtype=float, copy=True
    )
    attitude_covariance = pose_covariance_diag[3:6]
    if not np.all(np.isfinite(attitude_covariance)):
        raise ValueError("initialization pose attitude covariance must be finite")
    if np.any(attitude_covariance <= 0.0):
        raise ValueError(
            "initialization pose attitude covariance must be strictly positive"
        )

    median_dt = float(init_stats.median_dt)
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise ValueError("init_stats.median_dt must be finite and positive")
    gyro_noise_density = gyro_static_std * np.sqrt(median_dt)

    gyro_bias_random_walk_density = float(
        config.filter_noise.gyro_bias_random_walk_density
    )
    if (
        not np.isfinite(gyro_bias_random_walk_density)
        or gyro_bias_random_walk_density <= 0.0
    ):
        raise ValueError(
            "gyro_bias_random_walk_density must be finite and strictly positive"
        )

    return _CommonInitializationContext(
        imu_index=imu_index,
        pose_index=pose_index,
        timestamp=timestamp,
        q0_xyzw=q0,
        bg0_rad_s=bg0,
        pose_covariance_diag=pose_covariance_diag,
        static_sample_count=static_sample_count,
        gyro_static_std_rad_s=gyro_static_std,
        median_dt_s=median_dt,
        gyro_noise_density=gyro_noise_density,
        gyro_bias_random_walk_density=gyro_bias_random_walk_density,
    )


def initialize_eskf6d(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> ESKF6DInitialization:
    """Construct ``q0``, ``bg0``, ``P0``, and ``Qc`` at the frozen index ``k0``.

    误差态均值初始化为零，只表示名义状态当前没有待注入的修正；它不代表
    状态完全确定。因此 ``P0`` 必须保留初始姿态和零偏估计的不确定性。
    """

    common = _common_initialization_context(
        imu, pose, init_stats, alignment, config
    )

    # P0 前三维是初始姿态误差协方差，后三维是静止均值作为零偏估计时的
    # 均值不确定性。误差态均值为零并不意味着这些协方差也应为零。
    attitude_covariance = common.pose_covariance_diag[3:6]
    bias_covariance = (
        common.gyro_static_std_rad_s**2 / common.static_sample_count
    )
    P0 = np.diag(np.concatenate((attitude_covariance, bias_covariance)))

    # 静止段角速度标准差用于估计陀螺测量白噪声密度；它描述单次测量噪声，
    # 与下方描述零偏随时间缓慢漂移强度的 bias random walk density 不是一回事。
    gyro_measurement_noise = common.gyro_noise_density**2

    # 零偏随机游走密度来自显式工程配置，并非用本段静止 gyro std 冒充得到。
    bias_random_walk_noise = np.full(
        3, common.gyro_bias_random_walk_density**2, dtype=float
    )
    Qc = np.diag(
        np.concatenate((gyro_measurement_noise, bias_random_walk_noise))
    )

    _validate_positive_symmetric_matrix(P0, "P0", (6, 6))
    _validate_positive_symmetric_matrix(Qc, "Qc", (6, 6))

    return ESKF6DInitialization(
        imu_index=common.imu_index,
        pose_index=common.pose_index,
        timestamp=common.timestamp,
        q0_xyzw=common.q0_xyzw,
        bg0_rad_s=common.bg0_rad_s,
        P0=P0,
        Qc=Qc,
        static_sample_count=common.static_sample_count,
        gyro_static_std_rad_s=common.gyro_static_std_rad_s,
        median_dt_s=common.median_dt_s,
        gyro_noise_density=common.gyro_noise_density,
        gyro_bias_random_walk_density=common.gyro_bias_random_walk_density,
    )


def initialize_eskf15d(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> ESKF15DInitialization:
    """Construct the frozen 15D nominal state, ``P0``, and continuous ``Qc``.

    The error-state ordering is fixed as
    ``[delta_p, delta_v, delta_theta, delta_b_g, delta_b_a]``. This function
    initializes those quantities only; it performs no prediction or update.
    """

    common = _common_initialization_context(
        imu, pose, init_stats, alignment, config
    )

    pose_position = np.asarray(pose.position, dtype=float)
    pose_count = np.asarray(pose.quaternion_xyzw).shape[0]
    if pose_position.shape != (pose_count, 3):
        raise ValueError("pose.position must have shape (M, 3)")
    p0_W_m = _finite_vector(
        pose_position[common.pose_index],
        "initialization pose position",
    )
    v0_W_mps = np.zeros(3, dtype=float)

    acc_mean = _finite_vector(init_stats.acc_mean, "init_stats.acc_mean")
    acc_static_std = _finite_vector(init_stats.acc_std, "init_stats.acc_std")
    if np.any(acc_static_std <= 0.0):
        raise ValueError("init_stats.acc_std must be strictly positive")

    position_covariance = common.pose_covariance_diag[0:3]
    if not np.all(np.isfinite(position_covariance)):
        raise ValueError("initialization pose position covariance must be finite")
    if np.any(position_covariance <= 0.0):
        raise ValueError(
            "initialization pose position covariance must be strictly positive"
        )
    attitude_covariance = common.pose_covariance_diag[3:6]
    Ptheta0 = np.diag(attitude_covariance)

    initial_velocity_std_mps = float(
        config.initialization.initial_velocity_std_mps
    )
    if (
        not np.isfinite(initial_velocity_std_mps)
        or initial_velocity_std_mps <= 0.0
    ):
        raise ValueError(
            "initial_velocity_std_mps must be finite and strictly positive"
        )

    R0_WB = quaternion_to_rotation_matrix(common.q0_xyzw)
    gravity_B0_mps2 = R0_WB.T @ GRAVITY_WORLD_MPS2
    # 静止时 a_W≈0，因此 specific-force measurement bias 位于 Body/IMU frame：
    # b_a0 = mean(a_m) + R_WB(q0)^T g_W。
    ba0_mps2 = acc_mean + gravity_B0_mps2

    P0 = np.zeros((15, 15), dtype=float)
    P0[0:3, 0:3] = np.diag(position_covariance)
    P0[3:6, 3:6] = initial_velocity_std_mps**2 * np.eye(3, dtype=float)
    P0[6:9, 6:9] = Ptheta0
    P0[9:12, 9:12] = np.diag(
        common.gyro_static_std_rad_s**2 / common.static_sample_count
    )

    P_acc_mean = np.diag(acc_static_std**2 / common.static_sample_count)
    J_g = hat(gravity_B0_mps2)
    P_ba_gravity = J_g @ Ptheta0 @ J_g.T
    # 由 delta_b_a ≈ J_g delta_theta - n_bar_a 可知，使用不确定的初始姿态
    # 补偿重力时，delta_theta 与 delta_b_a 具有以下互协方差；其余没有可靠
    # 建模依据的初始 cross covariance 在第一版中仍保持为零。
    P0[6:9, 12:15] = Ptheta0 @ J_g.T
    P0[12:15, 6:9] = J_g @ Ptheta0
    P0[12:15, 12:15] = P_acc_mean + P_ba_gravity

    acc_noise_density = acc_static_std * np.sqrt(common.median_dt_s)
    accelerometer_bias_random_walk_density = float(
        config.filter_noise.accelerometer_bias_random_walk_density
    )
    if (
        not np.isfinite(accelerometer_bias_random_walk_density)
        or accelerometer_bias_random_walk_density <= 0.0
    ):
        raise ValueError(
            "accelerometer_bias_random_walk_density must be finite and "
            "strictly positive"
        )

    Qc = np.diag(
        np.concatenate(
            (
                acc_noise_density**2,
                common.gyro_noise_density**2,
                np.full(
                    3,
                    common.gyro_bias_random_walk_density**2,
                    dtype=float,
                ),
                np.full(
                    3,
                    accelerometer_bias_random_walk_density**2,
                    dtype=float,
                ),
            )
        )
    )

    _validate_positive_symmetric_matrix(P0, "P0", (15, 15))
    _validate_positive_symmetric_matrix(Qc, "Qc", (12, 12))

    return ESKF15DInitialization(
        imu_index=common.imu_index,
        pose_index=common.pose_index,
        timestamp=common.timestamp,
        p0_W_m=p0_W_m,
        v0_W_mps=v0_W_mps,
        q0_xyzw=common.q0_xyzw,
        bg0_rad_s=common.bg0_rad_s,
        ba0_mps2=ba0_mps2,
        P0=P0,
        Qc=Qc,
        static_sample_count=common.static_sample_count,
        gyro_static_std_rad_s=common.gyro_static_std_rad_s,
        acc_static_std_mps2=acc_static_std,
        median_dt_s=common.median_dt_s,
        gyro_noise_density=common.gyro_noise_density,
        acc_noise_density=acc_noise_density,
        gyro_bias_random_walk_density=common.gyro_bias_random_walk_density,
        accelerometer_bias_random_walk_density=(
            accelerometer_bias_random_walk_density
        ),
    )
