"""Project-wide conventions, preprocessing settings, and filter tuning.

Only fixed conventions, data-quality thresholds, numerical safety values, and
explicit engineering tuning parameters belong here. Values estimated from the
supplied recording are deliberately not stored in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


QUATERNION_ORDER = "xyzw"
ROTATION_MATRIX_CONVENTION = "R_WB maps body-frame vectors into world frame"
ERROR_CONVENTION = "right multiplicative: R_true = R_hat Exp(delta_theta^)"
GRAVITY_MPS2 = 9.81
GRAVITY_WORLD_MPS2 = np.array([0.0, 0.0, -GRAVITY_MPS2], dtype=float)


@dataclass(frozen=True)
class NumericalSafetyConfig:
    """Numerical guards; these are not ESKF tuning parameters.

    这些量只规定何时把近零四元数、明显负协方差或显著非对称矩阵判为数值
    错误，不能用来改变滤波器对物理噪声和观测可信度的判断。
    """

    quaternion_norm_epsilon: float = 1e-12
    minimum_valid_quaternion_norm: float = 1e-8
    maximum_quaternion_normalization_error: float = 1e-3
    small_angle_epsilon: float = 1e-8
    covariance_negative_tolerance: float = 1e-12
    covariance_symmetry_tolerance: float = 1e-12


@dataclass(frozen=True)
class PreprocessConfig:
    """Raw-file validation policy and timestamp anomaly thresholds."""

    invalid_sample_policy: Literal["raise", "drop"] = "drop"
    timestamp_small_step_ratio: float = 0.5
    timestamp_large_step_ratio: float = 1.8


@dataclass(frozen=True)
class StaticAnalysisConfig:
    """Transparent aggregate checks for the initial candidate interval."""

    candidate_duration_s: float = 2.0
    max_gyro_mean_norm_rad_s: float = 0.05
    max_gyro_std_component_rad_s: float = 0.02
    max_acc_norm_error_mps2: float = 0.5
    max_acc_std_component_mps2: float = 0.20
    max_acc_norm_std_mps2: float = 0.20


@dataclass(frozen=True)
class AlignmentConfig:
    """Nearest-timestamp matching settings."""

    tolerance_ratio: float = 0.5
    absolute_tolerance_s: float | None = None


@dataclass(frozen=True)
class FilterNoiseConfig:
    """Continuous-time ESKF bias-driving noise tuning.

    ``gyro_bias_random_walk_density`` is the amplitude of the white driving
    noise in the continuous gyro-bias random-walk model. The default is
    an engineering starting value, not a calibration result from this dataset.

    它描述“零偏随时间漂移得多快”，不是当前零偏 ``b_g`` 的数值，也不是
    静止段陀螺测量标准差；后两者分别属于名义状态估计和测量噪声统计。

    ``accelerometer_bias_random_walk_density`` 对应 ``dot(b_a)=n_ba``，描述
    加速度计零偏随机漂移的驱动噪声强度。它不是测量噪声、当前 ``b_a``、
    静止段 ``acc_std`` 或标定结果；默认值只是 15D 的工程起始值。
    """

    gyro_bias_random_walk_density: float = 1e-4
    accelerometer_bias_random_walk_density: float = 1e-3


@dataclass(frozen=True)
class InitializationConfig:
    """Prior uncertainty used only when constructing the initial state.

    ``initial_velocity_std_mps`` 是初始静止条件下 World-frame 速度先验的
    标准差，不是 process noise，也不是由当前约 2 秒数据标定得到的结果。
    """

    initial_velocity_std_mps: float = 0.1


@dataclass(frozen=True)
class ProjectConfig:
    numerical: NumericalSafetyConfig = field(default_factory=NumericalSafetyConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    static: StaticAnalysisConfig = field(default_factory=StaticAnalysisConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    filter_noise: FilterNoiseConfig = field(default_factory=FilterNoiseConfig)
    initialization: InitializationConfig = field(
        default_factory=InitializationConfig
    )


DEFAULT_CONFIG = ProjectConfig()
