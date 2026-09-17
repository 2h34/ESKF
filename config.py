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
    """Numerical guards; these are not ESKF tuning parameters."""

    quaternion_norm_epsilon: float = 1e-12
    minimum_valid_quaternion_norm: float = 1e-8
    maximum_quaternion_normalization_error: float = 1e-3
    small_angle_epsilon: float = 1e-8
    covariance_negative_tolerance: float = 1e-12


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
    """Continuous-time 6D ESKF noise tuning.

    ``gyro_bias_random_walk_density`` is the amplitude of the white driving
    noise in the later continuous gyro-bias random-walk model. The default is
    an engineering starting value, not a calibration result from this dataset.
    """

    gyro_bias_random_walk_density: float = 1e-4


@dataclass(frozen=True)
class ProjectConfig:
    numerical: NumericalSafetyConfig = field(default_factory=NumericalSafetyConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    static: StaticAnalysisConfig = field(default_factory=StaticAnalysisConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    filter_noise: FilterNoiseConfig = field(default_factory=FilterNoiseConfig)


DEFAULT_CONFIG = ProjectConfig()
