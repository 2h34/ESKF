"""Fixed conventions, input thresholds, and tuning for the offline 15D ESKF."""

import numpy as np

# xyzw Hamilton quaternions; R_WB maps Body to World; right-multiplicative error.
GRAVITY_MPS2 = 9.81
GRAVITY_WORLD_MPS2 = np.array([0.0, 0.0, -GRAVITY_MPS2])

# Numerical guards are not noise tuning. Bias-driving densities and initial
# velocity uncertainty are engineering priors, not calibration results.
# NumericalSafety
QUATERNION_NORM_EPSILON = 1e-12
MINIMUM_VALID_QUATERNION_NORM = 1e-08
MAXIMUM_QUATERNION_NORMALIZATION_ERROR = 0.001
SMALL_ANGLE_EPSILON = 1e-08
COVARIANCE_NEGATIVE_TOLERANCE = 1e-12
COVARIANCE_SYMMETRY_TOLERANCE = 1e-12

# Preprocess
INVALID_SAMPLE_POLICY = 'drop'
TIMESTAMP_SMALL_STEP_RATIO = 0.5
TIMESTAMP_LARGE_STEP_RATIO = 1.8

# StaticAnalysis
CANDIDATE_DURATION_S = 2.0
MAX_GYRO_MEAN_NORM_RAD_S = 0.05
MAX_GYRO_STD_COMPONENT_RAD_S = 0.02
MAX_ACC_NORM_ERROR_MPS2 = 0.5
MAX_ACC_STD_COMPONENT_MPS2 = 0.2
MAX_ACC_NORM_STD_MPS2 = 0.2

# Alignment
TOLERANCE_RATIO = 0.5
ABSOLUTE_TOLERANCE_S = None

# FilterNoise
GYRO_BIAS_RANDOM_WALK_DENSITY = 0.0001
ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY = 0.001

# Initialization
INITIAL_VELOCITY_STD_MPS = 0.1
