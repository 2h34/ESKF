"""15D ESKF constants and project conventions."""

import numpy as np

# xyzw Hamilton quaternion; R_WB maps body vectors to world coordinates.
GRAVITY_MPS2 = 9.81
GRAVITY_WORLD_MPS2 = np.array([0.0, 0.0, -GRAVITY_MPS2])

# Static initialization checks.
CANDIDATE_DURATION_S = 2.0
MAX_GYRO_MEAN_NORM_RAD_S = 0.05
MAX_GYRO_STD_COMPONENT_RAD_S = 0.02
MAX_ACC_NORM_ERROR_MPS2 = 0.5
MAX_ACC_STD_COMPONENT_MPS2 = 0.2
MAX_ACC_NORM_STD_MPS2 = 0.2

# Timestamp matching and filter priors.
TOLERANCE_RATIO = 0.5
INITIAL_VELOCITY_STD_MPS = 0.1
GYRO_BIAS_RANDOM_WALK_DENSITY = 1e-4
ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY = 1e-3
SMALL_ANGLE_EPSILON = 1e-8
GIMBAL_LOCK_EPSILON = 1e-12
