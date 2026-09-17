"""Rotation helpers using xyzw Hamilton quaternions.

Quaternions represent active body-to-world rotations ``R_WB``. Multiplication
returns ``q1 tensor-product q2``. This convention is compatible with the current
right-multiplicative error definition ``R_true = R_hat Exp(delta_theta^)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config import DEFAULT_CONFIG, NumericalSafetyConfig


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def hat(vector: ArrayLike) -> FloatArray:
    """Return the 3x3 skew matrix satisfying ``hat(a) @ b == cross(a, b)``."""

    x, y, z = _vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def normalize_quaternion(
    quaternion_xyzw: ArrayLike,
    numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
) -> FloatArray:
    """Normalize one finite quaternion with storage order ``[x, y, z, w]``."""

    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    minimum = numerical.minimum_valid_quaternion_norm
    if norm < minimum:
        raise ValueError(f"quaternion norm {norm:.3e} is below {minimum:.3e}")
    return q / norm


def quaternion_multiply(q1_xyzw: ArrayLike, q2_xyzw: ArrayLike) -> FloatArray:
    """Return Hamilton product ``q1 tensor-product q2`` in xyzw order."""

    q1 = _vector(q1_xyzw, 4, "q1_xyzw")
    q2 = _vector(q2_xyzw, 4, "q2_xyzw")
    v1, w1 = q1[:3], q1[3]
    v2, w2 = q2[:3], q2[3]
    vector = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    scalar = w1 * w2 - float(np.dot(v1, v2))
    return np.concatenate((vector, np.array([scalar], dtype=float)))


def quaternion_inverse(
    quaternion_xyzw: ArrayLike,
    numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
) -> FloatArray:
    """Return the multiplicative inverse of a finite xyzw quaternion."""

    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm_sq = float(np.dot(q, q))
    minimum_sq = numerical.minimum_valid_quaternion_norm**2
    if norm_sq < minimum_sq:
        raise ValueError("cannot invert a near-zero quaternion")
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float) / norm_sq


def rotvec_to_quaternion(
    rotation_vector_rad: ArrayLike,
    numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
) -> FloatArray:
    """Convert a rotation vector in radians, shape ``(3,)``, to xyzw."""

    phi = _vector(rotation_vector_rad, 3, "rotation_vector_rad")
    theta = float(np.linalg.norm(phi))
    half_theta = 0.5 * theta
    epsilon = numerical.small_angle_epsilon
    if theta < epsilon:
        # sin(theta/2)/theta = 1/2 - theta^2/48 + theta^4/3840 + O(theta^6)
        theta_sq = theta * theta
        scale = 0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
    else:
        scale = np.sin(half_theta) / theta
    quaternion = np.concatenate((scale * phi, np.array([np.cos(half_theta)])))
    return normalize_quaternion(quaternion, numerical)


def quaternion_to_rotvec(
    quaternion_xyzw: ArrayLike,
    numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
) -> FloatArray:
    """Convert an xyzw quaternion to the shortest rotation vector in radians."""

    q = normalize_quaternion(quaternion_xyzw, numerical)
    if q[3] < 0.0:
        q = -q
    vector = q[:3]
    vector_norm = float(np.linalg.norm(vector))
    epsilon = numerical.small_angle_epsilon
    if vector_norm < epsilon:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, q[3])
    return (angle / vector_norm) * vector


def quaternion_to_rotation_matrix(quaternion_xyzw: ArrayLike) -> FloatArray:
    """Return active ``R_WB`` (shape ``(3, 3)``) from an xyzw quaternion."""

    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_to_rpy(quaternion_xyzw: ArrayLike) -> FloatArray:
    """Return ``[roll, pitch, yaw]`` in radians using ZYX yaw-pitch-roll."""

    rotation = quaternion_to_rotation_matrix(quaternion_xyzw)
    pitch_argument = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = np.arcsin(pitch_argument)
    if abs(abs(pitch_argument) - 1.0) <= DEFAULT_CONFIG.numerical.quaternion_norm_epsilon:
        # At gimbal lock roll and yaw are not individually observable. Fix
        # roll=0 and return the equivalent, deterministic yaw representation.
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    else:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)
