"""使用 xyzw Hamilton 四元数的旋转工具。

四元数表示主动的机体到世界旋转 ``R_WB``。四元数乘法返回
``q1 ⊗ q2``。该约定与当前采用的右乘误差定义
``R_true = R_hat Exp(delta_theta^)`` 兼容。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

import config as cfg


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def hat(vector: ArrayLike) -> FloatArray:
    """返回满足 ``hat(a) @ b == cross(a, b)`` 的 3x3 反对称矩阵。"""

    x, y, z = _vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def normalize_quaternion(
    quaternion_xyzw: ArrayLike,
) -> FloatArray:
    """归一化一个有限四元数，存储顺序为 ``[x, y, z, w]``。"""

    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    minimum = cfg.MINIMUM_VALID_QUATERNION_NORM
    if norm < minimum:
        raise ValueError(f"quaternion norm {norm:.3e} is below {minimum:.3e}")
    return q / norm


def quaternion_multiply(q1_xyzw: ArrayLike, q2_xyzw: ArrayLike) -> FloatArray:
    """返回 xyzw 顺序下的 Hamilton 积 ``q1 ⊗ q2``。"""

    q1 = _vector(q1_xyzw, 4, "q1_xyzw")
    q2 = _vector(q2_xyzw, 4, "q2_xyzw")
    v1, w1 = q1[:3], q1[3]
    v2, w2 = q2[:3], q2[3]
    vector = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    scalar = w1 * w2 - float(np.dot(v1, v2))
    return np.concatenate((vector, np.array([scalar], dtype=float)))


def quaternion_inverse(
    quaternion_xyzw: ArrayLike,
) -> FloatArray:
    """返回一个有限 xyzw 四元数的乘法逆。"""

    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm_sq = float(np.dot(q, q))
    minimum_sq = cfg.MINIMUM_VALID_QUATERNION_NORM**2
    if norm_sq < minimum_sq:
        raise ValueError("cannot invert a near-zero quaternion")
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float) / norm_sq


def rotvec_to_quaternion(
    rotation_vector_rad: ArrayLike,
) -> FloatArray:
    """把弧度制旋转向量（形状 ``(3,)``）转换为 xyzw 四元数。"""

    phi = _vector(rotation_vector_rad, 3, "rotation_vector_rad")
    theta = float(np.linalg.norm(phi))
    half_theta = 0.5 * theta
    epsilon = cfg.SMALL_ANGLE_EPSILON
    if theta < epsilon:
        # sin(theta/2)/theta = 1/2 - theta^2/48 + theta^4/3840 + O(theta^6)
        theta_sq = theta * theta
        scale = 0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
    else:
        scale = np.sin(half_theta) / theta
    quaternion = np.concatenate((scale * phi, np.array([np.cos(half_theta)])))
    return normalize_quaternion(quaternion)


def quaternion_to_rotvec(
    quaternion_xyzw: ArrayLike,
) -> FloatArray:
    """把 xyzw 四元数转换为弧度制的最短旋转向量。"""

    q = normalize_quaternion(quaternion_xyzw)
    if q[3] < 0.0:
        q = -q
    vector = q[:3]
    vector_norm = float(np.linalg.norm(vector))
    epsilon = cfg.SMALL_ANGLE_EPSILON
    if vector_norm < epsilon:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, q[3])
    return (angle / vector_norm) * vector


def quaternion_to_rotation_matrix(quaternion_xyzw: ArrayLike) -> FloatArray:
    """由 xyzw 四元数返回主动旋转 ``R_WB``（形状 ``(3, 3)``）。"""

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
    """按 ZYX（yaw-pitch-roll）顺序返回弧度制 ``[roll, pitch, yaw]``。"""

    rotation = quaternion_to_rotation_matrix(quaternion_xyzw)
    pitch_argument = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = np.arcsin(pitch_argument)
    if abs(abs(pitch_argument) - 1.0) <= cfg.GIMBAL_LOCK_EPSILON:
        # 万向锁时 roll 与 yaw 无法各自分辨。固定 roll=0，返回等价且
        # 确定的 yaw 表示。
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    else:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)
