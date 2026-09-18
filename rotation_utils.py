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
    """Return the 3x3 skew matrix satisfying ``hat(a) @ b == cross(a, b)``.

    ``hat`` 把三维向量映射为李代数 so(3) 的反对称矩阵，使叉乘能够写成
    矩阵乘法；ESKF 的姿态误差动力学和 reset Jacobian 都使用这一表示。
    """

    x, y, z = _vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def normalize_quaternion(
    quaternion_xyzw: ArrayLike,
    numerical: NumericalSafetyConfig = DEFAULT_CONFIG.numerical,
) -> FloatArray:
    """Normalize one finite quaternion with storage order ``[x, y, z, w]``.

    单位四元数才表示纯旋转；归一化用于清除浮点累计误差，而近零四元数没有
    可恢复的旋转含义，因此必须明确拒绝，不能静默归一化。
    """

    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    minimum = numerical.minimum_valid_quaternion_norm
    if norm < minimum:
        raise ValueError(f"quaternion norm {norm:.3e} is below {minimum:.3e}")
    return q / norm


def quaternion_multiply(q1_xyzw: ArrayLike, q2_xyzw: ArrayLike) -> FloatArray:
    """Return Hamilton product ``q1 tensor-product q2`` in xyzw order.

    代码存储顺序是 ``xyzw``，但乘法仍是标准 Hamilton product；乘数顺序
    表示旋转复合顺序，不能因为更换存储布局而交换。
    """

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
    """Return the multiplicative inverse of a finite xyzw quaternion.

    对单位四元数它等于共轭；这里仍除以范数平方，使函数对有限的非单位输入
    保持真正的乘法逆，并对近零输入报错。
    """

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
    """Convert a rotation vector in radians, shape ``(3,)``, to xyzw.

    rotation vector 的方向是旋转轴，模长是旋转角；该函数是 SO(3) 指数映射
    的四元数实现，小角度分支用级数避免 ``sin(theta/2)/theta`` 数值不稳。
    """

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
    """Convert an xyzw quaternion to the shortest rotation vector in radians.

    这是 SO(3) 对数映射的工程实现。先选择标量部非负的等价四元数，以返回
    最短旋转；小角度时用 ``rotvec ≈ 2*q_xyz`` 避免除以极小量。
    """

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
    """Return ``[roll, pitch, yaw]`` in radians using ZYX yaw-pitch-roll.

    RPY 采用 ZYX（先 yaw、再 pitch、再 roll）的可读输出约定，只用于结果
    表达与检查；滤波内部始终用四元数，不用存在奇异性的欧拉角传播状态。
    """

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
