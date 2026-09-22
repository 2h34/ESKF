"""Quaternion and SO(3) operations used by the ESKF."""

import numpy as np
from numpy.typing import ArrayLike

import config as cfg


def normalize_quaternion(value: ArrayLike) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be a finite non-zero xyzw vector")
    return q / norm


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    q1, q2 = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    vector = q1[3] * q2[:3] + q2[3] * q1[:3] + np.cross(q1[:3], q2[:3])
    return np.r_[vector, q1[3] * q2[3] - float(np.dot(q1[:3], q2[:3]))]


def quaternion_inverse(value: ArrayLike) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    norm_sq = float(np.dot(q, q))
    if q.shape != (4,) or not np.isfinite(norm_sq) or norm_sq == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return np.r_[-q[:3], q[3]] / norm_sq


def hat(vector: ArrayLike) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rotvec_to_quaternion(rotation_vector_rad: ArrayLike) -> np.ndarray:
    phi = np.asarray(rotation_vector_rad, dtype=float)
    theta = float(np.linalg.norm(phi))
    half_theta = 0.5 * theta
    if theta < cfg.SMALL_ANGLE_EPSILON:
        theta_sq = theta * theta
        scale = 0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
    else:
        scale = np.sin(half_theta) / theta
    return normalize_quaternion(np.r_[scale * phi, np.cos(half_theta)])


def quaternion_to_rotvec(value: ArrayLike) -> np.ndarray:
    q = normalize_quaternion(value)
    if q[3] < 0.0:
        q = -q
    vector_norm = np.linalg.norm(q[:3])
    if vector_norm < cfg.SMALL_ANGLE_EPSILON:
        return 2.0 * q[:3]
    return q[:3] * (2.0 * np.arctan2(vector_norm, q[3]) / vector_norm)


def quaternion_to_rotation_matrix(value: ArrayLike) -> np.ndarray:
    x, y, z, w = normalize_quaternion(value)
    return np.array((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ))


def quaternion_to_rpy(value: ArrayLike) -> np.ndarray:
    R = quaternion_to_rotation_matrix(value)
    pitch_arg = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    if abs(abs(pitch_arg) - 1.0) <= cfg.GIMBAL_LOCK_EPSILON:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll, yaw = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    return np.array((roll, pitch, yaw))
