import unittest

import numpy as np

from rotation_utils import rotvec_to_quaternion
from scripts.analyze_dynamic_consistency import (
    compare_rotation_increment,
    lag_sweep,
    pose_increment_rotvec,
)


class DynamicConsistencyTests(unittest.TestCase):
    """Minimal convention and method tests for the Phase 2.4B-2 diagnostic."""

    def test_pose_increment_uses_prediction_to_observation_direction(self) -> None:
        omega = 0.7
        dt = 0.02
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        q1 = rotvec_to_quaternion(np.array([0.0, 0.0, omega * dt]))

        increment = pose_increment_rotvec(q0, q1)

        np.testing.assert_allclose(increment, [0.0, 0.0, omega * dt], atol=1e-14)

    def test_lag_sweep_sign_matches_delayed_pose_signal(self) -> None:
        tau = 0.010
        imu_time = np.arange(0.0, 5.0, 0.001)
        pose_time = np.arange(0.2, 4.8, 0.005)

        def signal(time: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (
                    np.sin(1.3 * time) + 0.07 * time,
                    0.8 * np.cos(0.7 * time + 0.2) + 0.03 * time * time,
                    np.sin(0.4 * time * time + 0.1),
                )
            )

        imu_omega = signal(imu_time)
        pose_omega = signal(pose_time - tau)
        lag_values = np.arange(-0.030, 0.0300001, 0.00025)

        rows = lag_sweep(
            imu_time, imu_omega, pose_time, pose_omega, lag_values
        )
        best = min(rows, key=lambda row: row["combined_RMSE"])

        self.assertAlmostEqual(float(best["lag_s"]), -tau, places=12)

    def test_trapezoid_matches_linear_same_axis_increment(self) -> None:
        dt = 0.02
        omega0 = 0.3
        alpha = 5.0
        true_angle = omega0 * dt + 0.5 * alpha * dt * dt
        q_pose_rel = rotvec_to_quaternion(np.array([0.0, 0.0, true_angle]))

        comparison = compare_rotation_increment(
            q_pose_rel,
            np.array([0.0, 0.0, omega0]),
            np.array([0.0, 0.0, omega0 + alpha * dt]),
            dt,
        )

        self.assertGreater(np.linalg.norm(comparison.e_left), 0.0)
        self.assertLess(np.linalg.norm(comparison.e_trap), 1e-14)
        np.testing.assert_allclose(
            comparison.zoh_correction,
            [0.0, 0.0, 0.5 * alpha * dt * dt],
            atol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
