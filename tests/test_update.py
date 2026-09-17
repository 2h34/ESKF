import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from eskf6d import ESKF6D
from initialization import ESKF6DInitialization, initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import hat, quaternion_to_rotvec, rotvec_to_quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class UpdateTests(unittest.TestCase):
    """Necessary development-time tests for attitude update/injection/reset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.imu = load_processed_imu(PROCESSED_ROOT / "imu_processed.npz")
        cls.pose = load_processed_pose(PROCESSED_ROOT / "pose_processed.npz")
        cls.stats = load_init_stats(
            PROCESSED_ROOT / "init_stats.npz",
            imu_sample_count=cls.imu.timestamp.size,
        )
        cls.alignment = load_alignment(
            PROCESSED_ROOT / "match_table.npz",
            imu_sample_count=cls.imu.timestamp.size,
            pose_sample_count=cls.pose.timestamp.size,
        )
        cls.real_initialization = initialize_eskf6d(
            cls.imu,
            cls.pose,
            cls.stats,
            cls.alignment,
            DEFAULT_CONFIG,
        )

    def _synthetic_initialization(self) -> ESKF6DInitialization:
        return replace(
            self.real_initialization,
            q0_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            bg0_rad_s=np.array([0.01, -0.02, 0.03]),
            P0=np.diag([1e-3, 2e-3, 3e-3, 4e-5, 5e-5, 6e-5]),
            Qc=np.diag([1e-6, 2e-6, 3e-6, 4e-8, 5e-8, 6e-8]),
        )

    @staticmethod
    def _R() -> np.ndarray:
        return np.diag([1e-4, 2e-4, 3e-4])

    def test_matching_observation_has_zero_correction_but_reduces_covariance(self) -> None:
        filter_6d = ESKF6D.from_initialization(self._synthetic_initialization())
        before = filter_6d.get_state()
        debug = filter_6d.update_attitude(before.q_xyzw, self._R())
        after = filter_6d.get_state()

        np.testing.assert_allclose(debug.r_theta_rad, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(debug.delta_x, np.zeros(6), atol=1e-15)
        np.testing.assert_allclose(after.q_xyzw, before.q_xyzw, atol=1e-15)
        np.testing.assert_array_equal(after.bg_rad_s, before.bg_rad_s)
        self.assertTrue(np.all(np.diag(after.P)[0:3] < np.diag(before.P)[0:3]))

    def test_quaternion_double_cover_produces_equivalent_update(self) -> None:
        q_obs = rotvec_to_quaternion(np.array([0.02, -0.01, 0.03]))
        filter_positive = ESKF6D.from_initialization(
            self._synthetic_initialization()
        )
        filter_negative = ESKF6D.from_initialization(
            self._synthetic_initialization()
        )
        positive = filter_positive.update_attitude(q_obs, self._R())
        negative = filter_negative.update_attitude(-q_obs, self._R())

        self.assertFalse(positive.quaternion_sign_flipped)
        self.assertTrue(negative.quaternion_sign_flipped)
        np.testing.assert_allclose(positive.r_theta_rad, negative.r_theta_rad)
        np.testing.assert_allclose(positive.delta_x, negative.delta_x)
        positive_state = filter_positive.get_state()
        negative_state = filter_negative.get_state()
        np.testing.assert_allclose(positive_state.q_xyzw, negative_state.q_xyzw)
        np.testing.assert_allclose(positive_state.bg_rad_s, negative_state.bg_rad_s)
        np.testing.assert_allclose(positive_state.P, negative_state.P)

    def test_positive_z_residual_moves_attitude_toward_observation(self) -> None:
        filter_6d = ESKF6D.from_initialization(self._synthetic_initialization())
        observation_angle = 0.1
        q_obs = rotvec_to_quaternion(np.array([0.0, 0.0, observation_angle]))
        debug = filter_6d.update_attitude(q_obs, self._R())
        after_rotvec = quaternion_to_rotvec(filter_6d.get_state().q_xyzw)

        self.assertGreater(debug.r_theta_rad[2], 0.0)
        self.assertGreater(debug.delta_theta_rad[2], 0.0)
        self.assertGreater(after_rotvec[2], 0.0)
        self.assertLess(after_rotvec[2], observation_angle)

    def test_reset_jacobian_uses_negative_half_hat(self) -> None:
        filter_6d = ESKF6D.from_initialization(self._synthetic_initialization())
        q_obs = rotvec_to_quaternion(np.array([0.02, -0.03, 0.04]))
        debug = filter_6d.update_attitude(q_obs, self._R())
        expected = np.eye(3) - 0.5 * hat(debug.delta_theta_rad)
        np.testing.assert_allclose(debug.G_reset[0:3, 0:3], expected)

    def test_nonpositive_observation_covariance_is_rejected_atomically(self) -> None:
        for invalid_diagonal in (
            np.array([1e-4, 0.0, 2e-4]),
            np.array([1e-4, -1e-6, 2e-4]),
        ):
            with self.subTest(invalid_diagonal=invalid_diagonal):
                filter_6d = ESKF6D.from_initialization(
                    self._synthetic_initialization()
                )
                before = filter_6d.get_state()
                with self.assertRaisesRegex(ValueError, "positive definite"):
                    filter_6d.update_attitude(
                        before.q_xyzw, np.diag(invalid_diagonal)
                    )
                after = filter_6d.get_state()
                np.testing.assert_array_equal(after.q_xyzw, before.q_xyzw)
                np.testing.assert_array_equal(after.bg_rad_s, before.bg_rad_s)
                np.testing.assert_array_equal(after.P, before.P)

    def test_updated_covariance_and_quaternion_are_numerically_valid(self) -> None:
        filter_6d = ESKF6D.from_initialization(self._synthetic_initialization())
        q_obs = rotvec_to_quaternion(np.array([0.01, 0.02, -0.03]))
        debug = filter_6d.update_attitude(q_obs, self._R())
        state = filter_6d.get_state()

        self.assertEqual(state.P.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(state.P)))
        np.testing.assert_allclose(state.P, state.P.T, rtol=0.0, atol=1e-15)
        self.assertTrue(
            np.all(
                np.diag(state.P)
                >= -DEFAULT_CONFIG.numerical.covariance_negative_tolerance
            )
        )
        self.assertAlmostEqual(float(np.linalg.norm(state.q_xyzw)), 1.0)
        self.assertLessEqual(
            debug.raw_covariance_symmetry_error,
            DEFAULT_CONFIG.numerical.covariance_symmetry_tolerance,
        )

    def test_real_first_prediction_and_update_uses_external_scheduler(self) -> None:
        filter_6d = ESKF6D.from_initialization(self.real_initialization)
        gyro_index = self.real_initialization.imu_index
        current_imu_index = gyro_index + 1
        self.assertEqual(gyro_index, 395)
        self.assertEqual(current_imu_index, 396)
        filter_6d.predict(
            self.imu.gyro_rad_s[gyro_index], self.imu.dt[current_imu_index]
        )

        pose_index = int(
            self.alignment.pose_index_for_imu[current_imu_index]
        )
        self.assertGreaterEqual(pose_index, 0)
        attitude_covariance = self.pose.cov_diag[pose_index, 3:6]
        self.assertTrue(np.all(attitude_covariance > 0.0))
        debug = filter_6d.update_attitude(
            self.pose.quaternion_xyzw[pose_index],
            np.diag(attitude_covariance),
        )

        self.assertAlmostEqual(debug.q_norm, 1.0)
        self.assertFalse(hasattr(filter_6d, "_imu_index"))
        self.assertFalse(hasattr(filter_6d, "_timestamp"))


if __name__ == "__main__":
    unittest.main()
