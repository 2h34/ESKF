import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG, GRAVITY_WORLD_MPS2
from initialization import initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import hat, quaternion_to_rotation_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class Initialization15DTests(unittest.TestCase):
    """Focused tests for the initialization-only 15D extension."""

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
        cls.result = initialize_eskf15d(
            cls.imu,
            cls.pose,
            cls.stats,
            cls.alignment,
            DEFAULT_CONFIG,
        )

    def test_real_data_nominal_initialization_and_shared_pose(self) -> None:
        result = self.result
        self.assertEqual(result.imu_index, 395)
        self.assertGreaterEqual(result.pose_index, 0)
        self.assertEqual(
            result.pose_index,
            int(self.alignment.pose_index_for_imu[result.imu_index]),
        )
        self.assertEqual(result.p0_W_m.shape, (3,))
        self.assertEqual(result.v0_W_mps.shape, (3,))
        self.assertEqual(result.q0_xyzw.shape, (4,))
        self.assertEqual(result.bg0_rad_s.shape, (3,))
        self.assertEqual(result.ba0_mps2.shape, (3,))
        for value in (
            result.p0_W_m,
            result.v0_W_mps,
            result.q0_xyzw,
            result.bg0_rad_s,
            result.ba0_mps2,
        ):
            self.assertTrue(np.all(np.isfinite(value)))
        self.assertAlmostEqual(float(np.linalg.norm(result.q0_xyzw)), 1.0)
        np.testing.assert_array_equal(result.v0_W_mps, np.zeros(3))
        np.testing.assert_allclose(result.bg0_rad_s, self.stats.gyro_mean)
        np.testing.assert_allclose(
            result.p0_W_m, self.pose.position[result.pose_index]
        )
        expected_q0 = self.pose.quaternion_xyzw[result.pose_index]
        expected_q0 = expected_q0 / np.linalg.norm(expected_q0)
        np.testing.assert_allclose(result.q0_xyzw, expected_q0)

    def test_accelerometer_bias_formula(self) -> None:
        expected_ba0 = (
            self.stats.acc_mean
            + quaternion_to_rotation_matrix(self.result.q0_xyzw).T
            @ GRAVITY_WORLD_MPS2
        )
        np.testing.assert_allclose(self.result.ba0_mps2, expected_ba0)

    def test_P0_blocks_and_zero_initial_cross_terms(self) -> None:
        result = self.result
        pose_covariance = self.pose.cov_diag[result.pose_index]
        Ptheta0 = np.diag(pose_covariance[3:6])
        gravity_B0 = (
            quaternion_to_rotation_matrix(result.q0_xyzw).T
            @ GRAVITY_WORLD_MPS2
        )
        J_g = hat(gravity_B0)
        expected = np.zeros((15, 15), dtype=float)
        expected[0:3, 0:3] = np.diag(pose_covariance[0:3])
        expected[3:6, 3:6] = (
            DEFAULT_CONFIG.initialization.initial_velocity_std_mps**2
            * np.eye(3)
        )
        expected[6:9, 6:9] = Ptheta0
        expected[9:12, 9:12] = np.diag(
            self.stats.gyro_std**2 / result.static_sample_count
        )
        expected[12:15, 12:15] = (
            np.diag(self.stats.acc_std**2 / result.static_sample_count)
            + J_g @ Ptheta0 @ J_g.T
        )
        self.assertEqual(result.P0.shape, (15, 15))
        self.assertTrue(np.all(np.isfinite(result.P0)))
        np.testing.assert_allclose(result.P0, result.P0.T, rtol=0.0, atol=1e-15)
        self.assertTrue(np.all(np.diag(result.P0) > 0.0))
        np.testing.assert_allclose(result.P0, expected)

    def test_Qc_block_ordering_and_values(self) -> None:
        result = self.result
        accelerometer_bias_rw = (
            DEFAULT_CONFIG.filter_noise.accelerometer_bias_random_walk_density
        )
        expected = np.diag(
            np.concatenate(
                (
                    (self.stats.acc_std * np.sqrt(self.stats.median_dt)) ** 2,
                    (self.stats.gyro_std * np.sqrt(self.stats.median_dt)) ** 2,
                    np.full(
                        3,
                        DEFAULT_CONFIG.filter_noise.gyro_bias_random_walk_density
                        ** 2,
                    ),
                    np.full(
                        3,
                        accelerometer_bias_rw**2,
                    ),
                )
            )
        )
        self.assertEqual(result.Qc.shape, (12, 12))
        self.assertTrue(np.all(np.isfinite(result.Qc)))
        np.testing.assert_allclose(result.Qc, result.Qc.T, rtol=0.0, atol=1e-15)
        self.assertTrue(np.all(np.diag(result.Qc) > 0.0))
        np.testing.assert_allclose(result.Qc, expected)

    def test_missing_pose_at_k0_is_rejected(self) -> None:
        pose_index = self.alignment.pose_index_for_imu.copy()
        pose_index[self.stats.static_end_idx - 1] = -1
        invalid_alignment = replace(
            self.alignment, pose_index_for_imu=pose_index
        )
        with self.assertRaisesRegex(ValueError, "has no matched FAST-LIO pose"):
            initialize_eskf15d(
                self.imu,
                self.pose,
                self.stats,
                invalid_alignment,
                DEFAULT_CONFIG,
            )

    def test_nonpositive_pose_covariance_is_rejected(self) -> None:
        for covariance_index, label in ((1, "position"), (4, "attitude")):
            with self.subTest(label=label):
                covariance = self.pose.cov_diag.copy()
                covariance[self.result.pose_index, covariance_index] = 0.0
                invalid_pose = replace(self.pose, cov_diag=covariance)
                with self.assertRaisesRegex(
                    ValueError, f"{label} covariance must be strictly positive"
                ):
                    initialize_eskf15d(
                        self.imu,
                        invalid_pose,
                        self.stats,
                        self.alignment,
                        DEFAULT_CONFIG,
                    )

    def test_invalid_provisional_configuration_is_rejected(self) -> None:
        invalid_velocity = replace(
            DEFAULT_CONFIG,
            initialization=replace(
                DEFAULT_CONFIG.initialization,
                initial_velocity_std_mps=0.0,
            ),
        )
        invalid_accel_rw = replace(
            DEFAULT_CONFIG,
            filter_noise=replace(
                DEFAULT_CONFIG.filter_noise,
                accelerometer_bias_random_walk_density=float("nan"),
            ),
        )
        for config, message in (
            (invalid_velocity, "initial_velocity_std_mps"),
            (invalid_accel_rw, "accelerometer_bias_random_walk_density"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    initialize_eskf15d(
                        self.imu,
                        self.pose,
                        self.stats,
                        self.alignment,
                        config,
                    )


if __name__ == "__main__":
    unittest.main()
