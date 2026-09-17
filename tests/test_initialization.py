import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from initialization import initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class InitializationTests(unittest.TestCase):
    """Development-time checks for the initialization-only stage."""

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

    def test_real_data_initialization(self) -> None:
        result = initialize_eskf6d(
            self.imu, self.pose, self.stats, self.alignment, DEFAULT_CONFIG
        )

        self.assertEqual(self.stats.static_end_idx, 396)
        self.assertEqual(result.imu_index, 395)
        self.assertGreaterEqual(result.pose_index, 0)
        self.assertEqual(result.q0_xyzw.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(result.q0_xyzw)), 1.0)
        np.testing.assert_allclose(result.bg0_rad_s, self.stats.gyro_mean)

        for matrix in (result.P0, result.Qc):
            self.assertEqual(matrix.shape, (6, 6))
            np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-15)
            self.assertTrue(np.all(np.diag(matrix) > 0.0))

    def test_missing_pose_at_k0_is_rejected(self) -> None:
        pose_index = self.alignment.pose_index_for_imu.copy()
        pose_index[self.stats.static_end_idx - 1] = -1
        invalid_alignment = replace(
            self.alignment, pose_index_for_imu=pose_index
        )
        with self.assertRaisesRegex(ValueError, "has no matched FAST-LIO pose"):
            initialize_eskf6d(
                self.imu,
                self.pose,
                self.stats,
                invalid_alignment,
                DEFAULT_CONFIG,
            )

    def test_nonpositive_attitude_covariance_at_k0_is_rejected(self) -> None:
        pose_index = int(
            self.alignment.pose_index_for_imu[self.stats.static_end_idx - 1]
        )
        covariance = self.pose.cov_diag.copy()
        covariance[pose_index, 4] = 0.0
        invalid_pose = replace(self.pose, cov_diag=covariance)
        with self.assertRaisesRegex(ValueError, "must be strictly positive"):
            initialize_eskf6d(
                self.imu,
                invalid_pose,
                self.stats,
                self.alignment,
                DEFAULT_CONFIG,
            )


if __name__ == "__main__":
    unittest.main()
