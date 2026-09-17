import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG, PreprocessConfig
from preprocess import (
    analyze_initial_static_segment,
    load_imu_csv,
    load_pose_csv,
    save_init_stats,
    save_processed_imu,
    save_processed_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.imu, cls.imu_diagnostics = load_imu_csv(PROJECT_ROOT / "data/raw/imu.csv")
        cls.pose, cls.pose_diagnostics = load_pose_csv(PROJECT_ROOT / "data/raw/pose_cov.csv")
        cls.stats = analyze_initial_static_segment(cls.imu)

    def test_imu_shapes_and_units(self) -> None:
        count = self.imu.timestamp.size
        self.assertEqual(self.imu.gyro_rad_s.shape, (count, 3))
        self.assertEqual(self.imu.acc_mps2.shape, (count, 3))
        self.assertTrue(np.isnan(self.imu.dt[0]))
        self.assertTrue(np.all(self.imu.dt[1:] > 0.0))
        self.assertTrue(np.all(np.diff(self.imu.timestamp) > 0.0))
        np.testing.assert_allclose(
            self.imu.acc_mps2[0],
            np.array([-0.010521283, -0.001322764, 0.996570110]) * 9.81,
        )

    def test_real_invalid_row_is_reported_not_repaired(self) -> None:
        self.assertEqual(self.imu_diagnostics.invalid_sample_count, 1)
        self.assertEqual(self.imu_diagnostics.invalid_source_rows, (25565,))
        self.assertEqual(
            self.imu_diagnostics.sample_count_raw,
            self.imu_diagnostics.sample_count_processed + 1,
        )

    def test_timestamp_anomaly_values_match_reported_indices(self) -> None:
        for index, reported_dt in zip(
            self.imu_diagnostics.large_step_destination_indices,
            self.imu_diagnostics.large_step_dt_s,
            strict=True,
        ):
            self.assertEqual(reported_dt, self.imu.dt[index])

    def test_strict_invalid_policy_raises(self) -> None:
        strict = replace(
            DEFAULT_CONFIG,
            preprocess=PreprocessConfig(invalid_sample_policy="raise"),
        )
        with self.assertRaises(ValueError):
            load_imu_csv(PROJECT_ROOT / "data/raw/imu.csv", strict)

    def test_no_nonfinite_processed_measurements(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.imu.timestamp)))
        self.assertTrue(np.all(np.isfinite(self.imu.gyro_rad_s)))
        self.assertTrue(np.all(np.isfinite(self.imu.acc_mps2)))
        self.assertTrue(np.all(np.isfinite(self.pose.timestamp)))
        self.assertTrue(np.all(np.isfinite(self.pose.position)))
        self.assertTrue(np.all(np.isfinite(self.pose.quaternion_xyzw)))
        self.assertTrue(np.all(np.isfinite(self.pose.cov_diag)))

    def test_static_acceleration_norm_is_near_gravity(self) -> None:
        self.assertTrue(self.stats.is_static)
        self.assertLess(abs(self.stats.acc_norm_mean - 9.81), 0.5)

    def test_pose_quaternions_and_covariance(self) -> None:
        norms = np.linalg.norm(self.pose.quaternion_xyzw, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)
        self.assertEqual(self.pose.quaternion_xyzw.shape[1], 4)
        self.assertEqual(self.pose.cov_diag.shape[1], 6)
        self.assertTrue(np.all(self.pose.cov_diag >= 0.0))
        self.assertEqual(self.pose_diagnostics.invalid_quaternion_count, 0)
        self.assertEqual(self.pose_diagnostics.invalid_covariance_count, 0)

    def test_npz_round_trip_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_processed_imu(root / "imu.npz", self.imu)
            save_processed_pose(root / "pose.npz", self.pose)
            save_init_stats(root / "stats.npz", self.stats)
            with np.load(root / "imu.npz") as loaded:
                self.assertEqual(loaded["gyro_rad_s"].shape, self.imu.gyro_rad_s.shape)
                self.assertEqual(loaded["acc_mps2"].shape, self.imu.acc_mps2.shape)
            with np.load(root / "pose.npz") as loaded:
                self.assertEqual(loaded["quaternion_xyzw"].shape, self.pose.quaternion_xyzw.shape)
                self.assertEqual(loaded["cov_diag"].shape, self.pose.cov_diag.shape)
            with np.load(root / "stats.npz") as loaded:
                self.assertEqual(loaded["gyro_mean"].shape, (3,))
                self.assertEqual(loaded["acc_mean"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
