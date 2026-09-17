import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from eskf6d import ESKF6D
from initialization import initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from runner6d import run_eskf6d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class Runner6DTests(unittest.TestCase):
    """Focused scheduling and full-run safety checks for Phase 2.4A."""

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
        cls.initialization = initialize_eskf6d(
            cls.imu,
            cls.pose,
            cls.stats,
            cls.alignment,
            DEFAULT_CONFIG,
        )
        cls.full_run = run_eskf6d(
            cls.imu,
            cls.pose,
            cls.stats,
            cls.alignment,
            cls.initialization,
            ESKF6D.from_initialization(
                cls.initialization, DEFAULT_CONFIG.numerical
            ),
            DEFAULT_CONFIG,
        )

    def _short_inputs(self):
        end = self.initialization.imu_index + 2
        imu = replace(
            self.imu,
            timestamp=self.imu.timestamp[:end].copy(),
            dt=self.imu.dt[:end].copy(),
            gyro_rad_s=self.imu.gyro_rad_s[:end].copy(),
            acc_mps2=self.imu.acc_mps2[:end].copy(),
        )
        alignment = replace(
            self.alignment,
            pose_index_for_imu=self.alignment.pose_index_for_imu[:end].copy(),
            time_error=self.alignment.time_error[:end].copy(),
        )
        return imu, alignment

    def test_initialization_row_is_logged_once_without_duplicate_update(self) -> None:
        k0 = self.initialization.imu_index
        rows = self.full_run.result_rows
        debug = self.full_run.debug_rows

        self.assertEqual(k0, 395)
        self.assertEqual(rows[0]["imu_index"], k0)
        self.assertEqual(sum(row["imu_index"] == k0 for row in rows), 1)
        self.assertFalse(rows[0]["update_applied"])
        self.assertEqual(
            debug[0]["skip_reason"],
            "initialization_observation_already_used",
        )
        self.assertFalse(self.full_run.summary["initialization_observation_reused"])

    def test_first_loop_uses_gyro_395_dt_396_and_match_at_396(self) -> None:
        first_step = self.full_run.debug_rows[1]
        self.assertEqual(first_step["imu_index"], 396)
        self.assertEqual(first_step["dt"], float(self.imu.dt[396]))
        self.assertEqual(
            first_step["pose_index"],
            int(self.alignment.pose_index_for_imu[396]),
        )
        expected_omega_hat = (
            self.imu.gyro_rad_s[395] - self.initialization.bg0_rad_s
        )
        np.testing.assert_allclose(
            [
                first_step["omega_hat_x"],
                first_step["omega_hat_y"],
                first_step["omega_hat_z"],
            ],
            expected_omega_hat,
            rtol=0.0,
            atol=1e-15,
        )

    def test_unmatched_pose_keeps_prediction_and_skips_update(self) -> None:
        imu, alignment = self._short_inputs()
        alignment.pose_index_for_imu[396] = -1
        alignment.time_error[396] = np.nan
        expected_filter = ESKF6D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        expected_filter.predict(imu.gyro_rad_s[395], imu.dt[396])
        expected = expected_filter.get_state()

        actual_filter = ESKF6D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        run = run_eskf6d(
            imu,
            self.pose,
            self.stats,
            alignment,
            self.initialization,
            actual_filter,
            DEFAULT_CONFIG,
        )
        actual = actual_filter.get_state()

        self.assertFalse(run.result_rows[1]["has_pose_match"])
        self.assertFalse(run.result_rows[1]["update_applied"])
        self.assertEqual(run.debug_rows[1]["skip_reason"], "unmatched_pose")
        np.testing.assert_allclose(actual.q_xyzw, expected.q_xyzw)
        np.testing.assert_allclose(actual.bg_rad_s, expected.bg_rad_s)
        np.testing.assert_allclose(actual.P, expected.P)

    def test_invalid_attitude_covariance_keeps_prediction_and_skips_update(self) -> None:
        imu, alignment = self._short_inputs()
        pose_index = int(alignment.pose_index_for_imu[396])
        self.assertGreaterEqual(pose_index, 0)
        covariance = self.pose.cov_diag.copy()
        covariance[pose_index, 4] = 0.0
        pose = replace(self.pose, cov_diag=covariance)

        expected_filter = ESKF6D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        expected_filter.predict(imu.gyro_rad_s[395], imu.dt[396])
        expected = expected_filter.get_state()

        actual_filter = ESKF6D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        run = run_eskf6d(
            imu,
            pose,
            self.stats,
            alignment,
            self.initialization,
            actual_filter,
            DEFAULT_CONFIG,
        )
        actual = actual_filter.get_state()

        self.assertTrue(run.result_rows[1]["has_pose_match"])
        self.assertFalse(run.result_rows[1]["update_applied"])
        self.assertEqual(
            run.debug_rows[1]["skip_reason"],
            "nonpositive_attitude_covariance",
        )
        np.testing.assert_allclose(actual.q_xyzw, expected.q_xyzw)
        np.testing.assert_allclose(actual.bg_rad_s, expected.bg_rad_s)
        np.testing.assert_allclose(actual.P, expected.P)

    def test_full_run_length_and_posterior_numerical_safety(self) -> None:
        rows = self.full_run.result_rows
        debug = self.full_run.debug_rows
        expected_count = self.imu.timestamp.size - self.initialization.imu_index
        self.assertEqual(len(rows), expected_count)
        self.assertEqual(len(debug), expected_count)

        timestamps = np.asarray([row["timestamp"] for row in rows])
        quaternions = np.asarray(
            [[row["qx"], row["qy"], row["qz"], row["qw"]] for row in rows]
        )
        result_numeric = np.asarray(
            [
                [
                    row["timestamp"],
                    row["qx"],
                    row["qy"],
                    row["qz"],
                    row["qw"],
                    row["roll_rad"],
                    row["pitch_rad"],
                    row["yaw_rad"],
                    row["bg_x_rad_s"],
                    row["bg_y_rad_s"],
                    row["bg_z_rad_s"],
                ]
                for row in rows
            ]
        )
        self.assertTrue(np.all(np.diff(timestamps) > 0.0))
        self.assertTrue(np.all(np.isfinite(result_numeric)))
        np.testing.assert_allclose(
            np.linalg.norm(quaternions, axis=1),
            np.ones(len(rows)),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(
            all(np.isfinite(row["P_min_eigenvalue"]) for row in debug)
        )
        self.assertGreaterEqual(
            min(row["P_min_eigenvalue"] for row in debug),
            -DEFAULT_CONFIG.numerical.covariance_negative_tolerance,
        )
        self.assertLessEqual(
            max(row["P_symmetry_error"] for row in debug),
            DEFAULT_CONFIG.numerical.covariance_symmetry_tolerance,
        )


if __name__ == "__main__":
    unittest.main()
