import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from eskf15d import ESKF15D
from initialization import initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from runner15d import run_eskf15d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class Runner15DTests(unittest.TestCase):
    """Short-data scheduling and logging checks for the 15D runner."""

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
        cls.initialization = initialize_eskf15d(
            cls.imu,
            cls.pose,
            cls.stats,
            cls.alignment,
            DEFAULT_CONFIG,
        )

    def _short_inputs(self, prediction_steps: int = 1):
        end = self.initialization.imu_index + prediction_steps + 1
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

    def _run(self, imu, alignment, pose=None):
        filter_15d = ESKF15D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        run = run_eskf15d(
            imu,
            self.pose if pose is None else pose,
            self.stats,
            alignment,
            self.initialization,
            filter_15d,
            DEFAULT_CONFIG,
        )
        return run, filter_15d

    def test_initialization_observation_is_not_reused(self) -> None:
        imu, alignment = self._short_inputs()
        run, _ = self._run(imu, alignment)
        k0 = self.initialization.imu_index

        self.assertEqual(k0, 395)
        self.assertEqual(run.result_rows[0]["imu_index"], k0)
        self.assertEqual(
            sum(row["imu_index"] == k0 for row in run.result_rows), 1
        )
        self.assertFalse(run.result_rows[0]["update_applied"])
        self.assertEqual(
            run.debug_rows[0]["skip_reason"],
            "initialization_observation_already_used",
        )
        self.assertFalse(run.summary["initialization_observation_reused"])

    def test_first_loop_uses_left_endpoint_imu_and_match_at_396(self) -> None:
        imu, alignment = self._short_inputs()
        run, actual_filter = self._run(imu, alignment)
        first_debug = run.debug_rows[1]

        manual_filter = ESKF15D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        manual_prediction = manual_filter.predict(
            imu.gyro_rad_s[395], imu.acc_mps2[395], imu.dt[396]
        )
        pose_index = int(alignment.pose_index_for_imu[396])
        manual_filter.update_pose(
            self.pose.position[pose_index],
            self.pose.quaternion_xyzw[pose_index],
            np.diag(self.pose.cov_diag[pose_index, 0:6]),
        )
        expected = manual_filter.get_state()
        actual = actual_filter.get_state()

        self.assertEqual(first_debug["imu_index"], 396)
        self.assertEqual(first_debug["dt"], float(imu.dt[396]))
        self.assertEqual(first_debug["pose_index"], pose_index)
        np.testing.assert_allclose(
            [
                first_debug["omega_hat_x"],
                first_debug["omega_hat_y"],
                first_debug["omega_hat_z"],
            ],
            manual_prediction.omega_hat_rad_s,
        )
        np.testing.assert_allclose(
            [
                first_debug["a_hat_W_x"],
                first_debug["a_hat_W_y"],
                first_debug["a_hat_W_z"],
            ],
            manual_prediction.a_hat_W_mps2,
        )
        self._assert_states_equal(actual, expected)

    def test_unmatched_pose_keeps_prediction(self) -> None:
        imu, alignment = self._short_inputs()
        alignment.pose_index_for_imu[396] = -1
        alignment.time_error[396] = np.nan
        manual_filter = ESKF15D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        manual_filter.predict(imu.gyro_rad_s[395], imu.acc_mps2[395], imu.dt[396])
        expected = manual_filter.get_state()

        run, actual_filter = self._run(imu, alignment)

        self.assertFalse(run.result_rows[1]["has_pose_match"])
        self.assertFalse(run.result_rows[1]["update_applied"])
        self.assertEqual(run.debug_rows[1]["skip_reason"], "unmatched_pose")
        self._assert_states_equal(actual_filter.get_state(), expected)

    def test_nonpositive_covariance_keeps_prediction(self) -> None:
        imu, alignment = self._short_inputs()
        pose_index = int(alignment.pose_index_for_imu[396])
        covariance = self.pose.cov_diag.copy()
        covariance[pose_index, 0] = 0.0
        pose = replace(self.pose, cov_diag=covariance)
        manual_filter = ESKF15D.from_initialization(
            self.initialization, DEFAULT_CONFIG.numerical
        )
        manual_filter.predict(imu.gyro_rad_s[395], imu.acc_mps2[395], imu.dt[396])
        expected = manual_filter.get_state()

        run, actual_filter = self._run(imu, alignment, pose)

        self.assertTrue(run.result_rows[1]["has_pose_match"])
        self.assertFalse(run.result_rows[1]["update_applied"])
        self.assertEqual(
            run.debug_rows[1]["skip_reason"],
            "nonpositive_pose_covariance",
        )
        self._assert_states_equal(actual_filter.get_state(), expected)

    def test_short_multistep_run_has_finite_posterior_rows(self) -> None:
        prediction_steps = 12
        imu, alignment = self._short_inputs(prediction_steps)
        run, _ = self._run(imu, alignment)
        rows = run.result_rows
        debug = run.debug_rows

        self.assertEqual(len(rows), prediction_steps + 1)
        self.assertEqual(len(debug), prediction_steps + 1)
        timestamps = np.asarray([row["timestamp"] for row in rows])
        quaternions = np.asarray(
            [[row["qx"], row["qy"], row["qz"], row["qw"]] for row in rows]
        )
        numeric_rows = np.asarray(
            [
                [
                    row[field]
                    for field in (
                        "px_m",
                        "py_m",
                        "pz_m",
                        "vx_mps",
                        "vy_mps",
                        "vz_mps",
                        "qx",
                        "qy",
                        "qz",
                        "qw",
                        "roll_rad",
                        "pitch_rad",
                        "yaw_rad",
                        "bg_x_rad_s",
                        "bg_y_rad_s",
                        "bg_z_rad_s",
                        "ba_x_mps2",
                        "ba_y_mps2",
                        "ba_z_mps2",
                    )
                ]
                for row in rows
            ]
        )
        self.assertTrue(np.all(np.diff(timestamps) > 0.0))
        self.assertTrue(np.all(np.isfinite(numeric_rows)))
        np.testing.assert_allclose(
            np.linalg.norm(quaternions, axis=1),
            np.ones(len(rows)),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(run.summary["prediction_steps"], 0)
        self.assertTrue(any(row["update_applied"] for row in rows[1:]))
        self.assertGreaterEqual(
            min(row["P_min_eigenvalue"] for row in debug),
            -DEFAULT_CONFIG.numerical.covariance_negative_tolerance,
        )
        self.assertLessEqual(
            max(row["P_symmetry_error"] for row in debug),
            DEFAULT_CONFIG.numerical.covariance_symmetry_tolerance,
        )

    def _assert_states_equal(self, actual, expected) -> None:
        np.testing.assert_allclose(actual.p_W_m, expected.p_W_m)
        np.testing.assert_allclose(actual.v_W_mps, expected.v_W_mps)
        np.testing.assert_allclose(actual.q_xyzw, expected.q_xyzw)
        np.testing.assert_allclose(actual.bg_rad_s, expected.bg_rad_s)
        np.testing.assert_allclose(actual.ba_mps2, expected.ba_mps2)
        np.testing.assert_allclose(actual.P, expected.P)


if __name__ == "__main__":
    unittest.main()
