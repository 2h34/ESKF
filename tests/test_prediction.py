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
from rotation_utils import quaternion_multiply, rotvec_to_quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class PredictionTests(unittest.TestCase):
    """Necessary development-time tests for prediction-only behavior."""

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

    def _synthetic_initialization(
        self,
        *,
        q0: np.ndarray | None = None,
        bg0: np.ndarray | None = None,
    ) -> ESKF6DInitialization:
        return replace(
            self.real_initialization,
            q0_xyzw=(
                np.array([0.0, 0.0, 0.0, 1.0]) if q0 is None else q0
            ),
            bg0_rad_s=(np.zeros(3) if bg0 is None else bg0),
            P0=np.diag([1e-4, 2e-4, 3e-4, 4e-6, 5e-6, 6e-6]),
            Qc=np.diag([1e-6, 2e-6, 3e-6, 4e-8, 5e-8, 6e-8]),
        )

    def test_zero_corrected_angular_velocity_keeps_attitude(self) -> None:
        bg = np.array([0.1, -0.2, 0.3])
        filter_6d = ESKF6D.from_initialization(
            self._synthetic_initialization(bg0=bg)
        )
        before = filter_6d.get_state()
        debug = filter_6d.predict(bg, 0.1)
        after = filter_6d.get_state()

        np.testing.assert_allclose(debug.omega_hat_rad_s, np.zeros(3))
        np.testing.assert_allclose(after.q_xyzw, before.q_xyzw, atol=1e-15)
        np.testing.assert_array_equal(after.bg_rad_s, before.bg_rad_s)

    def test_constant_angular_velocity_is_right_multiplied(self) -> None:
        q_old = rotvec_to_quaternion(np.array([0.3, 0.0, 0.0]))
        initialization = self._synthetic_initialization(q0=q_old)
        filter_6d = ESKF6D.from_initialization(initialization)
        filter_6d.predict(np.array([0.0, 0.0, 1.0]), 0.1)

        expected_delta = rotvec_to_quaternion(np.array([0.0, 0.0, 0.1]))
        expected = quaternion_multiply(q_old, expected_delta)
        np.testing.assert_allclose(
            filter_6d.get_state().q_xyzw, expected, rtol=0.0, atol=1e-14
        )

    def test_nominal_bias_stays_constant(self) -> None:
        filter_6d = ESKF6D.from_initialization(self.real_initialization)
        before = filter_6d.get_state().bg_rad_s
        filter_6d.predict(np.array([0.4, -0.3, 0.2]), 0.02)
        np.testing.assert_array_equal(filter_6d.get_state().bg_rad_s, before)

    def test_predicted_covariance_properties(self) -> None:
        filter_6d = ESKF6D.from_initialization(self.real_initialization)
        filter_6d.predict(np.array([0.4, -0.3, 0.2]), 0.02)
        covariance = filter_6d.get_state().P

        self.assertEqual(covariance.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(covariance)))
        np.testing.assert_allclose(
            covariance, covariance.T, rtol=0.0, atol=1e-15
        )
        self.assertTrue(np.all(np.diag(covariance) >= 0.0))

    def test_Qd_scales_linearly_with_dt(self) -> None:
        initialization = self._synthetic_initialization()
        gyro = np.array([0.1, 0.2, -0.3])
        debug_short = ESKF6D.from_initialization(initialization).predict(
            gyro, 0.01
        )
        debug_long = ESKF6D.from_initialization(initialization).predict(
            gyro, 0.02
        )
        np.testing.assert_allclose(
            debug_long.Qd, 2.0 * debug_short.Qd, rtol=1e-14, atol=0.0
        )

    def test_real_first_step_uses_gyro_395_and_dt_396(self) -> None:
        filter_6d = ESKF6D.from_initialization(self.real_initialization)
        k = self.real_initialization.imu_index
        self.assertEqual(k, 395)
        debug = filter_6d.predict(self.imu.gyro_rad_s[k], self.imu.dt[k + 1])

        self.assertEqual(debug.imu_index_before, 395)
        self.assertEqual(debug.imu_index_after, 396)
        np.testing.assert_array_equal(
            debug.gyro_measurement_rad_s, self.imu.gyro_rad_s[395]
        )
        self.assertEqual(debug.dt_s, float(self.imu.dt[396]))
        self.assertAlmostEqual(
            filter_6d.get_state().timestamp,
            float(self.imu.timestamp[396]),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
