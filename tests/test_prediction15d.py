import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG, GRAVITY_WORLD_MPS2
from eskf6d import ESKF6D
from eskf15d import ESKF15D
from initialization import initialize_eskf6d, initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import hat, quaternion_to_rotation_matrix, rotvec_to_quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class Prediction15DTests(unittest.TestCase):
    """Focused tests for the 15D nominal and covariance prediction."""

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
        cls.initialization15d = initialize_eskf15d(
            cls.imu, cls.pose, cls.stats, cls.alignment, DEFAULT_CONFIG
        )
        cls.initialization6d = initialize_eskf6d(
            cls.imu, cls.pose, cls.stats, cls.alignment, DEFAULT_CONFIG
        )

    def _initialization(
        self,
        *,
        p: np.ndarray | None = None,
        v: np.ndarray | None = None,
        q: np.ndarray | None = None,
        bg: np.ndarray | None = None,
        ba: np.ndarray | None = None,
    ):
        return replace(
            self.initialization15d,
            p0_W_m=np.zeros(3) if p is None else p,
            v0_W_mps=np.zeros(3) if v is None else v,
            q0_xyzw=(
                np.array([0.0, 0.0, 0.0, 1.0]) if q is None else q
            ),
            bg0_rad_s=np.zeros(3) if bg is None else bg,
            ba0_mps2=np.zeros(3) if ba is None else ba,
            P0=np.diag(np.linspace(1e-5, 15e-5, 15)),
            Qc=np.diag(np.linspace(1e-7, 12e-7, 12)),
        )

    def test_stationary_case_preserves_nominal_state_and_biases(self) -> None:
        q = rotvec_to_quaternion(np.array([0.2, -0.1, 0.3]))
        bg = np.array([0.01, -0.02, 0.03])
        ba = np.array([0.08, -0.04, 0.02])
        initialization = self._initialization(q=q, bg=bg, ba=ba)
        filter_15d = ESKF15D.from_initialization(initialization)
        before = filter_15d.get_state()
        R_before = quaternion_to_rotation_matrix(q)
        acc = -R_before.T @ GRAVITY_WORLD_MPS2 + ba

        filter_15d.predict(bg, acc, 0.01)
        after = filter_15d.get_state()

        np.testing.assert_allclose(after.p_W_m, before.p_W_m, atol=1e-15)
        np.testing.assert_allclose(after.v_W_mps, before.v_W_mps, atol=1e-15)
        np.testing.assert_allclose(after.q_xyzw, before.q_xyzw, atol=1e-15)
        np.testing.assert_array_equal(after.bg_rad_s, before.bg_rad_s)
        np.testing.assert_array_equal(after.ba_mps2, before.ba_mps2)

    def test_constant_world_acceleration_uses_old_velocity(self) -> None:
        p_before = np.array([1.0, -2.0, 0.5])
        v_before = np.array([0.4, -0.3, 0.2])
        ba = np.array([0.02, -0.03, 0.04])
        a_W = np.array([0.6, -0.4, 0.8])
        dt = 0.2
        initialization = self._initialization(p=p_before, v=v_before, ba=ba)
        filter_15d = ESKF15D.from_initialization(initialization)
        acc = a_W - GRAVITY_WORLD_MPS2 + ba

        filter_15d.predict(np.zeros(3), acc, dt)
        after = filter_15d.get_state()

        np.testing.assert_allclose(after.v_W_mps, v_before + a_W * dt)
        np.testing.assert_allclose(
            after.p_W_m,
            p_before + v_before * dt + 0.5 * a_W * dt**2,
        )

    def test_pure_rotation_matches_6d_attitude_prediction(self) -> None:
        q = rotvec_to_quaternion(np.array([0.3, -0.2, 0.1]))
        bg = np.array([0.01, 0.02, -0.03])
        ba = np.array([0.05, -0.02, 0.04])
        gyro = np.array([0.4, -0.3, 0.2])
        dt = 0.015
        filter_15d = ESKF15D.from_initialization(
            self._initialization(q=q, bg=bg, ba=ba)
        )
        filter_6d = ESKF6D.from_initialization(
            replace(self.initialization6d, q0_xyzw=q, bg0_rad_s=bg)
        )
        acc = (
            -quaternion_to_rotation_matrix(q).T @ GRAVITY_WORLD_MPS2 + ba
        )

        filter_15d.predict(gyro, acc, dt)
        filter_6d.predict(gyro, dt)

        q_after = filter_15d.get_state().q_xyzw
        np.testing.assert_allclose(
            q_after, filter_6d.get_state().q_xyzw, rtol=0.0, atol=1e-15
        )
        self.assertAlmostEqual(float(np.linalg.norm(q_after)), 1.0)

    def test_Fc_and_Gc_use_left_endpoint_quantities(self) -> None:
        q = rotvec_to_quaternion(np.array([0.2, 0.1, -0.15]))
        bg = np.array([0.02, -0.01, 0.03])
        ba = np.array([0.1, -0.2, 0.05])
        gyro = np.array([0.3, -0.4, 0.2])
        acc = np.array([0.5, -0.6, 9.7])
        debug = ESKF15D.from_initialization(
            self._initialization(q=q, bg=bg, ba=ba)
        ).predict(gyro, acc, 0.01)

        R_before = quaternion_to_rotation_matrix(q)
        omega_hat = gyro - bg
        f_hat_B = acc - ba
        expected_Fc = np.zeros((15, 15))
        expected_Fc[0:3, 3:6] = np.eye(3)
        expected_Fc[3:6, 6:9] = -R_before @ hat(f_hat_B)
        expected_Fc[3:6, 12:15] = -R_before
        expected_Fc[6:9, 6:9] = -hat(omega_hat)
        expected_Fc[6:9, 9:12] = -np.eye(3)
        expected_Gc = np.zeros((15, 12))
        expected_Gc[3:6, 0:3] = -R_before
        expected_Gc[6:9, 3:6] = -np.eye(3)
        expected_Gc[9:12, 6:9] = np.eye(3)
        expected_Gc[12:15, 9:12] = np.eye(3)

        np.testing.assert_allclose(debug.omega_hat_rad_s, omega_hat)
        np.testing.assert_allclose(debug.f_hat_B_mps2, f_hat_B)
        np.testing.assert_allclose(debug.Fc, expected_Fc)
        np.testing.assert_allclose(debug.Gc, expected_Gc)

    def test_first_order_covariance_prediction(self) -> None:
        initialization = self._initialization()
        filter_15d = ESKF15D.from_initialization(initialization)
        P_before = filter_15d.get_state().P
        dt = 0.02
        debug = filter_15d.predict(
            np.array([0.1, -0.2, 0.3]),
            np.array([0.2, -0.1, 9.75]),
            dt,
        )
        after = filter_15d.get_state()
        expected_Phi = np.eye(15) + debug.Fc * dt
        expected_Qd = debug.Gc @ initialization.Qc @ debug.Gc.T * dt
        expected_P = expected_Phi @ P_before @ expected_Phi.T + expected_Qd

        np.testing.assert_allclose(debug.Phi, expected_Phi)
        np.testing.assert_allclose(debug.Qd, expected_Qd)
        np.testing.assert_array_equal(debug.Qd[0:3, 0:3], np.zeros((3, 3)))
        np.testing.assert_allclose(after.P, expected_P, rtol=1e-14, atol=1e-18)
        np.testing.assert_allclose(after.P, after.P.T, rtol=0.0, atol=1e-15)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(after.P).min()), -1e-12)

    def test_nonpositive_dt_is_rejected(self) -> None:
        for dt in (0.0, -0.01):
            with self.subTest(dt=dt):
                filter_15d = ESKF15D.from_initialization(self._initialization())
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    filter_15d.predict(np.zeros(3), np.zeros(3), dt)


if __name__ == "__main__":
    unittest.main()
