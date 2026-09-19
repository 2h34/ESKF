import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from eskf6d import ESKF6D
from eskf15d import ESKF15D
from initialization import initialize_eskf6d, initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import (
    hat,
    normalize_quaternion,
    quaternion_multiply,
    rotvec_to_quaternion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class Update15DTests(unittest.TestCase):
    """Core tests for 15D FAST-LIO pose update, injection, and reset."""

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

    @staticmethod
    def _R_pose() -> np.ndarray:
        return np.diag([2e-4, 3e-4, 4e-4, 1e-4, 2e-4, 3e-4])

    def _initialization(
        self,
        *,
        p: np.ndarray | None = None,
        q: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ):
        return replace(
            self.initialization15d,
            p0_W_m=np.zeros(3) if p is None else p,
            v0_W_mps=np.zeros(3),
            q0_xyzw=(
                np.array([0.0, 0.0, 0.0, 1.0]) if q is None else q
            ),
            bg0_rad_s=np.array([0.01, -0.02, 0.03]),
            ba0_mps2=np.array([0.1, -0.05, 0.02]),
            P0=(
                np.diag(np.linspace(1e-3, 15e-3, 15))
                if P is None
                else P
            ),
            Qc=np.diag(np.linspace(1e-7, 12e-7, 12)),
        )

    def test_position_residual_moves_position_toward_observation(self) -> None:
        p_pred = np.array([1.0, -2.0, 0.5])
        p_obs = p_pred + np.array([0.2, -0.1, 0.05])
        filter_15d = ESKF15D.from_initialization(
            self._initialization(p=p_pred)
        )
        q_pred = filter_15d.get_state().q_xyzw

        debug = filter_15d.update_pose(p_obs, q_pred, self._R_pose())
        after = filter_15d.get_state()

        np.testing.assert_allclose(debug.r_position_m, p_obs - p_pred)
        np.testing.assert_allclose(debug.r_theta_rad, np.zeros(3), atol=1e-15)
        self.assertGreater(float(np.dot(after.p_W_m - p_pred, p_obs - p_pred)), 0.0)
        self.assertLess(
            float(np.linalg.norm(p_obs - after.p_W_m)),
            float(np.linalg.norm(p_obs - p_pred)),
        )

    def test_attitude_update_matches_6d_convention_and_injection(self) -> None:
        q_pred = rotvec_to_quaternion(np.array([0.2, -0.1, 0.3]))
        observation_increment = np.array([0.03, -0.02, 0.01])
        q_obs = quaternion_multiply(
            q_pred, rotvec_to_quaternion(observation_increment)
        )
        P15 = np.diag(np.linspace(1e-3, 15e-3, 15))
        attitude_bias_indices = [6, 7, 8, 9, 10, 11]
        P6 = P15[np.ix_(attitude_bias_indices, attitude_bias_indices)]
        filter_15d = ESKF15D.from_initialization(
            self._initialization(q=q_pred, P=P15)
        )
        filter_6d = ESKF6D.from_initialization(
            replace(
                self.initialization6d,
                q0_xyzw=q_pred,
                bg0_rad_s=np.array([0.01, -0.02, 0.03]),
                P0=P6,
            )
        )
        R_pose = self._R_pose()

        debug15 = filter_15d.update_pose(np.zeros(3), q_obs, R_pose)
        debug6 = filter_6d.update_attitude(q_obs, R_pose[3:6, 3:6])

        np.testing.assert_allclose(debug15.q_rel_xyzw, debug6.q_rel_xyzw)
        np.testing.assert_allclose(debug15.r_theta_rad, debug6.r_theta_rad)
        np.testing.assert_allclose(
            debug15.delta_theta_rad, debug6.delta_theta_rad
        )
        np.testing.assert_allclose(
            filter_15d.get_state().q_xyzw,
            filter_6d.get_state().q_xyzw,
        )
        expected_q = normalize_quaternion(
            quaternion_multiply(
                q_pred, rotvec_to_quaternion(debug15.delta_theta_rad)
            )
        )
        np.testing.assert_allclose(debug15.q_after_xyzw, expected_q)

    def test_position_cross_covariance_indirectly_corrects_velocity(self) -> None:
        P = np.diag(np.full(15, 1e-3))
        P[0, 3] = 5e-4
        P[3, 0] = 5e-4
        self.assertGreater(float(np.linalg.eigvalsh(P).min()), 0.0)
        filter_15d = ESKF15D.from_initialization(self._initialization(P=P))
        before = filter_15d.get_state()

        debug = filter_15d.update_pose(
            np.array([0.2, 0.0, 0.0]),
            before.q_xyzw,
            self._R_pose(),
        )
        after = filter_15d.get_state()

        self.assertGreater(debug.delta_v_mps[0], 0.0)
        np.testing.assert_allclose(
            after.v_W_mps, before.v_W_mps + debug.delta_v_mps
        )

    def test_H_gain_joseph_and_reset_match_frozen_formulas(self) -> None:
        initialization = self._initialization()
        filter_15d = ESKF15D.from_initialization(initialization)
        before = filter_15d.get_state()
        p_obs = np.array([0.1, -0.05, 0.02])
        q_obs = rotvec_to_quaternion(np.array([0.02, -0.03, 0.01]))
        R_pose = self._R_pose()

        debug = filter_15d.update_pose(p_obs, q_obs, R_pose)
        after = filter_15d.get_state()

        expected_H = np.zeros((6, 15))
        expected_H[0:3, 0:3] = np.eye(3)
        expected_H[3:6, 6:9] = np.eye(3)
        PHt = before.P @ expected_H.T
        expected_S = expected_H @ PHt + R_pose
        expected_K = np.linalg.solve(expected_S, PHt.T).T
        A = np.eye(15) - expected_K @ expected_H
        expected_P_joseph = A @ before.P @ A.T + expected_K @ R_pose @ expected_K.T
        expected_G_reset = np.eye(15)
        expected_G_reset[6:9, 6:9] = (
            np.eye(3) - 0.5 * hat(debug.delta_theta_rad)
        )
        expected_P_reset = (
            expected_G_reset @ expected_P_joseph @ expected_G_reset.T
        )
        expected_P_reset = 0.5 * (expected_P_reset + expected_P_reset.T)

        np.testing.assert_array_equal(debug.H, expected_H)
        np.testing.assert_allclose(debug.S, expected_S)
        np.testing.assert_allclose(debug.K, expected_K)
        np.testing.assert_allclose(
            debug.P_diag_after_joseph, np.diag(expected_P_joseph)
        )
        np.testing.assert_allclose(debug.G_reset, expected_G_reset)
        np.testing.assert_allclose(after.P, expected_P_reset)
        np.testing.assert_allclose(after.P, after.P.T, rtol=0.0, atol=1e-15)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(after.P).min()), -1e-12)

    def test_quaternion_double_cover_produces_equivalent_pose_update(self) -> None:
        q_obs = rotvec_to_quaternion(np.array([0.02, -0.01, 0.03]))
        positive_filter = ESKF15D.from_initialization(self._initialization())
        negative_filter = ESKF15D.from_initialization(self._initialization())
        p_obs = np.array([0.03, -0.02, 0.01])

        positive = positive_filter.update_pose(p_obs, q_obs, self._R_pose())
        negative = negative_filter.update_pose(p_obs, -q_obs, self._R_pose())

        self.assertFalse(positive.quaternion_sign_flipped)
        self.assertTrue(negative.quaternion_sign_flipped)
        np.testing.assert_allclose(positive.residual, negative.residual)
        np.testing.assert_allclose(positive.delta_x, negative.delta_x)
        positive_state = positive_filter.get_state()
        negative_state = negative_filter.get_state()
        for positive_value, negative_value in (
            (positive_state.p_W_m, negative_state.p_W_m),
            (positive_state.v_W_mps, negative_state.v_W_mps),
            (positive_state.q_xyzw, negative_state.q_xyzw),
            (positive_state.bg_rad_s, negative_state.bg_rad_s),
            (positive_state.ba_mps2, negative_state.ba_mps2),
            (positive_state.P, negative_state.P),
        ):
            np.testing.assert_allclose(positive_value, negative_value)


if __name__ == "__main__":
    unittest.main()
