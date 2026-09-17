import unittest
from pathlib import Path

import numpy as np

from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from scripts.check_pose_convention import check_pose_convention


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class PoseConventionCheckTests(unittest.TestCase):
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

    def test_both_hypotheses_produce_finite_statistics(self) -> None:
        result = check_pose_convention(
            self.imu, self.pose, self.stats, self.alignment
        )
        self.assertEqual(result.static_matched_sample_count, 367)
        self.assertEqual(result.excluded_nonpositive_attitude_covariance_count, 20)
        self.assertEqual(result.used_sample_count, 347)
        for hypothesis in (result.hypothesis_a, result.hypothesis_b):
            self.assertTrue(np.all(np.isfinite(hypothesis.angular_error_deg)))
            self.assertTrue(
                np.all(np.isfinite(hypothesis.euclidean_residual_mps2))
            )
        self.assertEqual(result.conclusion, "Strongly supports R_WB")

    def test_zero_covariance_frames_precede_filter_start(self) -> None:
        zero_covariance = np.flatnonzero(np.all(self.pose.cov_diag == 0.0, axis=1))
        self.assertEqual(zero_covariance.size, 20)
        self.assertLess(
            float(self.pose.timestamp[zero_covariance[-1]]),
            self.stats.static_end_time,
        )
        after_initialization = self.alignment.pose_index_for_imu[
            self.stats.static_end_idx :
        ]
        used_after_initialization = after_initialization[after_initialization >= 0]
        self.assertEqual(
            np.intersect1d(used_after_initialization, zero_covariance).size, 0
        )


if __name__ == "__main__":
    unittest.main()
