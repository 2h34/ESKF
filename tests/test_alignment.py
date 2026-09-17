import unittest

import numpy as np

from alignment import align_timestamps, summarize_alignment
from config import AlignmentConfig


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AlignmentConfig(absolute_tolerance_s=0.003)

    def test_exact_timestamps(self) -> None:
        imu = np.array([0.0, 0.01, 0.02])
        result = align_timestamps(imu, imu.copy(), self.config)
        np.testing.assert_array_equal(result.pose_index_for_imu, [0, 1, 2])
        np.testing.assert_allclose(result.time_error, 0.0)

    def test_nearest_timestamp_and_signed_error(self) -> None:
        imu = np.array([0.0, 0.01, 0.02])
        pose = np.array([0.001, 0.008, 0.022])
        result = align_timestamps(imu, pose, self.config)
        np.testing.assert_array_equal(result.pose_index_for_imu, [0, 1, 2])
        np.testing.assert_allclose(result.time_error, [0.001, -0.002, 0.002])

    def test_nearest_candidate_can_skip_an_older_pose(self) -> None:
        result = align_timestamps(
            [0.0, 0.01],
            [-0.002, 0.001, 0.01],
            AlignmentConfig(absolute_tolerance_s=0.003),
        )
        np.testing.assert_array_equal(result.pose_index_for_imu, [1, 2])
        np.testing.assert_allclose(result.time_error, [0.001, 0.0])

    def test_outside_tolerance_is_unmatched(self) -> None:
        result = align_timestamps(
            [0.0, 0.01, 0.02],
            [0.004, 0.014, 0.024],
            AlignmentConfig(absolute_tolerance_s=0.001),
        )
        np.testing.assert_array_equal(result.pose_index_for_imu, [-1, -1, -1])
        self.assertTrue(np.all(np.isnan(result.time_error)))

    def test_pose_is_not_reused(self) -> None:
        result = align_timestamps(
            [0.0, 0.001, 0.01],
            [0.0005, 0.0105],
            AlignmentConfig(absolute_tolerance_s=0.002),
        )
        matched = result.pose_index_for_imu[result.pose_index_for_imu >= 0]
        self.assertEqual(np.unique(matched).size, matched.size)
        self.assertEqual(summarize_alignment(result, 2).duplicate_pose_use_count, 0)

    def test_matches_remain_monotonic(self) -> None:
        result = align_timestamps(
            [0.0, 0.01, 0.02, 0.03],
            [0.002, 0.011, 0.029],
            AlignmentConfig(absolute_tolerance_s=0.003),
        )
        matched = result.pose_index_for_imu[result.pose_index_for_imu >= 0]
        self.assertTrue(np.all(np.diff(matched) > 0))

    def test_tolerance_is_derived_from_data(self) -> None:
        result = align_timestamps(
            [0.0, 0.01, 0.02],
            [0.0, 0.02, 0.04],
            AlignmentConfig(tolerance_ratio=0.25),
        )
        self.assertAlmostEqual(result.tolerance_s, 0.005)


if __name__ == "__main__":
    unittest.main()
