import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class ProcessedIOTests(unittest.TestCase):
    def test_current_npz_files_restore_dataclasses(self) -> None:
        imu = load_processed_imu(PROCESSED_ROOT / "imu_processed.npz")
        pose = load_processed_pose(PROCESSED_ROOT / "pose_processed.npz")
        stats = load_init_stats(
            PROCESSED_ROOT / "init_stats.npz",
            imu_sample_count=imu.timestamp.size,
        )
        alignment = load_alignment(
            PROCESSED_ROOT / "match_table.npz",
            imu_sample_count=imu.timestamp.size,
            pose_sample_count=pose.timestamp.size,
        )
        self.assertIsInstance(imu, ProcessedIMUData)
        self.assertIsInstance(pose, ProcessedPoseData)
        self.assertIsInstance(stats, InitStats)
        self.assertIsInstance(alignment, AlignmentResult)

    def test_bad_imu_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_imu.npz"
            np.savez(
                path,
                timestamp=np.array([0.0, 0.01]),
                dt=np.array([np.nan, 0.01]),
                gyro_rad_s=np.zeros((2, 2)),
                acc_mps2=np.zeros((2, 3)),
            )
            with self.assertRaisesRegex(ValueError, "gyro_rad_s must have shape"):
                load_processed_imu(path)

    def test_out_of_range_match_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_match.npz"
            np.savez(
                path,
                pose_index_for_imu=np.array([0, 2], dtype=np.int64),
                time_error=np.array([0.0, 0.0]),
                tolerance_s=0.001,
            )
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                load_alignment(path, imu_sample_count=2, pose_sample_count=2)

    def test_duplicate_pose_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate_match.npz"
            np.savez(
                path,
                pose_index_for_imu=np.array([0, 0], dtype=np.int64),
                time_error=np.array([0.0, 0.0]),
                tolerance_s=0.001,
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing and unique"):
                load_alignment(path, imu_sample_count=2, pose_sample_count=1)


if __name__ == "__main__":
    unittest.main()
