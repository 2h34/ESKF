"""Prepare SI measurements, static statistics, and one-to-one timestamp matches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess import (
    align_timestamps, analyze_initial_static_segment, load_imu_csv,
    load_pose_csv, save_processed_data,
)


def run(raw_directory: Path, output_directory: Path) -> dict:
    imu, imu_summary = load_imu_csv(raw_directory / "imu.csv")
    pose, pose_summary = load_pose_csv(raw_directory / "pose_cov.csv")
    stats = analyze_initial_static_segment(imu)
    alignment = align_timestamps(imu.timestamp, pose.timestamp)
    save_processed_data(output_directory, imu, pose, stats, alignment)

    matched = alignment.pose_index_for_imu >= 0
    match_count = int(np.count_nonzero(matched))
    report = {
        "imu": imu_summary, "pose": pose_summary,
        "static_segment": {
            "static_start_idx": stats.static_start_idx, "static_end_idx": stats.static_end_idx,
            "static_start_time": stats.static_start_time, "static_end_time": stats.static_end_time,
            "gyro_mean_rad_s": stats.gyro_mean.tolist(), "gyro_std_rad_s": stats.gyro_std.tolist(),
            "acc_mean_mps2": stats.acc_mean.tolist(), "acc_std_mps2": stats.acc_std.tolist(),
            "is_static": stats.is_static,
        },
        "alignment": {
            "valid_match_count": match_count, "unmatched_imu_count": int(matched.size - match_count),
            "unused_pose_count": int(pose.timestamp.size - match_count),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output_directory / "diagnostic_report.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    arguments = parser.parse_args()
    run(arguments.raw_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
