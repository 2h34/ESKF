"""Convert the two raw CSV files into data used by the ESKF."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess import (align_timestamps, analyze_initial_static_segment,
                        load_imu_csv, load_pose_csv, save_processed_data)


def run(raw_directory: Path, output_directory: Path) -> dict:
    imu, imu_info = load_imu_csv(raw_directory / "imu.csv")
    pose, pose_info = load_pose_csv(raw_directory / "pose_cov.csv")
    stats = analyze_initial_static_segment(imu)
    alignment = align_timestamps(imu.timestamp, pose.timestamp)
    save_processed_data(output_directory, imu, pose, stats, alignment)
    matched = int((alignment.pose_index_for_imu >= 0).sum())
    return {
        "imu": imu_info, "pose": pose_info,
        "static_interval": [stats.static_start_idx, stats.static_end_idx],
        "matched_imu": matched, "unmatched_imu": int(imu.timestamp.size - matched),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    print(json.dumps(run(args.raw_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
