"""Run the Phase 3.5 15D ESKF and write result diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from eskf15d import ESKF15D
from initialization import initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from runner15d import run_eskf15d, save_eskf15d_run


def run(processed_directory: Path, output_directory: Path) -> dict[str, object]:
    imu = load_processed_imu(processed_directory / "imu_processed.npz")
    pose = load_processed_pose(processed_directory / "pose_processed.npz")
    init_stats = load_init_stats(
        processed_directory / "init_stats.npz",
        imu_sample_count=imu.timestamp.size,
    )
    alignment = load_alignment(
        processed_directory / "match_table.npz",
        imu_sample_count=imu.timestamp.size,
        pose_sample_count=pose.timestamp.size,
    )
    initialization = initialize_eskf15d(
        imu, pose, init_stats, alignment, DEFAULT_CONFIG
    )
    eskf = ESKF15D.from_initialization(
        initialization, DEFAULT_CONFIG.numerical
    )
    run_result = run_eskf15d(
        imu,
        pose,
        init_stats,
        alignment,
        initialization,
        eskf,
        DEFAULT_CONFIG,
    )
    paths = save_eskf15d_run(run_result, output_directory)
    print(json.dumps(run_result.summary, ensure_ascii=False, indent=2))
    print(f"result_csv: {paths.result_csv}")
    print(f"debug_csv: {paths.debug_csv}")
    print(f"summary_json: {paths.summary_json}")
    return run_result.summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    arguments = parser.parse_args()
    run(arguments.processed_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
