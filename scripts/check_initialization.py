"""Print a compact real-data diagnostic for 6D ESKF initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from initialization import ESKF6DInitialization, initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)


def _array(value: np.ndarray) -> str:
    return np.array2string(value, precision=12, suppress_small=False)


def format_report(result: ESKF6DInitialization, next_dt_s: float) -> str:
    """Format only the quantities needed for initialization review."""

    lines = [
        "6D ESKF Initialization",
        "",
        f"IMU initialization index: {result.imu_index}",
        f"Pose initialization index: {result.pose_index}",
        f"Initialization timestamp: {result.timestamp:.9f} s",
        f"Future prediction boundary: {result.imu_index} -> {result.imu_index + 1}",
        f"Future prediction dt (not applied): {next_dt_s:.12f} s",
        "",
        f"q0 xyzw: {_array(result.q0_xyzw)}",
        f"bg0 rad/s: {_array(result.bg0_rad_s)}",
        "",
        f"P0 diagonal: {_array(np.diag(result.P0))}",
        "",
        f"Static gyro std rad/s: {_array(result.gyro_static_std_rad_s)}",
        f"Static sample count: {result.static_sample_count}",
        f"Median dt: {result.median_dt_s:.12f} s",
        f"Gyro noise density (data estimate): {_array(result.gyro_noise_density)}",
        "Gyro bias random-walk density (engineering initial value): "
        f"{result.gyro_bias_random_walk_density:.12g}",
        f"Qc diagonal: {_array(np.diag(result.Qc))}",
    ]
    return "\n".join(lines) + "\n"


def run(processed_directory: Path) -> ESKF6DInitialization:
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
    result = initialize_eskf6d(
        imu, pose, init_stats, alignment, DEFAULT_CONFIG
    )
    next_index = result.imu_index + 1
    if next_index >= imu.dt.size:
        raise ValueError("initialization leaves no following IMU sample")
    next_dt = float(imu.dt[next_index])
    if not np.isfinite(next_dt) or next_dt <= 0.0:
        raise ValueError("the first future prediction dt must be finite and positive")
    print(format_report(result, next_dt), end="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    arguments = parser.parse_args()
    run(arguments.processed_dir)


if __name__ == "__main__":
    main()
