"""Print the real-data Phase 3.1 15D initialization diagnostics only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, GRAVITY_WORLD_MPS2
from initialization import ESKF15DInitialization, initialize_eskf15d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import quaternion_to_rotation_matrix


def _array(value: np.ndarray) -> str:
    return np.array2string(value, precision=12, suppress_small=False)


def format_report(
    result: ESKF15DInitialization,
    acc_mean_mps2: np.ndarray,
) -> str:
    """Format the quantities required for Phase 3.1 human review."""

    gravity_B0 = (
        quaternion_to_rotation_matrix(result.q0_xyzw).T
        @ GRAVITY_WORLD_MPS2
    )
    P_diagonal = np.diag(result.P0)
    return "\n".join(
        [
            "15D ESKF Initialization",
            "",
            f"k0 IMU index: {result.imu_index}",
            f"j0 pose index: {result.pose_index}",
            f"timestamp: {result.timestamp:.9f} s",
            "",
            f"p0 World m: {_array(result.p0_W_m)}",
            f"v0 World m/s: {_array(result.v0_W_mps)}",
            f"q0_WB xyzw: {_array(result.q0_xyzw)}",
            f"bg0 Body rad/s: {_array(result.bg0_rad_s)}",
            f"ba0 Body m/s^2: {_array(result.ba0_mps2)}",
            "",
            f"static acc mean m/s^2: {_array(acc_mean_mps2)}",
            f"|static acc mean|: {np.linalg.norm(acc_mean_mps2):.12f} m/s^2",
            f"R0.T @ g_W: {_array(gravity_B0)}",
            f"|ba0|: {np.linalg.norm(result.ba0_mps2):.12f} m/s^2",
            f"static acc std m/s^2: {_array(result.acc_static_std_mps2)}",
            f"acc noise density: {_array(result.acc_noise_density)}",
            "",
            f"diag(Pp0): {_array(P_diagonal[0:3])}",
            f"diag(Pv0): {_array(P_diagonal[3:6])}",
            f"diag(Ptheta0): {_array(P_diagonal[6:9])}",
            f"diag(Pbg0): {_array(P_diagonal[9:12])}",
            f"diag(Pba0): {_array(P_diagonal[12:15])}",
            f"P0 shape: {result.P0.shape}",
            f"Qc shape: {result.Qc.shape}",
            "P0 symmetry error: "
            f"{np.max(np.abs(result.P0 - result.P0.T)):.3e}",
            "Qc symmetry error: "
            f"{np.max(np.abs(result.Qc - result.Qc.T)):.3e}",
        ]
    ) + "\n"


def run(processed_directory: Path) -> ESKF15DInitialization:
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
    result = initialize_eskf15d(
        imu, pose, init_stats, alignment, DEFAULT_CONFIG
    )
    print(format_report(result, init_stats.acc_mean), end="")
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
