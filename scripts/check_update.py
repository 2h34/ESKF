"""Diagnose the first real prediction plus attitude update/injection/reset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from eskf6d import ESKF6D
from initialization import initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)


def _array(value: np.ndarray) -> str:
    return np.array2string(value, precision=12, suppress_small=False)


def run(processed_directory: Path) -> None:
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
    initialization = initialize_eskf6d(
        imu, pose, init_stats, alignment, DEFAULT_CONFIG
    )
    filter_6d = ESKF6D.from_initialization(
        initialization, DEFAULT_CONFIG.numerical
    )

    gyro_index = initialization.imu_index
    current_imu_index = gyro_index + 1
    if gyro_index != 395 or current_imu_index != 396:
        raise RuntimeError("expected the frozen first transition 395 -> 396")
    filter_6d.predict(
        imu.gyro_rad_s[gyro_index], imu.dt[current_imu_index]
    )

    pose_index = int(alignment.pose_index_for_imu[current_imu_index])
    if pose_index < 0:
        raise RuntimeError(
            f"IMU index {current_imu_index} has no matched attitude observation"
        )
    attitude_covariance = np.asarray(pose.cov_diag[pose_index, 3:6], dtype=float)
    if not np.all(np.isfinite(attitude_covariance)) or np.any(
        attitude_covariance <= 0.0
    ):
        raise RuntimeError(
            "matched observation has non-positive attitude covariance and must "
            "be skipped by scheduler policy"
        )

    q_observation = pose.quaternion_xyzw[pose_index]
    R_attitude = np.diag(attitude_covariance)
    update = filter_6d.update_attitude(q_observation, R_attitude)

    lines = [
        "6D ESKF First Attitude Update Diagnostic",
        "",
        f"IMU transition: {gyro_index} -> {current_imu_index}",
        f"Gyro source index: {gyro_index}",
        f"dt source index: {current_imu_index}",
        f"Pose index from alignment: {pose_index}",
        "Timestamp from imu.timestamp[396]: "
        f"{float(imu.timestamp[current_imu_index]):.9f} s",
        "",
        f"q predicted xyzw: {_array(update.q_pred_xyzw)}",
        f"q observation raw xyzw: {_array(update.q_obs_raw_xyzw)}",
        f"q observation aligned xyzw: {_array(update.q_obs_aligned_xyzw)}",
        f"Quaternion sign flipped: {update.quaternion_sign_flipped}",
        f"q relative xyzw: {_array(update.q_rel_xyzw)}",
        f"r_theta rad: {_array(update.r_theta_rad)}",
        f"Residual norm: {update.residual_norm:.12e} rad",
        "",
        f"diag(R): {_array(np.diag(update.R_attitude))}",
        f"diag(S): {_array(np.diag(update.S))}",
        f"NIS: {update.NIS:.12e}",
        "",
        "K attitude block:",
        _array(update.K[0:3, :]),
        "K bias block:",
        _array(update.K[3:6, :]),
        f"delta_theta rad: {_array(update.delta_theta_rad)}",
        f"delta_bg rad/s: {_array(update.delta_bg_rad_s)}",
        "",
        f"q before injection xyzw: {_array(update.q_pred_xyzw)}",
        f"q after injection xyzw: {_array(update.q_after_xyzw)}",
        f"q norm: {update.q_norm:.15f}",
        f"bg before rad/s: {_array(update.bg_before_rad_s)}",
        f"bg after rad/s: {_array(update.bg_after_rad_s)}",
        "",
        f"diag(P predicted): {_array(update.P_diag_before)}",
        f"diag(P after Joseph): {_array(update.P_diag_after_joseph)}",
        f"diag(P after reset): {_array(update.P_diag_after_reset)}",
        "G_reset attitude block:",
        _array(update.G_reset[0:3, 0:3]),
        "Raw reset covariance symmetry error: "
        f"{update.raw_covariance_symmetry_error:.3e}",
        "Final covariance symmetry error: "
        f"{update.final_covariance_symmetry_error:.3e}",
    ]
    print("\n".join(lines))


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
