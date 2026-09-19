"""Run the first real-data 15D prediction and pose update sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def _array(value: np.ndarray) -> str:
    return np.array2string(value, precision=12, suppress_small=False)


def run(processed_directory: Path) -> ESKF15D:
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
    filter_15d = ESKF15D.from_initialization(
        initialization, DEFAULT_CONFIG.numerical
    )

    k0 = initialization.imu_index
    current_k = k0 + 1
    if current_k >= imu.timestamp.size:
        raise ValueError("initialization index has no following IMU sample")
    filter_15d.predict(
        imu.gyro_rad_s[k0],
        imu.acc_mps2[k0],
        float(imu.dt[current_k]),
    )

    pose_index = int(alignment.pose_index_for_imu[current_k])
    if pose_index < 0:
        raise RuntimeError(
            f"IMU index {current_k} has no matched FAST-LIO pose"
        )
    covariance_diag = pose.cov_diag[pose_index, 0:6]
    if not np.all(covariance_diag > 0.0):
        raise RuntimeError(
            f"pose index {pose_index} has nonpositive pose covariance"
        )

    predicted = filter_15d.get_state()
    p_obs = pose.position[pose_index]
    q_obs = pose.quaternion_xyzw[pose_index]
    R_pose = np.diag(covariance_diag)
    debug = filter_15d.update_pose(p_obs, q_obs, R_pose)
    after = filter_15d.get_state()
    minimum_eigenvalue = float(np.linalg.eigvalsh(after.P).min())

    print("15D ESKF First Prediction + Pose Update")
    print()
    print(f"IMU step: {k0} -> {current_k}")
    print(f"pose index: {pose_index}")
    print(f"timestamp: {float(imu.timestamp[current_k]):.9f} s")
    print(f"predicted position World m: {_array(predicted.p_W_m)}")
    print(f"observed position World m: {_array(p_obs)}")
    print(f"position residual m: {_array(debug.r_position_m)}")
    print(f"attitude residual rad: {_array(debug.r_theta_rad)}")
    print(f"delta_p m: {_array(debug.delta_p_m)}")
    print(f"delta_v m/s: {_array(debug.delta_v_mps)}")
    print(f"delta_theta rad: {_array(debug.delta_theta_rad)}")
    print(f"delta_bg rad/s: {_array(debug.delta_bg_rad_s)}")
    print(f"delta_ba m/s^2: {_array(debug.delta_ba_mps2)}")
    print(f"position after World m: {_array(after.p_W_m)}")
    print(f"velocity after World m/s: {_array(after.v_W_mps)}")
    print(f"quaternion after xyzw: {_array(after.q_xyzw)}")
    print(f"gyro bias after rad/s: {_array(after.bg_rad_s)}")
    print(f"accelerometer bias after m/s^2: {_array(after.ba_mps2)}")
    print(f"q norm: {debug.q_norm:.15f}")
    print(f"NIS: {debug.NIS:.12f}")
    print(
        "P raw/final symmetry error: "
        f"{debug.raw_covariance_symmetry_error:.3e} / "
        f"{debug.final_covariance_symmetry_error:.3e}"
    )
    print(f"P minimum eigenvalue: {minimum_eigenvalue:.12e}")
    return filter_15d


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
