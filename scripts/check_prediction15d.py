"""Run the first real-data 15D prediction step for development review."""

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
    next_k = k0 + 1
    if next_k >= imu.timestamp.size:
        raise ValueError("initialization index has no following IMU sample")
    gyro = imu.gyro_rad_s[k0]
    acc = imu.acc_mps2[k0]
    dt = float(imu.dt[next_k])

    debug = filter_15d.predict(gyro, acc, dt)
    after = filter_15d.get_state()
    minimum_eigenvalue = float(np.linalg.eigvalsh(after.P).min())

    print("15D ESKF First Prediction")
    print()
    print(f"k0: {k0}")
    print(f"step: {k0} -> {next_k}")
    print(f"dt[{next_k}]: {dt:.12f} s")
    print(f"p before World m: {_array(debug.p_before_W_m)}")
    print(f"p after World m: {_array(debug.p_after_W_m)}")
    print(f"v before World m/s: {_array(debug.v_before_W_mps)}")
    print(f"v after World m/s: {_array(debug.v_after_W_mps)}")
    print(f"q before xyzw: {_array(debug.q_before_xyzw)}")
    print(f"q after xyzw: {_array(debug.q_after_xyzw)}")
    print(f"q norm: {debug.q_norm:.15f}")
    print(f"bg Body rad/s: {_array(after.bg_rad_s)}")
    print(f"ba Body m/s^2: {_array(after.ba_mps2)}")
    print(f"omega_hat Body rad/s: {_array(debug.omega_hat_rad_s)}")
    print(f"f_hat Body m/s^2: {_array(debug.f_hat_B_mps2)}")
    print(f"a_hat World m/s^2: {_array(debug.a_hat_W_mps2)}")
    print(f"Fc shape: {debug.Fc.shape}")
    print(f"Gc shape: {debug.Gc.shape}")
    print(f"Phi shape: {debug.Phi.shape}")
    print(f"Qd shape: {debug.Qd.shape}")
    print(f"P symmetry error: {debug.P_symmetry_error:.3e}")
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
