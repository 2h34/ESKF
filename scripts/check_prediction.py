"""Review the first real 6D ESKF prediction and a 20-step prediction run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from eskf6d import ESKF6D, PredictionDebug
from initialization import initialize_eskf6d
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)


PREDICTION_STEPS = 20


def _array(value: np.ndarray) -> str:
    return np.array2string(value, precision=12, suppress_small=False)


def _first_step_report(
    debug: PredictionDebug,
    gyro_index: int,
    dt_index: int,
    old_timestamp: float,
    new_timestamp: float,
    bg_rad_s: np.ndarray,
) -> list[str]:
    return [
        "First real prediction (left-endpoint ZOH)",
        f"State transition: {gyro_index} -> {dt_index}",
        f"Gyro source index: {gyro_index}",
        f"dt source index: {dt_index}",
        f"Old timestamp: {old_timestamp:.9f} s",
        f"New timestamp: {new_timestamp:.9f} s",
        f"dt: {debug.dt_s:.12f} s",
        "",
        f"gyro measurement rad/s: {_array(debug.gyro_measurement_rad_s)}",
        f"bg rad/s: {_array(bg_rad_s)}",
        f"omega_hat rad/s: {_array(debug.omega_hat_rad_s)}",
        f"delta_theta rad: {_array(debug.delta_theta_rad)}",
        "",
        f"q before xyzw: {_array(debug.q_before_xyzw)}",
        f"q after xyzw: {_array(debug.q_after_xyzw)}",
        f"q norm: {debug.q_norm:.15f}",
        "",
        "Fc:",
        _array(debug.Fc),
        "Phi:",
        _array(debug.Phi),
        f"diag(Qd): {_array(np.diag(debug.Qd))}",
        f"diag(P before): {_array(debug.P_diag_before)}",
        f"diag(P after): {_array(debug.P_diag_after)}",
        f"P symmetry error: {debug.P_symmetry_error:.3e}",
    ]


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
    if initialization.imu_index + PREDICTION_STEPS >= imu.timestamp.size:
        raise ValueError("not enough IMU samples for the prediction diagnostic")

    filter_6d = ESKF6D.from_initialization(initialization)
    initial_bg = initialization.bg0_rad_s.copy()
    first_k = initialization.imu_index
    first_next_k = first_k + 1
    first_old_timestamp = float(imu.timestamp[first_k])
    first_new_timestamp = float(imu.timestamp[first_next_k])
    first_dt = float(imu.dt[first_next_k])
    first_debug = filter_6d.predict(
        imu.gyro_rad_s[first_k], first_dt
    )

    if not np.isclose(
        first_new_timestamp - first_old_timestamp,
        first_dt,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("dt[k+1] does not match the authoritative IMU timestamps")

    for gyro_index in range(first_k + 1, first_k + PREDICTION_STEPS):
        filter_6d.predict(
            imu.gyro_rad_s[gyro_index], imu.dt[gyro_index + 1]
        )

    final_state = filter_6d.get_state()
    final_imu_index = first_k + PREDICTION_STEPS
    final_timestamp = float(imu.timestamp[final_imu_index])
    final_symmetry_error = float(
        np.max(np.abs(final_state.P - final_state.P.T))
    )
    p_is_finite = bool(np.all(np.isfinite(final_state.P)))
    p_diagonal_nonnegative = bool(np.all(np.diag(final_state.P) >= 0.0))
    bg_change = float(np.max(np.abs(final_state.bg_rad_s - initial_bg)))

    lines = [
        "6D ESKF Prediction Diagnostic",
        "",
        *_first_step_report(
            first_debug,
            first_k,
            first_next_k,
            first_old_timestamp,
            first_new_timestamp,
            initial_bg,
        ),
        "",
        "Short prediction-only run",
        f"Prediction steps: {PREDICTION_STEPS}",
        f"Final IMU index (scheduler): {final_imu_index}",
        f"Final timestamp from imu.timestamp: {final_timestamp:.9f} s",
        f"Final quaternion norm: {np.linalg.norm(final_state.q_xyzw):.15f}",
        f"Maximum nominal bias change: {bg_change:.3e} rad/s",
        f"P finite: {p_is_finite}",
        f"P symmetry error: {final_symmetry_error:.3e}",
        f"P diagonal nonnegative: {p_diagonal_nonnegative}",
        f"Minimum P diagonal: {np.min(np.diag(final_state.P)):.12e}",
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
