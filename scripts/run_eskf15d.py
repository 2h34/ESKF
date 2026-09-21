"""Replay IMU and matched FAST-LIO poses; save posterior states and a run summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from eskf15d import initialize_eskf15d
from preprocess import load_alignment, load_init_stats, load_processed_imu, load_processed_pose
from rotation_utils import quaternion_to_rpy


RESULT_FIELDNAMES = (
    "timestamp",
    "imu_index",
    "px_m",
    "py_m",
    "pz_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "bg_x_rad_s",
    "bg_y_rad_s",
    "bg_z_rad_s",
    "ba_x_mps2",
    "ba_y_mps2",
    "ba_z_mps2",
    "has_pose_match",
    "update_applied",
)


def _statistics(values: list[float]) -> dict:
    if not values:
        return dict.fromkeys(("mean", "median", "p95", "max"))
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("summary statistics contain NaN or Inf")
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95.0)), "max": float(np.max(array))}


def _write_csv_atomic(path: Path, rows: list[list]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(RESULT_FIELDNAMES)
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, summary: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(processed_directory: Path, output_directory: Path) -> dict:
    imu = load_processed_imu(processed_directory / "imu_processed.npz")
    pose = load_processed_pose(processed_directory / "pose_processed.npz")
    stats = load_init_stats(processed_directory / "init_stats.npz", imu_sample_count=imu.timestamp.size)
    alignment = load_alignment(processed_directory / "match_table.npz",
                               imu_sample_count=imu.timestamp.size, pose_sample_count=pose.timestamp.size)
    eskf, k0 = initialize_eskf15d(imu, pose, stats, alignment)
    rows = []
    position_residuals, attitude_residuals, nis_values = [], [], []
    pose_matches = successful_updates = unmatched = invalid_covariance = 0
    minimum_P_eigenvalue, max_P_symmetry_error = float("inf"), 0.0

    for k in range(k0, imu.timestamp.size):
        j = int(alignment.pose_index_for_imu[k])
        update_applied = False
        pose_matches += int(j >= 0)
        # The initial row is already a posterior; do not reuse its observation.
        if k > k0:
            eskf.predict(imu.gyro_rad_s[k - 1], imu.acc_mps2[k - 1], float(imu.dt[k]))
            if j < 0:
                unmatched += 1
            elif np.any(pose.cov_diag[j] <= 0.0):
                invalid_covariance += 1
            else:
                r_position, r_attitude, nis = eskf.update_pose(
                    pose.position[j], pose.quaternion_xyzw[j], np.diag(pose.cov_diag[j]))
                position_residuals.append(r_position)
                attitude_residuals.append(r_attitude)
                nis_values.append(nis)
                update_applied = True
                successful_updates += 1

        # Check the posterior, retaining full-record numerical health metrics.
        if not np.all(np.isfinite(eskf.P)):
            raise RuntimeError(f"P is non-finite at IMU index {k}")
        symmetry = float(np.max(np.abs(eskf.P - eskf.P.T)))
        minimum = float(np.min(np.linalg.eigvalsh(eskf.P)))
        if symmetry > cfg.COVARIANCE_SYMMETRY_TOLERANCE or minimum < -cfg.COVARIANCE_NEGATIVE_TOLERANCE:
            raise RuntimeError(f"P is asymmetric or indefinite at IMU index {k}")
        max_P_symmetry_error = max(max_P_symmetry_error, symmetry)
        minimum_P_eigenvalue = min(minimum_P_eigenvalue, minimum)
        # Convert to scalar values now, so later updates cannot change saved rows.
        values = np.concatenate((eskf.p, eskf.v, eskf.q, quaternion_to_rpy(eskf.q), eskf.bg, eskf.ba))
        rows.append([float(imu.timestamp[k]), k, *values.tolist(), j >= 0, update_applied])

    result = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("formal result contains NaN or Inf")
    timestamps = result[:, 0]
    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("formal result timestamps must strictly increase")
    quaternions = result[:, 8:12]
    propagation_dt = imu.dt[k0 + 1:]
    large_dt_threshold = cfg.TIMESTAMP_LARGE_STEP_RATIO * float(np.median(imu.dt[1:]))
    summary = {
        "phase": "3.5",  # Retained for compatibility with saved summaries.
        "start_index": k0, "end_index": int(imu.timestamp.size - 1),
        "start_timestamp": float(timestamps[0]), "end_timestamp": float(timestamps[-1]),
        "duration": float(timestamps[-1] - timestamps[0]),
        "total_output_frames": len(rows), "prediction_steps": len(rows) - 1,
        "pose_matches": pose_matches, "successful_updates": successful_updates,
        "unmatched_pose_frames": unmatched, "invalid_covariance_skips": invalid_covariance,
        "initialization_observation_reused": False,
        "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0))),
        "max_P_symmetry_error": max_P_symmetry_error, "minimum_P_eigenvalue": minimum_P_eigenvalue,
        "formal_result_all_finite": True,
        "position_initial": result[0, 2:5].tolist(), "position_final": result[-1, 2:5].tolist(),
        "velocity_initial": result[0, 5:8].tolist(), "velocity_final": result[-1, 5:8].tolist(),
        "bg_initial": result[0, 15:18].tolist(), "bg_final": result[-1, 15:18].tolist(),
        "ba_initial": result[0, 18:21].tolist(), "ba_final": result[-1, 18:21].tolist(),
        "position_residual_norm": _statistics(position_residuals),
        "attitude_residual_norm": _statistics(attitude_residuals), "NIS": _statistics(nis_values),
        "max_dt": float(np.max(propagation_dt)), "large_dt_threshold": large_dt_threshold,
        "large_dt_count": int(np.count_nonzero(propagation_dt > large_dt_threshold)),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_directory / "eskf15d_result.csv", rows)
    _write_json_atomic(output_directory / "eskf15d_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    arguments = parser.parse_args()
    summary = run(arguments.processed_dir, arguments.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result_csv: {arguments.output_dir / 'eskf15d_result.csv'}")
    print(f"summary_json: {arguments.output_dir / 'eskf15d_summary.json'}")


if __name__ == "__main__":
    main()
