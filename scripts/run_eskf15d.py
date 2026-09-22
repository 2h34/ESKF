"""Run the 15D ESKF and save posterior states plus a short summary."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eskf15d import initialize_eskf15d
from preprocess import (load_alignment, load_init_stats, load_processed_imu,
                        load_processed_pose)
from rotation_utils import quaternion_to_rpy

FIELDS = (
    "timestamp", "imu_index", "px_m", "py_m", "pz_m", "vx_mps", "vy_mps", "vz_mps",
    "qx", "qy", "qz", "qw", "roll_rad", "pitch_rad", "yaw_rad",
    "bg_x_rad_s", "bg_y_rad_s", "bg_z_rad_s", "ba_x_mps2", "ba_y_mps2", "ba_z_mps2",
    "has_pose_match", "update_applied",
)


def run(processed_directory: Path, output_directory: Path) -> dict:
    imu = load_processed_imu(processed_directory / "imu_processed.npz")
    pose = load_processed_pose(processed_directory / "pose_processed.npz")
    stats = load_init_stats(processed_directory / "init_stats.npz")
    alignment = load_alignment(processed_directory / "match_table.npz")
    eskf, k0 = initialize_eskf15d(imu, pose, stats, alignment)
    rows = []
    successful_updates = unmatched = invalid_covariance = 0

    for k in range(k0, imu.timestamp.size):
        j = int(alignment.pose_index_for_imu[k])
        updated = False
        if k > k0:
            eskf.predict(imu.gyro_rad_s[k - 1], imu.acc_mps2[k - 1], float(imu.dt[k]))
            if j < 0:
                unmatched += 1
            elif np.any(pose.cov_diag[j] <= 0.0):
                invalid_covariance += 1
            else:
                eskf.update_pose(
                    pose.position[j], pose.quaternion_xyzw[j], np.diag(pose.cov_diag[j]))
                successful_updates += 1
                updated = True
        values = np.r_[eskf.p, eskf.v, eskf.q, quaternion_to_rpy(eskf.q), eskf.bg, eskf.ba]
        rows.append([float(imu.timestamp[k]), k, *values.tolist(), j >= 0, updated])

    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "eskf15d_result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)
    result = np.asarray(rows, dtype=float)
    dt = imu.dt[k0 + 1:]
    summary = {
        "phase": "3.5", "start_index": k0, "end_index": int(imu.timestamp.size - 1),
        "start_timestamp": float(result[0, 0]), "end_timestamp": float(result[-1, 0]),
        "duration": float(result[-1, 0] - result[0, 0]),
        "total_output_frames": len(rows), "prediction_steps": len(rows) - 1,
        "pose_matches": int(np.count_nonzero(alignment.pose_index_for_imu[k0:] >= 0)),
        "successful_updates": successful_updates, "unmatched_pose_frames": unmatched,
        "invalid_covariance_skips": invalid_covariance, "initialization_observation_reused": False,
        "max_quaternion_norm_error": float(np.max(abs(np.linalg.norm(result[:, 8:12], axis=1) - 1))),
        "formal_result_all_finite": bool(np.all(np.isfinite(result))),
        "position_initial": result[0, 2:5].tolist(), "position_final": result[-1, 2:5].tolist(),
        "velocity_initial": result[0, 5:8].tolist(), "velocity_final": result[-1, 5:8].tolist(),
        "bg_initial": result[0, 15:18].tolist(), "bg_final": result[-1, 15:18].tolist(),
        "ba_initial": result[0, 18:21].tolist(), "ba_final": result[-1, 18:21].tolist(),
        "max_dt": float(np.max(dt)),
    }
    (output_directory / "eskf15d_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(run(args.processed_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
