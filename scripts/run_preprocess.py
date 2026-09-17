"""Run phase-1 preprocessing and write processed data plus diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment import align_timestamps, save_alignment, summarize_alignment
from config import DEFAULT_CONFIG
from preprocess import (
    analyze_initial_static_segment,
    load_imu_csv,
    load_pose_csv,
    save_init_stats,
    save_processed_imu,
    save_processed_pose,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _format_vector(value: np.ndarray) -> str:
    return np.array2string(value, precision=9, separator=", ")


def build_text_report(report: dict[str, Any]) -> str:
    imu = report["imu"]
    static = report["static_segment"]
    pose = report["pose"]
    alignment = report["alignment"]
    lines = [
        "Phase-1 preprocessing diagnostic report",
        "",
        "IMU",
        f"  samples: raw={imu['sample_count_raw']}, processed={imu['sample_count_processed']}",
        f"  time range: {imu['start_timestamp']:.9f} to {imu['end_timestamp']:.9f}",
        f"  duration: {imu['duration_s']:.9f} s",
        f"  dt mean/median/min/max: {imu['mean_dt']:.9f} / {imu['median_dt']:.9f} / {imu['min_dt']:.9f} / {imu['max_dt']:.9f} s",
        f"  estimated frequency: {imu['estimated_frequency_hz']:.6f} Hz",
        f"  invalid rows dropped: {imu['invalid_sample_count']} (CSV rows {imu['invalid_source_rows']})",
        f"  small/large timestamp steps: {imu['small_step_count']} / {imu['large_step_count']}",
        f"  large-step destination indices: {imu['large_step_destination_indices']}",
        f"  large-step dt values: {imu['large_step_dt_s']} s",
        "",
        "Static segment",
        f"  candidate index range [start, end): [{static['candidate_start_idx']}, {static['candidate_end_idx']})",
        f"  selected index range [start, end): [{static['static_start_idx']}, {static['static_end_idx']})",
        f"  selected time range: {static['static_start_time']:.9f} to {static['static_end_time']:.9f}",
        f"  duration: {static['static_end_time'] - static['static_start_time']:.9f} s",
        f"  sample count: {static['static_end_idx'] - static['static_start_idx']}",
        f"  gyro mean rad/s: {_format_vector(np.asarray(static['gyro_mean']))}",
        f"  gyro std rad/s: {_format_vector(np.asarray(static['gyro_std']))}",
        f"  acc mean m/s^2: {_format_vector(np.asarray(static['acc_mean']))}",
        f"  acc std m/s^2: {_format_vector(np.asarray(static['acc_std']))}",
        f"  |acc| mean/std: {static['acc_norm_mean']:.9f} / {static['acc_norm_std']:.9f} m/s^2",
        f"  accepted as static: {static['is_static']}",
        "  reasons: " + "; ".join(static["decision_reasons"]),
        "",
        "FAST-LIO pose",
        f"  samples: raw={pose['sample_count_raw']}, processed={pose['sample_count_processed']}",
        f"  time range: {pose['start_timestamp']:.9f} to {pose['end_timestamp']:.9f}",
        f"  duration: {pose['duration_s']:.9f} s",
        f"  median dt / frequency: {pose['median_dt']:.9f} s / {pose['estimated_frequency_hz']:.6f} Hz",
        f"  raw quaternion norm mean/min/max: {pose['quaternion_norm_mean_raw']:.12f} / {pose['quaternion_norm_min_raw']:.12f} / {pose['quaternion_norm_max_raw']:.12f}",
        f"  invalid quaternion frames: {pose['invalid_quaternion_count']}",
        f"  invalid covariance frames: {pose['invalid_covariance_count']}",
        f"  all-zero covariance frames: {pose['zero_covariance_frame_count']}",
        f"  covariance diagonal min: {_format_vector(np.asarray(pose['covariance_diag_min']))}",
        f"  covariance diagonal mean: {_format_vector(np.asarray(pose['covariance_diag_mean']))}",
        f"  covariance diagonal median: {_format_vector(np.asarray(pose['covariance_diag_median']))}",
        f"  covariance diagonal max: {_format_vector(np.asarray(pose['covariance_diag_max']))}",
        "",
        "Timestamp alignment",
        f"  tolerance: {alignment['tolerance_s']:.9f} s",
        f"  valid matches: {alignment['valid_match_count']}",
        f"  unmatched IMU: {alignment['unmatched_imu_count']}",
        f"  unused pose: {alignment['unused_pose_count']}",
        f"  match ratio: {alignment['match_ratio']:.9%}",
        f"  absolute error mean/median/max: {alignment['mean_abs_time_error_s']:.9f} / {alignment['median_abs_time_error_s']:.9f} / {alignment['max_abs_time_error_s']:.9f} s",
        f"  duplicate pose use: {alignment['duplicate_pose_use_count']}",
        f"  unmatched IMU indices: {alignment['unmatched_imu_indices']}",
        f"  unused pose indices: {alignment['unused_pose_indices']}",
    ]
    return "\n".join(lines) + "\n"


def run(raw_directory: Path, output_directory: Path) -> dict[str, Any]:
    imu, imu_diagnostics = load_imu_csv(raw_directory / "imu.csv")
    pose, pose_diagnostics = load_pose_csv(raw_directory / "pose_cov.csv")
    init_stats = analyze_initial_static_segment(imu)
    alignment = align_timestamps(imu.timestamp, pose.timestamp)
    alignment_diagnostics = summarize_alignment(alignment, pose.timestamp.size)

    output_directory.mkdir(parents=True, exist_ok=True)
    save_processed_imu(output_directory / "imu_processed.npz", imu)
    save_processed_pose(output_directory / "pose_processed.npz", pose)
    save_init_stats(output_directory / "init_stats.npz", init_stats)
    save_alignment(output_directory / "match_table.npz", alignment)

    report = {
        "conventions": {
            "quaternion_order": "xyzw",
            "rotation_matrix": "R_WB maps body vectors to world vectors",
            "error": "right multiplicative",
            "gravity_world_mps2": [0.0, 0.0, -9.81],
            "dt_first_sample": "NaN",
            "time_error": "pose_timestamp[j] - imu_timestamp[k]",
        },
        "imu": asdict(imu_diagnostics),
        "static_segment": asdict(init_stats),
        "pose": asdict(pose_diagnostics),
        "alignment": asdict(alignment_diagnostics),
    }
    with (output_directory / "diagnostic_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, default=_json_value, ensure_ascii=False, indent=2)
        stream.write("\n")
    text_report = build_text_report(report)
    (output_directory / "diagnostic_report.txt").write_text(text_report, encoding="utf-8")
    print(text_report, end="")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    arguments = parser.parse_args()
    run(arguments.raw_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
