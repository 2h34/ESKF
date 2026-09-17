"""Prepare submission-ready RPY data and figures from frozen 6D ESKF output."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from processed_io import load_alignment, load_processed_pose
from rotation_utils import quaternion_to_rpy


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

RPY_FIELDNAMES = (
    "timestamp",
    "time_s",
    "imu_index",
    "roll_rad",
    "pitch_rad",
    "yaw_raw_rad",
    "yaw_unwrapped_rad",
    "roll_deg",
    "pitch_deg",
    "yaw_raw_deg",
    "yaw_unwrapped_deg",
)

ESKF_COLOR = "#1F4E79"
FAST_LIO_COLOR = "#D97706"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#1F2937"
FIGURE_DPI = 200


@dataclass(frozen=True)
class RPYResultData:
    timestamp: FloatArray
    time_s: FloatArray
    imu_index: IntArray
    roll_rad: FloatArray
    pitch_rad: FloatArray
    yaw_raw_rad: FloatArray
    yaw_unwrapped_rad: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    yaw_raw_deg: FloatArray
    yaw_unwrapped_deg: FloatArray


@dataclass(frozen=True)
class PlotResultPaths:
    rpy_csv: Path
    roll_figure: Path
    pitch_figure: Path
    yaw_figure: Path
    overview_figure: Path
    fastlio_comparison_figure: Path
    experiment_summary_json: Path


def _one_dimensional_finite(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def prepare_rpy_data(
    timestamp: ArrayLike,
    imu_index: ArrayLike,
    roll_rad: ArrayLike,
    pitch_rad: ArrayLike,
    yaw_rad: ArrayLike,
) -> RPYResultData:
    """Validate formal RPY samples and add relative time/visualization columns."""

    time = _one_dimensional_finite(timestamp, "timestamp")
    roll = _one_dimensional_finite(roll_rad, "roll_rad")
    pitch = _one_dimensional_finite(pitch_rad, "pitch_rad")
    yaw = _one_dimensional_finite(yaw_rad, "yaw_rad")
    indices_raw = np.asarray(imu_index)
    if indices_raw.ndim != 1 or not np.issubdtype(indices_raw.dtype, np.integer):
        raise ValueError("imu_index must be a one-dimensional integer array")
    indices = indices_raw.astype(np.int64, copy=False)
    sample_count = time.size
    if any(value.size != sample_count for value in (indices, roll, pitch, yaw)):
        raise ValueError("timestamp, imu_index, roll, pitch, and yaw lengths must match")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamp must be strictly increasing")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("imu_index must be strictly increasing")

    relative_time = time - time[0]
    if relative_time[0] != 0.0:
        raise RuntimeError("the first relative timestamp must be exactly zero")
    yaw_unwrapped = np.unwrap(yaw)
    return RPYResultData(
        timestamp=time,
        time_s=relative_time,
        imu_index=indices,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_raw_rad=yaw,
        yaw_unwrapped_rad=yaw_unwrapped,
        roll_deg=np.rad2deg(roll),
        pitch_deg=np.rad2deg(pitch),
        yaw_raw_deg=np.rad2deg(yaw),
        yaw_unwrapped_deg=np.rad2deg(yaw_unwrapped),
    )


def load_formal_result(path: Path) -> RPYResultData:
    """Read the frozen formal result without modifying or resampling it."""

    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    required = {
        "timestamp",
        "imu_index",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    try:
        timestamp = np.asarray([float(row["timestamp"]) for row in rows], dtype=float)
        imu_index = np.asarray([int(row["imu_index"]) for row in rows], dtype=np.int64)
        roll = np.asarray([float(row["roll_rad"]) for row in rows], dtype=float)
        pitch = np.asarray([float(row["pitch_rad"]) for row in rows], dtype=float)
        yaw = np.asarray([float(row["yaw_rad"]) for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains an invalid formal RPY row") from exc
    return prepare_rpy_data(timestamp, imu_index, roll, pitch, yaw)


def _rpy_rows(data: RPYResultData) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for position in range(data.timestamp.size):
        rows.append(
            {
                "timestamp": float(data.timestamp[position]),
                "time_s": float(data.time_s[position]),
                "imu_index": int(data.imu_index[position]),
                "roll_rad": float(data.roll_rad[position]),
                "pitch_rad": float(data.pitch_rad[position]),
                "yaw_raw_rad": float(data.yaw_raw_rad[position]),
                "yaw_unwrapped_rad": float(data.yaw_unwrapped_rad[position]),
                "roll_deg": float(data.roll_deg[position]),
                "pitch_deg": float(data.pitch_deg[position]),
                "yaw_raw_deg": float(data.yaw_raw_deg[position]),
                "yaw_unwrapped_deg": float(data.yaw_unwrapped_deg[position]),
            }
        )
    return rows


def save_rpy_csv(path: Path, data: RPYResultData) -> None:
    """Write one row per formal result sample, preserving all raw RPY radians."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RPY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(_rpy_rows(data))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"figure was not generated correctly: {path}")


def _plot_single(
    time_s: FloatArray,
    angle_deg: FloatArray,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 4.4))
    axis.plot(time_s, angle_deg, color=ESKF_COLOR, linewidth=1.15)
    axis.set_title(title, color=TEXT_COLOR, fontsize=13, pad=10)
    axis.set_xlabel("Time [s]")
    _style_axis(axis, "Angle [deg]")
    _save_figure(figure, path)


def generate_required_figures(data: RPYResultData, figure_directory: Path) -> dict[str, Path]:
    """Generate the three required plots plus the RPY overview."""

    roll_path = figure_directory / "roll_time.png"
    pitch_path = figure_directory / "pitch_time.png"
    yaw_path = figure_directory / "yaw_time.png"
    overview_path = figure_directory / "rpy_overview.png"
    _plot_single(data.time_s, data.roll_deg, "Roll - Time", roll_path)
    _plot_single(data.time_s, data.pitch_deg, "Pitch - Time", pitch_path)
    _plot_single(
        data.time_s,
        data.yaw_unwrapped_deg,
        "Yaw - Time (unwrapped for visualization)",
        yaw_path,
    )

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.4), sharex=True)
    series = (
        (data.roll_deg, "Roll [deg]", "Roll"),
        (data.pitch_deg, "Pitch [deg]", "Pitch"),
        (data.yaw_unwrapped_deg, "Yaw [deg]", "Yaw (unwrapped for visualization)"),
    )
    for axis, (values, ylabel, title) in zip(axes, series):
        axis.plot(data.time_s, values, color=ESKF_COLOR, linewidth=1.05)
        axis.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR)
        _style_axis(axis, ylabel)
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("6D ESKF Attitude Result", fontsize=14, color=TEXT_COLOR, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        overview_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    if not overview_path.is_file() or overview_path.stat().st_size == 0:
        raise RuntimeError(f"figure was not generated correctly: {overview_path}")
    return {
        "roll": roll_path,
        "pitch": pitch_path,
        "yaw": yaw_path,
        "overview": overview_path,
    }


def unwrap_with_nan_gaps(angle_rad: ArrayLike) -> FloatArray:
    """Unwrap each contiguous finite run without filling or bridging NaN gaps."""

    angle = np.asarray(angle_rad, dtype=float)
    if angle.ndim != 1:
        raise ValueError("angle_rad must be one-dimensional")
    output = np.full(angle.shape, np.nan, dtype=float)
    finite = np.isfinite(angle)
    if not np.any(finite):
        return output
    starts = np.flatnonzero(finite & np.concatenate(([True], ~finite[:-1])))
    ends = np.flatnonzero(finite & np.concatenate((~finite[1:], [True]))) + 1
    for start, end in zip(starts, ends):
        output[start:end] = np.unwrap(angle[start:end])
    return output


def fastlio_rpy_for_result(
    data: RPYResultData,
    pose_path: Path,
    alignment_path: Path,
) -> tuple[FloatArray, IntArray]:
    """Map FAST-LIO observations through the saved match table only."""

    pose = load_processed_pose(pose_path)
    maximum_index = int(np.max(data.imu_index))
    alignment = load_alignment(
        alignment_path,
        imu_sample_count=maximum_index + 1,
        pose_sample_count=pose.timestamp.size,
    )
    rpy = np.full((data.imu_index.size, 3), np.nan, dtype=float)
    pose_indices = np.full(data.imu_index.size, -1, dtype=np.int64)
    for position, imu_index in enumerate(data.imu_index):
        pose_index = int(alignment.pose_index_for_imu[int(imu_index)])
        pose_indices[position] = pose_index
        if pose_index >= 0:
            rpy[position] = quaternion_to_rpy(pose.quaternion_xyzw[pose_index])
    return rpy, pose_indices


def generate_fastlio_comparison_figure(
    data: RPYResultData,
    fastlio_rpy_rad: ArrayLike,
    path: Path,
) -> None:
    """Plot ESKF and matched FAST-LIO observations without interpolation."""

    observation = np.asarray(fastlio_rpy_rad, dtype=float)
    if observation.shape != (data.timestamp.size, 3):
        raise ValueError("fastlio_rpy_rad must have shape (sample_count, 3)")
    observation_deg = np.rad2deg(observation)
    observation_yaw_deg = np.rad2deg(unwrap_with_nan_gaps(observation[:, 2]))
    estimated = (data.roll_deg, data.pitch_deg, data.yaw_unwrapped_deg)
    observed = (observation_deg[:, 0], observation_deg[:, 1], observation_yaw_deg)
    labels = ("Roll [deg]", "Pitch [deg]", "Yaw [deg]")

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.7), sharex=True)
    for axis, estimate, reference, ylabel in zip(axes, estimated, observed, labels):
        axis.plot(
            data.time_s,
            estimate,
            color=ESKF_COLOR,
            linewidth=1.15,
            label="ESKF Estimate",
        )
        axis.plot(
            data.time_s,
            reference,
            color=FAST_LIO_COLOR,
            linewidth=0.9,
            alpha=0.82,
            label="FAST-LIO Observation",
        )
        _style_axis(axis, ylabel)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle(
        "ESKF Estimate and FAST-LIO Observation Reference",
        fontsize=14,
        color=TEXT_COLOR,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Yaw is unwrapped for visualization only. FAST-LIO is not independent ground truth.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.975))
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"figure was not generated correctly: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_experiment_summary(
    data: RPYResultData,
    eskf_summary: Mapping[str, Any],
    nis_summary: Mapping[str, Any],
    dynamic_summary: Mapping[str, Any],
    pose_indices: IntArray,
) -> dict[str, Any]:
    """Consolidate existing run/diagnostic facts without inventing new tuning claims."""

    expected_frames = int(eskf_summary["total_output_frames"])
    if data.timestamp.size != expected_frames:
        raise ValueError("formal result row count disagrees with eskf6d_summary.json")
    duration = float(data.time_s[-1])
    if not np.isclose(duration, float(eskf_summary["duration"]), rtol=0.0, atol=1e-9):
        raise ValueError("formal result duration disagrees with eskf6d_summary.json")
    if int(nis_summary["update_count"]) != int(eskf_summary["successful_updates"]):
        raise ValueError("NIS update count disagrees with the formal run summary")
    if dynamic_summary.get("phase") != "2.4B-2":
        raise ValueError("dynamic diagnostic summary is not Phase 2.4B-2")

    return {
        "phase": "2.5",
        "purpose": "submission-ready RPY data and experiment-result summary",
        "duration_s": duration,
        "output_frame_count": expected_frames,
        "prediction_step_count": int(eskf_summary["prediction_steps"]),
        "successful_update_count": int(eskf_summary["successful_updates"]),
        "max_quaternion_norm_error": float(
            eskf_summary["max_quaternion_norm_error"]
        ),
        "minimum_P_eigenvalue": float(eskf_summary["minimum_P_eigenvalue"]),
        "max_P_symmetry_error": float(eskf_summary["max_P_symmetry_error"]),
        "all_formal_results_finite": bool(
            eskf_summary["formal_result_all_finite"]
            and np.all(np.isfinite(data.roll_rad))
            and np.all(np.isfinite(data.pitch_rad))
            and np.all(np.isfinite(data.yaw_raw_rad))
        ),
        "initial_bg": [float(value) for value in eskf_summary["bias_initial"]],
        "final_bg": [float(value) for value in eskf_summary["bias_final"]],
        "residual_norm": {
            key: float(eskf_summary["residual_norm"][key])
            for key in ("median", "p95", "max")
        },
        "NIS": {
            key: float(eskf_summary["NIS"][key])
            for key in ("median", "p95", "max")
        },
        "RPY_range": {
            "roll_deg": [float(np.min(data.roll_deg)), float(np.max(data.roll_deg))],
            "pitch_deg": [float(np.min(data.pitch_deg)), float(np.max(data.pitch_deg))],
            "yaw_raw_deg": [
                float(np.min(data.yaw_raw_deg)),
                float(np.max(data.yaw_raw_deg)),
            ],
            "yaw_unwrapped_deg": [
                float(np.min(data.yaw_unwrapped_deg)),
                float(np.max(data.yaw_unwrapped_deg)),
            ],
        },
        "fastlio_observation_reference": {
            "matched_frame_count": int(np.count_nonzero(pose_indices >= 0)),
            "unmatched_frame_count": int(np.count_nonzero(pose_indices < 0)),
            "independent_ground_truth": False,
            "comparison_purpose": "trend and observation-consistency visualization only",
        },
        "dynamic_diagnostic": {
            "high_NIS_mainly_during_dynamic_rotation": True,
            "local_yz_residual_more_evident": True,
            "left_endpoint_ZOH_confirmed_partial_contribution": True,
            "offline_trapezoid_improves_most_typical_intervals": True,
            "extreme_high_NIS_tail_remains": True,
            "fixed_latency_hypothesis_weakened": True,
            "unique_physical_root_cause_identified": False,
            "formal_filter_unchanged": True,
        },
        "yaw_unwrap_scope": "visualization and submission CSV only; not filter state",
        "smoothing_applied": False,
        "resampling_applied": False,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_submission_results(
    formal_result_csv: Path,
    pose_path: Path,
    alignment_path: Path,
    eskf_summary_path: Path,
    nis_summary_path: Path,
    dynamic_summary_path: Path,
    output_directory: Path,
) -> tuple[PlotResultPaths, dict[str, Any]]:
    """Create all Phase 2.5 outputs from frozen formal and processed data."""

    data = load_formal_result(formal_result_csv)
    rpy_path = output_directory / "rpy_result.csv"
    figure_directory = output_directory / "figures"
    save_rpy_csv(rpy_path, data)
    required = generate_required_figures(data, figure_directory)
    fastlio_rpy, pose_indices = fastlio_rpy_for_result(
        data, pose_path, alignment_path
    )
    comparison_path = figure_directory / "rpy_fastlio_comparison.png"
    generate_fastlio_comparison_figure(data, fastlio_rpy, comparison_path)

    summary = build_experiment_summary(
        data,
        _read_json(eskf_summary_path),
        _read_json(nis_summary_path),
        _read_json(dynamic_summary_path),
        pose_indices,
    )
    summary_path = output_directory / "experiment_result_summary.json"
    _write_json_atomic(summary_path, summary)
    paths = PlotResultPaths(
        rpy_csv=rpy_path,
        roll_figure=required["roll"],
        pitch_figure=required["pitch"],
        yaw_figure=required["yaw"],
        overview_figure=required["overview"],
        fastlio_comparison_figure=comparison_path,
        experiment_summary_json=summary_path,
    )
    return paths, summary
