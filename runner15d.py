"""Scheduler, diagnostics, and file output for the 15D ESKF.

This module owns IMU indices, authoritative IMU timestamps, and precomputed
pose matches. The :class:`ESKF15D` object remains responsible only for the
mathematical prediction and pose update.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config import DEFAULT_CONFIG, ProjectConfig
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from eskf15d import ESKF15D, ESKF15DState, PredictionDebug, UpdateDebug
from initialization import ESKF15DInitialization
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

DEBUG_FIELDNAMES = (
    "imu_index",
    "timestamp",
    "dt",
    "pose_index",
    "has_pose_match",
    "update_applied",
    "skip_reason",
    "omega_hat_x",
    "omega_hat_y",
    "omega_hat_z",
    "a_hat_W_x",
    "a_hat_W_y",
    "a_hat_W_z",
    "q_norm",
    "P_min_eigenvalue",
    "P_symmetry_error",
    "r_position_x",
    "r_position_y",
    "r_position_z",
    "r_theta_x",
    "r_theta_y",
    "r_theta_z",
    "position_residual_norm",
    "attitude_residual_norm",
    "NIS",
    "delta_p_x",
    "delta_p_y",
    "delta_p_z",
    "delta_v_x",
    "delta_v_y",
    "delta_v_z",
    "delta_theta_x",
    "delta_theta_y",
    "delta_theta_z",
    "delta_bg_x",
    "delta_bg_y",
    "delta_bg_z",
    "delta_ba_x",
    "delta_ba_y",
    "delta_ba_z",
)


@dataclass(frozen=True)
class ESKF15DRunResult:
    """In-memory formal rows, development diagnostics, and run statistics."""

    result_rows: tuple[dict[str, Any], ...]
    debug_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ESKF15DOutputPaths:
    """Paths written by :func:`save_eskf15d_run`."""

    result_csv: Path
    debug_csv: Path
    summary_json: Path


def _validate_runner_inputs(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    initialization: ESKF15DInitialization,
    eskf: ESKF15D,
    config: ProjectConfig,
) -> int:
    sample_count = int(np.asarray(imu.timestamp).size)
    if sample_count < 2:
        raise ValueError("runner requires at least two IMU samples")
    if np.asarray(imu.timestamp).shape != (sample_count,):
        raise ValueError("imu.timestamp must have shape (N,)")
    if np.asarray(imu.dt).shape != (sample_count,):
        raise ValueError("imu.dt must have shape (N,)")
    if np.asarray(imu.gyro_rad_s).shape != (sample_count, 3):
        raise ValueError("imu.gyro_rad_s must have shape (N, 3)")
    if np.asarray(imu.acc_mps2).shape != (sample_count, 3):
        raise ValueError("imu.acc_mps2 must have shape (N, 3)")

    pose_index_for_imu = np.asarray(alignment.pose_index_for_imu)
    if pose_index_for_imu.shape != (sample_count,):
        raise ValueError("alignment must have one pose index per IMU sample")

    pose_count = int(np.asarray(pose.timestamp).size)
    if np.asarray(pose.position).shape != (pose_count, 3):
        raise ValueError("pose.position must have shape (M, 3)")
    if np.asarray(pose.quaternion_xyzw).shape != (pose_count, 4):
        raise ValueError("pose.quaternion_xyzw must have shape (M, 4)")
    if np.asarray(pose.cov_diag).shape != (pose_count, 6):
        raise ValueError("pose.cov_diag must have shape (M, 6)")
    if np.any(pose_index_for_imu < -1) or np.any(
        pose_index_for_imu >= pose_count
    ):
        raise ValueError("alignment contains an out-of-range pose index")

    k0 = int(initialization.imu_index)
    expected_k0 = int(init_stats.static_end_idx) - 1
    if k0 != expected_k0:
        raise ValueError(
            f"initialization IMU index {k0} does not equal static_end_idx - 1 "
            f"({expected_k0})"
        )
    if not (0 <= k0 < sample_count - 1):
        raise ValueError("runner requires an IMU sample after initialization")
    if float(initialization.timestamp) != float(imu.timestamp[k0]):
        raise ValueError("initialization timestamp must come from imu.timestamp[k0]")
    if int(pose_index_for_imu[k0]) != int(initialization.pose_index):
        raise ValueError("initialization pose index must come from the match table")

    state = eskf.get_state()
    tolerance = config.numerical.covariance_symmetry_tolerance

    def require_match(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
        if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
            raise ValueError(f"ESKF {label} does not match the supplied initialization")

    require_match(state.p_W_m, initialization.p0_W_m, "position")
    require_match(state.v_W_mps, initialization.v0_W_mps, "velocity")
    require_match(state.q_xyzw, initialization.q0_xyzw, "quaternion")
    require_match(state.bg_rad_s, initialization.bg0_rad_s, "gyro bias")
    require_match(state.ba_mps2, initialization.ba0_mps2, "accelerometer bias")
    require_match(state.P, initialization.P0, "covariance")
    return k0


def _covariance_metrics(
    state: ESKF15DState,
    config: ProjectConfig,
    imu_index: int,
) -> tuple[float, float]:
    covariance = np.asarray(state.P, dtype=float)
    if covariance.shape != (15, 15) or not np.all(np.isfinite(covariance)):
        raise RuntimeError(f"P is invalid at IMU index {imu_index}")
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if symmetry_error > config.numerical.covariance_symmetry_tolerance:
        raise RuntimeError(
            f"P symmetry error {symmetry_error:.3e} exceeds tolerance at "
            f"IMU index {imu_index}"
        )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue < -config.numerical.covariance_negative_tolerance:
        raise RuntimeError(
            f"P minimum eigenvalue {minimum_eigenvalue:.3e} is below tolerance "
            f"at IMU index {imu_index}"
        )
    return minimum_eigenvalue, symmetry_error


def _result_row(
    *,
    imu_index: int,
    timestamp: float,
    state: ESKF15DState,
    has_pose_match: bool,
    update_applied: bool,
) -> dict[str, Any]:
    rpy = quaternion_to_rpy(state.q_xyzw)
    return {
        "timestamp": timestamp,
        "imu_index": imu_index,
        "px_m": float(state.p_W_m[0]),
        "py_m": float(state.p_W_m[1]),
        "pz_m": float(state.p_W_m[2]),
        "vx_mps": float(state.v_W_mps[0]),
        "vy_mps": float(state.v_W_mps[1]),
        "vz_mps": float(state.v_W_mps[2]),
        "qx": float(state.q_xyzw[0]),
        "qy": float(state.q_xyzw[1]),
        "qz": float(state.q_xyzw[2]),
        "qw": float(state.q_xyzw[3]),
        "roll_rad": float(rpy[0]),
        "pitch_rad": float(rpy[1]),
        "yaw_rad": float(rpy[2]),
        "bg_x_rad_s": float(state.bg_rad_s[0]),
        "bg_y_rad_s": float(state.bg_rad_s[1]),
        "bg_z_rad_s": float(state.bg_rad_s[2]),
        "ba_x_mps2": float(state.ba_mps2[0]),
        "ba_y_mps2": float(state.ba_mps2[1]),
        "ba_z_mps2": float(state.ba_mps2[2]),
        "has_pose_match": has_pose_match,
        "update_applied": update_applied,
    }


def _debug_row(
    *,
    imu_index: int,
    timestamp: float,
    dt_s: float,
    pose_index: int,
    has_pose_match: bool,
    update_applied: bool,
    skip_reason: str,
    state: ESKF15DState,
    minimum_eigenvalue: float,
    symmetry_error: float,
    prediction: PredictionDebug | None,
    update: UpdateDebug | None,
) -> dict[str, Any]:
    missing = float("nan")
    omega_hat = prediction.omega_hat_rad_s if prediction is not None else None
    a_hat_W = prediction.a_hat_W_mps2 if prediction is not None else None

    if update is None:
        r_position = r_theta = None
        delta_p = delta_v = delta_theta = delta_bg = delta_ba = None
        position_residual_norm = attitude_residual_norm = NIS = missing
    else:
        r_position = update.r_position_m
        r_theta = update.r_theta_rad
        delta_p = update.delta_p_m
        delta_v = update.delta_v_mps
        delta_theta = update.delta_theta_rad
        delta_bg = update.delta_bg_rad_s
        delta_ba = update.delta_ba_mps2
        position_residual_norm = float(np.linalg.norm(r_position))
        attitude_residual_norm = float(np.linalg.norm(r_theta))
        NIS = update.NIS

    def component(vector: np.ndarray | None, index: int) -> float:
        return missing if vector is None else float(vector[index])

    return {
        "imu_index": imu_index,
        "timestamp": timestamp,
        "dt": dt_s,
        "pose_index": pose_index,
        "has_pose_match": has_pose_match,
        "update_applied": update_applied,
        "skip_reason": skip_reason,
        "omega_hat_x": component(omega_hat, 0),
        "omega_hat_y": component(omega_hat, 1),
        "omega_hat_z": component(omega_hat, 2),
        "a_hat_W_x": component(a_hat_W, 0),
        "a_hat_W_y": component(a_hat_W, 1),
        "a_hat_W_z": component(a_hat_W, 2),
        "q_norm": float(np.linalg.norm(state.q_xyzw)),
        "P_min_eigenvalue": minimum_eigenvalue,
        "P_symmetry_error": symmetry_error,
        "r_position_x": component(r_position, 0),
        "r_position_y": component(r_position, 1),
        "r_position_z": component(r_position, 2),
        "r_theta_x": component(r_theta, 0),
        "r_theta_y": component(r_theta, 1),
        "r_theta_z": component(r_theta, 2),
        "position_residual_norm": position_residual_norm,
        "attitude_residual_norm": attitude_residual_norm,
        "NIS": NIS,
        "delta_p_x": component(delta_p, 0),
        "delta_p_y": component(delta_p, 1),
        "delta_p_z": component(delta_p, 2),
        "delta_v_x": component(delta_v, 0),
        "delta_v_y": component(delta_v, 1),
        "delta_v_z": component(delta_v, 2),
        "delta_theta_x": component(delta_theta, 0),
        "delta_theta_y": component(delta_theta, 1),
        "delta_theta_z": component(delta_theta, 2),
        "delta_bg_x": component(delta_bg, 0),
        "delta_bg_y": component(delta_bg, 1),
        "delta_bg_z": component(delta_bg, 2),
        "delta_ba_x": component(delta_ba, 0),
        "delta_ba_y": component(delta_ba, 1),
        "delta_ba_z": component(delta_ba, 2),
    }


def _validate_prediction_update_boundary(
    prediction: PredictionDebug,
    update: UpdateDebug,
    config: ProjectConfig,
    imu_index: int,
) -> None:
    prediction_diagonal = np.asarray(prediction.P_diag_after, dtype=float)
    update_diagonal = np.asarray(update.P_diag_before, dtype=float)
    tolerance = config.numerical.covariance_symmetry_tolerance
    if not np.allclose(
        prediction_diagonal,
        update_diagonal,
        rtol=1e-12,
        atol=tolerance,
    ):
        maximum_difference = float(
            np.max(np.abs(prediction_diagonal - update_diagonal))
        )
        raise RuntimeError(
            "prediction/update P^- diagonal mismatch at IMU index "
            f"{imu_index}: max difference {maximum_difference:.3e}"
        )


def _statistics(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    if not np.all(np.isfinite(array)):
        raise RuntimeError("summary statistic input contains NaN or Inf")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def run_eskf15d(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    initialization: ESKF15DInitialization,
    eskf: ESKF15D,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> ESKF15DRunResult:
    """Run the initialization posterior through the final supplied IMU sample."""

    k0 = _validate_runner_inputs(
        imu, pose, init_stats, alignment, initialization, eskf, config
    )
    sample_count = int(imu.timestamp.size)
    result_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    position_residual_norms: list[float] = []
    attitude_residual_norms: list[float] = []
    nis_values: list[float] = []

    initial_pose_index = int(alignment.pose_index_for_imu[k0])
    initial_state = eskf.get_state()
    initial_min_eigenvalue, initial_symmetry_error = _covariance_metrics(
        initial_state, config, k0
    )
    result_rows.append(
        _result_row(
            imu_index=k0,
            timestamp=float(imu.timestamp[k0]),
            state=initial_state,
            has_pose_match=initial_pose_index >= 0,
            update_applied=False,
        )
    )
    debug_rows.append(
        _debug_row(
            imu_index=k0,
            timestamp=float(imu.timestamp[k0]),
            dt_s=float("nan"),
            pose_index=initial_pose_index,
            has_pose_match=initial_pose_index >= 0,
            update_applied=False,
            skip_reason="initialization_observation_already_used",
            state=initial_state,
            minimum_eigenvalue=initial_min_eigenvalue,
            symmetry_error=initial_symmetry_error,
            prediction=None,
            update=None,
        )
    )

    for k in range(k0, sample_count - 1):
        current_k = k + 1
        prediction = eskf.predict(
            imu.gyro_rad_s[k],
            imu.acc_mps2[k],
            float(imu.dt[current_k]),
        )

        pose_index = int(alignment.pose_index_for_imu[current_k])
        has_pose_match = pose_index >= 0
        update: UpdateDebug | None = None

        if not has_pose_match:
            skip_reason = "unmatched_pose"
        else:
            covariance_diagonal = np.asarray(
                pose.cov_diag[pose_index, 0:6], dtype=float
            )
            if not np.all(np.isfinite(covariance_diagonal)):
                raise RuntimeError(
                    f"pose covariance is non-finite at pose index {pose_index}"
                )
            if np.any(covariance_diagonal <= 0.0):
                skip_reason = "nonpositive_pose_covariance"
            else:
                update = eskf.update_pose(
                    pose.position[pose_index],
                    pose.quaternion_xyzw[pose_index],
                    np.diag(covariance_diagonal),
                )
                _validate_prediction_update_boundary(
                    prediction, update, config, current_k
                )
                skip_reason = ""
                position_residual_norms.append(
                    float(np.linalg.norm(update.r_position_m))
                )
                attitude_residual_norms.append(
                    float(np.linalg.norm(update.r_theta_rad))
                )
                nis_values.append(update.NIS)

        state = eskf.get_state()
        minimum_eigenvalue, symmetry_error = _covariance_metrics(
            state, config, current_k
        )
        timestamp = float(imu.timestamp[current_k])
        update_applied = update is not None
        result_rows.append(
            _result_row(
                imu_index=current_k,
                timestamp=timestamp,
                state=state,
                has_pose_match=has_pose_match,
                update_applied=update_applied,
            )
        )
        debug_rows.append(
            _debug_row(
                imu_index=current_k,
                timestamp=timestamp,
                dt_s=float(imu.dt[current_k]),
                pose_index=pose_index,
                has_pose_match=has_pose_match,
                update_applied=update_applied,
                skip_reason=skip_reason,
                state=state,
                minimum_eigenvalue=minimum_eigenvalue,
                symmetry_error=symmetry_error,
                prediction=prediction,
                update=update,
            )
        )

    timestamps = np.asarray([row["timestamp"] for row in result_rows], dtype=float)
    positions = np.asarray(
        [[row["px_m"], row["py_m"], row["pz_m"]] for row in result_rows],
        dtype=float,
    )
    velocities = np.asarray(
        [[row["vx_mps"], row["vy_mps"], row["vz_mps"]] for row in result_rows],
        dtype=float,
    )
    quaternions = np.asarray(
        [[row["qx"], row["qy"], row["qz"], row["qw"]] for row in result_rows],
        dtype=float,
    )
    rpy = np.asarray(
        [[row["roll_rad"], row["pitch_rad"], row["yaw_rad"]] for row in result_rows],
        dtype=float,
    )
    gyro_biases = np.asarray(
        [
            [row["bg_x_rad_s"], row["bg_y_rad_s"], row["bg_z_rad_s"]]
            for row in result_rows
        ],
        dtype=float,
    )
    accelerometer_biases = np.asarray(
        [
            [row["ba_x_mps2"], row["ba_y_mps2"], row["ba_z_mps2"]]
            for row in result_rows
        ],
        dtype=float,
    )
    formal_arrays = (
        timestamps,
        positions,
        velocities,
        quaternions,
        rpy,
        gyro_biases,
        accelerometer_biases,
    )
    if not all(np.all(np.isfinite(array)) for array in formal_arrays):
        raise RuntimeError("formal result contains NaN or Inf")
    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("formal result timestamps are not strictly increasing")

    propagation_dt = np.asarray(imu.dt[k0 + 1 :], dtype=float)
    median_dt = float(np.median(np.asarray(imu.dt[1:], dtype=float)))
    large_dt_threshold = config.preprocess.timestamp_large_step_ratio * median_dt
    pose_matches = sum(bool(row["has_pose_match"]) for row in result_rows)
    successful_updates = sum(bool(row["update_applied"]) for row in result_rows)
    unmatched_pose_frames = sum(
        row["skip_reason"] == "unmatched_pose" for row in debug_rows
    )
    invalid_covariance_skips = sum(
        row["skip_reason"] == "nonpositive_pose_covariance"
        for row in debug_rows
    )
    summary: dict[str, Any] = {
        "phase": "3.5",
        "start_index": k0,
        "end_index": sample_count - 1,
        "start_timestamp": float(timestamps[0]),
        "end_timestamp": float(timestamps[-1]),
        "duration": float(timestamps[-1] - timestamps[0]),
        "total_output_frames": len(result_rows),
        "prediction_steps": sample_count - 1 - k0,
        "pose_matches": int(pose_matches),
        "successful_updates": int(successful_updates),
        "unmatched_pose_frames": int(unmatched_pose_frames),
        "invalid_covariance_skips": int(invalid_covariance_skips),
        "initialization_observation_reused": False,
        "max_quaternion_norm_error": float(
            np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0))
        ),
        "max_P_symmetry_error": max(
            float(row["P_symmetry_error"]) for row in debug_rows
        ),
        "minimum_P_eigenvalue": min(
            float(row["P_min_eigenvalue"]) for row in debug_rows
        ),
        "formal_result_all_finite": True,
        "position_initial": positions[0].tolist(),
        "position_final": positions[-1].tolist(),
        "velocity_initial": velocities[0].tolist(),
        "velocity_final": velocities[-1].tolist(),
        "bg_initial": gyro_biases[0].tolist(),
        "bg_final": gyro_biases[-1].tolist(),
        "ba_initial": accelerometer_biases[0].tolist(),
        "ba_final": accelerometer_biases[-1].tolist(),
        "position_residual_norm": _statistics(position_residual_norms),
        "attitude_residual_norm": _statistics(attitude_residual_norms),
        "NIS": _statistics(nis_values),
        "max_dt": float(np.max(propagation_dt)),
        "large_dt_threshold": float(large_dt_threshold),
        "large_dt_count": int(
            np.count_nonzero(propagation_dt > large_dt_threshold)
        ),
    }

    return ESKF15DRunResult(
        result_rows=tuple(result_rows),
        debug_rows=tuple(debug_rows),
        summary=summary,
    )


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_eskf15d_run(
    run_result: ESKF15DRunResult,
    output_directory: str | Path,
) -> ESKF15DOutputPaths:
    """Write formal results, development diagnostics, and JSON statistics."""

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = ESKF15DOutputPaths(
        result_csv=output_root / "eskf15d_result.csv",
        debug_csv=output_root / "eskf15d_debug.csv",
        summary_json=output_root / "eskf15d_summary.json",
    )
    _write_csv_atomic(paths.result_csv, RESULT_FIELDNAMES, run_result.result_rows)
    _write_csv_atomic(paths.debug_csv, DEBUG_FIELDNAMES, run_result.debug_rows)

    temporary_summary = paths.summary_json.with_name(paths.summary_json.name + ".tmp")
    try:
        with temporary_summary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                run_result.summary,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_summary, paths.summary_json)
    finally:
        if temporary_summary.exists():
            temporary_summary.unlink()
    return paths
