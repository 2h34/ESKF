"""Full-data scheduler, diagnostics, and file output for the 6D ESKF.

This module owns IMU indices, authoritative IMU timestamps, and precomputed
pose matches.  It deliberately keeps those concerns outside :class:`ESKF6D`,
which remains a mathematical attitude/bias filter only.
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
from eskf6d import ESKF6D, ESKF6DState, PredictionDebug, UpdateDebug
from initialization import ESKF6DInitialization
from rotation_utils import quaternion_to_rpy


RESULT_FIELDNAMES = (
    "timestamp",
    "imu_index",
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
    "q_norm",
    "P_theta_x",
    "P_theta_y",
    "P_theta_z",
    "P_bg_x",
    "P_bg_y",
    "P_bg_z",
    "P_min_eigenvalue",
    "P_symmetry_error",
    "r_theta_x",
    "r_theta_y",
    "r_theta_z",
    "residual_norm",
    "R_x",
    "R_y",
    "R_z",
    "NIS",
    "delta_theta_x",
    "delta_theta_y",
    "delta_theta_z",
    "delta_bg_x",
    "delta_bg_y",
    "delta_bg_z",
    "K_theta_x",
    "K_theta_y",
    "K_theta_z",
    "K_bg_x",
    "K_bg_y",
    "K_bg_z",
)


@dataclass(frozen=True)
class ESKF6DRunResult:
    """In-memory formal rows, development diagnostics, and run statistics."""

    result_rows: tuple[dict[str, Any], ...]
    debug_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ESKF6DOutputPaths:
    """Paths written by :func:`save_eskf6d_run`."""

    result_csv: Path
    debug_csv: Path
    summary_json: Path


def _validate_runner_inputs(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    initialization: ESKF6DInitialization,
    eskf: ESKF6D,
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
    if np.asarray(alignment.pose_index_for_imu).shape != (sample_count,):
        raise ValueError("alignment must have one pose index per IMU sample")

    pose_count = int(np.asarray(pose.timestamp).size)
    if np.asarray(pose.quaternion_xyzw).shape != (pose_count, 4):
        raise ValueError("pose.quaternion_xyzw must have shape (M, 4)")
    if np.asarray(pose.cov_diag).shape != (pose_count, 6):
        raise ValueError("pose.cov_diag must have shape (M, 6)")

    k0 = int(initialization.imu_index)
    expected_k0 = int(init_stats.static_end_idx) - 1
    if k0 != expected_k0:
        raise ValueError(
            f"initialization IMU index {k0} does not equal static_end_idx - 1 "
            f"({expected_k0})"
        )
    if not (0 <= k0 < sample_count):
        raise ValueError("initialization IMU index is outside the IMU data")
    if float(initialization.timestamp) != float(imu.timestamp[k0]):
        raise ValueError("initialization timestamp must come from imu.timestamp[k0]")
    if int(alignment.pose_index_for_imu[k0]) != int(initialization.pose_index):
        raise ValueError("initialization pose index must come from the match table")

    state = eskf.get_state()
    if not np.allclose(state.q_xyzw, initialization.q0_xyzw, rtol=0.0, atol=1e-15):
        raise ValueError("ESKF quaternion does not match the supplied initialization")
    if not np.array_equal(state.bg_rad_s, initialization.bg0_rad_s):
        raise ValueError("ESKF bias does not match the supplied initialization")
    if not np.array_equal(state.P, initialization.P0):
        raise ValueError("ESKF covariance does not match the supplied initialization")
    return k0


def _covariance_metrics(
    state: ESKF6DState,
    config: ProjectConfig,
    imu_index: int,
) -> tuple[float, float]:
    covariance = np.asarray(state.P, dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
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
    state: ESKF6DState,
    has_pose_match: bool,
    update_applied: bool,
) -> dict[str, Any]:
    rpy = quaternion_to_rpy(state.q_xyzw)
    return {
        "timestamp": timestamp,
        "imu_index": imu_index,
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
    state: ESKF6DState,
    minimum_eigenvalue: float,
    symmetry_error: float,
    prediction: PredictionDebug | None,
    update: UpdateDebug | None,
) -> dict[str, Any]:
    missing = float("nan")
    omega_hat = prediction.omega_hat_rad_s if prediction is not None else None
    covariance_diagonal = np.diag(state.P)

    if update is None:
        residual = delta_theta = delta_bg = None
        R_diagonal = None
        K_theta = K_bg = None
        residual_norm = NIS = missing
    else:
        residual = update.r_theta_rad
        delta_theta = update.delta_theta_rad
        delta_bg = update.delta_bg_rad_s
        R_diagonal = np.diag(update.R_attitude)
        K_theta = np.diag(update.K[0:3, :])
        K_bg = np.array(
            [update.K[3, 0], update.K[4, 1], update.K[5, 2]], dtype=float
        )
        residual_norm = update.residual_norm
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
        "q_norm": float(np.linalg.norm(state.q_xyzw)),
        "P_theta_x": float(covariance_diagonal[0]),
        "P_theta_y": float(covariance_diagonal[1]),
        "P_theta_z": float(covariance_diagonal[2]),
        "P_bg_x": float(covariance_diagonal[3]),
        "P_bg_y": float(covariance_diagonal[4]),
        "P_bg_z": float(covariance_diagonal[5]),
        "P_min_eigenvalue": minimum_eigenvalue,
        "P_symmetry_error": symmetry_error,
        "r_theta_x": component(residual, 0),
        "r_theta_y": component(residual, 1),
        "r_theta_z": component(residual, 2),
        "residual_norm": residual_norm,
        "R_x": component(R_diagonal, 0),
        "R_y": component(R_diagonal, 1),
        "R_z": component(R_diagonal, 2),
        "NIS": NIS,
        "delta_theta_x": component(delta_theta, 0),
        "delta_theta_y": component(delta_theta, 1),
        "delta_theta_z": component(delta_theta, 2),
        "delta_bg_x": component(delta_bg, 0),
        "delta_bg_y": component(delta_bg, 1),
        "delta_bg_z": component(delta_bg, 2),
        "K_theta_x": component(K_theta, 0),
        "K_theta_y": component(K_theta, 1),
        "K_theta_z": component(K_theta, 2),
        "K_bg_x": component(K_bg, 0),
        "K_bg_y": component(K_bg, 1),
        "K_bg_z": component(K_bg, 2),
    }


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


def run_eskf6d(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
    initialization: ESKF6DInitialization,
    eskf: ESKF6D,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> ESKF6DRunResult:
    """Run initialization posterior through the final IMU sample.

    The initialization pose at ``k0`` has already contributed to ``q0/P0`` and
    is therefore logged but never passed to ``update_attitude`` again.
    """

    k0 = _validate_runner_inputs(
        imu, pose, init_stats, alignment, initialization, eskf
    )
    sample_count = int(imu.timestamp.size)
    result_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    residual_norms: list[float] = []
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
        prediction = eskf.predict(imu.gyro_rad_s[k], float(imu.dt[current_k]))
        pose_index = int(alignment.pose_index_for_imu[current_k])
        has_pose_match = pose_index >= 0
        update: UpdateDebug | None = None

        if not has_pose_match:
            skip_reason = "unmatched_pose"
        else:
            attitude_covariance = np.asarray(
                pose.cov_diag[pose_index, 3:6], dtype=float
            )
            if not np.all(np.isfinite(attitude_covariance)):
                raise RuntimeError(
                    f"attitude covariance is non-finite at pose index {pose_index}"
                )
            if np.any(attitude_covariance <= 0.0):
                skip_reason = "nonpositive_attitude_covariance"
            else:
                update = eskf.update_attitude(
                    pose.quaternion_xyzw[pose_index],
                    np.diag(attitude_covariance),
                )
                skip_reason = ""
                residual_norms.append(update.residual_norm)
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
    quaternions = np.asarray(
        [[row["qx"], row["qy"], row["qz"], row["qw"]] for row in result_rows],
        dtype=float,
    )
    biases = np.asarray(
        [
            [row["bg_x_rad_s"], row["bg_y_rad_s"], row["bg_z_rad_s"]]
            for row in result_rows
        ],
        dtype=float,
    )
    rpy = np.asarray(
        [[row["roll_rad"], row["pitch_rad"], row["yaw_rad"]] for row in result_rows],
        dtype=float,
    )
    if not (
        np.all(np.isfinite(timestamps))
        and np.all(np.isfinite(quaternions))
        and np.all(np.isfinite(biases))
        and np.all(np.isfinite(rpy))
    ):
        raise RuntimeError("formal result contains NaN or Inf")
    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("formal result timestamps are not strictly increasing")

    propagation_dt = np.asarray(imu.dt[k0 + 1 :], dtype=float)
    full_data_median_dt = float(np.median(np.asarray(imu.dt[1:], dtype=float)))
    large_dt_threshold = (
        config.preprocess.timestamp_large_step_ratio * full_data_median_dt
    )
    pose_matches = sum(bool(row["has_pose_match"]) for row in result_rows)
    successful_updates = sum(bool(row["update_applied"]) for row in result_rows)
    invalid_covariance_skips = sum(
        row["skip_reason"] == "nonpositive_attitude_covariance"
        for row in debug_rows
    )
    unmatched_pose_frames = sum(
        row["skip_reason"] == "unmatched_pose" for row in debug_rows
    )
    P_minimum_eigenvalue = min(
        float(row["P_min_eigenvalue"]) for row in debug_rows
    )
    max_P_symmetry_error = max(
        float(row["P_symmetry_error"]) for row in debug_rows
    )
    summary: dict[str, Any] = {
        "phase": "2.4A",
        "interpretation": (
            "完整运行成功只验证 6D ESKF 闭环能够稳定运行并输出诊断量；"
            "Q/R/P0 及其他参数的合理性留待下一阶段人工分析。"
        ),
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
        "max_P_symmetry_error": max_P_symmetry_error,
        "minimum_P_eigenvalue": P_minimum_eigenvalue,
        "formal_result_all_finite": True,
        "bias_initial": biases[0].tolist(),
        "bias_final": biases[-1].tolist(),
        "bias_min": np.min(biases, axis=0).tolist(),
        "bias_max": np.max(biases, axis=0).tolist(),
        "residual_norm": _statistics(residual_norms),
        "NIS": _statistics(nis_values),
        "RPY_rad": {
            "min": {
                "roll": float(np.min(rpy[:, 0])),
                "pitch": float(np.min(rpy[:, 1])),
                "yaw": float(np.min(rpy[:, 2])),
            },
            "max": {
                "roll": float(np.max(rpy[:, 0])),
                "pitch": float(np.max(rpy[:, 1])),
                "yaw": float(np.max(rpy[:, 2])),
            },
        },
        "max_dt": float(np.max(propagation_dt)),
        "large_dt_threshold": float(large_dt_threshold),
        "large_dt_count": int(np.count_nonzero(propagation_dt > large_dt_threshold)),
    }

    return ESKF6DRunResult(
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


def save_eskf6d_run(
    run_result: ESKF6DRunResult,
    output_directory: str | Path,
) -> ESKF6DOutputPaths:
    """Write formal results, development diagnostics, and JSON statistics."""

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = ESKF6DOutputPaths(
        result_csv=output_root / "eskf6d_result.csv",
        debug_csv=output_root / "eskf6d_debug.csv",
        summary_json=output_root / "eskf6d_summary.json",
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
