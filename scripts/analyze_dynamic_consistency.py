"""Offline dynamic-consistency diagnostics for Phase 2.4B-2.

This script reads processed inputs and already-saved 6D ESKF outputs.  It does
not rerun or modify the filter, tune parameters, reject observations, or write
the formal result/debug CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import (
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)


CHI2_DF3_95 = 7.8147279
CHI2_DF3_99 = 11.3448667
DYNAMIC_OMEGA_THRESHOLD_RAD_S = 0.05
LAG_MIN_S = -0.030
LAG_MAX_S = 0.030
LAG_STEP_S = 0.00025
LOG_EPSILON = 1e-15
DT_RTOL = 1e-10
DT_ATOL = 1e-12


FloatArray = NDArray[np.float64]


INCREMENT_FIELDNAMES = (
    "imu_index_prev",
    "imu_index_curr",
    "timestamp_prev",
    "timestamp_curr",
    "dt",
    "pose_index_prev",
    "pose_index_curr",
    "pose_timestamp_prev",
    "pose_timestamp_curr",
    "dt_pose",
    "pose_interval_midpoint",
    "NIS_curr",
    "normal_nis_curr",
    "high_nis_curr",
    "omega_x",
    "omega_y",
    "omega_z",
    "omega_norm",
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "alpha_norm",
    "omega_pose_x",
    "omega_pose_y",
    "omega_pose_z",
    "omega_pose_norm",
    "phi_pose_x",
    "phi_pose_y",
    "phi_pose_z",
    "phi_imu_left_x",
    "phi_imu_left_y",
    "phi_imu_left_z",
    "phi_imu_trap_x",
    "phi_imu_trap_y",
    "phi_imu_trap_z",
    "e_left_x",
    "e_left_y",
    "e_left_z",
    "e_left_norm",
    "e_trap_x",
    "e_trap_y",
    "e_trap_z",
    "e_trap_norm",
    "left_minus_trap_x",
    "left_minus_trap_y",
    "left_minus_trap_z",
    "left_minus_trap_norm",
    "zoh_correction_x",
    "zoh_correction_y",
    "zoh_correction_z",
    "zoh_correction_norm",
    "zoh_approximation_error_norm",
    "improvement_ratio",
)


LAG_FIELDNAMES = (
    "subset",
    "bias_version",
    "lag_s",
    "lag_ms",
    "sample_count",
    "RMSE_x",
    "RMSE_y",
    "RMSE_z",
    "combined_RMSE",
    "yz_only_RMSE",
)


KEY_SEGMENT_FIELDNAMES = (
    "segment_name",
    "segment_id",
    "start_imu_index",
    "end_imu_index",
    "start_timestamp",
    "end_timestamp",
    "duration_s",
    "interval_count",
    "bias_version",
    "median_omega_norm",
    "max_omega_norm",
    "median_alpha_norm",
    "max_alpha_norm",
    "median_e_left_norm",
    "p95_e_left_norm",
    "max_e_left_norm",
    "median_e_trap_norm",
    "p95_e_trap_norm",
    "max_e_trap_norm",
    "median_abs_e_left_x",
    "median_abs_e_left_y",
    "median_abs_e_left_z",
    "median_abs_e_trap_x",
    "median_abs_e_trap_y",
    "median_abs_e_trap_z",
    "median_improvement_ratio",
    "positive_improvement_fraction",
    "best_lag_s",
    "best_lag_ms",
    "zero_lag_rmse",
    "best_lag_rmse",
    "rmse_reduction_fraction",
    "yz_best_lag_s",
    "yz_best_lag_ms",
    "yz_zero_lag_rmse",
    "yz_best_lag_rmse",
    "yz_rmse_reduction_fraction",
)


@dataclass(frozen=True)
class IncrementComparison:
    phi_imu_left: FloatArray
    phi_imu_trap: FloatArray
    e_left: FloatArray
    e_trap: FloatArray
    zoh_correction: FloatArray


@dataclass(frozen=True)
class DynamicAnalysisPaths:
    summary_json: Path
    increment_csv: Path
    lag_sweep_csv: Path
    key_segments_csv: Path


def pose_increment_rotvec(q_prev_xyzw: ArrayLike, q_curr_xyzw: ArrayLike) -> FloatArray:
    """Return shortest local increment ``Log(q_prev^-1 tensor q_curr)``."""

    q_prev = normalize_quaternion(q_prev_xyzw)
    q_curr = normalize_quaternion(q_curr_xyzw)
    if float(np.dot(q_prev, q_curr)) < 0.0:
        q_curr = -q_curr
    q_rel = normalize_quaternion(
        quaternion_multiply(quaternion_inverse(q_prev), q_curr)
    )
    return quaternion_to_rotvec(q_rel)


def compare_rotation_increment(
    q_pose_rel_xyzw: ArrayLike,
    omega_prev_rad_s: ArrayLike,
    omega_curr_rad_s: ArrayLike,
    dt_s: float,
) -> IncrementComparison:
    """Compare one pose increment with left-endpoint and trapezoidal gyro steps."""

    q_pose_rel = normalize_quaternion(q_pose_rel_xyzw)
    omega_prev = np.asarray(omega_prev_rad_s, dtype=float)
    omega_curr = np.asarray(omega_curr_rad_s, dtype=float)
    if omega_prev.shape != (3,) or omega_curr.shape != (3,):
        raise ValueError("omega_prev_rad_s and omega_curr_rad_s must have shape (3,)")
    if not np.all(np.isfinite(omega_prev)) or not np.all(np.isfinite(omega_curr)):
        raise ValueError("angular velocities must be finite")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    phi_left = omega_prev * dt_s
    phi_trap = 0.5 * (omega_prev + omega_curr) * dt_s
    dq_left = rotvec_to_quaternion(phi_left)
    dq_trap = rotvec_to_quaternion(phi_trap)
    e_left = quaternion_to_rotvec(
        quaternion_multiply(quaternion_inverse(dq_left), q_pose_rel)
    )
    e_trap = quaternion_to_rotvec(
        quaternion_multiply(quaternion_inverse(dq_trap), q_pose_rel)
    )
    zoh_correction = 0.5 * (omega_curr - omega_prev) * dt_s
    return IncrementComparison(phi_left, phi_trap, e_left, e_trap, zoh_correction)


def lag_sweep(
    imu_timestamp_s: ArrayLike,
    imu_omega_rad_s: ArrayLike,
    pose_midpoint_s: ArrayLike,
    pose_omega_rad_s: ArrayLike,
    lag_values_s: ArrayLike,
) -> list[dict[str, float | int]]:
    """Compare pose interval velocity at t with interpolated IMU velocity at t+lag."""

    imu_time = np.asarray(imu_timestamp_s, dtype=float)
    imu_omega = np.asarray(imu_omega_rad_s, dtype=float)
    pose_time = np.asarray(pose_midpoint_s, dtype=float)
    pose_omega = np.asarray(pose_omega_rad_s, dtype=float)
    lags = np.asarray(lag_values_s, dtype=float)
    if imu_time.ndim != 1 or pose_time.ndim != 1 or lags.ndim != 1:
        raise ValueError("time and lag inputs must be one-dimensional")
    if imu_omega.shape != (imu_time.size, 3):
        raise ValueError("imu_omega_rad_s must have shape (len(imu_timestamp_s), 3)")
    if pose_omega.shape != (pose_time.size, 3):
        raise ValueError("pose_omega_rad_s must have shape (len(pose_midpoint_s), 3)")
    if imu_time.size < 2 or pose_time.size == 0 or lags.size == 0:
        raise ValueError("lag sweep inputs must be non-empty")
    if not all(
        np.all(np.isfinite(value))
        for value in (imu_time, imu_omega, pose_time, pose_omega, lags)
    ):
        raise ValueError("lag sweep inputs must be finite")
    if np.any(np.diff(imu_time) <= 0.0):
        raise ValueError("imu_timestamp_s must be strictly increasing")

    # Relative time reduces floating-point cancellation for Unix-like timestamps.
    origin = float(imu_time[0])
    imu_relative = imu_time - origin
    pose_relative = pose_time - origin
    rows: list[dict[str, float | int]] = []
    for lag in lags:
        query = pose_relative + float(lag)
        valid = (query >= imu_relative[0]) & (query <= imu_relative[-1])
        if not np.any(valid):
            raise ValueError(f"lag {lag} leaves no samples inside the IMU time range")
        interpolated = np.column_stack(
            [
                np.interp(query[valid], imu_relative, imu_omega[:, axis])
                for axis in range(3)
            ]
        )
        error = pose_omega[valid] - interpolated
        axis_rmse = np.sqrt(np.mean(error * error, axis=0))
        combined = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        yz_only = float(np.sqrt(np.mean(np.sum(error[:, 1:] ** 2, axis=1))))
        rows.append(
            {
                "lag_s": float(lag),
                "lag_ms": float(lag * 1000.0),
                "sample_count": int(np.count_nonzero(valid)),
                "RMSE_x": float(axis_rmse[0]),
                "RMSE_y": float(axis_rmse[1]),
                "RMSE_z": float(axis_rmse[2]),
                "combined_RMSE": combined,
                "yz_only_RMSE": yz_only,
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _boolean(row: Mapping[str, str], key: str) -> bool:
    value = row.get(key)
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{key} must contain True or False")


def _statistics(values: ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("statistics require a non-empty finite vector")
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _pearson(x: ArrayLike, y: ArrayLike) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _result_biases(
    rows: Sequence[Mapping[str, str]],
    imu_timestamp: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not rows:
        raise ValueError("eskf6d_result.csv contains no rows")
    indices: list[int] = []
    timestamps: list[float] = []
    biases: list[list[float]] = []
    for row in rows:
        index = _integer(row, "imu_index")
        timestamp = _finite_float(row, "timestamp")
        if index < 0 or index >= imu_timestamp.size:
            raise ValueError(f"result IMU index {index} is out of range")
        if not np.isclose(timestamp, imu_timestamp[index], rtol=0.0, atol=1e-9):
            raise ValueError(f"result timestamp mismatch at IMU index {index}")
        indices.append(index)
        timestamps.append(timestamp)
        biases.append(
            [
                _finite_float(row, "bg_x_rad_s"),
                _finite_float(row, "bg_y_rad_s"),
                _finite_float(row, "bg_z_rad_s"),
            ]
        )
    index_array = np.asarray(indices, dtype=np.int64)
    if np.any(np.diff(index_array) != 1):
        raise ValueError("result IMU indices must be consecutive")
    return (
        index_array,
        np.asarray(timestamps, dtype=float),
        np.asarray(biases, dtype=float),
    )


def _nis_by_index(rows: Sequence[Mapping[str, str]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        index = _integer(row, "imu_index")
        if _boolean(row, "update_applied"):
            values[index] = _finite_float(row, "NIS")
    return values


def _pose_relative_quaternion(q_prev: FloatArray, q_curr: FloatArray) -> FloatArray:
    previous = normalize_quaternion(q_prev)
    current = normalize_quaternion(q_curr)
    if float(np.dot(previous, current)) < 0.0:
        current = -current
    return normalize_quaternion(
        quaternion_multiply(quaternion_inverse(previous), current)
    )


def _comparison_values(comparison: IncrementComparison) -> dict[str, Any]:
    left_minus_trap = comparison.e_left - comparison.e_trap
    left_norm = float(np.linalg.norm(comparison.e_left))
    trap_norm = float(np.linalg.norm(comparison.e_trap))
    improvement = (left_norm - trap_norm) / left_norm if left_norm > 0.0 else np.nan
    return {
        "phi_left": comparison.phi_imu_left,
        "phi_trap": comparison.phi_imu_trap,
        "e_left": comparison.e_left,
        "e_trap": comparison.e_trap,
        "e_left_norm": left_norm,
        "e_trap_norm": trap_norm,
        "left_minus_trap": left_minus_trap,
        "left_minus_trap_norm": float(np.linalg.norm(left_minus_trap)),
        "zoh_correction": comparison.zoh_correction,
        "zoh_correction_norm": float(np.linalg.norm(comparison.zoh_correction)),
        "zoh_approximation_error_norm": float(
            np.linalg.norm(left_minus_trap - comparison.zoh_correction)
        ),
        "improvement_ratio": float(improvement),
    }


def _build_interval_records(
    imu: Any,
    pose: Any,
    alignment: Any,
    result_indices: NDArray[np.int64],
    estimated_bias: FloatArray,
    fixed_bg0: FloatArray,
    nis_by_index: Mapping[int, float],
) -> list[dict[str, Any]]:
    bias_by_index = {
        int(index): estimated_bias[position]
        for position, index in enumerate(result_indices)
    }
    records: list[dict[str, Any]] = []
    for current_index in range(int(result_indices[0]) + 1, int(result_indices[-1]) + 1):
        previous_index = current_index - 1
        pose_previous = int(alignment.pose_index_for_imu[previous_index])
        pose_current = int(alignment.pose_index_for_imu[current_index])
        if pose_previous < 0 or pose_current < 0 or pose_current != pose_previous + 1:
            continue

        dt = float(imu.timestamp[current_index] - imu.timestamp[previous_index])
        if dt <= 0.0:
            raise RuntimeError("IMU timestamps are not strictly increasing")
        if not np.isclose(dt, imu.dt[current_index], rtol=DT_RTOL, atol=DT_ATOL):
            raise RuntimeError(f"imu.dt mismatch at destination index {current_index}")
        dt_pose = float(pose.timestamp[pose_current] - pose.timestamp[pose_previous])
        if dt_pose <= 0.0:
            raise RuntimeError("pose timestamps are not strictly increasing")

        q_pose_rel = _pose_relative_quaternion(
            pose.quaternion_xyzw[pose_previous], pose.quaternion_xyzw[pose_current]
        )
        phi_pose = quaternion_to_rotvec(q_pose_rel)
        omega_pose = phi_pose / dt_pose

        estimated_omega_previous = (
            imu.gyro_rad_s[previous_index] - bias_by_index[previous_index]
        )
        estimated_omega_current = (
            imu.gyro_rad_s[current_index] - bias_by_index[current_index]
        )
        fixed_omega_previous = imu.gyro_rad_s[previous_index] - fixed_bg0
        fixed_omega_current = imu.gyro_rad_s[current_index] - fixed_bg0

        estimated_comparison = _comparison_values(
            compare_rotation_increment(
                q_pose_rel, estimated_omega_previous, estimated_omega_current, dt
            )
        )
        fixed_comparison = _comparison_values(
            compare_rotation_increment(
                q_pose_rel, fixed_omega_previous, fixed_omega_current, dt
            )
        )
        estimated_alpha = (estimated_omega_current - estimated_omega_previous) / dt
        fixed_alpha = (fixed_omega_current - fixed_omega_previous) / dt
        nis = float(nis_by_index.get(current_index, np.nan))
        records.append(
            {
                "imu_index_prev": previous_index,
                "imu_index_curr": current_index,
                "timestamp_prev": float(imu.timestamp[previous_index]),
                "timestamp_curr": float(imu.timestamp[current_index]),
                "dt": dt,
                "pose_index_prev": pose_previous,
                "pose_index_curr": pose_current,
                "pose_timestamp_prev": float(pose.timestamp[pose_previous]),
                "pose_timestamp_curr": float(pose.timestamp[pose_current]),
                "dt_pose": dt_pose,
                "pose_interval_midpoint": float(
                    0.5 * (pose.timestamp[pose_previous] + pose.timestamp[pose_current])
                ),
                "NIS_curr": nis,
                "normal_nis_curr": bool(np.isfinite(nis) and nis <= CHI2_DF3_95),
                "high_nis_curr": bool(np.isfinite(nis) and nis > CHI2_DF3_99),
                "phi_pose": phi_pose,
                "omega_pose": omega_pose,
                "omega_pose_norm": float(np.linalg.norm(omega_pose)),
                "estimated_bias": {
                    "omega_prev": estimated_omega_previous,
                    "omega_curr": estimated_omega_current,
                    "omega_norm": float(np.linalg.norm(estimated_omega_previous)),
                    "alpha": estimated_alpha,
                    "alpha_norm": float(np.linalg.norm(estimated_alpha)),
                    **estimated_comparison,
                },
                "fixed_bg0": {
                    "omega_prev": fixed_omega_previous,
                    "omega_curr": fixed_omega_current,
                    "omega_norm": float(np.linalg.norm(fixed_omega_previous)),
                    "alpha": fixed_alpha,
                    "alpha_norm": float(np.linalg.norm(fixed_alpha)),
                    **fixed_comparison,
                },
            }
        )
    if not records:
        raise ValueError("no reliable consecutive matched pose intervals were found")
    return records


def _increment_csv_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        estimated = record["estimated_bias"]
        row: dict[str, Any] = {
            key: record[key]
            for key in (
                "imu_index_prev",
                "imu_index_curr",
                "timestamp_prev",
                "timestamp_curr",
                "dt",
                "pose_index_prev",
                "pose_index_curr",
                "pose_timestamp_prev",
                "pose_timestamp_curr",
                "dt_pose",
                "pose_interval_midpoint",
                "NIS_curr",
                "normal_nis_curr",
                "high_nis_curr",
                "omega_pose_norm",
            )
        }
        for prefix, vector in (
            ("omega", estimated["omega_prev"]),
            ("alpha", estimated["alpha"]),
            ("omega_pose", record["omega_pose"]),
            ("phi_pose", record["phi_pose"]),
            ("phi_imu_left", estimated["phi_left"]),
            ("phi_imu_trap", estimated["phi_trap"]),
            ("e_left", estimated["e_left"]),
            ("e_trap", estimated["e_trap"]),
            ("left_minus_trap", estimated["left_minus_trap"]),
            ("zoh_correction", estimated["zoh_correction"]),
        ):
            for axis, value in zip("xyz", vector):
                row[f"{prefix}_{axis}"] = float(value)
        row.update(
            {
                "omega_norm": estimated["omega_norm"],
                "alpha_norm": estimated["alpha_norm"],
                "e_left_norm": estimated["e_left_norm"],
                "e_trap_norm": estimated["e_trap_norm"],
                "left_minus_trap_norm": estimated["left_minus_trap_norm"],
                "zoh_correction_norm": estimated["zoh_correction_norm"],
                "zoh_approximation_error_norm": estimated[
                    "zoh_approximation_error_norm"
                ],
                "improvement_ratio": estimated["improvement_ratio"],
            }
        )
        rows.append(row)
    return rows


def _group_records(
    records: Sequence[Mapping[str, Any]], group_name: str
) -> list[Mapping[str, Any]]:
    if group_name == "all":
        return list(records)
    if group_name == "normal":
        return [record for record in records if record["normal_nis_curr"]]
    if group_name == "high":
        return [record for record in records if record["high_nis_curr"]]
    raise ValueError(f"unknown increment group {group_name}")


def _increment_summary(
    records: Sequence[Mapping[str, Any]], bias_version: str
) -> dict[str, Any]:
    if not records:
        raise ValueError("increment summary group is empty")
    values = [record[bias_version] for record in records]
    improvement = np.asarray(
        [value["improvement_ratio"] for value in values], dtype=float
    )
    finite_improvement = improvement[np.isfinite(improvement)]
    summary: dict[str, Any] = {
        "count": len(records),
        "e_left_norm": _statistics([value["e_left_norm"] for value in values]),
        "e_trap_norm": _statistics([value["e_trap_norm"] for value in values]),
        "abs_e_left": {},
        "abs_e_trap": {},
        "improvement_ratio": {
            "finite_count": int(finite_improvement.size),
            "median": float(np.median(finite_improvement)),
            "p95": float(np.percentile(finite_improvement, 95.0)),
            "max": float(np.max(finite_improvement)),
            "positive_fraction": float(np.mean(finite_improvement > 0.0)),
        },
        "omega_norm": _statistics([value["omega_norm"] for value in values]),
        "alpha_norm": _statistics([value["alpha_norm"] for value in values]),
    }
    for axis_index, axis in enumerate("xyz"):
        summary["abs_e_left"][axis] = _statistics(
            [abs(float(value["e_left"][axis_index])) for value in values]
        )
        summary["abs_e_trap"][axis] = _statistics(
            [abs(float(value["e_trap"][axis_index])) for value in values]
        )
    return summary


def _correlation_summary(
    records: Sequence[Mapping[str, Any]], bias_version: str
) -> dict[str, Any]:
    values = [record[bias_version] for record in records]
    left_norm = np.asarray([value["e_left_norm"] for value in values], dtype=float)
    omega_norm = np.asarray([value["omega_norm"] for value in values], dtype=float)
    alpha_norm = np.asarray([value["alpha_norm"] for value in values], dtype=float)
    result: dict[str, Any] = {
        "norm": {
            "log_error_vs_log_omega": _pearson(
                np.log(left_norm + LOG_EPSILON),
                np.log(omega_norm + LOG_EPSILON),
            ),
            "log_error_vs_log_alpha": _pearson(
                np.log(left_norm + LOG_EPSILON),
                np.log(alpha_norm + LOG_EPSILON),
            ),
        },
        "axes": {},
    }
    for axis_index, axis in enumerate("xyz"):
        axis_error = np.asarray(
            [abs(float(value["e_left"][axis_index])) for value in values],
            dtype=float,
        )
        axis_omega = np.asarray(
            [abs(float(value["omega_prev"][axis_index])) for value in values],
            dtype=float,
        )
        axis_alpha = np.asarray(
            [abs(float(value["alpha"][axis_index])) for value in values],
            dtype=float,
        )
        result["axes"][axis] = {
            "log_abs_error_vs_log_abs_omega": _pearson(
                np.log(axis_error + LOG_EPSILON),
                np.log(axis_omega + LOG_EPSILON),
            ),
            "log_abs_error_vs_log_abs_alpha": _pearson(
                np.log(axis_error + LOG_EPSILON),
                np.log(axis_alpha + LOG_EPSILON),
            ),
        }
    return result


def _zoh_approximation_summary(
    records: Sequence[Mapping[str, Any]], bias_version: str
) -> dict[str, Any]:
    values = [record[bias_version] for record in records]
    difference_norm = np.asarray(
        [value["left_minus_trap_norm"] for value in values], dtype=float
    )
    correction_norm = np.asarray(
        [value["zoh_correction_norm"] for value in values], dtype=float
    )
    approximation_error = np.asarray(
        [value["zoh_approximation_error_norm"] for value in values], dtype=float
    )
    cosines: list[float] = []
    for value in values:
        a = np.asarray(value["left_minus_trap"], dtype=float)
        b = np.asarray(value["zoh_correction"], dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator > 0.0:
            cosines.append(float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))
    return {
        "small_step_approximation": "0.5 * (omega_curr - omega_prev) * dt; not a strict high-order SO(3) error formula",
        "left_minus_trap_norm": _statistics(difference_norm),
        "zoh_correction_norm": _statistics(correction_norm),
        "approximation_error_norm": _statistics(approximation_error),
        "norm_correlation": _pearson(difference_norm, correction_norm),
        "median_cosine_similarity": (
            float(np.median(np.asarray(cosines, dtype=float))) if cosines else None
        ),
    }


def _lag_values() -> FloatArray:
    count = int(round((LAG_MAX_S - LAG_MIN_S) / LAG_STEP_S)) + 1
    return LAG_MIN_S + np.arange(count, dtype=float) * LAG_STEP_S


def _best_lag_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("lag rows are empty")
    zero = min(rows, key=lambda row: abs(float(row["lag_s"])))
    if abs(float(zero["lag_s"])) > 1e-15:
        raise RuntimeError("lag grid does not contain zero")
    best = min(
        rows,
        key=lambda row: (
            float(row["combined_RMSE"]),
            abs(float(row["lag_s"])),
            float(row["lag_s"]),
        ),
    )
    yz_best = min(
        rows,
        key=lambda row: (
            float(row["yz_only_RMSE"]),
            abs(float(row["lag_s"])),
            float(row["lag_s"]),
        ),
    )
    zero_rmse = float(zero["combined_RMSE"])
    yz_zero_rmse = float(zero["yz_only_RMSE"])
    return {
        "input_interval_count": int(max(int(row["sample_count"]) for row in rows)),
        "best_lag_s": float(best["lag_s"]),
        "best_lag_ms": float(best["lag_ms"]),
        "best_lag_sample_count": int(best["sample_count"]),
        "zero_lag_rmse": zero_rmse,
        "best_lag_rmse": float(best["combined_RMSE"]),
        "rmse_reduction_fraction": float(
            (zero_rmse - float(best["combined_RMSE"])) / zero_rmse
        ),
        "yz_only_best_lag_s": float(yz_best["lag_s"]),
        "yz_only_best_lag_ms": float(yz_best["lag_ms"]),
        "yz_only_best_lag_sample_count": int(yz_best["sample_count"]),
        "yz_only_zero_lag_rmse": yz_zero_rmse,
        "yz_only_best_lag_rmse": float(yz_best["yz_only_RMSE"]),
        "yz_only_rmse_reduction_fraction": float(
            (yz_zero_rmse - float(yz_best["yz_only_RMSE"])) / yz_zero_rmse
        ),
        "axis_rmse_at_combined_best": {
            axis: float(best[f"RMSE_{axis}"]) for axis in "xyz"
        },
    }


def _validate_key_segments(
    nis_summary: Mapping[str, Any],
    segment_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    if not segment_rows:
        raise ValueError("high_nis_segments.csv contains no rows")
    csv_by_id = {_integer(row, "segment_id"): row for row in segment_rows}
    output: dict[str, dict[str, Any]] = {}
    for name, summary_key in (
        ("longest_high_NIS_segment", "longest_high_NIS_segment"),
        ("max_NIS_segment", "max_NIS_segment"),
    ):
        value = nis_summary.get(summary_key)
        if not isinstance(value, dict):
            raise ValueError(f"nis_analysis_summary.json is missing {summary_key}")
        segment_id = int(value["segment_id"])
        csv_row = csv_by_id.get(segment_id)
        if csv_row is None:
            raise ValueError(f"segment {segment_id} is missing from high_nis_segments.csv")
        for key in ("start_imu_index", "end_imu_index"):
            if int(value[key]) != _integer(csv_row, key):
                raise ValueError(f"segment {segment_id} {key} disagrees across inputs")
        output[name] = dict(value)
    return output


def _lag_subsets(
    records: Sequence[Mapping[str, Any]],
    key_segments: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    subsets = {
        "all_valid_dynamic": [
            record
            for record in records
            if float(record["omega_pose_norm"]) > DYNAMIC_OMEGA_THRESHOLD_RAD_S
        ],
        "high_NIS": [record for record in records if record["high_nis_curr"]],
    }
    for name, segment in key_segments.items():
        start = int(segment["start_imu_index"])
        end = int(segment["end_imu_index"])
        subsets[name] = [
            record
            for record in records
            if start <= int(record["imu_index_curr"]) <= end
        ]
    for name, subset in subsets.items():
        if not subset:
            raise ValueError(f"lag subset {name} is empty")
    return subsets


def _run_lag_analyses(
    records: Sequence[Mapping[str, Any]],
    key_segments: Mapping[str, Mapping[str, Any]],
    corrected_gyro: Mapping[str, tuple[FloatArray, FloatArray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lag_values = _lag_values()
    subsets = _lag_subsets(records, key_segments)
    summary: dict[str, Any] = {bias: {} for bias in corrected_gyro}
    all_rows: list[dict[str, Any]] = []
    for bias_version, (imu_time, imu_omega) in corrected_gyro.items():
        for subset_name, subset in subsets.items():
            pose_time = np.asarray(
                [record["pose_interval_midpoint"] for record in subset], dtype=float
            )
            pose_omega = np.asarray(
                [record["omega_pose"] for record in subset], dtype=float
            )
            rows = lag_sweep(imu_time, imu_omega, pose_time, pose_omega, lag_values)
            labeled = [
                {
                    "subset": subset_name,
                    "bias_version": bias_version,
                    **row,
                }
                for row in rows
            ]
            all_rows.extend(labeled)
            subset_summary = _best_lag_summary(labeled)
            subset_summary["selected_interval_count"] = len(subset)
            subset_summary["dynamic_threshold_rad_s"] = (
                DYNAMIC_OMEGA_THRESHOLD_RAD_S
                if subset_name == "all_valid_dynamic"
                else None
            )
            summary[bias_version][subset_name] = subset_summary
    return summary, all_rows


def _key_segment_rows(
    records: Sequence[Mapping[str, Any]],
    key_segments: Mapping[str, Mapping[str, Any]],
    lag_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_name, segment in key_segments.items():
        start = int(segment["start_imu_index"])
        end = int(segment["end_imu_index"])
        subset = [
            record
            for record in records
            if start <= int(record["imu_index_curr"]) <= end
        ]
        for bias_version in ("estimated_bias", "fixed_bg0"):
            stats = _increment_summary(subset, bias_version)
            lag = lag_summary[bias_version][segment_name]
            rows.append(
                {
                    "segment_name": segment_name,
                    "segment_id": int(segment["segment_id"]),
                    "start_imu_index": start,
                    "end_imu_index": end,
                    "start_timestamp": float(segment["start_timestamp"]),
                    "end_timestamp": float(segment["end_timestamp"]),
                    "duration_s": float(
                        segment["end_timestamp"] - segment["start_timestamp"]
                    ),
                    "interval_count": len(subset),
                    "bias_version": bias_version,
                    "median_omega_norm": stats["omega_norm"]["median"],
                    "max_omega_norm": stats["omega_norm"]["max"],
                    "median_alpha_norm": stats["alpha_norm"]["median"],
                    "max_alpha_norm": stats["alpha_norm"]["max"],
                    "median_e_left_norm": stats["e_left_norm"]["median"],
                    "p95_e_left_norm": stats["e_left_norm"]["p95"],
                    "max_e_left_norm": stats["e_left_norm"]["max"],
                    "median_e_trap_norm": stats["e_trap_norm"]["median"],
                    "p95_e_trap_norm": stats["e_trap_norm"]["p95"],
                    "max_e_trap_norm": stats["e_trap_norm"]["max"],
                    **{
                        f"median_abs_e_left_{axis}": stats["abs_e_left"][axis][
                            "median"
                        ]
                        for axis in "xyz"
                    },
                    **{
                        f"median_abs_e_trap_{axis}": stats["abs_e_trap"][axis][
                            "median"
                        ]
                        for axis in "xyz"
                    },
                    "median_improvement_ratio": stats["improvement_ratio"][
                        "median"
                    ],
                    "positive_improvement_fraction": stats["improvement_ratio"][
                        "positive_fraction"
                    ],
                    "best_lag_s": lag["best_lag_s"],
                    "best_lag_ms": lag["best_lag_ms"],
                    "zero_lag_rmse": lag["zero_lag_rmse"],
                    "best_lag_rmse": lag["best_lag_rmse"],
                    "rmse_reduction_fraction": lag["rmse_reduction_fraction"],
                    "yz_best_lag_s": lag["yz_only_best_lag_s"],
                    "yz_best_lag_ms": lag["yz_only_best_lag_ms"],
                    "yz_zero_lag_rmse": lag["yz_only_zero_lag_rmse"],
                    "yz_best_lag_rmse": lag["yz_only_best_lag_rmse"],
                    "yz_rmse_reduction_fraction": lag[
                        "yz_only_rmse_reduction_fraction"
                    ],
                }
            )
    return rows


def analyze_dynamic_consistency(
    imu_path: Path,
    pose_path: Path,
    alignment_path: Path,
    init_stats_path: Path,
    result_csv: Path,
    debug_csv: Path,
    nis_summary_json: Path,
    high_nis_segments_csv: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    imu = load_processed_imu(imu_path)
    pose = load_processed_pose(pose_path)
    alignment = load_alignment(
        alignment_path,
        imu_sample_count=imu.timestamp.size,
        pose_sample_count=pose.timestamp.size,
    )
    init_stats = load_init_stats(init_stats_path, imu_sample_count=imu.timestamp.size)
    result_rows = _read_csv(result_csv)
    debug_rows = _read_csv(debug_csv)
    nis_summary = _read_json(nis_summary_json)
    segment_rows = _read_csv(high_nis_segments_csv)

    result_indices, result_timestamps, estimated_bias = _result_biases(
        result_rows, imu.timestamp
    )
    fixed_bg0 = np.asarray(init_stats.gyro_mean, dtype=float)
    records = _build_interval_records(
        imu,
        pose,
        alignment,
        result_indices,
        estimated_bias,
        fixed_bg0,
        _nis_by_index(debug_rows),
    )
    key_segments = _validate_key_segments(nis_summary, segment_rows)

    gyro_for_result_range = imu.gyro_rad_s[result_indices]
    corrected_gyro = {
        "estimated_bias": (
            result_timestamps,
            gyro_for_result_range - estimated_bias,
        ),
        "fixed_bg0": (
            result_timestamps,
            gyro_for_result_range - fixed_bg0,
        ),
    }
    lag_summary, lag_rows = _run_lag_analyses(
        records, key_segments, corrected_gyro
    )
    increment_rows = _increment_csv_rows(records)
    key_rows = _key_segment_rows(records, key_segments, lag_summary)

    increment_summary: dict[str, Any] = {}
    correlation: dict[str, Any] = {}
    axis_analysis: dict[str, Any] = {}
    left_vs_trapezoid: dict[str, Any] = {}
    zoh_approximation: dict[str, Any] = {}
    for bias_version in ("estimated_bias", "fixed_bg0"):
        increment_summary[bias_version] = {}
        correlation[bias_version] = {}
        axis_analysis[bias_version] = {}
        left_vs_trapezoid[bias_version] = {}
        zoh_approximation[bias_version] = {}
        for group_name in ("all", "normal", "high"):
            group = _group_records(records, group_name)
            group_summary = _increment_summary(group, bias_version)
            increment_summary[bias_version][group_name] = group_summary
            correlation[bias_version][group_name] = _correlation_summary(
                group, bias_version
            )
            axis_analysis[bias_version][group_name] = {
                "abs_e_left": group_summary["abs_e_left"],
                "abs_e_trap": group_summary["abs_e_trap"],
            }
            left_vs_trapezoid[bias_version][group_name] = {
                "left_median": group_summary["e_left_norm"]["median"],
                "left_p95": group_summary["e_left_norm"]["p95"],
                "trap_median": group_summary["e_trap_norm"]["median"],
                "trap_p95": group_summary["e_trap_norm"]["p95"],
                "median_improvement_ratio": group_summary["improvement_ratio"][
                    "median"
                ],
                "positive_improvement_fraction": group_summary[
                    "improvement_ratio"
                ]["positive_fraction"],
            }
            zoh_approximation[bias_version][group_name] = (
                _zoh_approximation_summary(group, bias_version)
            )

    key_segment_summary: dict[str, Any] = {}
    for name, segment in key_segments.items():
        subset = [
            record
            for record in records
            if int(segment["start_imu_index"])
            <= int(record["imu_index_curr"])
            <= int(segment["end_imu_index"])
        ]
        key_segment_summary[name] = {
            "segment_id": int(segment["segment_id"]),
            "start_imu_index": int(segment["start_imu_index"]),
            "end_imu_index": int(segment["end_imu_index"]),
            "start_timestamp": float(segment["start_timestamp"]),
            "end_timestamp": float(segment["end_timestamp"]),
            "duration_s": float(
                segment["end_timestamp"] - segment["start_timestamp"]
            ),
            "interval_count": len(subset),
            "increment_error": {
                bias: _increment_summary(subset, bias)
                for bias in ("estimated_bias", "fixed_bg0")
            },
            "lag_analysis": {
                bias: lag_summary[bias][name]
                for bias in ("estimated_bias", "fixed_bg0")
            },
        }

    estimated_all = increment_summary["estimated_bias"]["all"]
    fixed_all = increment_summary["fixed_bg0"]["all"]
    summary = {
        "phase": "2.4B-2",
        "scope": "offline diagnostic only; no filter modification; no tuning",
        "valid_interval_definition": (
            "Both IMU endpoints have matched poses and the current pose index is "
            "exactly the previous pose index plus one."
        ),
        "valid_interval_count": len(records),
        "normal_NIS_interval_count": int(
            sum(bool(record["normal_nis_curr"]) for record in records)
        ),
        "high_NIS_interval_count": int(
            sum(bool(record["high_nis_curr"]) for record in records)
        ),
        "thresholds": {
            "normal_NIS_max_reference": CHI2_DF3_95,
            "high_NIS_min_reference": CHI2_DF3_99,
            "dynamic_omega_pose_norm_rad_s": DYNAMIC_OMEGA_THRESHOLD_RAD_S,
            "lag_min_s": LAG_MIN_S,
            "lag_max_s": LAG_MAX_S,
            "lag_step_s": LAG_STEP_S,
            "reference_only": True,
        },
        "bias_versions": {
            "estimated_bias": "posterior bg[k] from eskf6d_result.csv",
            "fixed_bg0": "init_stats.gyro_mean held constant",
            "bg0_rad_s": [float(value) for value in fixed_bg0],
        },
        "increment_error": increment_summary,
        "left_vs_trapezoid": left_vs_trapezoid,
        "correlation": correlation,
        "axis_analysis": axis_analysis,
        "zoh_small_step_diagnostic": zoh_approximation,
        "lag_analysis": lag_summary,
        "bias_sensitivity": {
            "all_interval_left_median_difference_fixed_minus_estimated": float(
                fixed_all["e_left_norm"]["median"]
                - estimated_all["e_left_norm"]["median"]
            ),
            "all_interval_trap_median_difference_fixed_minus_estimated": float(
                fixed_all["e_trap_norm"]["median"]
                - estimated_all["e_trap_norm"]["median"]
            ),
            "all_dynamic_best_lag_difference_fixed_minus_estimated_s": float(
                lag_summary["fixed_bg0"]["all_valid_dynamic"]["best_lag_s"]
                - lag_summary["estimated_bias"]["all_valid_dynamic"][
                    "best_lag_s"
                ]
            ),
        },
        "key_segments": key_segment_summary,
        "interpretation_limits": [
            "A stable nonzero lag is consistent with effective latency but does not prove hardware delay.",
            "Trapezoidal improvement supports a time-varying-angular-velocity contribution but cannot establish a unique root cause.",
            "Pose-derived angular velocity is an interval average, not a strict instantaneous angular velocity.",
            "The ZOH correction vector is a small-step diagnostic approximation, not a strict high-order SO(3) error formula.",
        ],
        "filter_modified": False,
        "parameters_tuned": False,
        "gating_or_rejection_added": False,
    }
    return summary, increment_rows, lag_rows, key_rows


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_dynamic_analysis(
    output_directory: Path,
    summary: Mapping[str, Any],
    increment_rows: Sequence[Mapping[str, Any]],
    lag_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
) -> DynamicAnalysisPaths:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = DynamicAnalysisPaths(
        summary_json=output_directory / "dynamic_consistency_summary.json",
        increment_csv=output_directory / "increment_consistency.csv",
        lag_sweep_csv=output_directory / "lag_sweep.csv",
        key_segments_csv=output_directory / "key_dynamic_segments.csv",
    )
    _write_csv_atomic(paths.increment_csv, INCREMENT_FIELDNAMES, increment_rows)
    _write_csv_atomic(paths.lag_sweep_csv, LAG_FIELDNAMES, lag_rows)
    _write_csv_atomic(paths.key_segments_csv, KEY_SEGMENT_FIELDNAMES, key_rows)

    temporary = paths.summary_json.with_name(paths.summary_json.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, paths.summary_json)
    finally:
        if temporary.exists():
            temporary.unlink()
    return paths


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    summary, increment_rows, lag_rows, key_rows = analyze_dynamic_consistency(
        arguments.imu,
        arguments.pose,
        arguments.alignment,
        arguments.init_stats,
        arguments.result_csv,
        arguments.debug_csv,
        arguments.nis_summary,
        arguments.high_nis_segments,
    )
    paths = save_dynamic_analysis(
        arguments.output_dir, summary, increment_rows, lag_rows, key_rows
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json: {paths.summary_json}")
    print(f"increment_csv: {paths.increment_csv}")
    print(f"lag_sweep_csv: {paths.lag_sweep_csv}")
    print(f"key_segments_csv: {paths.key_segments_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imu", type=Path, default=PROJECT_ROOT / "data/processed/imu_processed.npz"
    )
    parser.add_argument(
        "--pose", type=Path, default=PROJECT_ROOT / "data/processed/pose_processed.npz"
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=PROJECT_ROOT / "data/processed/match_table.npz",
    )
    parser.add_argument(
        "--init-stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/init_stats.npz",
    )
    parser.add_argument(
        "--result-csv", type=Path, default=PROJECT_ROOT / "results/eskf6d_result.csv"
    )
    parser.add_argument(
        "--debug-csv", type=Path, default=PROJECT_ROOT / "results/eskf6d_debug.csv"
    )
    parser.add_argument(
        "--nis-summary",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/nis_analysis_summary.json",
    )
    parser.add_argument(
        "--high-nis-segments",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/high_nis_segments.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/analysis"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
