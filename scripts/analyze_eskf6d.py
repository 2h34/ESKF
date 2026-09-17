"""Analyze saved 6D ESKF NIS/innovation diagnostics without rerunning the filter."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHI2_DF3_95 = 7.8147279
CHI2_DF3_99 = 11.3448667
TAIL_THRESHOLDS = (20.0, 50.0, 100.0)
TOP_FRAME_COUNT = 30
LOG_EPSILON = 1e-15
NIS_RECONSTRUCTION_RTOL = 1e-10
NIS_RECONSTRUCTION_ATOL = 1e-12


FloatArray = NDArray[np.float64]


TOP_FRAME_FIELDNAMES = (
    "rank",
    "imu_index",
    "timestamp",
    "pose_index",
    "dt",
    "NIS",
    "residual_norm",
    "r_theta_x",
    "r_theta_y",
    "r_theta_z",
    "standardized_diag_x",
    "standardized_diag_y",
    "standardized_diag_z",
    "R_x",
    "R_y",
    "R_z",
    "P_pred_theta_x",
    "P_pred_theta_y",
    "P_pred_theta_z",
    "S_xx",
    "S_yy",
    "S_zz",
    "rho_xy",
    "rho_xz",
    "rho_yz",
    "K_theta_x",
    "K_theta_y",
    "K_theta_z",
    "K_bg_x",
    "K_bg_y",
    "K_bg_z",
    "omega_hat_norm",
    "bg_x",
    "bg_y",
    "bg_z",
    "delta_bg_per_frame_x",
    "delta_bg_per_frame_y",
    "delta_bg_per_frame_z",
)


SEGMENT_FIELDNAMES = (
    "segment_id",
    "start_imu_index",
    "end_imu_index",
    "start_timestamp",
    "end_timestamp",
    "frame_count",
    "max_NIS",
    "median_NIS",
    "max_residual_norm",
    "median_residual_norm",
    "median_R_x",
    "median_R_y",
    "median_R_z",
    "median_P_pred_theta_x",
    "median_P_pred_theta_y",
    "median_P_pred_theta_z",
    "median_S_xx",
    "median_S_yy",
    "median_S_zz",
    "median_abs_standardized_x",
    "median_abs_standardized_y",
    "median_abs_standardized_z",
    "median_K_theta_x",
    "median_K_theta_y",
    "median_K_theta_z",
    "median_abs_K_bg_x",
    "median_abs_K_bg_y",
    "median_abs_K_bg_z",
    "median_omega_hat_norm",
)


@dataclass(frozen=True)
class AnalysisOutputPaths:
    summary_json: Path
    top_nis_csv: Path
    top_residual_csv: Path
    high_nis_segments_csv: Path


@dataclass(frozen=True)
class AnalysisResult:
    summary: dict[str, Any]
    top_nis_rows: tuple[dict[str, Any], ...]
    top_residual_rows: tuple[dict[str, Any], ...]
    segment_rows: tuple[dict[str, Any], ...]


def reconstruct_nis(
    innovation_covariance: ArrayLike,
    residual: ArrayLike,
) -> tuple[float, float, FloatArray]:
    """Return solve-based NIS, Cholesky whitened norm, and whitened coordinates."""

    S = np.asarray(innovation_covariance, dtype=float)
    r = np.asarray(residual, dtype=float)
    if S.shape != (3, 3) or r.shape != (3,):
        raise ValueError("S and residual must have shapes (3, 3) and (3,)")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(r)):
        raise ValueError("S and residual must contain only finite values")
    if not np.allclose(S, S.T, rtol=0.0, atol=1e-15):
        raise ValueError("S must be symmetric")
    try:
        L = np.linalg.cholesky(S)
        innovation_solution = np.linalg.solve(S, r)
        whitened = np.linalg.solve(L, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("S must be positive definite and solvable") from exc
    nis = float(r @ innovation_solution)
    whitened_norm = float(whitened @ whitened)
    return nis, whitened_norm, whitened


def segment_contiguous_indices(indices: Sequence[int]) -> list[tuple[int, int]]:
    """Return half-open record-position ranges for strictly increasing indices."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if values.size == 0:
        return []
    if np.any(np.diff(values) <= 0):
        raise ValueError("indices must be strictly increasing and unique")
    split_positions = np.flatnonzero(np.diff(values) != 1) + 1
    starts = np.concatenate((np.array([0]), split_positions))
    ends = np.concatenate((split_positions, np.array([values.size])))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


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


def _require_columns(
    rows: Sequence[Mapping[str, str]],
    required: set[str],
    label: str,
) -> None:
    if not rows:
        raise ValueError(f"{label} contains no data rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


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
    try:
        value = row[key]
    except KeyError as exc:
        raise ValueError(f"missing boolean column {key}") from exc
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
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def _pearson_correlation(x: ArrayLike, y: ArrayLike) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        raise ValueError("correlation inputs must be matching vectors of length >= 2")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("correlation inputs must be finite")
    if np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        raise ValueError("correlation is undefined for a constant vector")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _geometric_mean_three(values: ArrayLike, label: str) -> float:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite length-three vector")
    if np.any(vector <= 0.0):
        raise ValueError(f"{label} entries must be strictly positive")
    return float(np.exp(np.mean(np.log(vector))))


def _result_bias_maps(
    result_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[int, FloatArray], dict[int, FloatArray], dict[int, float]]:
    required = {
        "imu_index",
        "timestamp",
        "bg_x_rad_s",
        "bg_y_rad_s",
        "bg_z_rad_s",
    }
    _require_columns(result_rows, required, "eskf6d_result.csv")

    biases: dict[int, FloatArray] = {}
    delta_biases: dict[int, FloatArray] = {}
    timestamps: dict[int, float] = {}
    previous_index: int | None = None
    previous_bias: FloatArray | None = None
    previous_timestamp: float | None = None
    for row in result_rows:
        imu_index = _integer(row, "imu_index")
        timestamp = _finite_float(row, "timestamp")
        bias = np.array(
            [
                _finite_float(row, "bg_x_rad_s"),
                _finite_float(row, "bg_y_rad_s"),
                _finite_float(row, "bg_z_rad_s"),
            ],
            dtype=float,
        )
        if imu_index in biases:
            raise ValueError(f"duplicate result IMU index {imu_index}")
        if previous_index is not None:
            if imu_index != previous_index + 1:
                raise ValueError("result IMU indices must be consecutive")
            if timestamp <= float(previous_timestamp):
                raise ValueError("result timestamps must be strictly increasing")
            delta_biases[imu_index] = bias - np.asarray(previous_bias)
        biases[imu_index] = bias
        timestamps[imu_index] = timestamp
        previous_index = imu_index
        previous_bias = bias
        previous_timestamp = timestamp
    return biases, delta_biases, timestamps


def _update_records(
    debug_rows: Sequence[Mapping[str, str]],
    result_rows: Sequence[Mapping[str, str]],
    source_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float, float]:
    required = {
        "imu_index",
        "timestamp",
        "dt",
        "pose_index",
        "update_applied",
        "omega_hat_x",
        "omega_hat_y",
        "omega_hat_z",
        "r_theta_x",
        "r_theta_y",
        "r_theta_z",
        "residual_norm",
        "R_x",
        "R_y",
        "R_z",
        "P_pred_theta_x",
        "P_pred_theta_y",
        "P_pred_theta_z",
        "S_xx",
        "S_yy",
        "S_zz",
        "S_xy",
        "S_xz",
        "S_yz",
        "NIS",
        "K_theta_x",
        "K_theta_y",
        "K_theta_z",
        "K_bg_x",
        "K_bg_y",
        "K_bg_z",
    }
    _require_columns(debug_rows, required, "eskf6d_debug.csv")
    biases, delta_biases, result_timestamps = _result_bias_maps(result_rows)

    records: list[dict[str, Any]] = []
    maximum_nis_error = 0.0
    maximum_whitened_error = 0.0
    previous_debug_index: int | None = None
    for row in debug_rows:
        imu_index = _integer(row, "imu_index")
        if previous_debug_index is not None and imu_index != previous_debug_index + 1:
            raise ValueError("debug IMU indices must be consecutive")
        previous_debug_index = imu_index
        if not _boolean(row, "update_applied"):
            continue
        if imu_index not in biases or imu_index not in delta_biases:
            raise ValueError(f"result join is missing IMU index {imu_index}")

        timestamp = _finite_float(row, "timestamp")
        if not np.isclose(
            timestamp,
            result_timestamps[imu_index],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"result/debug timestamp mismatch at IMU index {imu_index}")

        residual = np.array(
            [
                _finite_float(row, "r_theta_x"),
                _finite_float(row, "r_theta_y"),
                _finite_float(row, "r_theta_z"),
            ],
            dtype=float,
        )
        R_diagonal = np.array(
            [
                _finite_float(row, "R_x"),
                _finite_float(row, "R_y"),
                _finite_float(row, "R_z"),
            ],
            dtype=float,
        )
        predicted_attitude_diagonal = np.array(
            [
                _finite_float(row, "P_pred_theta_x"),
                _finite_float(row, "P_pred_theta_y"),
                _finite_float(row, "P_pred_theta_z"),
            ],
            dtype=float,
        )
        S = np.array(
            [
                [
                    _finite_float(row, "S_xx"),
                    _finite_float(row, "S_xy"),
                    _finite_float(row, "S_xz"),
                ],
                [
                    _finite_float(row, "S_xy"),
                    _finite_float(row, "S_yy"),
                    _finite_float(row, "S_yz"),
                ],
                [
                    _finite_float(row, "S_xz"),
                    _finite_float(row, "S_yz"),
                    _finite_float(row, "S_zz"),
                ],
            ],
            dtype=float,
        )
        S_diagonal = np.diag(S)
        saved_nis = _finite_float(row, "NIS")
        reconstructed_nis, whitened_norm, _ = reconstruct_nis(S, residual)
        nis_error = abs(reconstructed_nis - saved_nis)
        whitened_error = abs(whitened_norm - saved_nis)
        maximum_nis_error = max(maximum_nis_error, nis_error)
        maximum_whitened_error = max(maximum_whitened_error, whitened_error)
        if not np.isclose(
            reconstructed_nis,
            saved_nis,
            rtol=NIS_RECONSTRUCTION_RTOL,
            atol=NIS_RECONSTRUCTION_ATOL,
        ):
            raise RuntimeError(
                f"NIS reconstruction mismatch at IMU index {imu_index}: "
                f"saved={saved_nis:.16e}, reconstructed={reconstructed_nis:.16e}"
            )
        if not np.isclose(
            whitened_norm,
            saved_nis,
            rtol=NIS_RECONSTRUCTION_RTOL,
            atol=NIS_RECONSTRUCTION_ATOL,
        ):
            raise RuntimeError(
                f"whitened norm mismatch at IMU index {imu_index}: "
                f"saved={saved_nis:.16e}, whitened={whitened_norm:.16e}"
            )

        standardized = residual / np.sqrt(S_diagonal)
        rho = np.array(
            [
                S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]),
                S[0, 2] / np.sqrt(S[0, 0] * S[2, 2]),
                S[1, 2] / np.sqrt(S[1, 1] * S[2, 2]),
            ],
            dtype=float,
        )
        omega_hat = np.array(
            [
                _finite_float(row, "omega_hat_x"),
                _finite_float(row, "omega_hat_y"),
                _finite_float(row, "omega_hat_z"),
            ],
            dtype=float,
        )
        bias = biases[imu_index]
        delta_bias = delta_biases[imu_index]
        record: dict[str, Any] = {
            "imu_index": imu_index,
            "timestamp": timestamp,
            "pose_index": _integer(row, "pose_index"),
            "dt": _finite_float(row, "dt"),
            "NIS": saved_nis,
            "residual_norm": _finite_float(row, "residual_norm"),
            "r_theta_x": float(residual[0]),
            "r_theta_y": float(residual[1]),
            "r_theta_z": float(residual[2]),
            "standardized_diag_x": float(standardized[0]),
            "standardized_diag_y": float(standardized[1]),
            "standardized_diag_z": float(standardized[2]),
            "R_x": float(R_diagonal[0]),
            "R_y": float(R_diagonal[1]),
            "R_z": float(R_diagonal[2]),
            "P_pred_theta_x": float(predicted_attitude_diagonal[0]),
            "P_pred_theta_y": float(predicted_attitude_diagonal[1]),
            "P_pred_theta_z": float(predicted_attitude_diagonal[2]),
            "S_xx": float(S_diagonal[0]),
            "S_yy": float(S_diagonal[1]),
            "S_zz": float(S_diagonal[2]),
            "rho_xy": float(rho[0]),
            "rho_xz": float(rho[1]),
            "rho_yz": float(rho[2]),
            "K_theta_x": _finite_float(row, "K_theta_x"),
            "K_theta_y": _finite_float(row, "K_theta_y"),
            "K_theta_z": _finite_float(row, "K_theta_z"),
            "K_bg_x": _finite_float(row, "K_bg_x"),
            "K_bg_y": _finite_float(row, "K_bg_y"),
            "K_bg_z": _finite_float(row, "K_bg_z"),
            "omega_hat_norm": float(np.linalg.norm(omega_hat)),
            "bg_x": float(bias[0]),
            "bg_y": float(bias[1]),
            "bg_z": float(bias[2]),
            "delta_bg_per_frame_x": float(delta_bias[0]),
            "delta_bg_per_frame_y": float(delta_bias[1]),
            "delta_bg_per_frame_z": float(delta_bias[2]),
            "R_geometric_mean": _geometric_mean_three(R_diagonal, "R diagonal"),
            "Ppred_geometric_mean": _geometric_mean_three(
                predicted_attitude_diagonal, "predicted attitude P diagonal"
            ),
            "S_geometric_mean": _geometric_mean_three(S_diagonal, "S diagonal"),
        }
        records.append(record)

    if not records:
        raise ValueError("debug CSV contains no successful updates")
    expected_update_count = int(source_summary.get("successful_updates", -1))
    if len(records) != expected_update_count:
        raise ValueError(
            f"update count {len(records)} does not match source summary "
            f"{expected_update_count}"
        )
    if np.any(np.diff([record["imu_index"] for record in records]) <= 0):
        raise ValueError("successful update IMU indices must be strictly increasing")
    return records, maximum_nis_error, maximum_whitened_error


def _ranked_rows(
    records: Sequence[Mapping[str, Any]],
    sort_key: str,
) -> tuple[dict[str, Any], ...]:
    ranked = sorted(
        records,
        key=lambda record: (-float(record[sort_key]), int(record["imu_index"])),
    )[:TOP_FRAME_COUNT]
    return tuple(
        {
            "rank": rank,
            **{field: record[field] for field in TOP_FRAME_FIELDNAMES if field != "rank"},
        }
        for rank, record in enumerate(ranked, start=1)
    )


def _median(records: Sequence[Mapping[str, Any]], key: str, *, absolute: bool = False) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=float)
    if absolute:
        values = np.abs(values)
    return float(np.median(values))


def _group_comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normal = [record for record in records if record["NIS"] <= CHI2_DF3_95]
    high = [record for record in records if record["NIS"] > CHI2_DF3_99]

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not group:
            raise ValueError("normal/high comparison group is empty")
        return {
            "count": len(group),
            "median_residual_norm": _median(group, "residual_norm"),
            "median_abs_r_theta": {
                axis: _median(group, f"r_theta_{axis}", absolute=True)
                for axis in "xyz"
            },
            "median_R": {
                axis: _median(group, f"R_{axis}") for axis in "xyz"
            },
            "median_P_pred_theta": {
                axis: _median(group, f"P_pred_theta_{axis}") for axis in "xyz"
            },
            "median_S_diag": {
                axis: _median(group, f"S_{axis}{axis}") for axis in "xyz"
            },
            "median_abs_standardized_diag": {
                axis: _median(group, f"standardized_diag_{axis}", absolute=True)
                for axis in "xyz"
            },
            "median_K_theta": {
                axis: _median(group, f"K_theta_{axis}") for axis in "xyz"
            },
            "median_abs_K_bg": {
                axis: _median(group, f"K_bg_{axis}", absolute=True)
                for axis in "xyz"
            },
            "median_omega_hat_norm": _median(group, "omega_hat_norm"),
        }

    return {
        "definition": {
            "normal": f"NIS <= {CHI2_DF3_95}",
            "high": f"NIS > {CHI2_DF3_99}",
            "middle_excluded": True,
        },
        "normal": summarize(normal),
        "high": summarize(high),
    }


def _segment_rows(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    high = [record for record in records if record["NIS"] > CHI2_DF3_99]
    ranges = segment_contiguous_indices([record["imu_index"] for record in high])
    rows: list[dict[str, Any]] = []
    for segment_id, (start, end) in enumerate(ranges, start=1):
        segment = high[start:end]
        rows.append(
            {
                "segment_id": segment_id,
                "start_imu_index": int(segment[0]["imu_index"]),
                "end_imu_index": int(segment[-1]["imu_index"]),
                "start_timestamp": float(segment[0]["timestamp"]),
                "end_timestamp": float(segment[-1]["timestamp"]),
                "frame_count": len(segment),
                "max_NIS": max(float(record["NIS"]) for record in segment),
                "median_NIS": _median(segment, "NIS"),
                "max_residual_norm": max(
                    float(record["residual_norm"]) for record in segment
                ),
                "median_residual_norm": _median(segment, "residual_norm"),
                **{
                    f"median_R_{axis}": _median(segment, f"R_{axis}")
                    for axis in "xyz"
                },
                **{
                    f"median_P_pred_theta_{axis}": _median(
                        segment, f"P_pred_theta_{axis}"
                    )
                    for axis in "xyz"
                },
                **{
                    f"median_S_{axis}{axis}": _median(
                        segment, f"S_{axis}{axis}"
                    )
                    for axis in "xyz"
                },
                **{
                    f"median_abs_standardized_{axis}": _median(
                        segment, f"standardized_diag_{axis}", absolute=True
                    )
                    for axis in "xyz"
                },
                **{
                    f"median_K_theta_{axis}": _median(
                        segment, f"K_theta_{axis}"
                    )
                    for axis in "xyz"
                },
                **{
                    f"median_abs_K_bg_{axis}": _median(
                        segment, f"K_bg_{axis}", absolute=True
                    )
                    for axis in "xyz"
                },
                "median_omega_hat_norm": _median(segment, "omega_hat_norm"),
            }
        )
    return tuple(rows)


def _exceedance(values: FloatArray, threshold: float) -> dict[str, float | int]:
    count = int(np.count_nonzero(values > threshold))
    return {
        "threshold": threshold,
        "count": count,
        "fraction": float(count / values.size),
    }


def analyze_saved_results(
    result_csv: str | Path,
    debug_csv: str | Path,
    summary_json: str | Path,
) -> AnalysisResult:
    """Read saved Phase 2.4A outputs and derive Phase 2.4B-1 evidence."""

    result_path = Path(result_csv)
    debug_path = Path(debug_csv)
    summary_path = Path(summary_json)
    result_rows = _read_csv(result_path)
    debug_rows = _read_csv(debug_path)
    source_summary = _read_json(summary_path)
    records, maximum_nis_error, maximum_whitened_error = _update_records(
        debug_rows, result_rows, source_summary
    )

    nis = np.asarray([record["NIS"] for record in records], dtype=float)
    residual_norm = np.asarray(
        [record["residual_norm"] for record in records], dtype=float
    )
    omega_hat_norm = np.asarray(
        [record["omega_hat_norm"] for record in records], dtype=float
    )
    correlations = np.asarray(
        [
            [record["rho_xy"], record["rho_xz"], record["rho_yz"]]
            for record in records
        ],
        dtype=float,
    )
    R_geometric_mean = np.asarray(
        [record["R_geometric_mean"] for record in records], dtype=float
    )
    Ppred_geometric_mean = np.asarray(
        [record["Ppred_geometric_mean"] for record in records], dtype=float
    )
    S_geometric_mean = np.asarray(
        [record["S_geometric_mean"] for record in records], dtype=float
    )

    top_nis_rows = _ranked_rows(records, "NIS")
    top_residual_rows = _ranked_rows(records, "residual_norm")
    segment_rows = _segment_rows(records)
    longest_segment = max(
        segment_rows,
        key=lambda row: (int(row["frame_count"]), float(row["max_NIS"])),
    )
    maximum_nis_segment = max(
        segment_rows,
        key=lambda row: float(row["max_NIS"]),
    )

    log_nis = np.log(nis + LOG_EPSILON)
    summary: dict[str, Any] = {
        "phase": "2.4B-1",
        "analysis_scope": (
            "只分析保存的 successful Update；卡方阈值仅作标准独立 Gaussian "
            "Kalman 假设下的 diagnostic reference，不用于 gating 或调参。"
        ),
        "update_count": len(records),
        "NIS_reference_thresholds": {
            "chi2_df3_95": CHI2_DF3_95,
            "chi2_df3_99": CHI2_DF3_99,
            "reference_only": True,
        },
        "exceedance_counts_and_fractions": {
            "NIS_gt_chi2_df3_95": _exceedance(nis, CHI2_DF3_95),
            "NIS_gt_chi2_df3_99": _exceedance(nis, CHI2_DF3_99),
            **{
                f"NIS_gt_{int(threshold)}": _exceedance(nis, threshold)
                for threshold in TAIL_THRESHOLDS
            },
        },
        "NIS_statistics": _statistics(nis),
        "residual_statistics": _statistics(residual_norm),
        "max_abs_NIS_reconstruction_error": maximum_nis_error,
        "max_abs_whitened_norm_error": maximum_whitened_error,
        "innovation_off_diagonal_correlation": {
            "max_abs": float(np.max(np.abs(correlations))),
            "median_abs": float(np.median(np.abs(correlations))),
            "pairs": ["rho_xy", "rho_xz", "rho_yz"],
        },
        "normal_vs_high_comparison": _group_comparison(records),
        "correlations": {
            "log1p_NIS_vs_omega_hat_norm": _pearson_correlation(
                np.log1p(nis), omega_hat_norm
            ),
            "log1p_NIS_vs_residual_norm": _pearson_correlation(
                np.log1p(nis), residual_norm
            ),
            "log_NIS_vs_log_residual_norm": _pearson_correlation(
                log_nis, np.log(residual_norm + LOG_EPSILON)
            ),
            "log_NIS_vs_log_R_geometric_mean": _pearson_correlation(
                log_nis, np.log(R_geometric_mean + LOG_EPSILON)
            ),
            "log_NIS_vs_log_Ppred_geometric_mean": _pearson_correlation(
                log_nis, np.log(Ppred_geometric_mean + LOG_EPSILON)
            ),
            "log_NIS_vs_log_S_geometric_mean": _pearson_correlation(
                log_nis, np.log(S_geometric_mean + LOG_EPSILON)
            ),
            "interpretation": "Pearson correlation is descriptive and does not establish causality.",
            "log_epsilon": LOG_EPSILON,
        },
        "standardized_diag_note": (
            "standardized_diag uses r_i/sqrt(S_ii) as an axis-wise diagnostic; "
            "it is not the Cholesky-whitened physical-axis contribution when S "
            "has off-diagonal terms."
        ),
        "whitened_innovation_note": (
            "Cholesky-whitened coordinates are used only to verify NIS; their "
            "components are not interpreted as Body x/y/z contributions."
        ),
        "FAST_LIO_dependence_note": (
            "FAST-LIO uses IMU data, so prediction and observation are not "
            "strictly independent."
        ),
        "top_NIS": {
            "imu_index": int(top_nis_rows[0]["imu_index"]),
            "timestamp": float(top_nis_rows[0]["timestamp"]),
            "NIS": float(top_nis_rows[0]["NIS"]),
        },
        "top_residual": {
            "imu_index": int(top_residual_rows[0]["imu_index"]),
            "timestamp": float(top_residual_rows[0]["timestamp"]),
            "residual_norm": float(top_residual_rows[0]["residual_norm"]),
        },
        "high_NIS_segment_count": len(segment_rows),
        "longest_high_NIS_segment": dict(longest_segment),
        "max_NIS_segment": dict(maximum_nis_segment),
    }
    return AnalysisResult(
        summary=summary,
        top_nis_rows=top_nis_rows,
        top_residual_rows=top_residual_rows,
        segment_rows=segment_rows,
    )


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_analysis(
    analysis: AnalysisResult,
    output_directory: str | Path,
) -> AnalysisOutputPaths:
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = AnalysisOutputPaths(
        summary_json=output_root / "nis_analysis_summary.json",
        top_nis_csv=output_root / "top_nis_frames.csv",
        top_residual_csv=output_root / "top_residual_frames.csv",
        high_nis_segments_csv=output_root / "high_nis_segments.csv",
    )
    _write_csv_atomic(paths.top_nis_csv, TOP_FRAME_FIELDNAMES, analysis.top_nis_rows)
    _write_csv_atomic(
        paths.top_residual_csv,
        TOP_FRAME_FIELDNAMES,
        analysis.top_residual_rows,
    )
    _write_csv_atomic(
        paths.high_nis_segments_csv,
        SEGMENT_FIELDNAMES,
        analysis.segment_rows,
    )

    temporary_summary = paths.summary_json.with_name(paths.summary_json.name + ".tmp")
    try:
        with temporary_summary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                analysis.summary,
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


def run(
    result_csv: Path,
    debug_csv: Path,
    summary_json: Path,
    output_directory: Path,
) -> dict[str, Any]:
    analysis = analyze_saved_results(result_csv, debug_csv, summary_json)
    paths = save_analysis(analysis, output_directory)
    print(json.dumps(analysis.summary, ensure_ascii=False, indent=2))
    print(f"summary_json: {paths.summary_json}")
    print(f"top_nis_csv: {paths.top_nis_csv}")
    print(f"top_residual_csv: {paths.top_residual_csv}")
    print(f"high_nis_segments_csv: {paths.high_nis_segments_csv}")
    return analysis.summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "eskf6d_result.csv",
    )
    parser.add_argument(
        "--debug-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "eskf6d_debug.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "eskf6d_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis",
    )
    arguments = parser.parse_args()
    run(
        arguments.result_csv,
        arguments.debug_csv,
        arguments.summary_json,
        arguments.output_dir,
    )


if __name__ == "__main__":
    main()
