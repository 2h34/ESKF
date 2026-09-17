"""Compare R_WB and R_BW interpretations against static accelerometer data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GRAVITY_WORLD_MPS2
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData
from processed_io import (
    load_alignment,
    load_init_stats,
    load_processed_imu,
    load_processed_pose,
)
from rotation_utils import normalize_quaternion, quaternion_to_rotation_matrix


FloatArray = NDArray[np.float64]

STRONG_MEDIAN_MARGIN_DEG = 0.5
STRONG_WIN_FRACTION = 0.75


@dataclass(frozen=True)
class HypothesisStats:
    angular_error_deg: FloatArray
    euclidean_residual_mps2: FloatArray
    mean_angular_error_deg: float
    median_angular_error_deg: float
    max_angular_error_deg: float
    mean_euclidean_residual_mps2: float
    median_euclidean_residual_mps2: float
    max_euclidean_residual_mps2: float


@dataclass(frozen=True)
class PoseConventionCheckResult:
    static_matched_sample_count: int
    excluded_nonpositive_attitude_covariance_count: int
    used_sample_count: int
    hypothesis_a: HypothesisStats
    hypothesis_b: HypothesisStats
    hypothesis_a_better_fraction: float
    representative_quaternion_xyzw: FloatArray
    static_acc_mean_mps2: FloatArray
    representative_prediction_a_mps2: FloatArray
    representative_prediction_b_mps2: FloatArray
    representative_angle_a_deg: float
    representative_angle_b_deg: float
    conclusion: str


def _direction_angle_deg(measured: FloatArray, predicted: FloatArray) -> FloatArray:
    measured_norm = np.linalg.norm(measured, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    if np.any(measured_norm <= 0.0) or np.any(predicted_norm <= 0.0):
        raise ValueError("direction comparison requires non-zero vectors")
    cosine = np.sum(measured * predicted, axis=1) / (measured_norm * predicted_norm)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _summarize(measured: FloatArray, predicted: FloatArray) -> HypothesisStats:
    angles = _direction_angle_deg(measured, predicted)
    residual = np.linalg.norm(measured - predicted, axis=1)
    return HypothesisStats(
        angular_error_deg=angles,
        euclidean_residual_mps2=residual,
        mean_angular_error_deg=float(np.mean(angles)),
        median_angular_error_deg=float(np.median(angles)),
        max_angular_error_deg=float(np.max(angles)),
        mean_euclidean_residual_mps2=float(np.mean(residual)),
        median_euclidean_residual_mps2=float(np.median(residual)),
        max_euclidean_residual_mps2=float(np.max(residual)),
    )


def _average_quaternion(quaternion_xyzw: FloatArray) -> FloatArray:
    if quaternion_xyzw.ndim != 2 or quaternion_xyzw.shape[1] != 4:
        raise ValueError("quaternion_xyzw must have shape (N, 4)")
    accumulator = quaternion_xyzw.T @ quaternion_xyzw
    _, eigenvectors = np.linalg.eigh(accumulator)
    average = eigenvectors[:, -1]
    if np.dot(average, quaternion_xyzw[0]) < 0.0:
        average = -average
    return normalize_quaternion(average)


def _single_angle_deg(measured: FloatArray, predicted: FloatArray) -> float:
    return float(_direction_angle_deg(measured[None, :], predicted[None, :])[0])


def check_pose_convention(
    imu: ProcessedIMUData,
    pose: ProcessedPoseData,
    init_stats: InitStats,
    alignment: AlignmentResult,
) -> PoseConventionCheckResult:
    """Evaluate both quaternion-direction hypotheses on the initial static range."""

    static_imu_index = np.arange(
        init_stats.static_start_idx, init_stats.static_end_idx, dtype=np.int64
    )
    matched_pose_index = alignment.pose_index_for_imu[static_imu_index]
    matched = matched_pose_index >= 0
    static_imu_index = static_imu_index[matched]
    matched_pose_index = matched_pose_index[matched]
    static_matched_count = int(matched_pose_index.size)
    if static_matched_count == 0:
        raise ValueError("the static interval contains no matched pose observations")

    attitude_covariance = pose.cov_diag[matched_pose_index, 3:6]
    usable_covariance = np.all(np.isfinite(attitude_covariance), axis=1) & np.all(
        attitude_covariance > 0.0, axis=1
    )
    excluded_count = int(np.count_nonzero(~usable_covariance))
    static_imu_index = static_imu_index[usable_covariance]
    matched_pose_index = matched_pose_index[usable_covariance]
    if matched_pose_index.size == 0:
        raise ValueError("no static matches remain after covariance validity checks")

    measured_acc = imu.acc_mps2[static_imu_index]
    quaternions = pose.quaternion_xyzw[matched_pose_index]
    rotation = np.asarray(
        [quaternion_to_rotation_matrix(value) for value in quaternions], dtype=float
    )
    predicted_a = np.asarray(
        [-matrix.T @ GRAVITY_WORLD_MPS2 for matrix in rotation], dtype=float
    )
    predicted_b = np.asarray(
        [-matrix @ GRAVITY_WORLD_MPS2 for matrix in rotation], dtype=float
    )
    stats_a = _summarize(measured_acc, predicted_a)
    stats_b = _summarize(measured_acc, predicted_b)
    a_better_fraction = float(
        np.mean(stats_a.angular_error_deg < stats_b.angular_error_deg)
    )

    representative_quaternion = _average_quaternion(quaternions)
    representative_rotation = quaternion_to_rotation_matrix(representative_quaternion)
    representative_a = -representative_rotation.T @ GRAVITY_WORLD_MPS2
    representative_b = -representative_rotation @ GRAVITY_WORLD_MPS2
    representative_angle_a = _single_angle_deg(init_stats.acc_mean, representative_a)
    representative_angle_b = _single_angle_deg(init_stats.acc_mean, representative_b)

    median_margin = (
        stats_b.median_angular_error_deg - stats_a.median_angular_error_deg
    )
    if (
        median_margin >= STRONG_MEDIAN_MARGIN_DEG
        and a_better_fraction >= STRONG_WIN_FRACTION
    ):
        conclusion = "Strongly supports R_WB"
    elif (
        median_margin <= -STRONG_MEDIAN_MARGIN_DEG
        and a_better_fraction <= 1.0 - STRONG_WIN_FRACTION
    ):
        conclusion = "Strongly supports R_BW"
    else:
        conclusion = "Still ambiguous"

    return PoseConventionCheckResult(
        static_matched_sample_count=static_matched_count,
        excluded_nonpositive_attitude_covariance_count=excluded_count,
        used_sample_count=int(matched_pose_index.size),
        hypothesis_a=stats_a,
        hypothesis_b=stats_b,
        hypothesis_a_better_fraction=a_better_fraction,
        representative_quaternion_xyzw=representative_quaternion,
        static_acc_mean_mps2=init_stats.acc_mean,
        representative_prediction_a_mps2=representative_a,
        representative_prediction_b_mps2=representative_b,
        representative_angle_a_deg=representative_angle_a,
        representative_angle_b_deg=representative_angle_b,
        conclusion=conclusion,
    )


def _format_hypothesis(name: str, stats: HypothesisStats) -> list[str]:
    return [
        name,
        f"  mean angular error: {stats.mean_angular_error_deg:.9f} deg",
        f"  median angular error: {stats.median_angular_error_deg:.9f} deg",
        f"  max angular error: {stats.max_angular_error_deg:.9f} deg",
        f"  mean Euclidean residual: {stats.mean_euclidean_residual_mps2:.9f} m/s^2",
        f"  median Euclidean residual: {stats.median_euclidean_residual_mps2:.9f} m/s^2",
        f"  max Euclidean residual: {stats.max_euclidean_residual_mps2:.9f} m/s^2",
    ]


def format_report(result: PoseConventionCheckResult) -> str:
    lines = [
        "FAST-LIO quaternion direction gravity sanity check",
        "",
        f"Static matched samples before covariance exclusion: {result.static_matched_sample_count}",
        "Excluded samples with non-positive attitude covariance: "
        f"{result.excluded_nonpositive_attitude_covariance_count}",
        f"Static matched samples used: {result.used_sample_count}",
        "",
        *_format_hypothesis("Hypothesis A: q = R_WB", result.hypothesis_a),
        "",
        *_format_hypothesis("Hypothesis B: q = R_BW", result.hypothesis_b),
        "",
        f"Fraction of samples where A has lower angular error: {result.hypothesis_a_better_fraction:.9%}",
        "Strong-evidence rule: median angular-error margin >= "
        f"{STRONG_MEDIAN_MARGIN_DEG:.3f} deg and per-sample win fraction >= "
        f"{STRONG_WIN_FRACTION:.1%}.",
        "",
        "Representative static-orientation check",
        "  mean quaternion xyzw: "
        + np.array2string(result.representative_quaternion_xyzw, precision=9),
        "  measured static acc mean: "
        + np.array2string(result.static_acc_mean_mps2, precision=9)
        + " m/s^2",
        "  predicted A: "
        + np.array2string(result.representative_prediction_a_mps2, precision=9)
        + " m/s^2",
        "  predicted B: "
        + np.array2string(result.representative_prediction_b_mps2, precision=9)
        + " m/s^2",
        f"  angular error A/B: {result.representative_angle_a_deg:.9f} / {result.representative_angle_b_deg:.9f} deg",
        "",
        f"Conclusion: {result.conclusion}",
        "Gravity constrains roll/pitch direction but does not independently verify yaw semantics.",
    ]
    return "\n".join(lines) + "\n"


def run(processed_directory: Path) -> PoseConventionCheckResult:
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
    result = check_pose_convention(imu, pose, init_stats, alignment)
    print(format_report(result), end="")
    return result


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
