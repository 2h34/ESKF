"""Read raw data, prepare the four arrays needed by the 15D ESKF."""

import csv
from dataclasses import fields
from pathlib import Path

import numpy as np

import config as cfg
from data_types import AlignmentResult, InitStats, ProcessedIMUData, ProcessedPoseData

IMU_COLUMNS = ("time", "ax", "ay", "az", "gx", "gy", "gz")
POSE_BASE_COLUMNS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")
POSE_COLUMNS = POSE_BASE_COLUMNS + tuple(f"cov_{r}{c}" for r in range(6) for c in range(6))
COV_DIAG_INDICES = tuple(len(POSE_BASE_COLUMNS) + 7 * i for i in range(6))


def _read_csv(path: Path, columns: tuple[str, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    rows, dropped = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(columns):
            raise ValueError(f"{path} columns do not match the expected CSV schema")
        for row_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in columns]
                if not np.all(np.isfinite(values)):
                    raise ValueError
            except (TypeError, ValueError):
                dropped.append(row_number)
            else:
                rows.append(values)
    if not rows:
        raise ValueError(f"{path} has no usable numeric rows")
    return np.asarray(rows), tuple(dropped)


def _timestamps(timestamp: np.ndarray, name: str) -> np.ndarray:
    if timestamp.ndim != 1 or timestamp.size < 2 or not np.all(np.isfinite(timestamp)):
        raise ValueError(f"{name} needs at least two finite timestamps")
    dt = np.diff(timestamp)
    if np.any(dt <= 0.0):
        raise ValueError(f"{name} timestamps must strictly increase")
    return dt


def load_imu_csv(path: Path) -> tuple[ProcessedIMUData, dict]:
    values, dropped = _read_csv(path, IMU_COLUMNS)
    timestamp = values[:, 0]
    differences = _timestamps(timestamp, "IMU")
    dt = np.r_[np.nan, differences]
    return ProcessedIMUData(timestamp, dt, values[:, [4, 5, 6]], values[:, [1, 2, 3]] * cfg.GRAVITY_MPS2), {
        "sample_count": int(timestamp.size), "dropped_rows": dropped,
        "median_dt": float(np.median(differences)),
    }


def load_pose_csv(path: Path) -> tuple[ProcessedPoseData, dict]:
    values, dropped = _read_csv(path, POSE_COLUMNS)
    timestamp = values[:, 0]
    _timestamps(timestamp, "pose")
    quaternion = values[:, 4:8]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("pose contains a zero quaternion")
    covariance = values[:, COV_DIAG_INDICES]
    if np.any(covariance < 0.0):
        raise ValueError("pose contains a negative covariance diagonal")
    return ProcessedPoseData(timestamp, values[:, 1:4], quaternion / norm[:, None], covariance), {
        "sample_count": int(timestamp.size), "dropped_rows": dropped,
    }


def analyze_initial_static_segment(imu: ProcessedIMUData) -> InitStats:
    stop = int(np.searchsorted(imu.timestamp, imu.timestamp[0] + cfg.CANDIDATE_DURATION_S, side="right"))
    if stop < 2:
        raise ValueError("initial static interval has fewer than two IMU samples")
    gyro, acc = imu.gyro_rad_s[:stop], imu.acc_mps2[:stop]
    gyro_mean, gyro_std = np.mean(gyro, axis=0), np.std(gyro, axis=0)
    acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
    acc_norm = np.linalg.norm(acc, axis=1)
    checks = (
        (np.linalg.norm(gyro_mean) <= cfg.MAX_GYRO_MEAN_NORM_RAD_S, "gyro mean"),
        (np.max(gyro_std) <= cfg.MAX_GYRO_STD_COMPONENT_RAD_S, "gyro variation"),
        (abs(np.mean(acc_norm) - cfg.GRAVITY_MPS2) <= cfg.MAX_ACC_NORM_ERROR_MPS2, "accelerometer norm"),
        (np.max(acc_std) <= cfg.MAX_ACC_STD_COMPONENT_MPS2, "accelerometer variation"),
        (np.std(acc_norm) <= cfg.MAX_ACC_NORM_STD_MPS2, "accelerometer norm variation"),
    )
    failed = tuple(name for passed, name in checks if not passed)
    if failed:
        raise ValueError("initial static check failed: " + ", ".join(failed))
    return InitStats(0, stop, float(imu.timestamp[0]), float(imu.timestamp[stop - 1]),
                     gyro_mean, gyro_std, acc_mean, acc_std,
                     float(np.median(imu.dt[1:])), True, tuple(name for _, name in checks))


def align_timestamps(imu_timestamp: np.ndarray, pose_timestamp: np.ndarray) -> AlignmentResult:
    imu_dt, pose_dt = _timestamps(imu_timestamp, "IMU"), _timestamps(pose_timestamp, "pose")
    tolerance = cfg.TOLERANCE_RATIO * max(np.median(imu_dt), np.median(pose_dt))
    matched = np.full(imu_timestamp.shape, -1, dtype=np.int64)
    cursor = 0
    for index, timestamp in enumerate(imu_timestamp):
        while cursor < pose_timestamp.size and pose_timestamp[cursor] < timestamp - tolerance:
            cursor += 1
        if cursor == pose_timestamp.size:
            break
        candidate = cursor
        if cursor + 1 < pose_timestamp.size and abs(pose_timestamp[cursor + 1] - timestamp) < abs(pose_timestamp[cursor] - timestamp):
            candidate += 1
        if abs(pose_timestamp[candidate] - timestamp) <= tolerance:
            matched[index] = candidate
            cursor = candidate + 1
    return AlignmentResult(matched)


def save_processed_data(directory: Path, imu: ProcessedIMUData, pose: ProcessedPoseData,
                        stats: InitStats, alignment: AlignmentResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in (("imu_processed.npz", imu), ("pose_processed.npz", pose),
                            ("init_stats.npz", stats), ("match_table.npz", alignment)):
        np.savez(directory / filename, **{item.name: getattr(value, item.name) for item in fields(value)})


def load_processed_imu(path: Path) -> ProcessedIMUData:
    with np.load(path) as data:
        return ProcessedIMUData(data["timestamp"], data["dt"], data["gyro_rad_s"], data["acc_mps2"])


def load_processed_pose(path: Path) -> ProcessedPoseData:
    with np.load(path) as data:
        return ProcessedPoseData(data["timestamp"], data["position"], data["quaternion_xyzw"], data["cov_diag"])


def load_init_stats(path: Path) -> InitStats:
    with np.load(path) as data:
        return InitStats(int(data["static_start_idx"]), int(data["static_end_idx"]),
                         float(data["static_start_time"]), float(data["static_end_time"]),
                         data["gyro_mean"], data["gyro_std"], data["acc_mean"], data["acc_std"],
                         float(data["median_dt"]), bool(data["is_static"]),
                         tuple(str(x) for x in data["decision_reasons"]))


def load_alignment(path: Path) -> AlignmentResult:
    with np.load(path) as data:
        return AlignmentResult(data["pose_index_for_imu"].astype(np.int64))
