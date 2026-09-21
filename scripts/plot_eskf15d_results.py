"""Plot position and ZYX roll/pitch/yaw from the 15D result CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DPI = 200
LINE_COLOR = "#1F4E79"
GRID_COLOR = "#D1D5DB"

PLOT_SPECS = (
    ("px_m", "Position X (m)", "15D ESKF Position X - Time", "x_time.png"),
    ("py_m", "Position Y (m)", "15D ESKF Position Y - Time", "y_time.png"),
    ("pz_m", "Position Z (m)", "15D ESKF Position Z - Time", "z_time.png"),
    ("roll_rad", "Roll (rad)", "15D ESKF Roll - Time", "roll_time.png"),
    ("pitch_rad", "Pitch (rad)", "15D ESKF Pitch - Time", "pitch_time.png"),
    ("yaw_rad", "Yaw (rad)", "15D ESKF Yaw - Time", "yaw_time.png"),
)


def _load_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc

    if len(rows) < 2:
        raise ValueError("result CSV needs at least two rows")

    required = {"timestamp", "imu_index", *(spec[0] for spec in PLOT_SPECS)}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"result CSV is missing columns: {sorted(missing)}")

    try:
        timestamp = np.asarray(
            [float(row["timestamp"]) for row in rows], dtype=float
        )
        imu_index = np.asarray(
            [int(row["imu_index"]) for row in rows], dtype=np.int64
        )
        values = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field, _, _, _ in PLOT_SPECS
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("result CSV contains an invalid plotting value") from exc

    if not np.all(np.isfinite(timestamp)):
        raise ValueError("timestamp must contain only finite values")
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError("timestamp must be strictly increasing")
    for field, array in values.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must contain only finite values")
    return timestamp, imu_index, values


def _save_plot(
    time_s: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 4.5))
    axis.plot(time_s, values, color=LINE_COLOR, linewidth=1.1)
    axis.set_title(title, fontsize=13, pad=10)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(float(time_s[0]), float(time_s[-1]))
    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_plots(result_csv: Path, output_directory: Path) -> dict[str, object]:
    timestamp, _, values = _load_result(result_csv)
    time_s = timestamp - timestamp[0]
    output_paths = {}
    for field, ylabel, title, filename in PLOT_SPECS:
        path = output_directory / filename
        _save_plot(time_s, values[field], ylabel, title, path)
        output_paths[field] = str(path.resolve())
    # Preserve raw yaw values, including the +/-pi representation boundary.
    return {
        "row_count": int(timestamp.size), "start_time_s": float(time_s[0]),
        "end_time_s": float(time_s[-1]), "figure_paths": output_paths,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "eskf15d_result.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures15d",
    )
    arguments = parser.parse_args()
    summary = generate_plots(arguments.result_csv, arguments.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
