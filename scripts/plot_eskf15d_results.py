"""Plot the required position and attitude curves from the result CSV."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PLOTS = (
    ("px_m", "Position X (m)", "x_time.png"), ("py_m", "Position Y (m)", "y_time.png"),
    ("pz_m", "Position Z (m)", "z_time.png"), ("roll_rad", "Roll (rad)", "roll_time.png"),
    ("pitch_rad", "Pitch (rad)", "pitch_time.png"), ("yaw_rad", "Yaw (rad)", "yaw_time.png"),
)


def generate_plots(result_csv: Path, output_directory: Path) -> dict:
    with result_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("result CSV needs at least two rows")
    time = np.asarray([float(row["timestamp"]) for row in rows])
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("result timestamps must increase")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field, ylabel, filename in PLOTS:
        values = np.asarray([float(row[field]) for row in rows])
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(time - time[0], values, linewidth=1.1)
        axis.set(xlabel="Time (s)", ylabel=ylabel, title=f"15D ESKF {ylabel} - Time")
        axis.grid(True)
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=200)
        plt.close(figure)
        paths[field] = str(path.resolve())
    return {"row_count": len(rows), "figure_paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-csv", type=Path, default=ROOT / "results" / "eskf15d_result.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures15d")
    args = parser.parse_args()
    print(generate_plots(args.result_csv, args.output_dir))


if __name__ == "__main__":
    main()
