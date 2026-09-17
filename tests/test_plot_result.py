import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_result import generate_required_figures, prepare_rpy_data, save_rpy_csv


class PlotResultTests(unittest.TestCase):
    def _synthetic_data(self):
        return prepare_rpy_data(
            timestamp=np.array([100.0, 100.01, 100.025]),
            imu_index=np.array([10, 11, 12], dtype=np.int64),
            roll_rad=np.array([0.1, 0.2, 0.3]),
            pitch_rad=np.array([-0.1, -0.2, -0.3]),
            yaw_rad=np.deg2rad(np.array([178.0, 179.0, -179.0])),
        )

    def test_relative_time_starts_at_zero_and_preserves_timestamp_differences(self):
        data = self._synthetic_data()
        self.assertEqual(float(data.time_s[0]), 0.0)
        np.testing.assert_allclose(data.time_s, [0.0, 0.01, 0.025])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            prepare_rpy_data(
                [1.0, 1.0],
                np.array([0, 1], dtype=np.int64),
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            )

    def test_yaw_unwrap_turns_179_to_minus_179_into_181_degrees(self):
        data = prepare_rpy_data(
            [1.0, 2.0],
            np.array([0, 1], dtype=np.int64),
            [0.0, 0.0],
            [0.0, 0.0],
            np.deg2rad([179.0, -179.0]),
        )
        np.testing.assert_allclose(data.yaw_unwrapped_deg, [179.0, 181.0])

    def test_csv_preserves_raw_radians_elementwise(self):
        data = self._synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rpy_result.csv"
            save_rpy_csv(output, data)
            with output.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        np.testing.assert_array_equal(
            [float(row["roll_rad"]) for row in rows], data.roll_rad
        )
        np.testing.assert_array_equal(
            [float(row["pitch_rad"]) for row in rows], data.pitch_rad
        )
        np.testing.assert_array_equal(
            [float(row["yaw_raw_rad"]) for row in rows], data.yaw_raw_rad
        )

    def test_required_headless_figures_are_nonempty(self):
        data = self._synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_required_figures(data, Path(directory))
            for name in ("roll", "pitch", "yaw"):
                self.assertTrue(paths[name].is_file())
                self.assertGreater(paths[name].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
