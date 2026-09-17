import unittest

import numpy as np

from scripts.analyze_eskf6d import reconstruct_nis, segment_contiguous_indices


class Analysis6DTests(unittest.TestCase):
    """Minimal checks for the Phase 2.4B-1 analysis primitives."""

    def test_nis_reconstruction_matches_solve_and_cholesky_norm(self) -> None:
        S = np.array(
            [
                [2.0, 0.3, -0.1],
                [0.3, 1.5, 0.2],
                [-0.1, 0.2, 1.0],
            ],
            dtype=float,
        )
        residual = np.array([0.5, -0.2, 0.1], dtype=float)
        expected = float(residual @ np.linalg.solve(S, residual))

        reconstructed, whitened_norm, _ = reconstruct_nis(S, residual)

        self.assertAlmostEqual(reconstructed, expected, places=15)
        self.assertAlmostEqual(whitened_norm, expected, places=15)

    def test_contiguous_high_nis_indices_are_not_merged_across_gaps(self) -> None:
        ranges = segment_contiguous_indices([10, 11, 14, 15, 16, 20])
        self.assertEqual(ranges, [(0, 2), (2, 5), (5, 6)])


if __name__ == "__main__":
    unittest.main()
