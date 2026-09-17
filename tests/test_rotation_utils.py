import math
import unittest

import numpy as np

from rotation_utils import (
    hat,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    quaternion_to_rotvec,
    quaternion_to_rpy,
    rotvec_to_quaternion,
)


class RotationUtilsTests(unittest.TestCase):
    def test_hat_cross_product(self) -> None:
        a = np.array([1.2, -0.4, 0.7])
        b = np.array([-0.3, 0.9, 2.0])
        np.testing.assert_allclose(hat(a) @ b, np.cross(a, b))

    def test_identity(self) -> None:
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(quaternion_to_rotation_matrix(identity), np.eye(3))
        np.testing.assert_allclose(quaternion_to_rpy(identity), np.zeros(3))

    def test_inverse_product(self) -> None:
        q = normalize_quaternion([0.2, -0.3, 0.1, 0.9])
        product = quaternion_multiply(q, quaternion_inverse(q))
        np.testing.assert_allclose(product, [0.0, 0.0, 0.0, 1.0], atol=1e-14)

    def test_hamilton_product_matches_active_rotation_composition(self) -> None:
        q1 = rotvec_to_quaternion([0.2, 0.0, 0.0])
        q2 = rotvec_to_quaternion([0.0, -0.3, 0.0])
        composed = quaternion_multiply(q1, q2)
        np.testing.assert_allclose(
            quaternion_to_rotation_matrix(composed),
            quaternion_to_rotation_matrix(q1) @ quaternion_to_rotation_matrix(q2),
            atol=1e-14,
        )

    def test_zero_rotvec(self) -> None:
        np.testing.assert_allclose(rotvec_to_quaternion(np.zeros(3)), [0, 0, 0, 1])

    def test_rotvec_round_trip(self) -> None:
        for phi in (
            np.array([1e-6, -2e-6, 3e-6]),
            np.array([0.1, -0.2, 0.05]),
            np.array([-0.4, 0.3, 0.2]),
        ):
            np.testing.assert_allclose(
                quaternion_to_rotvec(rotvec_to_quaternion(phi)), phi, atol=1e-12
            )

    def test_axis_rotations(self) -> None:
        half = math.sqrt(0.5)
        cases = (
            (
                np.array([math.pi / 2, 0.0, 0.0]),
                np.array([half, 0.0, 0.0, half]),
                np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                np.array([math.pi / 2, 0.0, 0.0]),
            ),
            (
                np.array([0.0, math.pi / 2, 0.0]),
                np.array([0.0, half, 0.0, half]),
                np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                np.array([0.0, math.pi / 2, 0.0]),
            ),
            (
                np.array([0.0, 0.0, math.pi / 2]),
                np.array([0.0, 0.0, half, half]),
                np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                np.array([0.0, 0.0, math.pi / 2]),
            ),
        )
        for phi, expected_q, expected_rotation, expected_rpy in cases:
            q = rotvec_to_quaternion(phi)
            np.testing.assert_allclose(q, expected_q, atol=1e-12)
            np.testing.assert_allclose(
                quaternion_to_rotation_matrix(q), expected_rotation, atol=1e-12
            )
            np.testing.assert_allclose(quaternion_to_rpy(q), expected_rpy, atol=1e-12)

    def test_tiny_rotvec_is_finite(self) -> None:
        q = rotvec_to_quaternion([1e-10, 0.0, 0.0])
        self.assertTrue(np.all(np.isfinite(q)))
        recovered = quaternion_to_rotvec(q)
        self.assertTrue(np.all(np.isfinite(recovered)))
        np.testing.assert_allclose(recovered, [1e-10, 0.0, 0.0], atol=1e-20)

    def test_quaternion_sign_equivalence(self) -> None:
        q = normalize_quaternion([0.2, -0.1, 0.3, 0.9])
        np.testing.assert_allclose(
            quaternion_to_rotation_matrix(q), quaternion_to_rotation_matrix(-q)
        )
        np.testing.assert_allclose(quaternion_to_rotvec(q), quaternion_to_rotvec(-q))

    def test_invalid_quaternion_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_quaternion([0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
