"""Tests for unmix.matrix – linear algebra helpers."""

import math
import unittest

from unmix.matrix import (
    frobenius_norm,
    mat_add,
    mat_mul,
    mat_scale,
    mat_sub,
    shape,
    soft_threshold,
    svd,
    transpose,
    zeros,
)


class TestBasicOps(unittest.TestCase):
    def test_zeros(self):
        m = zeros(3, 4)
        self.assertEqual(shape(m), (3, 4))
        self.assertTrue(all(v == 0.0 for row in m for v in row))

    def test_transpose(self):
        m = [[1, 2, 3], [4, 5, 6]]
        mt = transpose(m)
        self.assertEqual(shape(mt), (3, 2))
        self.assertEqual(mt[0], [1, 4])

    def test_add_sub(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        s = mat_add(a, b)
        self.assertEqual(s, [[6, 8], [10, 12]])
        d = mat_sub(a, b)
        self.assertEqual(d, [[-4, -4], [-4, -4]])

    def test_scale(self):
        m = [[1, 2], [3, 4]]
        self.assertEqual(mat_scale(m, 2), [[2, 4], [6, 8]])

    def test_matmul(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        c = mat_mul(a, b)
        self.assertEqual(c, [[19, 22], [43, 50]])

    def test_frobenius(self):
        m = [[3, 4]]
        self.assertAlmostEqual(frobenius_norm(m), 5.0)


class TestSoftThreshold(unittest.TestCase):
    def test_positive(self):
        m = [[5.0]]
        r = soft_threshold(m, 2.0)
        self.assertAlmostEqual(r[0][0], 3.0)

    def test_below_threshold(self):
        m = [[1.0]]
        r = soft_threshold(m, 2.0)
        self.assertAlmostEqual(r[0][0], 0.0)

    def test_negative(self):
        m = [[-5.0]]
        r = soft_threshold(m, 2.0)
        self.assertAlmostEqual(r[0][0], -3.0)


class TestSVD(unittest.TestCase):
    def test_identity(self):
        I = [[1, 0], [0, 1]]
        U, sigma, Vt = svd(I)
        for s in sigma:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_reconstruction(self):
        A = [[1, 2], [3, 4], [5, 6]]
        U, sigma, Vt = svd(A)
        m, n = shape(A)
        k = len(sigma)
        recon = zeros(m, n)
        for i in range(m):
            for j in range(n):
                for d in range(k):
                    recon[i][j] += U[i][d] * sigma[d] * Vt[d][j]
        for i in range(m):
            for j in range(n):
                self.assertAlmostEqual(recon[i][j], A[i][j], places=4,
                                       msg=f"({i},{j})")


if __name__ == "__main__":
    unittest.main()
