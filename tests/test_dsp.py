"""Tests for unmix.dsp – FFT, STFT, iSTFT."""

import cmath
import math
import unittest

from unmix.dsp import fft, hann_window, ifft, istft, stft


class TestFFT(unittest.TestCase):
    """Verify the FFT against known DFT results."""

    def _naive_dft(self, x):
        """Reference O(N²) DFT."""
        n = len(x)
        return [
            sum(x[k] * cmath.exp(-2j * cmath.pi * k * m / n) for k in range(n))
            for m in range(n)
        ]

    def _assert_spectra_close(self, a, b, places=6):
        self.assertEqual(len(a), len(b))
        for i, (ai, bi) in enumerate(zip(a, b)):
            self.assertAlmostEqual(abs(ai - bi), 0.0, places=places,
                                   msg=f"bin {i}")

    def test_power_of_two(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        self._assert_spectra_close(fft(x), self._naive_dft(x))

    def test_non_power_of_two(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        self._assert_spectra_close(fft(x), self._naive_dft(x), places=4)

    def test_single_element(self):
        self.assertEqual(len(fft([42.0])), 1)
        self.assertAlmostEqual(fft([42.0])[0].real, 42.0)

    def test_empty(self):
        self.assertEqual(fft([]), [])


class TestIFFT(unittest.TestCase):
    def test_round_trip(self):
        x = [1, -2, 3, -4, 5, -6, 7, -8]
        X = fft(x)
        y = ifft(X)
        for i in range(len(x)):
            self.assertAlmostEqual(y[i].real, x[i], places=6)


class TestSTFTRoundTrip(unittest.TestCase):
    """STFT → iSTFT should approximately reconstruct the original signal."""

    def test_sine_reconstruction(self):
        sr = 8000
        n = 4096
        sig = [math.sin(2 * math.pi * 440 * i / sr) for i in range(n)]
        frame_size = 512
        hop_size = 128
        frames = stft(sig, frame_size, hop_size)
        rec = istft(frames, frame_size, hop_size, length=n)
        # Allow some boundary error; check interior
        start = frame_size
        end = n - frame_size
        for i in range(start, end):
            self.assertAlmostEqual(rec[i], sig[i], places=2,
                                   msg=f"sample {i}")


class TestHannWindow(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(hann_window(256)), 256)

    def test_endpoints_near_zero(self):
        w = hann_window(256)
        self.assertAlmostEqual(w[0], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
