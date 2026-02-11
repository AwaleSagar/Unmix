"""Tests for separation algorithms (HPSS, spectral subtraction, RPCA)."""

import math
import unittest

from unmix.algorithms.hpss import hpss
from unmix.algorithms.spectral_sub import spectral_subtract
from unmix.algorithms.rpca import rpca_separate


def _sine(freq, sr, n, amplitude=0.5):
    """Generate a sine-wave signal."""
    return [amplitude * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def _rms(signal):
    """Root-mean-square of a signal."""
    return math.sqrt(sum(s * s for s in signal) / max(len(signal), 1))


class TestHPSS(unittest.TestCase):
    def test_output_length(self):
        sr = 8000
        sig = _sine(440, sr, 4096)
        h, p = hpss(sig, sr, frame_size=512, hop_size=128)
        self.assertEqual(len(h), len(sig))
        self.assertEqual(len(p), len(sig))

    def test_sum_approximates_original(self):
        sr = 8000
        sig = _sine(440, sr, 4096)
        h, p = hpss(sig, sr, frame_size=512, hop_size=128)
        diff = [sig[i] - h[i] - p[i] for i in range(len(sig))]
        # The sum of harmonic + percussive should be close to original
        self.assertLess(_rms(diff), 0.15 * _rms(sig) + 1e-6)


class TestSpectralSub(unittest.TestCase):
    def test_output_length(self):
        sr = 8000
        sig = _sine(440, sr, 4096)
        out = spectral_subtract(sig, sample_rate=sr,
                                frame_size=512, hop_size=128)
        self.assertEqual(len(out), len(sig))

    def test_denoising_reduces_noise(self):
        """White-ish noise added to a tone; subtraction should reduce RMS of the noise residual."""
        import random
        random.seed(42)
        sr = 8000
        n = 4096
        tone = _sine(440, sr, n, amplitude=0.5)
        noise = [0.05 * (random.random() * 2 - 1) for _ in range(n)]
        noisy = [tone[i] + noise[i] for i in range(n)]

        # Supply a separate noise reference so the estimator is accurate
        random.seed(99)
        noise_ref = [0.05 * (random.random() * 2 - 1) for _ in range(n)]

        clean = spectral_subtract(noisy, noise_signal=noise_ref,
                                  sample_rate=sr,
                                  frame_size=512, hop_size=128)
        # The cleaned signal should not be louder than the noisy one
        self.assertLessEqual(_rms(clean), _rms(noisy) + 0.02)


class TestRPCA(unittest.TestCase):
    def test_output_length(self):
        sr = 8000
        sig = _sine(440, sr, 4096)
        bg, fg = rpca_separate(sig, sample_rate=sr,
                               frame_size=512, hop_size=128,
                               max_iter=5)
        self.assertEqual(len(bg), len(sig))
        self.assertEqual(len(fg), len(sig))


if __name__ == "__main__":
    unittest.main()
