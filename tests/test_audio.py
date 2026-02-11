"""Tests for unmix.audio – WAV I/O."""

import math
import os
import tempfile
import unittest

from unmix.audio import read_wav, to_mono, write_wav


class TestWavRoundTrip(unittest.TestCase):
    """Write a WAV and read it back; values should survive the round-trip."""

    def _round_trip(self, channels, sample_rate=44100, sampwidth=2):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            write_wav(path, channels, sample_rate, sampwidth)
            out_ch, out_sr, out_nc = read_wav(path)
            self.assertEqual(out_sr, sample_rate)
            self.assertEqual(out_nc, len(channels))
            for c in range(len(channels)):
                self.assertEqual(len(out_ch[c]), len(channels[c]))
                for i in range(len(channels[c])):
                    self.assertAlmostEqual(
                        out_ch[c][i], channels[c][i], places=3,
                        msg=f"ch={c} sample={i}",
                    )
        finally:
            os.unlink(path)

    def test_mono_silence(self):
        self._round_trip([[0.0] * 100])

    def test_mono_sine(self):
        sr = 16000
        freq = 440
        n = sr  # 1 second
        sine = [0.5 * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]
        self._round_trip([sine], sr)

    def test_stereo(self):
        n = 200
        left = [0.3 * math.sin(2 * math.pi * 5 * i / n) for i in range(n)]
        right = [0.6 * math.sin(2 * math.pi * 10 * i / n) for i in range(n)]
        self._round_trip([left, right], 22050)


class TestToMono(unittest.TestCase):
    def test_averages_channels(self):
        left = [1.0, 0.0, -1.0]
        right = [0.0, 1.0, 0.5]
        mono = to_mono([left, right])
        self.assertAlmostEqual(mono[0], 0.5)
        self.assertAlmostEqual(mono[1], 0.5)
        self.assertAlmostEqual(mono[2], -0.25)


if __name__ == "__main__":
    unittest.main()
