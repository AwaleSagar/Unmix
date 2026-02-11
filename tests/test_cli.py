"""Tests for unmix.cli – argument parsing and sub-commands."""

import math
import os
import tempfile
import unittest

from unmix.audio import write_wav
from unmix.cli import build_parser, main


def _make_test_wav(path, sr=8000, duration_samples=4096):
    """Create a small mono WAV test fixture."""
    sig = [0.5 * math.sin(2 * math.pi * 440 * i / sr) for i in range(duration_samples)]
    write_wav(path, [sig], sr)


class TestCLIParsing(unittest.TestCase):
    def test_version(self):
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_command_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)


class TestCLISubcommands(unittest.TestCase):
    """Integration tests that run each sub-command on a tiny WAV file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wav_path = os.path.join(self.tmpdir, "test.wav")
        _make_test_wav(self.wav_path)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_hpss(self):
        main(["hpss", self.wav_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "test_harmonic.wav")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "test_percussive.wav")))

    def test_spectral_sub(self):
        main(["spectral-sub", self.wav_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "test_clean.wav")))

    def test_rpca(self):
        main(["rpca", self.wav_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128", "--max-iter", "3"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "test_background.wav")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "test_foreground.wav")))


if __name__ == "__main__":
    unittest.main()
