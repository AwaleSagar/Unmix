"""Tests for unmix.cli – argument parsing and sub-commands."""

import math
import os
import subprocess
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


class TestCLIFormatConversion(unittest.TestCase):
    """Integration tests for CLI format conversion support."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wav_path = os.path.join(self.tmpdir, "test.wav")
        _make_test_wav(self.wav_path)
        
        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            self.ffmpeg_available = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.ffmpeg_available = False

        # Create MP3 test file if ffmpeg is available
        if self.ffmpeg_available:
            self.mp3_path = os.path.join(self.tmpdir, "test.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", self.wav_path, self.mp3_path],
                capture_output=True,
                check=True
            )

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_hpss_with_mp3_input(self):
        """Test HPSS with MP3 input produces MP3 output."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        main(["hpss", self.mp3_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128"])
        
        # Check that MP3 files were created
        harmonic_path = os.path.join(self.tmpdir, "test_harmonic.mp3")
        percussive_path = os.path.join(self.tmpdir, "test_percussive.mp3")
        
        self.assertTrue(os.path.isfile(harmonic_path))
        self.assertTrue(os.path.isfile(percussive_path))
        self.assertGreater(os.path.getsize(harmonic_path), 0)
        self.assertGreater(os.path.getsize(percussive_path), 0)

    def test_spectral_sub_with_mp3_input(self):
        """Test spectral subtraction with MP3 input produces MP3 output."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        main(["spectral-sub", self.mp3_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128"])
        
        clean_path = os.path.join(self.tmpdir, "test_clean.mp3")
        self.assertTrue(os.path.isfile(clean_path))
        self.assertGreater(os.path.getsize(clean_path), 0)

    def test_rpca_with_mp3_input(self):
        """Test RPCA with MP3 input produces MP3 output."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        main(["rpca", self.mp3_path, "-o", self.tmpdir,
              "--frame-size", "512", "--hop-size", "128", "--max-iter", "3"])
        
        bg_path = os.path.join(self.tmpdir, "test_background.mp3")
        fg_path = os.path.join(self.tmpdir, "test_foreground.mp3")
        
        self.assertTrue(os.path.isfile(bg_path))
        self.assertTrue(os.path.isfile(fg_path))
        self.assertGreater(os.path.getsize(bg_path), 0)
        self.assertGreater(os.path.getsize(fg_path), 0)
