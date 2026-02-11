"""Tests for audio format conversion functionality."""

import math
import os
import subprocess
import tempfile
import unittest

from unmix.audio import (
    convert_from_wav,
    convert_to_wav,
    detect_audio_format,
    read_wav,
    write_wav,
)


def _make_test_wav(path, sr=8000, duration_samples=4096):
    """Create a small mono WAV test fixture."""
    sig = [0.5 * math.sin(2 * math.pi * 440 * i / sr) for i in range(duration_samples)]
    write_wav(path, [sig], sr)


class TestFormatDetection(unittest.TestCase):
    """Tests for audio format detection."""

    def test_detect_wav_format(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_test_wav(path)
            fmt = detect_audio_format(path)
            self.assertEqual(fmt.lower(), "wav")
        finally:
            os.unlink(path)

    def test_detect_mp3_format_by_extension(self):
        # Just test extension-based detection
        # Use a portable path that doesn't need to exist
        fmt = detect_audio_format(os.path.join(tempfile.gettempdir(), "test.mp3"))
        self.assertEqual(fmt.lower(), "mp3")

    def test_detect_flac_format_by_extension(self):
        fmt = detect_audio_format(os.path.join(tempfile.gettempdir(), "test.flac"))
        self.assertEqual(fmt.lower(), "flac")

    def test_detect_no_extension(self):
        # File doesn't exist, should return None for files without extension
        fmt = detect_audio_format(os.path.join(tempfile.gettempdir(), "test_no_ext"))
        self.assertIsNone(fmt)


class TestAudioConversion(unittest.TestCase):
    """Tests for audio format conversion using ffmpeg."""

    def setUp(self):
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

    def test_convert_wav_to_wav(self):
        """Test converting WAV to WAV (identity operation)."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            _make_test_wav(input_path)
            result_path = convert_to_wav(input_path, output_path)
            self.assertEqual(result_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify it's a valid WAV
            channels, sr, nc = read_wav(output_path)
            self.assertEqual(nc, 1)
            self.assertEqual(sr, 8000)
            self.assertGreater(len(channels[0]), 0)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_to_mp3_and_back(self):
        """Test converting WAV to MP3 and back to WAV."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav2_path = f.name

        try:
            # Create original WAV
            _make_test_wav(wav_path)
            original_channels, original_sr, _ = read_wav(wav_path)

            # Convert to MP3
            convert_from_wav(wav_path, mp3_path, "mp3")
            self.assertTrue(os.path.exists(mp3_path))
            self.assertGreater(os.path.getsize(mp3_path), 0)

            # Convert back to WAV
            convert_to_wav(mp3_path, wav2_path)
            self.assertTrue(os.path.exists(wav2_path))

            # Verify the converted WAV is valid
            channels, sr, nc = read_wav(wav2_path)
            self.assertEqual(nc, 1)
            # Sample rate should be preserved (or close to it)
            self.assertEqual(sr, original_sr)
            self.assertGreater(len(channels[0]), 0)
        finally:
            for path in [wav_path, mp3_path, wav2_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_convert_to_wav_creates_temp_file(self):
        """Test that convert_to_wav creates a temp file if output_path is None."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name

        try:
            _make_test_wav(input_path)
            temp_path = convert_to_wav(input_path)
            
            self.assertTrue(os.path.exists(temp_path))
            self.assertTrue(temp_path.endswith(".wav"))
            
            # Verify it's a valid WAV
            channels, sr, nc = read_wav(temp_path)
            self.assertGreater(len(channels[0]), 0)
            
            # Clean up temp file
            os.unlink(temp_path)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    def test_convert_invalid_file_raises_error(self):
        """Test that converting an invalid file raises RuntimeError."""
        if not self.ffmpeg_available:
            self.skipTest("ffmpeg not available")

        with self.assertRaises(RuntimeError):
            convert_to_wav("/nonexistent/file.mp3")


if __name__ == "__main__":
    unittest.main()
