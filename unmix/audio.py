"""WAV audio I/O using the standard-library ``wave`` module."""

import os
import struct
import subprocess
import tempfile
import wave


def read_wav(path):
    """Read a WAV file and return (samples, sample_rate, num_channels).

    *samples* is a list of lists – one inner list per channel, each containing
    float values in the range [-1.0, 1.0].
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        fmt = "B"
        offset, scale = 128, 128.0
    elif sampwidth == 2:
        fmt = "<h"
        offset, scale = 0, 32768.0
    elif sampwidth == 3:
        fmt = None
        offset, scale = 0, 8388608.0
    elif sampwidth == 4:
        fmt = "<i"
        offset, scale = 0, 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    total_samples = n_frames * n_channels
    if fmt is not None:
        sample_size = struct.calcsize(fmt)
        int_samples = [
            struct.unpack_from(fmt, raw, i * sample_size)[0]
            for i in range(total_samples)
        ]
    else:
        # 24-bit samples – unpack manually
        int_samples = []
        for i in range(total_samples):
            b = raw[3 * i : 3 * i + 3]
            val = b[0] | (b[1] << 8) | (b[2] << 16)
            if val >= 0x800000:
                val -= 0x1000000
            int_samples.append(val)

    channels = [[] for _ in range(n_channels)]
    for idx, val in enumerate(int_samples):
        channels[idx % n_channels].append((val - offset) / scale)

    return channels, sample_rate, n_channels


def write_wav(path, channels, sample_rate, sampwidth=2):
    """Write a WAV file from a list-of-lists of float samples in [-1, 1]."""
    n_channels = len(channels)
    n_frames = len(channels[0])

    if sampwidth == 2:
        fmt = "<h"
        scale = 32767.0
    elif sampwidth == 4:
        fmt = "<i"
        scale = 2147483647.0
    else:
        raise ValueError("Only 16-bit or 32-bit output is supported")

    frames = bytearray()
    for i in range(n_frames):
        for ch in range(n_channels):
            val = channels[ch][i] if i < len(channels[ch]) else 0.0
            clamped = max(-1.0, min(1.0, val))
            frames.extend(struct.pack(fmt, int(clamped * scale)))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def to_mono(channels):
    """Down-mix multi-channel audio to mono by averaging."""
    n = len(channels[0])
    k = len(channels)
    return [sum(channels[ch][i] for ch in range(k)) / k for i in range(n)]


def detect_audio_format(path):
    """Detect the audio format of a file.
    
    Returns the file extension (e.g., 'wav', 'mp3', 'flac') or None if detection fails.
    First tries file extension, then falls back to ffprobe if available.
    """
    # Try file extension first
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext:
        return ext
    
    # Fall back to ffprobe if extension is missing
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path],
            capture_output=True,
            text=True,
            check=True
        )
        # Try to extract format from ffprobe output
        import json
        data = json.loads(result.stdout)
        format_name = data.get('format', {}).get('format_name', '').split(',')[0]
        return format_name if format_name else None
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError):
        return None


def convert_to_wav(input_path, output_path=None):
    """Convert an audio file to WAV format using ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        output_path: Optional path for the output WAV file. If None, creates a temp file.
    
    Returns:
        Path to the converted WAV file
    
    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails
    """
    if output_path is None:
        # Create a temporary file for the WAV
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    try:
        # Use ffmpeg to convert to WAV
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-acodec', 'pcm_s16le', output_path],
            capture_output=True,
            check=True
        )
        return output_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert {input_path} to WAV: {e.stderr.decode()}")


def convert_from_wav(wav_path, output_path, target_format=None):
    """Convert a WAV file to another audio format using ffmpeg.
    
    Args:
        wav_path: Path to the input WAV file
        output_path: Path for the output file
        target_format: Optional target format. If None, inferred from output_path extension.
    
    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails
    """
    try:
        cmd = ['ffmpeg', '-y', '-i', wav_path]
        
        # Add format-specific options if needed
        if target_format:
            cmd.extend(['-f', target_format])
        
        cmd.append(output_path)
        
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert WAV to {target_format or 'target format'}: {e.stderr.decode()}")
