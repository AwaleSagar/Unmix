"""WAV audio I/O using the standard-library ``wave`` module."""

import struct
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
