"""Spectral Subtraction for noise reduction.

A classical approach: estimate the noise power spectrum from a known
noise-only segment (or a separate noise file) and subtract it from
the mixture spectrum, using magnitude-domain half-wave rectification
to avoid negative power values.

Reference:
    Boll, S. (1979) "Suppression of Acoustic Noise in Speech Using
    Spectral Subtraction."  IEEE Trans. ASSP-27(2).
"""

import math

from ..dsp import (
    hann_window,
    magnitude,
    phase,
    polar_to_complex,
    stft,
    istft,
)


def spectral_subtract(signal, noise_signal=None, sample_rate=44100,
                       frame_size=2048, hop_size=512,
                       noise_frames=10, oversubtraction=1.0,
                       spectral_floor=0.01):
    """Perform spectral subtraction on *signal*.

    Parameters
    ----------
    signal : list[float]
        Mono audio samples of the noisy signal.
    noise_signal : list[float] | None
        Optional separate noise-only recording.  If *None*, the first
        *noise_frames* STFT frames of *signal* are used as the noise
        estimate.
    sample_rate : int
        Not used directly; kept for API consistency.
    frame_size, hop_size : int
        STFT parameters.
    noise_frames : int
        Number of leading frames to treat as noise when *noise_signal*
        is not provided.
    oversubtraction : float
        Factor α ≥ 1 that controls how aggressively the noise is
        subtracted.  Higher values remove more noise but may introduce
        distortion.
    spectral_floor : float
        Minimum spectral magnitude (relative to noise) to avoid
        "musical noise" artefacts.

    Returns
    -------
    list[float]
        Denoised signal of the same length as *signal*.
    """
    window = hann_window(frame_size)
    frames = stft(signal, frame_size, hop_size, window)
    n_frames = len(frames)
    n_bins = frame_size

    # --- Noise power estimate ---
    if noise_signal is not None:
        noise_frames_stft = stft(noise_signal, frame_size, hop_size, window)
        nf_count = len(noise_frames_stft)
    else:
        noise_frames_stft = frames[:noise_frames]
        nf_count = min(noise_frames, n_frames)

    noise_power = [0.0] * n_bins
    for f in range(n_bins):
        for t in range(nf_count):
            noise_power[f] += abs(noise_frames_stft[t][f]) ** 2
        noise_power[f] /= max(nf_count, 1)

    noise_mag = [math.sqrt(p) for p in noise_power]

    # --- Subtraction ---
    clean_frames = []
    for t in range(n_frames):
        mag_t = magnitude(frames[t])
        pha_t = phase(frames[t])

        clean_mag = []
        for f in range(n_bins):
            subtracted = mag_t[f] ** 2 - oversubtraction * noise_power[f]
            floor = (spectral_floor * noise_mag[f]) ** 2
            clean_mag.append(math.sqrt(max(subtracted, floor)))

        clean_frames.append(polar_to_complex(clean_mag, pha_t))

    return istft(clean_frames, frame_size, hop_size, window, len(signal))
