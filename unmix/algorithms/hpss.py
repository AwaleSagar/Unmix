"""Harmonic-Percussive Source Separation (HPSS).

Separates an audio signal into *harmonic* (tonal, sustained) and *percussive*
(transient, broadband) components using median filtering on the magnitude
spectrogram, as described in:

    Fitzgerald, D. (2010) "Harmonic/Percussive Separation using Median
    Filtering."  Proc. of DAFx-10.
"""

from statistics import median

from ..dsp import (
    hann_window,
    istft,
    magnitude,
    phase,
    polar_to_complex,
    stft,
)


def _median_filter_rows(mat, kernel):
    """Apply a 1-D median filter along each row (time axis)."""
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    half = kernel // 2
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            lo = max(0, c - half)
            hi = min(cols, c + half + 1)
            out[r][c] = median(mat[r][lo:hi])
    return out


def _median_filter_cols(mat, kernel):
    """Apply a 1-D median filter along each column (frequency axis)."""
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    half = kernel // 2
    out = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        col = [mat[r][c] for r in range(rows)]
        for r in range(rows):
            lo = max(0, r - half)
            hi = min(rows, r + half + 1)
            out[r][c] = median(col[lo:hi])
    return out


def hpss(signal, sample_rate, frame_size=2048, hop_size=512,
         harmonic_kernel=31, percussive_kernel=31):
    """Run HPSS and return *(harmonic, percussive)* time-domain signals.

    Parameters
    ----------
    signal : list[float]
        Mono audio samples.
    sample_rate : int
        Sample rate (unused here, kept for API consistency).
    frame_size, hop_size : int
        STFT parameters.
    harmonic_kernel, percussive_kernel : int
        Median-filter kernel sizes (odd integers).  Harmonic kernel runs
        along time; percussive kernel runs along frequency.

    Returns
    -------
    (list[float], list[float])
        Harmonic and percussive signals of the same length as *signal*.
    """
    window = hann_window(frame_size)
    frames = stft(signal, frame_size, hop_size, window)
    n_frames = len(frames)
    n_bins = frame_size

    # Build magnitude / phase matrices  (rows = freq bins, cols = time frames)
    mag = [[0.0] * n_frames for _ in range(n_bins)]
    pha = [[0.0] * n_frames for _ in range(n_bins)]
    for t in range(n_frames):
        for f in range(n_bins):
            mag[f][t] = abs(frames[t][f])
            pha[f][t] = phase([frames[t][f]])[0]

    # Median filtering
    H = _median_filter_rows(mag, harmonic_kernel)    # enhance horizontal (harmonic)
    P = _median_filter_cols(mag, percussive_kernel)   # enhance vertical (percussive)

    # Wiener-like soft masks
    eps = 1e-10
    mask_h = [[0.0] * n_frames for _ in range(n_bins)]
    mask_p = [[0.0] * n_frames for _ in range(n_bins)]
    for f in range(n_bins):
        for t in range(n_frames):
            h2 = H[f][t] ** 2
            p2 = P[f][t] ** 2
            denom = h2 + p2 + eps
            mask_h[f][t] = h2 / denom
            mask_p[f][t] = p2 / denom

    # Apply masks and reconstruct STFT frames
    h_frames = []
    p_frames = []
    for t in range(n_frames):
        hf = polar_to_complex(
            [mag[f][t] * mask_h[f][t] for f in range(n_bins)],
            [pha[f][t] for f in range(n_bins)],
        )
        pf = polar_to_complex(
            [mag[f][t] * mask_p[f][t] for f in range(n_bins)],
            [pha[f][t] for f in range(n_bins)],
        )
        h_frames.append(hf)
        p_frames.append(pf)

    # Inverse STFT
    length = len(signal)
    harmonic_signal = istft(h_frames, frame_size, hop_size, window, length)
    percussive_signal = istft(p_frames, frame_size, hop_size, window, length)

    return harmonic_signal, percussive_signal
