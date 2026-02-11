"""Pure-Python DSP primitives: FFT, STFT, inverse STFT, and windowing."""

import cmath
import math


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def hann_window(length):
    """Return a periodic Hann window of the given *length*."""
    return [
        0.5 * (1.0 - math.cos(2.0 * math.pi * n / length))
        for n in range(length)
    ]


# ---------------------------------------------------------------------------
# FFT / IFFT  (Cooley-Tukey radix-2, with Bluestein fallback)
# ---------------------------------------------------------------------------

def _next_pow2(n):
    """Return the smallest power-of-two >= *n*."""
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(x):
    """Compute the DFT of sequence *x* (list of float/complex).

    Supports arbitrary lengths via Bluestein's algorithm, but is fastest
    when ``len(x)`` is a power of two (pure Cooley-Tukey).
    """
    n = len(x)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_radix2(x)
    return _bluestein(x)


def ifft(X):
    """Compute the inverse DFT of spectrum *X*."""
    n = len(X)
    if n == 0:
        return []
    # IDFT via conjugate trick: ifft(X) = conj(fft(conj(X))) / N
    conj_X = [z.conjugate() if isinstance(z, complex) else complex(z) for z in X]
    result = fft(conj_X)
    return [z.conjugate() / n for z in result]


def _fft_radix2(x):
    """Radix-2 Cooley-Tukey FFT.  *len(x)* must be a power of two."""
    n = len(x)
    if n <= 1:
        return [complex(v) for v in x]

    even = _fft_radix2(x[0::2])
    odd = _fft_radix2(x[1::2])

    half = n // 2
    result = [complex(0)] * n
    for k in range(half):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + half] = even[k] - t
    return result


def _bluestein(x):
    """Bluestein's algorithm – FFT for arbitrary lengths."""
    n = len(x)
    m = _next_pow2(2 * n - 1)

    # Chirp sequence
    chirp = [cmath.exp(-1j * cmath.pi * k * k / n) for k in range(n)]

    # Zero-padded sequences
    a = [complex(0)] * m
    b = [complex(0)] * m
    for k in range(n):
        a[k] = complex(x[k]) * chirp[k]
        b[k] = chirp[k].conjugate()
    for k in range(1, n):
        b[m - k] = chirp[k].conjugate()

    fa = _fft_radix2(a)
    fb = _fft_radix2(b)
    fc = [fa[i] * fb[i] for i in range(m)]
    # m is a power of two, so use radix-2 IFFT directly (conjugate trick)
    conj_fc = [z.conjugate() for z in fc]
    c_raw = _fft_radix2(conj_fc)
    c = [z.conjugate() / m for z in c_raw]

    return [c[k] * chirp[k] for k in range(n)]


# ---------------------------------------------------------------------------
# STFT / iSTFT
# ---------------------------------------------------------------------------

def stft(signal, frame_size=2048, hop_size=512, window=None):
    """Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    signal : list[float]
        Mono audio samples.
    frame_size : int
        FFT frame length (should be a power of two for speed).
    hop_size : int
        Hop length in samples.
    window : list[float] | None
        Analysis window.  Defaults to a Hann window of *frame_size*.

    Returns
    -------
    list[list[complex]]
        A list of frames, each a list of *frame_size* complex DFT bins.
    """
    if window is None:
        window = hann_window(frame_size)

    n = len(signal)
    frames = []
    pos = 0
    while pos + frame_size <= n:
        frame = [signal[pos + i] * window[i] for i in range(frame_size)]
        frames.append(fft(frame))
        pos += hop_size
    return frames


def istft(frames, frame_size=2048, hop_size=512, window=None, length=None):
    """Compute the inverse STFT via overlap-add.

    Parameters
    ----------
    frames : list[list[complex]]
        STFT frames (as returned by :func:`stft`).
    frame_size : int
        FFT frame length.
    hop_size : int
        Hop length in samples.
    window : list[float] | None
        Synthesis window.  Defaults to Hann.
    length : int | None
        Desired output length.  If *None*, computed from frame count.

    Returns
    -------
    list[float]
        Reconstructed time-domain signal.
    """
    if window is None:
        window = hann_window(frame_size)

    n_frames = len(frames)
    if length is None:
        length = frame_size + (n_frames - 1) * hop_size

    output = [0.0] * length
    win_sum = [0.0] * length

    for idx, F in enumerate(frames):
        time_frame = ifft(F)
        start = idx * hop_size
        for i in range(frame_size):
            pos = start + i
            if pos < length:
                output[pos] += time_frame[i].real * window[i]
                win_sum[pos] += window[i] * window[i]

    # Normalise by the window overlap sum (avoid division by zero).
    # Use max(win_sum) as a reference so edge samples with negligible
    # window coverage are zeroed out rather than amplified.
    peak = max(win_sum) if win_sum else 1.0
    threshold = peak * 1e-6
    output = [
        output[i] / win_sum[i] if win_sum[i] > threshold else 0.0
        for i in range(length)
    ]
    return output


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------

def magnitude(spectrum):
    """Return the magnitude of each DFT bin."""
    return [abs(z) for z in spectrum]


def phase(spectrum):
    """Return the phase angle of each DFT bin."""
    return [cmath.phase(z) for z in spectrum]


def polar_to_complex(mag, pha):
    """Reconstruct complex spectrum from magnitude and phase."""
    return [m * cmath.exp(1j * p) for m, p in zip(mag, pha)]
