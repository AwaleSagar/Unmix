"""Robust Principal Component Analysis (RPCA) for singing-voice separation.

Decomposes the magnitude spectrogram **X** into a *low-rank* component **L**
(repeating / background music) and a *sparse* component **S** (vocals /
transients) by solving:

    min  ‖L‖_* + λ ‖S‖₁   s.t.  X = L + S

via the Inexact Augmented Lagrange Multiplier (IALM) method.

Reference:
    Huang, P., et al. (2012) "Singing-Voice Separation from Monaural
    Recordings using Robust Principal Component Analysis."  Proc. ICASSP.
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
from ..matrix import (
    frobenius_norm,
    mat_add,
    mat_scale,
    mat_sub,
    shape,
    soft_threshold,
    svd_shrink,
    zeros,
)


def rpca_separate(signal, sample_rate=44100,
                  frame_size=2048, hop_size=512,
                  lam=None, mu=None, tol=1e-6, max_iter=50):
    """Separate *signal* into background (low-rank) and foreground (sparse).

    Parameters
    ----------
    signal : list[float]
        Mono audio samples.
    sample_rate : int
        Not used directly; kept for API consistency.
    frame_size, hop_size : int
        STFT parameters.
    lam : float | None
        Sparsity penalty weight.  Defaults to 1 / sqrt(max(m, n)).
    mu : float | None
        Initial ALM penalty parameter.  Defaults to n*m / (4 * ‖X‖₁).
    tol : float
        Convergence tolerance on ‖X − L − S‖_F / ‖X‖_F.
    max_iter : int
        Maximum IALM iterations.

    Returns
    -------
    (list[float], list[float])
        Background (low-rank) and foreground (sparse) time-domain signals.
    """
    window = hann_window(frame_size)
    frames = stft(signal, frame_size, hop_size, window)
    n_frames = len(frames)
    n_bins = frame_size

    # Build magnitude spectrogram  (rows = freq, cols = time)
    X = [[abs(frames[t][f]) for t in range(n_frames)] for f in range(n_bins)]
    pha_mat = [[phase([frames[t][f]])[0] for t in range(n_frames)] for f in range(n_bins)]

    m, n = shape(X)
    if lam is None:
        lam = 1.0 / math.sqrt(max(m, n))

    # L1 norm of X
    x_l1 = sum(abs(X[r][c]) for r in range(m) for c in range(n))
    if mu is None:
        mu = (m * n) / (4.0 * max(x_l1, 1e-10))

    x_fro = frobenius_norm(X)

    L = zeros(m, n)
    S = zeros(m, n)
    Y = zeros(m, n)  # dual variable

    for _ in range(max_iter):
        # L step: singular-value thresholding
        temp = mat_sub(mat_sub(X, S), mat_scale(Y, 1.0 / mu))
        L = svd_shrink(temp, 1.0 / mu)

        # S step: soft thresholding
        temp = mat_sub(mat_sub(X, L), mat_scale(Y, 1.0 / mu))
        S = soft_threshold(temp, lam / mu)

        # Dual update
        residual = mat_sub(mat_sub(X, L), S)
        Y = mat_add(Y, mat_scale(residual, mu))

        # Convergence check
        res_norm = frobenius_norm(residual)
        if res_norm / max(x_fro, 1e-10) < tol:
            break

    # Reconstruct time-domain signals via soft masks
    eps = 1e-10
    bg_frames = []
    fg_frames = []
    for t in range(n_frames):
        bg_mag = []
        fg_mag = []
        pha_t = [pha_mat[f][t] for f in range(n_bins)]
        for f in range(n_bins):
            l_val = max(L[f][t], 0.0)
            s_val = max(S[f][t], 0.0)
            denom = l_val + s_val + eps
            mask_l = l_val / denom
            mask_s = s_val / denom
            bg_mag.append(X[f][t] * mask_l)
            fg_mag.append(X[f][t] * mask_s)
        bg_frames.append(polar_to_complex(bg_mag, pha_t))
        fg_frames.append(polar_to_complex(fg_mag, pha_t))

    length = len(signal)
    background = istft(bg_frames, frame_size, hop_size, window, length)
    foreground = istft(fg_frames, frame_size, hop_size, window, length)
    return background, foreground
