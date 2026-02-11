# Unmix

Deterministic audio source separation using **only the Python standard library** — no external dependencies required.

Unmix implements several classical DSP algorithms described in the companion paper *"Mathematical Foundations and Deterministic Architectures for Blind and Non-Blind Audio Source Separation"*:

| Algorithm | Description |
|-----------|-------------|
| **HPSS** | Harmonic-Percussive Source Separation via median filtering |
| **Spectral Subtraction** | Noise reduction by subtracting an estimated noise spectrum |
| **RPCA** | Robust PCA (low-rank + sparse) for vocal/background separation |

## Requirements

* Python 3.8+
* No third-party packages

## Quick Start

```bash
# Harmonic-Percussive separation
python -m unmix hpss song.wav -o output/

# Noise reduction (auto-estimate from first frames)
python -m unmix spectral-sub noisy.wav -o output/

# Noise reduction with a separate noise reference
python -m unmix spectral-sub noisy.wav --noise room_tone.wav -o output/

# Vocal / background separation via RPCA
python -m unmix rpca song.wav -o output/
```

## CLI Reference

```
usage: unmix [-h] [--version] {hpss,spectral-sub,rpca} ...
```

### `unmix hpss`

Separate audio into harmonic (tonal) and percussive (transient) components.

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Input WAV file |
| `-o`, `--output-dir` | same as input | Output directory |
| `--frame-size` | 2048 | STFT frame size |
| `--hop-size` | 512 | STFT hop size |
| `--harmonic-kernel` | 31 | Median filter kernel for harmonic |
| `--percussive-kernel` | 31 | Median filter kernel for percussive |

### `unmix spectral-sub`

Remove background noise from audio via spectral subtraction.

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Input WAV file (noisy signal) |
| `-n`, `--noise` | — | Optional noise-only WAV reference |
| `-o`, `--output-dir` | same as input | Output directory |
| `--frame-size` | 2048 | STFT frame size |
| `--hop-size` | 512 | STFT hop size |
| `--noise-frames` | 10 | Frames used for noise estimate |
| `--oversubtraction` | 1.0 | Over-subtraction factor α |
| `--spectral-floor` | 0.01 | Spectral floor β |

### `unmix rpca`

Separate audio into low-rank background and sparse foreground (e.g. vocals).

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Input WAV file |
| `-o`, `--output-dir` | same as input | Output directory |
| `--frame-size` | 2048 | STFT frame size |
| `--hop-size` | 512 | STFT hop size |
| `--tol` | 1e-6 | Convergence tolerance |
| `--max-iter` | 50 | Maximum IALM iterations |

## Running Tests

```bash
python -m pytest tests/
# or
python -m unittest discover -s tests
```

## Project Structure

```
unmix/
├── __init__.py          # package metadata
├── __main__.py          # python -m unmix entry point
├── cli.py               # argparse CLI
├── audio.py             # WAV I/O (stdlib wave module)
├── dsp.py               # FFT, STFT, iSTFT, windowing
├── matrix.py            # pure-Python linear algebra helpers
└── algorithms/
    ├── __init__.py
    ├── hpss.py           # Harmonic-Percussive Source Separation
    ├── spectral_sub.py   # Spectral Subtraction
    └── rpca.py           # Robust PCA (IALM)
```

## License

MIT — see [LICENSE](LICENSE).
