"""Command-line interface for Unmix."""

import argparse
import os
import sys
import tempfile
import time

from . import __version__
from .audio import (
    convert_from_wav,
    convert_to_wav,
    detect_audio_format,
    read_wav,
    to_mono,
    write_wav,
)


def _resolve_output(input_path, suffix, output_dir, original_format=None):
    """Build an output path by appending *suffix* before the extension.
    
    Args:
        input_path (str): Original input file path
        suffix (str): Suffix to add to the base name
        output_dir (str): Output directory
        original_format (str, optional): Original audio format extension (e.g., 'mp3', 'flac').
                                        If None, defaults to 'wav'.
    
    Returns:
        str: Output file path
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = original_format if original_format else "wav"
    name = f"{base}_{suffix}.{ext}"
    return os.path.join(output_dir, name)


def _ensure_output_dir(args):
    """Return the output directory, creating it if needed."""
    out_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_output(path, channels, sample_rate, target_format):
    """Write audio output, converting to target format if needed.
    
    Args:
        path (str): Output file path
        channels (list): Audio channel data
        sample_rate (int): Sample rate in Hz
        target_format (str): Target format extension (e.g., 'mp3', 'flac', 'wav')
    """
    if target_format.lower() == 'wav':
        # Direct WAV write
        write_wav(path, channels, sample_rate)
    else:
        # Write to temporary WAV, then convert
        fd, temp_wav = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            write_wav(temp_wav, channels, sample_rate)
            convert_from_wav(temp_wav, path, target_format)
        finally:
            if os.path.exists(temp_wav):
                os.unlink(temp_wav)


def _load_mono(path):
    """Load an audio file and return (mono_samples, sample_rate, format, temp_wav_path).
    
    Automatically converts non-WAV formats to WAV using ffmpeg.
    
    Args:
        path (str): Path to the input audio file
    
    Returns:
        tuple: (mono_samples, sample_rate, original_format, temp_wav_path)
               If temp_wav_path is not None, caller should clean it up after use.
    """
    # Detect the audio format
    audio_format = detect_audio_format(path)
    temp_wav = None
    
    # Convert to WAV if needed
    if audio_format and audio_format.lower() != 'wav':
        print(f"[unmix]   Converting {audio_format.upper()} to WAV for processing...")
        temp_wav = convert_to_wav(path)
        wav_path = temp_wav
    else:
        # Default to 'wav' if format detection failed or it's already WAV
        wav_path = path
        audio_format = audio_format or 'wav'
    
    # Load the WAV file
    channels, sr, nc = read_wav(wav_path)
    mono = to_mono(channels) if nc > 1 else channels[0]
    
    return mono, sr, audio_format, temp_wav


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_hpss(args):
    from .algorithms.hpss import hpss

    print(f"[unmix] Loading {args.input} …")
    mono, sr, original_format, temp_wav = _load_mono(args.input)
    print(f"[unmix]   {len(mono)} samples @ {sr} Hz")

    try:
        print("[unmix] Running HPSS …")
        t0 = time.time()
        harmonic, percussive = hpss(
            mono, sr,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            harmonic_kernel=args.harmonic_kernel,
            percussive_kernel=args.percussive_kernel,
        )
        elapsed = time.time() - t0
        print(f"[unmix]   done in {elapsed:.1f}s")

        out_dir = _ensure_output_dir(args)
        h_path = _resolve_output(args.input, "harmonic", out_dir, original_format)
        p_path = _resolve_output(args.input, "percussive", out_dir, original_format)
        
        _write_output(h_path, [harmonic], sr, original_format)
        _write_output(p_path, [percussive], sr, original_format)
        
        print(f"[unmix] Wrote {h_path}")
        print(f"[unmix] Wrote {p_path}")
    finally:
        # Clean up temporary WAV file if created
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


def _cmd_spectral_sub(args):
    from .algorithms.spectral_sub import spectral_subtract

    print(f"[unmix] Loading {args.input} …")
    mono, sr, original_format, temp_wav = _load_mono(args.input)
    print(f"[unmix]   {len(mono)} samples @ {sr} Hz")

    temp_noise_wav = None
    try:
        noise = None
        if args.noise:
            print(f"[unmix] Loading noise reference {args.noise} …")
            noise, _, _, temp_noise_wav = _load_mono(args.noise)

        print("[unmix] Running spectral subtraction …")
        t0 = time.time()
        clean = spectral_subtract(
            mono,
            noise_signal=noise,
            sample_rate=sr,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            noise_frames=args.noise_frames,
            oversubtraction=args.oversubtraction,
            spectral_floor=args.spectral_floor,
        )
        elapsed = time.time() - t0
        print(f"[unmix]   done in {elapsed:.1f}s")

        out_dir = _ensure_output_dir(args)
        out_path = _resolve_output(args.input, "clean", out_dir, original_format)
        
        _write_output(out_path, [clean], sr, original_format)
        
        print(f"[unmix] Wrote {out_path}")
    finally:
        # Clean up temporary WAV files if created
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)
        if temp_noise_wav and os.path.exists(temp_noise_wav):
            os.unlink(temp_noise_wav)


def _cmd_rpca(args):
    from .algorithms.rpca import rpca_separate

    print(f"[unmix] Loading {args.input} …")
    mono, sr, original_format, temp_wav = _load_mono(args.input)
    print(f"[unmix]   {len(mono)} samples @ {sr} Hz")

    try:
        print("[unmix] Running RPCA separation …")
        t0 = time.time()
        background, foreground = rpca_separate(
            mono,
            sample_rate=sr,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            tol=args.tol,
            max_iter=args.max_iter,
        )
        elapsed = time.time() - t0
        print(f"[unmix]   done in {elapsed:.1f}s")

        out_dir = _ensure_output_dir(args)
        bg_path = _resolve_output(args.input, "background", out_dir, original_format)
        fg_path = _resolve_output(args.input, "foreground", out_dir, original_format)
        
        _write_output(bg_path, [background], sr, original_format)
        _write_output(fg_path, [foreground], sr, original_format)
        
        print(f"[unmix] Wrote {bg_path}")
        print(f"[unmix] Wrote {fg_path}")
    finally:
        # Clean up temporary WAV file if created
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    """Construct and return the top-level :class:`ArgumentParser`."""
    parser = argparse.ArgumentParser(
        prog="unmix",
        description="Deterministic audio source separation – no external dependencies.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="separation algorithm")

    # --- HPSS ---------------------------------------------------------------
    p_hpss = sub.add_parser(
        "hpss",
        help="Harmonic-Percussive Source Separation",
        description="Separate audio into harmonic (tonal) and percussive (transient) components.",
    )
    p_hpss.add_argument("input", help="input audio file (WAV, MP3, FLAC, etc.)")
    p_hpss.add_argument("-o", "--output-dir", default=None, help="output directory (default: same as input)")
    p_hpss.add_argument("--frame-size", type=int, default=2048, help="STFT frame size (default: 2048)")
    p_hpss.add_argument("--hop-size", type=int, default=512, help="STFT hop size (default: 512)")
    p_hpss.add_argument("--harmonic-kernel", type=int, default=31, help="median filter kernel for harmonic (default: 31)")
    p_hpss.add_argument("--percussive-kernel", type=int, default=31, help="median filter kernel for percussive (default: 31)")
    p_hpss.set_defaults(func=_cmd_hpss)

    # --- Spectral Subtraction ----------------------------------------------
    p_ss = sub.add_parser(
        "spectral-sub",
        help="Spectral subtraction (noise reduction)",
        description="Remove background noise from audio via spectral subtraction.",
    )
    p_ss.add_argument("input", help="input audio file (WAV, MP3, FLAC, etc.)")
    p_ss.add_argument("-n", "--noise", default=None, help="optional noise-only audio reference")
    p_ss.add_argument("-o", "--output-dir", default=None, help="output directory")
    p_ss.add_argument("--frame-size", type=int, default=2048)
    p_ss.add_argument("--hop-size", type=int, default=512)
    p_ss.add_argument("--noise-frames", type=int, default=10, help="frames used for noise estimate (default: 10)")
    p_ss.add_argument("--oversubtraction", type=float, default=1.0, help="oversubtraction factor α (default: 1.0)")
    p_ss.add_argument("--spectral-floor", type=float, default=0.01, help="spectral floor β (default: 0.01)")
    p_ss.set_defaults(func=_cmd_spectral_sub)

    # --- RPCA ---------------------------------------------------------------
    p_rpca = sub.add_parser(
        "rpca",
        help="Robust PCA (low-rank + sparse decomposition)",
        description="Separate audio into low-rank background and sparse foreground (e.g. vocals).",
    )
    p_rpca.add_argument("input", help="input audio file (WAV, MP3, FLAC, etc.)")
    p_rpca.add_argument("-o", "--output-dir", default=None, help="output directory")
    p_rpca.add_argument("--frame-size", type=int, default=2048)
    p_rpca.add_argument("--hop-size", type=int, default=512)
    p_rpca.add_argument("--tol", type=float, default=1e-6, help="convergence tolerance (default: 1e-6)")
    p_rpca.add_argument("--max-iter", type=int, default=50, help="maximum IALM iterations (default: 50)")
    p_rpca.set_defaults(func=_cmd_rpca)

    return parser


def main(argv=None):
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)
