"""
Audio Enhancement Experiment
==============================
Compares two denoising/enhancement approaches on every WAV in /input:

  Approach 1 (model)  — DeepFilterNet3 via deepfilternet (CPU)
  Approach 2 (script) — noisereduce + bandpass + pedalboard chain

Outputs:
  output_model/   — DeepFilterNet3 results
  output_script/  — signal-processing results
  results.csv     — per-file metrics
  report_summary.txt — aggregate comparison and recommendation

Optional clean references in input/clean/ enable true PESQ/STOI improvement
measurement (clean vs noisy / clean vs enhanced). Without clean refs, before
scores use the noisy file as its own reference (perfect baseline).

Usage:
    python generate_test_audio.py   # optional: create 10 synthetic test files
    python run_experiment.py
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import librosa
import numpy as np
import psutil
import soundfile as sf
import torch
from pedalboard import Compressor, NoiseGate, Pedalboard
from pystoi import stoi
from scipy.signal import butter, sosfiltfilt

import noisereduce as nr


def check_dependencies() -> None:
    """Verify optional compiled packages and exit with install hints if missing."""
    missing: list[str] = []

    try:
        from pesq import pesq as _pesq  # noqa: F401
    except ImportError:
        missing.append(
            "pesq -> pip install pesq "
            "(on Windows/Python 3.12 you may need MSVC Build Tools or Python 3.10/3.11)"
        )

    try:
        from df.enhance import enhance as _enhance, init_df as _init_df  # noqa: F401
    except ImportError:
        missing.append(
            "deepfilternet -> pip install deepfilternet "
            "(requires torch; on Windows use Python 3.10/3.11 or install Rust)"
        )

    if missing:
        print("Missing required packages:\n", file=sys.stderr)
        for line in missing:
            print(f"  • {line}", file=sys.stderr)
        print("\nSee README.md for full setup instructions.", file=sys.stderr)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
CLEAN_DIR = INPUT_DIR / "clean"
OUTPUT_MODEL_DIR = BASE_DIR / "output_model"
OUTPUT_SCRIPT_DIR = BASE_DIR / "output_script"
RESULTS_CSV = BASE_DIR / "results.csv"
REPORT_TXT = BASE_DIR / "report_summary.txt"

# 16 kHz is required for wideband PESQ and keeps processing tractable on CPU.
METRIC_SR = 16_000

# DeepFilterNet expects 48 kHz internally; process long files in chunks to limit RAM.
DFN_CHUNK_SEC = 30

# Supported audio extensions scanned in /input.
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


@dataclass
class ExperimentResult:
    """One row of experiment metrics for a single file and approach."""

    filename: str
    duration_sec: float
    approach: str
    processing_time_sec: float
    cpu_percent_avg: float
    pesq_before: float
    pesq_after: float
    stoi_before: float
    stoi_after: float
    output_path: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def discover_input_files(input_dir: Path) -> list[Path]:
    """Return sorted list of audio files in /input (excluding /input/clean)."""
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def load_audio_mono(path: Path, target_sr: int) -> tuple[np.ndarray, float]:
    """Load audio with librosa as mono float32 and return (samples, duration_sec)."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    duration = len(y) / sr
    return y.astype(np.float32), float(duration)


def find_clean_reference(noisy_path: Path) -> Optional[np.ndarray]:
    """
    Look for a matching clean WAV in input/clean/ with the same stem.
    Returns resampled mono audio at METRIC_SR, or None if not found.
    """
    clean_path = CLEAN_DIR / noisy_path.name
    if not clean_path.exists():
        alt = CLEAN_DIR / f"{noisy_path.stem}.wav"
        clean_path = alt if alt.exists() else None
    if clean_path is None or not clean_path.exists():
        return None
    y, _ = librosa.load(clean_path, sr=METRIC_SR, mono=True)
    return y.astype(np.float32)


def align_signals(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim two signals to the same length (required by PESQ/STOI)."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_pesq(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """
    Wideband PESQ in [0.5, 4.5]; higher is better perceptual quality.
    pesq expects int16-scale float inputs internally handled by the library.
    """
    from pesq import pesq

    ref, deg = align_signals(reference, degraded)
    if len(ref) < sr // 2:
        return float("nan")
    return float(pesq(sr, ref, deg, "wb"))


def compute_stoi(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """STOI in [0, 1]; higher means better intelligibility relative to reference."""
    ref, deg = align_signals(reference, degraded)
    if len(ref) < sr // 2:
        return float("nan")
    return float(stoi(ref, deg, sr, extended=False))


def measure_quality(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean_ref: Optional[np.ndarray],
    sr: int,
) -> tuple[float, float, float, float]:
    """
    Compute before/after PESQ and STOI.

    With clean reference:
      before = metric(clean, noisy), after = metric(clean, enhanced)

    Without clean reference (fallback):
      before = metric(noisy, noisy) → perfect baseline
      after  = metric(noisy, enhanced) → speech preservation vs original
    """
    if clean_ref is not None:
        clean_ref, noisy = align_signals(clean_ref, noisy)
        _, enhanced = align_signals(clean_ref, enhanced)
        pesq_before = compute_pesq(clean_ref, noisy, sr)
        pesq_after = compute_pesq(clean_ref, enhanced, sr)
        stoi_before = compute_stoi(clean_ref, noisy, sr)
        stoi_after = compute_stoi(clean_ref, enhanced, sr)
    else:
        pesq_before = compute_pesq(noisy, noisy, sr)
        pesq_after = compute_pesq(noisy, enhanced, sr)
        stoi_before = compute_stoi(noisy, noisy, sr)
        stoi_after = compute_stoi(noisy, enhanced, sr)
    return pesq_before, pesq_after, stoi_before, stoi_after


class CpuMonitor:
    """Sample system CPU usage in a background thread during processing."""

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Prime psutil's moving average, then poll while work runs.
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=0.2))

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if not self._samples:
            return 0.0
        return float(sum(self._samples) / len(self._samples))


def run_timed_with_cpu(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Execute enhancement fn(), returning (audio, elapsed_sec, avg_cpu_percent)."""
    monitor = CpuMonitor()
    monitor.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    cpu_avg = monitor.stop()
    return result, elapsed, cpu_avg


def normalize_peak(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Prevent clipping when saving enhanced audio."""
    max_val = np.max(np.abs(y))
    if max_val < 1e-8:
        return y
    return (y / max_val * peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Approach 1 — DeepFilterNet3 (model-based, CPU)
# ---------------------------------------------------------------------------


def init_deepfilternet_cpu():
    """
    Load DeepFilterNet3 and force CPU execution for a CPU-friendly benchmark.
    Returns (model, df_state) ready for enhance().
    """
    from df.enhance import init_df

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model, df_state, _, _ = init_df(model_base_dir="DeepFilterNet3", log_level="ERROR")
    model = model.to("cpu").eval()
    return model, df_state


def enhance_deepfilternet(
    y: np.ndarray,
    sr: int,
    model,
    df_state,
) -> np.ndarray:
    """
    Run DeepFilterNet3 enhancement.

    Audio is resampled to the model sample rate, processed in fixed-length
    chunks for memory safety on long (up to 60 min) files, then resampled back.
    """
    from df.enhance import enhance

    model_sr = df_state.sr

    if sr != model_sr:
        y_model = librosa.resample(y, orig_sr=sr, target_sr=model_sr)
    else:
        y_model = y.copy()

    chunk_len = int(DFN_CHUNK_SEC * model_sr)
    enhanced_chunks: list[np.ndarray] = []

    for start in range(0, len(y_model), chunk_len):
        chunk = y_model[start : start + chunk_len]
        # enhance() expects a torch tensor shaped [1, samples].
        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.inference_mode():
            out = enhance(model, df_state, chunk_tensor)
        enhanced_chunks.append(out.squeeze().cpu().numpy())

    enhanced_model_sr = np.concatenate(enhanced_chunks).astype(np.float32)

    if model_sr != sr:
        enhanced = librosa.resample(enhanced_model_sr, orig_sr=model_sr, target_sr=sr)
    else:
        enhanced = enhanced_model_sr

    # Match original length in case resampling shifted sample count slightly.
    n = min(len(y), len(enhanced))
    return enhanced[:n]


# ---------------------------------------------------------------------------
# Approach 2 — Classical signal-processing chain (no ML model)
# ---------------------------------------------------------------------------


def bandpass_voice(y: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """
    Butterworth band-pass filter keeping the typical telephony/voice band.
    sosfiltfilt applies zero-phase filtering (no time delay distortion).
    """
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def enhance_signal_processing(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Sequential classical pipeline:
      1. noisereduce  — spectral-gating noise suppression
      2. scipy        — 300–3400 Hz band-pass (voice range)
      3. pedalboard   — noise gate then compressor for dynamics cleanup
    """
    # Step 1: spectral subtraction / gating via noisereduce.
    denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=False,
        prop_decrease=0.75,
    )

    # Step 2: band-pass filter to attenuate out-of-band rumble and hiss.
    filtered = bandpass_voice(denoised, sr)

    # Step 3: pedalboard effects — gate quiet noise floor, then compress dynamics.
    board = Pedalboard(
        [
            NoiseGate(threshold_db=-35.0, ratio=2.0, release_ms=250.0),
            Compressor(
                threshold_db=-18.0,
                ratio=3.0,
                attack_ms=5.0,
                release_ms=100.0,
            ),
        ]
    )
    processed = board(filtered, sr)

    if processed.ndim > 1:
        processed = np.mean(processed, axis=0)

    n = min(len(y), len(processed))
    return processed[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def process_file(
    input_path: Path,
    approach: str,
    enhance_fn: Callable[[np.ndarray, int], np.ndarray],
    output_dir: Path,
) -> ExperimentResult:
    """Load one file, run one approach, save output, and collect metrics."""
    print(f"  [{approach}] {input_path.name}")

    # Load noisy/input audio at the metric sample rate.
    noisy, duration = load_audio_mono(input_path, METRIC_SR)
    clean_ref = find_clean_reference(input_path)

    # Time the enhancement while sampling CPU usage.
    enhanced, proc_time, cpu_avg = run_timed_with_cpu(
        lambda: enhance_fn(noisy, METRIC_SR)
    )

    enhanced = normalize_peak(enhanced)

    # Save enhanced WAV with soundfile.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_enhanced.wav"
    sf.write(out_path, enhanced, METRIC_SR, subtype="PCM_16")

    # Quality metrics: PESQ and STOI before vs after enhancement.
    pesq_before, pesq_after, stoi_before, stoi_after = measure_quality(
        noisy, enhanced, clean_ref, METRIC_SR
    )

    return ExperimentResult(
        filename=input_path.name,
        duration_sec=round(duration, 2),
        approach=approach,
        processing_time_sec=round(proc_time, 3),
        cpu_percent_avg=round(cpu_avg, 2),
        pesq_before=round(pesq_before, 4),
        pesq_after=round(pesq_after, 4),
        stoi_before=round(stoi_before, 4),
        stoi_after=round(stoi_after, 4),
        output_path=str(out_path),
    )


def write_results_csv(results: Iterable[ExperimentResult], path: Path) -> None:
    """Write all experiment rows to results.csv."""
    rows = list(results)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not np.isnan(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def generate_report(results: list[ExperimentResult], path: Path) -> str:
    """
    Build a human-readable summary comparing both approaches and pick a winner.
    Returns the report text (also written to report_summary.txt).
    """
    model_rows = [r for r in results if r.approach == "DeepFilterNet3"]
    script_rows = [r for r in results if r.approach == "SignalProcessing"]

    def summarize(label: str, rows: list[ExperimentResult]) -> dict[str, float]:
        return {
            "files": len(rows),
            "avg_time_sec": _mean([r.processing_time_sec for r in rows]),
            "avg_cpu_pct": _mean([r.cpu_percent_avg for r in rows]),
            "avg_pesq_gain": _mean([r.pesq_after - r.pesq_before for r in rows]),
            "avg_stoi_gain": _mean([r.stoi_after - r.stoi_before for r in rows]),
            "avg_pesq_after": _mean([r.pesq_after for r in rows]),
            "avg_stoi_after": _mean([r.stoi_after for r in rows]),
        }

    m = summarize("DeepFilterNet3", model_rows)
    s = summarize("SignalProcessing", script_rows)

    lines = [
        "=" * 72,
        "AUDIO ENHANCEMENT EXPERIMENT — SUMMARY REPORT",
        "=" * 72,
        "",
        f"Files processed per approach: {m['files']}",
        "",
        "AGGREGATE METRICS (mean across all files)",
        "-" * 72,
        f"{'Metric':<28} {'DeepFilterNet3':>18} {'Signal Processing':>18}",
        f"{'Processing time (s)':<28} {m['avg_time_sec']:>18.2f} {s['avg_time_sec']:>18.2f}",
        f"{'CPU usage (%)':<28} {m['avg_cpu_pct']:>18.2f} {s['avg_cpu_pct']:>18.2f}",
        f"{'PESQ gain (after-before)':<28} {m['avg_pesq_gain']:>18.4f} {s['avg_pesq_gain']:>18.4f}",
        f"{'STOI gain (after-before)':<28} {m['avg_stoi_gain']:>18.4f} {s['avg_stoi_gain']:>18.4f}",
        f"{'PESQ after':<28} {m['avg_pesq_after']:>18.4f} {s['avg_pesq_after']:>18.4f}",
        f"{'STOI after':<28} {m['avg_stoi_after']:>18.4f} {s['avg_stoi_after']:>18.4f}",
        "",
    ]

    # Score each approach: quality (PESQ+STOI gains) vs efficiency (time+CPU).
    quality_model = m["avg_pesq_gain"] + m["avg_stoi_gain"]
    quality_script = s["avg_pesq_gain"] + s["avg_stoi_gain"]
    speed_model = m["avg_time_sec"] * (1 + m["avg_cpu_pct"] / 100)
    speed_script = s["avg_time_sec"] * (1 + s["avg_cpu_pct"] / 100)

    if quality_model > quality_script and speed_model <= speed_script * 1.5:
        recommendation = (
            "RECOMMENDATION: DeepFilterNet3 (model-based)\n"
            "DeepFilterNet3 delivers higher average PESQ/STOI improvement with acceptable\n"
            "CPU cost for offline batch enhancement. Prefer it when quality is the priority."
        )
    elif quality_script >= quality_model and speed_script < speed_model:
        recommendation = (
            "RECOMMENDATION: Signal-processing pipeline (no model)\n"
            "The classical chain matches or beats quality scores while running faster with\n"
            "lower CPU usage. Prefer it for lightweight or real-time-adjacent workloads."
        )
    elif quality_model > quality_script:
        recommendation = (
            "RECOMMENDATION: DeepFilterNet3 (model-based)\n"
            "DeepFilterNet3 achieves better perceptual quality (PESQ/STOI gains) despite\n"
            "higher compute cost. Use the signal-processing chain only when CPU/time is tight."
        )
    else:
        recommendation = (
            "RECOMMENDATION: Signal-processing pipeline (no model)\n"
            "The noisereduce + bandpass + pedalboard chain offers the best balance of speed,\n"
            "CPU efficiency, and quality for this dataset."
        )

    lines.extend(
        [
            "RECOMMENDATION",
            "-" * 72,
            recommendation,
            "",
            "NOTES",
            "-" * 72,
            "- Place clean references in input/clean/ (same filename as noisy) for true",
            "  PESQ/STOI improvement vs a known clean target.",
            "- Without clean refs, 'before' scores are self-references (baseline).",
            "- Long files (>30 min) may take significant time with DeepFilterNet3 on CPU.",
            "",
        ]
    )

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    """Run the full experiment on all files in /input."""
    check_dependencies()

    print("Audio Enhancement Experiment")
    print("=" * 40)

    input_files = discover_input_files(INPUT_DIR)
    if not input_files:
        print(
            f"No audio files found in {INPUT_DIR}.\n"
            "Add WAV files to /input or run: python generate_test_audio.py",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(input_files)} input file(s).")
    has_clean = CLEAN_DIR.exists() and any(CLEAN_DIR.glob("*.wav"))
    print(
        "Clean references:",
        "found in input/clean/" if has_clean else "not found (using fallback metrics)",
    )

    # Load DeepFilterNet3 once and reuse for every file.
    print("\nLoading DeepFilterNet3 on CPU...")
    dfn_model, dfn_state = init_deepfilternet_cpu()
    print("Model ready.\n")

    results: list[ExperimentResult] = []

    for idx, path in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

        # Approach 1 — model-based enhancement.
        results.append(
            process_file(
                path,
                approach="DeepFilterNet3",
                enhance_fn=lambda y, sr, m=dfn_model, s=dfn_state: enhance_deepfilternet(
                    y, sr, m, s
                ),
                output_dir=OUTPUT_MODEL_DIR,
            )
        )

        # Approach 2 — classical signal processing (no neural model).
        results.append(
            process_file(
                path,
                approach="SignalProcessing",
                enhance_fn=enhance_signal_processing,
                output_dir=OUTPUT_SCRIPT_DIR,
            )
        )
        print()

    write_results_csv(results, RESULTS_CSV)
    report = generate_report(results, REPORT_TXT)

    print(report)
    print(f"Results CSV : {RESULTS_CSV}")
    print(f"Report      : {REPORT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
