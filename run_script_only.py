"""
Audio Enhancement Experiment — Script/Signal Processing Only
=============================================================
Uses noisereduce + bandpass filter + pedalboard chain.
No ML model, no Rust, works on Python 3.12.

Outputs:
  output_script/     — enhanced WAV files
  results.csv        — per-file metrics
  report_summary.txt — full report
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import psutil
import soundfile as sf
from pedalboard import Compressor, NoiseGate, Pedalboard
from pystoi import stoi
from scipy.signal import butter, sosfiltfilt
import noisereduce as nr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
CLEAN_DIR = INPUT_DIR / "clean"
OUTPUT_SCRIPT_DIR = BASE_DIR / "output_script"
RESULTS_CSV = BASE_DIR / "results.csv"
REPORT_TXT = BASE_DIR / "report_summary.txt"

METRIC_SR = 16_000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


@dataclass
class ExperimentResult:
    filename: str
    duration_sec: float
    approach: str
    processing_time_sec: float
    cpu_percent_avg: float
    stoi_before: float
    stoi_after: float
    output_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_input_files(input_dir: Path) -> list[Path]:
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def load_audio_mono(path: Path, target_sr: int) -> tuple[np.ndarray, float]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), float(len(y) / sr)


def find_clean_reference(noisy_path: Path) -> Optional[np.ndarray]:
    for candidate in [CLEAN_DIR / noisy_path.name, CLEAN_DIR / f"{noisy_path.stem}.wav"]:
        if candidate.exists():
            y, _ = librosa.load(candidate, sr=METRIC_SR, mono=True)
            return y.astype(np.float32)
    return None


def align_signals(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_stoi(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    ref, deg = align_signals(reference, degraded)
    if len(ref) < sr // 2:
        return float("nan")
    return float(stoi(ref, deg, sr, extended=False))


def normalize_peak(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_val = np.max(np.abs(y))
    if max_val < 1e-8:
        return y
    return (y / max_val * peak).astype(np.float32)


class CpuMonitor:
    def __init__(self):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=0.2))

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return float(sum(self._samples) / len(self._samples)) if self._samples else 0.0


def run_timed_with_cpu(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float, float]:
    monitor = CpuMonitor()
    monitor.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    cpu_avg = monitor.stop()
    return result, elapsed, cpu_avg


# ---------------------------------------------------------------------------
# Signal Processing Pipeline
# ---------------------------------------------------------------------------

def bandpass_voice(y: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def enhance_signal_processing(y: np.ndarray, sr: int) -> np.ndarray:
    # Step 1: spectral noise gating
    denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75)
    # Step 2: bandpass filter (voice range 300-3400 Hz)
    filtered = bandpass_voice(denoised, sr)
    # Step 3: noise gate + compressor via pedalboard
    board = Pedalboard([
        NoiseGate(threshold_db=-35.0, ratio=2.0, release_ms=250.0),
        Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=100.0),
    ])
    processed = board(filtered, sr)
    if processed.ndim > 1:
        processed = np.mean(processed, axis=0)
    n = min(len(y), len(processed))
    return processed[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Experiment Driver
# ---------------------------------------------------------------------------

def process_file(input_path: Path) -> ExperimentResult:
    print(f"  Processing: {input_path.name}")
    noisy, duration = load_audio_mono(input_path, METRIC_SR)
    clean_ref = find_clean_reference(input_path)

    enhanced, proc_time, cpu_avg = run_timed_with_cpu(
        lambda: enhance_signal_processing(noisy, METRIC_SR)
    )
    enhanced = normalize_peak(enhanced)

    OUTPUT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_SCRIPT_DIR / f"{input_path.stem}_enhanced.wav"
    sf.write(out_path, enhanced, METRIC_SR, subtype="PCM_16")

    ref = clean_ref if clean_ref is not None else noisy
    stoi_before = compute_stoi(ref, noisy, METRIC_SR)
    stoi_after = compute_stoi(ref, enhanced, METRIC_SR)

    return ExperimentResult(
        filename=input_path.name,
        duration_sec=round(duration, 2),
        approach="SignalProcessing (noisereduce + bandpass + pedalboard)",
        processing_time_sec=round(proc_time, 3),
        cpu_percent_avg=round(cpu_avg, 2),
        stoi_before=round(stoi_before, 4),
        stoi_after=round(stoi_after, 4),
        output_path=str(out_path),
    )


def write_results_csv(results: list[ExperimentResult]) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # filter nan
    return float(sum(clean) / len(clean)) if clean else float("nan")


def generate_report(results: list[ExperimentResult]) -> str:
    avg_time = _mean([r.processing_time_sec for r in results])
    avg_cpu = _mean([r.cpu_percent_avg for r in results])
    avg_stoi_before = _mean([r.stoi_before for r in results])
    avg_stoi_after = _mean([r.stoi_after for r in results])
    avg_stoi_gain = _mean([r.stoi_after - r.stoi_before for r in results])

    lines = [
        "=" * 72,
        "AUDIO ENHANCEMENT EXPERIMENT — SIGNAL PROCESSING REPORT",
        "=" * 72,
        "",
        f"Files processed     : {len(results)}",
        f"Approach            : noisereduce + bandpass filter + pedalboard",
        f"Python version      : 3.12 (no ML model required)",
        "",
        "PER-FILE RESULTS",
        "-" * 72,
        f"{'File':<35} {'Duration':>8} {'Time(s)':>8} {'CPU%':>6} {'STOI Before':>12} {'STOI After':>11}",
        "-" * 72,
    ]

    for r in results:
        dur_min = f"{r.duration_sec/60:.1f}m"
        lines.append(
            f"{r.filename:<35} {dur_min:>8} {r.processing_time_sec:>8.2f} "
            f"{r.cpu_percent_avg:>6.1f} {r.stoi_before:>12.4f} {r.stoi_after:>11.4f}"
        )

    lines += [
        "-" * 72,
        f"{'AVERAGE':<35} {'':>8} {avg_time:>8.2f} {avg_cpu:>6.1f} {avg_stoi_before:>12.4f} {avg_stoi_after:>11.4f}",
        "",
        "AGGREGATE METRICS",
        "-" * 72,
        f"  Avg processing time : {avg_time:.2f} seconds",
        f"  Avg CPU usage       : {avg_cpu:.1f}%",
        f"  Avg STOI before     : {avg_stoi_before:.4f}",
        f"  Avg STOI after      : {avg_stoi_after:.4f}",
        f"  Avg STOI gain       : {avg_stoi_gain:+.4f}",
        "",
        "PIPELINE DETAILS",
        "-" * 72,
        "  Step 1 — noisereduce  : Spectral gating to suppress background noise",
        "  Step 2 — bandpass     : Butterworth filter keeping 300-3400 Hz (voice range)",
        "  Step 3 — NoiseGate    : Silences low-level noise floor (threshold -35 dB)",
        "  Step 4 — Compressor   : Evens out volume dynamics (ratio 3:1)",
        "",
        "STRENGTHS",
        "-" * 72,
        "  + No ML model — lightweight, fast, no GPU or Rust required",
        "  + Works on Python 3.12, no extra toolchain needed",
        "  + Consistent processing speed regardless of audio content",
        "  + Low memory footprint even on 60-minute files",
        "",
        "LIMITATIONS",
        "-" * 72,
        "  - Cannot reconstruct degraded speech (unlike neural models)",
        "  - Less effective on highly variable or non-stationary noise",
        "  - Does not improve speech clarity beyond noise removal",
        "  - Quality ceiling lower than DeepFilterNet3",
        "",
        "=" * 72,
    ]

    report = "\n".join(lines)
    REPORT_TXT.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    print("Audio Enhancement Experiment — Script Only")
    print("=" * 40)

    input_files = discover_input_files(INPUT_DIR)
    if not input_files:
        print(f"No audio files found in {INPUT_DIR}. Add WAV/MP3/M4A files to input/")
        return 1

    print(f"Found {len(input_files)} file(s).\n")

    results = []
    for idx, path in enumerate(input_files, start=1):
        size_mb = path.stat().st_size / 1e6
        print(f"[{idx}/{len(input_files)}] {path.name} ({size_mb:.1f} MB)")
        results.append(process_file(path))
        print()

    write_results_csv(results)
    report = generate_report(results)
    print(report)
    print(f"Results CSV  : {RESULTS_CSV}")
    print(f"Report       : {REPORT_TXT}")
    print(f"Enhanced WAVs: {OUTPUT_SCRIPT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
