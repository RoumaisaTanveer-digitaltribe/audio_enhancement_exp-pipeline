"""
Generate 10 synthetic speech-like test files (1–60 min) with additive noise.

Creates paired clean/noisy WAV files so PESQ and STOI can measure true improvement.
Run once before the experiment if /input is empty:

    python generate_test_audio.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

# Match experiment sample rate used for metrics and saving.
SAMPLE_RATE = 16_000

# Target durations in minutes for the 10-file benchmark set.
DURATIONS_MIN = [1, 2, 3, 5, 8, 10, 15, 20, 30, 60]


def _speech_like_signal(duration_sec: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Build a simple voiced signal (harmonics + amplitude envelope) as a clean proxy."""
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float64) / sr

    # Slow formant-like amplitude modulation mimicking syllables.
    envelope = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * math.pi * 2.5 * t))
    envelope *= 0.5 + 0.5 * np.sin(2 * math.pi * 0.7 * t + 0.3)

    f0 = 120.0 + 15.0 * np.sin(2 * math.pi * 0.2 * t)
    phase = np.cumsum(2 * math.pi * f0 / sr)
    voiced = np.sin(phase)
    for harmonic in (2, 3, 4):
        voiced += (0.45 / harmonic) * np.sin(harmonic * phase)

    clean = (envelope * voiced).astype(np.float32)
    clean /= np.max(np.abs(clean)) + 1e-8
    return clean


def _add_noise(clean: np.ndarray, rng: np.random.Generator, snr_db: float = 8.0) -> np.ndarray:
    """Mix white + low-frequency rumble to simulate realistic background noise."""
    n = len(clean)
    white = rng.standard_normal(n).astype(np.float32)
    rumble = np.convolve(
        rng.standard_normal(n).astype(np.float32),
        np.ones(800, dtype=np.float32) / 800,
        mode="same",
    )
    noise = 0.7 * white + 0.3 * rumble
    signal_power = np.mean(clean**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    scale = math.sqrt(signal_power / noise_power) / (10 ** (snr_db / 20))
    noisy = clean + scale * noise
    noisy /= np.max(np.abs(noisy)) + 1e-8
    return noisy.astype(np.float32)


def main() -> None:
    base = Path(__file__).resolve().parent
    input_dir = base / "input"
    clean_dir = input_dir / "clean"
    input_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for idx, minutes in enumerate(DURATIONS_MIN, start=1):
        duration_sec = minutes * 60.0
        print(f"Generating sample_{idx:02d} ({minutes} min)...")

        clean = _speech_like_signal(duration_sec, SAMPLE_RATE, rng)
        noisy = _add_noise(clean, rng, snr_db=6.0 + idx * 0.5)

        stem = f"sample_{idx:02d}_{minutes}min"
        sf.write(input_dir / f"{stem}.wav", noisy, SAMPLE_RATE, subtype="PCM_16")
        sf.write(clean_dir / f"{stem}.wav", clean, SAMPLE_RATE, subtype="PCM_16")

    print(f"Done. Created {len(DURATIONS_MIN)} noisy/clean pairs under {input_dir}")


if __name__ == "__main__":
    main()
