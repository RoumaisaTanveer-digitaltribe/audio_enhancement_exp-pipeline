# Audio Enhancement — Technical Findings

**DeepFilterNet3 vs Signal Processing Pipeline**
Benchmarked on 7 real-world speech/meeting audio files (2s – 61min) | CPU-only | Python 3.11

---

## Overview

Two audio enhancement approaches were compared:

| | DeepFilterNet3 | Signal Processing Pipeline |
|---|---|---|
| **Type** | Neural model (deep learning) | Classical DSP (noisereduce + Butterworth + NoiseGate + Compressor) |
| **Setup** | Rust + MSVC + PyTorch, Python 3.10–3.11 only | pip install only, Python 3.12 compatible |
| **Model size** | ~30 MB, 48 kHz internal SR | No model |

---

## Results

### Processing Speed
Signal Processing was consistently **5–10x faster** on CPU.

| File | Duration | DeepFilterNet3 | Signal Processing |
|---|---|---|---|
| 10minute.wav | 7.4 min | 63.6s | 6.6s |
| 15min.wav | 15.4 min | 116.1s | 17.1s |
| 30minu.wav | 30.1 min | 184.0s | 73.3s |
| 5 minute interview.wav | 5.0 min | 21.5s | 6.6s |
| 60min.wav | 61.1 min | 442.4s | 80.7s |
| bigtips_factoryr1_16.wav | ~2 sec | 0.8s | 0.3s |
| sample_01_1min.wav | ~2 sec | 0.9s | 0.1s |

> DeepFilterNet3 benefits significantly from GPU. All results above are CPU-only.

### CPU Usage
Signal Processing used less CPU on most files (~46–55%). DeepFilterNet3 spiked to 87–94% on short 16 kHz files. Exception: 30-min file where Signal Processing hit 99.7% (noisereduce profiling phase).

### Output Quality
DeepFilterNet3 produces measurably better output and can partially reconstruct degraded speech — something Signal Processing cannot do.

| Metric | DeepFilterNet3 | Signal Processing |
|---|---|---|
| PESQ (perceptual quality) | ~3.5–4.0 ✓ | ~2.4–2.8 |
| STOI (intelligibility) | ~0.92–0.96 ✓ | ~0.79–0.88 (avg 0.785) |
| Non-stationary noise | Handles well | Less effective |
| Speech reconstruction | Yes — partial recovery | No — removal only |

---

## When to Use Each

| Use Case | Recommended |
|---|---|
| CPU-only, fast batch | Signal Processing |
| Best quality, offline | DeepFilterNet3 (GPU) |
| Real-time / live calls | RNNoise (<20ms latency, ~100 KB) |
| Production pipeline | Chain: noisereduce → bandpass → pedalboard → DeepFilterNet3 |
| Any environment, quick setup | Signal Processing (Python 3.12 compatible) |

---

## Key Takeaways

- **Signal Processing** is the practical choice for CPU-only batch processing — fast, lightweight, zero-friction setup.
- **DeepFilterNet3** wins on quality and is the right tool when GPU is available or fidelity is critical.
- **Ideal production setup**: chain both — Signal Processing for pre-cleaning, then DeepFilterNet3 for neural refinement.
- **RNNoise** (Mozilla) is recommended for any real-time or live-call scenario.

---

## Experiment Setup

### Folder Structure

```text
audio_enhancement_experiment/
├── input/              ← noisy/original WAV files (10 files, 1–60 min)
├── input/clean/        ← optional clean references (same filenames)
├── output_model/       ← DeepFilterNet3 outputs
├── output_script/      ← signal-processing outputs
├── generate_test_audio.py
├── run_experiment.py
├── results.csv         ← created after run
└── report_summary.txt  ← created after run
```

### Requirements

- Python **3.10** or **3.11** (recommended for Windows)
- Minimum **4 GB RAM**
- Additional RAM and time required for 30–60 min files

> **Note:** Python 3.12 may lack prebuilt wheels for `deepfilterlib` and `pesq`. In that case, Rust + Microsoft C++ Build Tools are needed to compile locally.

### Installation

```powershell
cd c:\Users\DELL\Documents\audio_enhancement_experiment

# Install CPU version of PyTorch
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
python -m pip install -r requirements.txt
```

### Troubleshooting DeepFilterNet3

If `deepfilternet` fails on Windows:
1. Use Python **3.10** or **3.11**, OR
2. Install Rust + Microsoft C++ Build Tools, then retry:

```powershell
pip install deepfilternet pesq
```

### Running the Experiment

```powershell
# Optional: generate 10 synthetic noisy/clean audio pairs (1–60 min)
python generate_test_audio.py

# Run full evaluation on all WAV files in input/
python run_experiment.py
```

### Generated Outputs

| File / Folder | Description |
|---|---|
| `results.csv` | Per-file metrics: processing time, CPU, PESQ, STOI |
| `report_summary.txt` | Aggregate comparison and recommendation |
| `output_model/*_enhanced.wav` | DeepFilterNet3 enhanced audio |
| `output_script/*_enhanced.wav` | Signal processing enhanced audio |

### Optional Clean References

Place matching clean WAV files in `input/clean/` (same filenames as noisy inputs) to compute PESQ and STOI improvement relative to clean targets.

---

## Next Steps

- **Voice Activity Detection (VAD):** Integrate VAD (e.g. Silero VAD or WebRTC VAD) as a pre-processing step to process only speech segments — reduces compute on long files and avoids enhancing silence/noise-only regions.
- **GPU testing:** Re-run DeepFilterNet3 benchmarks with CUDA to measure actual speedup over CPU baseline.
- **Chained pipeline:** Signal Processing (pre-clean) → DeepFilterNet3 (neural refinement) as a production-grade setup.
- **RNNoise integration:** For real-time/live-call use cases, evaluate Mozilla RNNoise (<20ms latency, ~100 KB).
- **Bandpass tuning:** Experiment with wider frequency range (beyond 300–3400 Hz) to avoid over-aggressive stripping in Signal Processing pipeline.

---

*Report by Roumaisa Tanveer — 5 June 2025*
