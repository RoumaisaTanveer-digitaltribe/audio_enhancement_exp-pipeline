# Audio Enhancement Experiment

Compare **DeepFilterNet3** (deep learning model) vs **classical signal processing** on WAV files stored in the `input/` directory.

---

## Folder Structure

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

---

## Requirements

- Python **3.10** or **3.11** (recommended for Windows)
- Minimum **4 GB RAM**
- Additional RAM and processing time may be required for long audio files (30–60 minutes)

### Important Note

Python **3.12** may not have prebuilt wheels available for:

- `deepfilterlib`
- `pesq`

In that case, you may need:

- Rust
- Microsoft C++ Build Tools

to compile dependencies locally.

---

## Installation

Open PowerShell and run:

```powershell
cd c:\Users\DELL\Documents\audio_enhancement_experiment

# Install CPU version of PyTorch (required by DeepFilterNet3)
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all project dependencies
python -m pip install -r requirements.txt
```

---

## Troubleshooting DeepFilterNet3 Installation

If `deepfilternet` fails to install on Windows:

1. Use Python **3.10** or **3.11**, OR
2. Install:
   - Rust
   - Microsoft C++ Build Tools

Then retry:

```powershell
pip install deepfilternet pesq
```

---

## Running the Experiment

### Optional: Generate Test Data

Create 10 synthetic noisy/clean audio pairs with durations ranging from 1 to 60 minutes:

```powershell
python generate_test_audio.py
```

### Run Full Evaluation

Process every WAV file in the `input/` directory using both enhancement approaches:

```powershell
python run_experiment.py
```

---

## Performance Notes

- DeepFilterNet3 can be computationally intensive on CPU.
- Processing 30–60 minute recordings may take a significant amount of time.
- It is recommended to test the pipeline first using shorter audio files.

---

## Generated Outputs

| File / Folder | Description |
|---------------|-------------|
| `results.csv` | Per-file metrics including processing time, CPU usage, PESQ, and STOI scores |
| `report_summary.txt` | Aggregate comparison and final recommendation |
| `output_model/*_enhanced.wav` | Audio enhanced using DeepFilterNet3 |
| `output_script/*_enhanced.wav` | Audio enhanced using classical signal processing |

---

## Optional Clean References

For objective quality evaluation, place matching clean WAV files inside:

```text
input/clean/
```

The filenames must match the corresponding noisy files.

When clean references are available, the experiment computes:

- PESQ improvement
- STOI improvement

relative to the clean target audio.
