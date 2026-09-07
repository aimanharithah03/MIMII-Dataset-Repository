# MIMII 6_dB_fan — Anomaly Classification Pipeline Documentation

Companion documentation for `mimii_fan_classifier.py`. Describes what each
stage of the pipeline does, why it's structured this way, and how to run it.

---

## 1. Overview

This pipeline trains a binary classifier (normal vs. abnormal) on fan sound
recordings from the MIMII dataset (6 dB SNR condition), evaluates it with
accuracy, F1-score, and a confusion matrix, and saves a model file suitable
for either further analysis (`.joblib`) or edge deployment (`.h5`).

The dataset has 4 physical fan units (`id_00`, `id_02`, `id_04`, `id_06`),
each with its own `normal/` and `abnormal/` recordings. A central question
this pipeline is built to answer is not just "how accurate is the model,"
but **"does it generalize to a fan it has never heard before, or did it
just memorize these 4 specific machines?"** — see Section 4.

---

## 2. Dataset & Folder Structure

The script expects the standard MIMII layout:

```
6_dB_fan/fan/id_00/normal/*.wav
6_dB_fan/fan/id_00/abnormal/*.wav
6_dB_fan/fan/id_02/normal/*.wav
6_dB_fan/fan/id_02/abnormal/*.wav
... (id_04, id_06)
```

`CONFIG["DATASET_DIR"]` should point at the folder that directly contains
the `id_*` subfolders (e.g. `.../6_dB_fan/fan`).

---

## 3. Pipeline Steps

Each training run — regardless of mode — goes through the same stages:

| Step | What happens |
|---|---|
| 1. File discovery | Walks `id_*/normal` and `id_*/abnormal`, collecting file paths and labels (0 = normal, 1 = abnormal). |
| 2. Feature extraction | Each `.wav` is loaded and summarized into one fixed-length vector: MFCCs (40 coefficients) plus spectral centroid, bandwidth, rolloff, zero-crossing rate, and RMS energy — each feature's mean **and** standard deviation over time, giving a 90-dimensional vector per file. |
| 2b. Per-machine normalization | Each machine's feature vectors are z-scored using the mean/std of **that machine's own normal samples only** (see Section 5 for why). |
| 3. Splitting | Data is split into train / validation / test. How, depends on the evaluation mode (Section 4). |
| 4. Training | A classifier is trained on the train split — either a RandomForest or a small neural network (Section 6). |
| 5. Validation sanity check | Quick accuracy/F1 check at the default 0.5 probability threshold. |
| 5b. Threshold tuning | Sweeps thresholds from 0.05–0.95 against the **validation set only**, picks the one that maximizes F1 on the abnormal class. Never touches the test set, to avoid leaking test information into model selection. |
| 6. Test evaluation | Final accuracy, F1, and full classification report, computed on the untouched test set using the tuned threshold. |
| 7. Confusion matrix | Plotted and saved as a PNG. |
| 8. Model saving | Model + chosen threshold saved to disk (format depends on backend — Section 6). |

---

## 4. Evaluation Modes

Set via `CONFIG["EVAL_MODE"]`.

### `"pooled"`
All files from the selected machine(s) are pooled together, then randomly
split into train/val/test. **The same machines appear in all three
splits.** This measures "can it detect faults on machines it has already
seen" — not whether it generalizes to a new unit.

### `"leave_one_group_out"`
Trains on every machine ID **except** `CONFIG["HOLDOUT_MACHINE_ID"]`, and
tests only on that held-out machine, which the model never sees during
training. This is the real test of generalization to an unseen fan.

### `"leave_one_group_out_all"`
Runs `leave_one_group_out` once per machine ID (each machine gets a turn
being the held-out one), then prints and saves a summary comparing all
four — so you can see whether generalization is consistent across
machines, or whether some are notably harder than others.

### Why this distinction matters (a real finding from this project)
Pooled evaluation on this dataset originally scored **~99.6% accuracy**.
Leave-one-out on the same setup scored **22.89%** — worse than just always
guessing "normal." The pooled model had learned to recognize *which
machine* a clip came from rather than genuine fault acoustics, and that
shortcut evaporates the moment it's tested on a machine it hasn't seen.
This is why `leave_one_group_out_all` is the default mode — it's the
number that actually reflects real-world deployment to a new unit.

---

## 5. Per-Machine Baseline Normalization

`CONFIG["NORMALIZE_PER_MACHINE"]` (default `True`).

Raw acoustic features carry a lot of "which physical unit is this" signal
— different fans have different baseline hums, resonances, and motor
characteristics. Z-scoring each machine's samples against **that
machine's own normal-sample mean/std** removes this absolute-scale
signal and leaves "how far is this from what's normal *for this unit*" —
a signal far more likely to transfer to a new, unseen unit.

This mirrors a realistic deployment step: record a short known-normal
calibration period on a newly installed unit, then compare live audio
against that unit's own baseline.

**Measured effect on this project:** with normalization, the leave-one-out
score for the previously-failing holdout machine went from 22.89%
accuracy (0.37 F1) to 86.56% accuracy (0.66 F1) — confirming the model had
been leaning on machine identity rather than fault acoustics.

---

## 6. Model Backends

Set via `CONFIG["MODEL_BACKEND"]`.

| Backend | Saved as | Notes |
|---|---|---|
| `"neural_net"` (default) | `.h5` | Small Keras feed-forward network (Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)), trained with class-balanced weights and early stopping. Required for edge/embedded deployment pipelines (e.g. TensorFlow Lite for Microcontrollers, Infineon ML tooling for PSoC6), which expect Keras/TF models — not scikit-learn trees. |
| `"random_forest"` | `.joblib` | The original classifier, kept for comparison. `.joblib` can bundle the model and threshold together in one file; `.h5` cannot, so the neural-net backend saves the threshold to a companion `..._threshold.txt` file. |

---

## 7. Decision Threshold

By default (`THRESHOLD_MODE = "auto_f1"`), the classifier's cutoff for
"abnormal" is not the default 0.5 — it's chosen by sweeping thresholds
against the validation set and picking whichever maximizes F1 on the
abnormal class. This matters because missing a real fault is usually
costlier than a false alarm, so the default 0.5 cutoff (which "balances"
precision/recall equally) isn't necessarily the right operating point.

**Measured effect on this project:** on the same held-out machine, moving
from threshold 0.5 to the tuned 0.35 raised abnormal-class recall from
0.49 to 0.84 (catching 304 of 361 real faults instead of 178), while
precision only dropped from 0.99 to 0.95.

---

## 8. Outputs Produced

For a single run (`pooled` or `leave_one_group_out`):
- `<model>.h5` (or `.joblib`) + `<model>_threshold.txt` (neural-net backend only)
- `confusion_matrix.png`
- `confusion_matrix_threshold_curve.png`

For `leave_one_group_out_all` (4 runs), all of the above per machine ID
(suffixed `_loo_id_00`, `_loo_id_02`, etc.), plus:
- `leave_one_out_summary.csv` — accuracy/precision/recall/F1/threshold per machine, and the mean
- `leave_one_out_summary.png` — grouped bar chart comparing all machines

---

## 9. Configuration Reference

| Key | Purpose |
|---|---|
| `DATASET_DIR` | Folder containing the `id_*` subfolders |
| `MACHINE_IDS` | `None` (auto-detect) or a specific list of IDs |
| `EVAL_MODE` | `"pooled"` / `"leave_one_group_out"` / `"leave_one_group_out_all"` |
| `HOLDOUT_MACHINE_ID` | Which machine to hold out (single-holdout mode only) |
| `MODEL_BACKEND` | `"neural_net"` (.h5) or `"random_forest"` (.joblib) |
| `NN_HIDDEN_UNITS`, `NN_DROPOUT`, `NN_EPOCHS`, `NN_BATCH_SIZE`, `NN_PATIENCE` | Neural-net training hyperparameters |
| `NORMALIZE_PER_MACHINE` | Whether to z-score each machine against its own normal baseline |
| `THRESHOLD_MODE` | `"auto_f1"` (tune on validation) or `"fixed"` |
| `FIXED_THRESHOLD` | Used only when `THRESHOLD_MODE = "fixed"` |
| `SR`, `N_MFCC` | Audio resample rate and MFCC coefficient count |
| `TEST_SIZE`, `VAL_SIZE` | Split fractions (pooled mode uses both; leave-one-out modes use `VAL_SIZE` only, since the held-out machine is the test set) |
| `RANDOM_STATE` | Seed for reproducible splits |
| `MODEL_OUT`, `CONFUSION_MATRIX_OUT` | Base output paths (extensions/suffixes are added automatically) |

---

## 10. How to Run

1. Install dependencies: `pip install librosa scikit-learn matplotlib joblib tensorflow`
2. Edit the `CONFIG` block at the top of `mimii_fan_classifier.py` — at minimum, confirm `DATASET_DIR` points at your `6_dB_fan/fan` folder.
3. Run the script (Thonny, or `python mimii_fan_classifier.py`).
4. In `leave_one_group_out_all` mode, plot windows appear at several points per fold (threshold curve, confusion matrix) and will pause execution until closed — expect roughly 9 windows across a full run.
5. Review the printed summary table, `leave_one_out_summary.csv`, and `leave_one_out_summary.png` for the overall generalization picture.

---

## 11. Deploying the `.h5` Model to PSoC 6 — DEEPCRAFT™ Model Converter

Training produces a `.h5` file, but the PSoC 6 doesn't run Keras models directly.
[DEEPCRAFT™ Model Converter](https://developer.imagimob.com/deepcraft-model-converter)
(Infineon/Imagimob) is the tool that bridges that gap: it converts the `.h5`
into a `model.c` / `model.h` pair that gets imported into a ModusToolbox™
project and compiled onto the board.

### 11.1 Before converting: check `.h5` compatibility

Model Converter accepts `.h5` files, but only in the pre-Keras-3 HDF5
convention. **If the `.h5` was saved using TensorFlow 2.16+ / Keras 3, it
should be re-saved in `.keras` format instead** — Model Converter supports
`.keras` directly, and mixing conventions is a known source of load errors.
This is the same Keras 3 / TF 2.16+ `.h5` serialization issue already
worked around in the autoencoder pipeline (via `TF_USE_LEGACY_KERAS` and
`compile=False` on load) — worth checking which TensorFlow/Keras version
produced this classifier's `.h5` before feeding it to the converter.

### 11.2 Install & launch

1. Download the installer for your OS from the
   [Infineon Developer Center](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.deepcraftmodelconverter).
2. Run it, choose Quick or Custom installation, accept the license agreement.
3. Once installed, both interfaces are available:
   - **GUI** — launches automatically after install.
   - **CLI** — open a command prompt and run `dcmc --help` to confirm it's on PATH.

### 11.3 What the converter needs — and an important gap

Model Converter only converts **the neural network itself** (the Dense /
Dropout layers). It assumes the input already arrives in exactly the
format the model expects — in this project's case, the 90-dimensional
MFCC + spectral feature vector.

**The MFCC/spectral feature extraction step (Section 3, Step 2) is not
part of what gets converted.** That preprocessing currently runs in
Python via `librosa`. Getting this classifier fully running on-device
means either:
- reimplementing that same feature extraction in C/embedded firmware, or
- checking whether ModusToolbox's ML/DSP libraries can compute
  equivalent features on-device.

This is realistically the next significant piece of engineering work
after code generation — worth planning for, not a detail to discover
later.

### 11.4 Generating the code (CLI)

Basic command, targeting PSoC 6, no quantization (float32):

```
dcmc -m <path to fan_classifier.h5> -o <path to output directory> -t Psoc6
```

This produces `model.c` and `model.h` in the output directory, plus a
`code_generation_report.md` summarizing memory usage and (if calibration/
validation data was supplied) accuracy impact.

### 11.5 Optional: quantization

Quantization shrinks the model and speeds up inference, at some accuracy
cost — generally worth trying on a memory-constrained target like PSoC 6.

```
dcmc -m <path to fan_classifier.h5> -o <output dir> -t Psoc6 --c-data <calibration data path> --quantize-type int8x8
```

- `int8x8` (weights + activations both 8-bit): smallest and fastest, largest potential accuracy loss.
- `int16x8` (activations 16-bit, weights 8-bit): usually higher accuracy, small increase in size/compute.
- The docs recommend generating both and comparing the validation report side by side rather than guessing which is better for a given model.

**Calibration/validation data must be the 90-dim feature vectors this
project already extracts (`X_train`/`X_val`/`X_test`), not raw audio** —
the model's actual input is the feature vector, not the waveform. Supported
formats for this single-input/single-output model: `.npz`, `.csv`, or a
recursive directory (see Model Converter's *Supported Data Formats* docs).
Keep validation data separate from calibration data to avoid biased results.

### 11.6 Validation before flashing

```
dcmc -m <path to fan_classifier.h5> -o <output dir> -t Psoc6 --v-data <validation data path> --v-max-samples <n>
```

Validates the quantized (or non-quantized) model against reference outputs
on the desktop, without needing the physical board — worth doing before
flashing anything.

### 11.7 Dependencies for PSoC 6

The generated code relies on Infineon's `ml-middleware 3.0.1` and
`ml-tflite-micro 3.0.1` libraries specifically for the PSoC 6 target
(PSoC Edge targets use the 3.1.0 versions instead), importable through
the ModusToolbox™ Eclipse IDE library manager.

### 11.8 Reference

Full, current documentation: <https://developer.imagimob.com/deepcraft-model-converter>
