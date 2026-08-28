"""
MIMII 6_dB_fan — Normal vs Abnormal Classification Pipeline
=============================================================
Train / Validation / Test with Accuracy, F1-score, and Confusion Matrix.

Three evaluation modes (set CONFIG["EVAL_MODE"]):

  "pooled"                 - files from all selected machine IDs are
                              pooled, then randomly split into
                              train/val/test.

  "leave_one_group_out"    - trains on every machine ID EXCEPT
                              CONFIG["HOLDOUT_MACHINE_ID"], tests only on
                              that held-out machine.

  "leave_one_group_out_all"- runs leave_one_group_out once PER machine ID,
                              then prints/saves a summary comparing all of
                              them.

Model backend (CONFIG["MODEL_BACKEND"]):

  "neural_net"    (default) - a small Keras feed-forward network, saved as
                               a .h5 file. Use this if you need a model
                               file for edge/embedded deployment (e.g.
                               TensorFlow Lite for Microcontrollers, or
                               Infineon's ML tooling for PSoC6) — those
                               pipelines expect Keras/TF models, not
                               scikit-learn trees. Because .h5 only stores
                               model architecture + weights (no extra
                               metadata), the chosen decision threshold is
                               saved alongside it in a small companion
                               "<model>_threshold.txt" file.

  "random_forest"            - the original scikit-learn RandomForest,
                               saved as .joblib (model + threshold
                               bundled together in one file). Kept for
                               comparison against the neural net.

Per-machine baseline normalization (CONFIG["NORMALIZE_PER_MACHINE"]):

  When True (default), each machine's feature vectors are z-scored using
  the mean/std of THAT MACHINE'S OWN normal samples, before pooling or
  training. Mirrors a realistic deployment: record a short known-normal
  calibration period on a new unit, then compare live audio against that
  unit's own baseline. This also gives the neural net well-scaled inputs,
  which helps training regardless of the generalization benefit.

Decision-threshold tuning (CONFIG["THRESHOLD_MODE"]):

  "auto_f1" (default) - sweeps thresholds from 0.05 to 0.95 on the
                         VALIDATION set only (never the test machine), and
                         picks the threshold that maximizes F1 for the
                         abnormal class. A precision/recall/F1-vs-threshold
                         plot is saved per run.

  "fixed"             - uses CONFIG["FIXED_THRESHOLD"] as-is.

Expected folder structure (standard MIMII layout):

    6_dB_fan/fan/id_00/normal/*.wav
    6_dB_fan/fan/id_00/abnormal/*.wav
    ... (id_02, id_04, id_06, etc.)

No CLI arguments — just edit the CONFIG block below and run in Thonny
(or any Python environment).

Dependencies (install once):
    pip install librosa scikit-learn matplotlib joblib tensorflow
"""

import os
import csv
import glob
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from tensorflow import keras

# ============================================================
# CONFIG — edit these before running
# ============================================================
CONFIG = {
    # Folder that directly contains the id_00, id_02, id_04, id_06 subfolders
    "DATASET_DIR": r"D:\MIMII Dataset_Big\6_dB_fan\fan",

    # Set to None to auto-detect every id_* folder, or list specific ones,
    # e.g. ["id_00", "id_02"]. This is the candidate pool of machines.
    "MACHINE_IDS": None,

    # "pooled", "leave_one_group_out", or "leave_one_group_out_all"
    "EVAL_MODE": "leave_one_group_out_all",

    # Only used when EVAL_MODE == "leave_one_group_out" (single-holdout mode)
    "HOLDOUT_MACHINE_ID": "id_06",

    # "neural_net" (saves .h5, for edge deployment) or "random_forest"
    # (saves .joblib). See module docstring above.
    "MODEL_BACKEND": "neural_net",
    "NN_HIDDEN_UNITS": (64, 32),   # sizes of the two hidden Dense layers
    "NN_DROPOUT": 0.3,
    "NN_EPOCHS": 60,
    "NN_BATCH_SIZE": 32,
    "NN_PATIENCE": 8,              # early-stopping patience, in epochs

    # z-score each machine's features against its OWN normal-sample
    # baseline before training/testing. See module docstring above.
    "NORMALIZE_PER_MACHINE": True,

    # "auto_f1" tunes the decision threshold on the validation set;
    # "fixed" uses FIXED_THRESHOLD as-is. See module docstring above.
    "THRESHOLD_MODE": "auto_f1",
    "FIXED_THRESHOLD": 0.5,
    "THRESHOLD_SWEEP": [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)],

    "SR": 16000,          # audio is resampled to this rate before feature extraction
    "N_MFCC": 40,          # number of MFCC coefficients to compute

    "TEST_SIZE": 0.15,     # ("pooled" mode only) fraction of data reserved for test
    "VAL_SIZE": 0.15,      # fraction of data reserved for validation, all modes

    "RANDOM_STATE": 42,    # fixed seed so splits/results are reproducible

    "MODEL_OUT": r"D:\MIMII Dataset_Big\6_dB_fan\fan_classifier.h5",
    "CONFUSION_MATRIX_OUT": r"D:\MIMII Dataset_Big\6_dB_fan\confusion_matrix.png",
}


# ============================================================
# Step 1 — Find every .wav file and its label
# ============================================================
def list_machine_ids(dataset_dir):
    """Returns every id_* folder found directly under dataset_dir, sorted."""
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(dataset_dir, "id_*"))
        if os.path.isdir(p)
    )


def find_wav_files(dataset_dir, machine_ids=None):
    """Walks id_*/normal and id_*/abnormal folders and returns
    (filepaths, labels, machine_ids_used). label 0 = normal, 1 = abnormal."""
    if machine_ids is None:
        machine_ids = list_machine_ids(dataset_dir)

    filepaths, labels = [], []
    for mid in machine_ids:
        for folder_name, label_value in [("normal", 0), ("abnormal", 1)]:
            folder = os.path.join(dataset_dir, mid, folder_name)
            wavs = glob.glob(os.path.join(folder, "*.wav"))
            filepaths.extend(wavs)
            labels.extend([label_value] * len(wavs))

    return filepaths, labels, machine_ids


# ============================================================
# Step 2 — Turn one audio file into a fixed-length feature vector
# ============================================================
def extract_features(filepath, sr, n_mfcc):
    """Loads a wav file and summarizes it into a single feature vector
    using MFCCs plus a few standard spectral descriptors. Each feature's
    mean AND standard deviation over time are kept, which is why the
    vector length is double the number of raw features."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    parts = []
    for feat in (mfcc, centroid, bandwidth, rolloff, zcr, rms):
        parts.append(feat.mean(axis=1))
        parts.append(feat.std(axis=1))

    return np.concatenate(parts)


def build_dataset(filepaths, labels, sr, n_mfcc):
    """Runs extract_features over every file and stacks the results
    into an (X, y) pair ready for training."""
    X, y = [], []
    for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
        try:
            X.append(extract_features(fp, sr, n_mfcc))
            y.append(lbl)
        except Exception as e:
            print(f"  [skipped] {fp} -> {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(filepaths):
            print(f"  extracted {i + 1}/{len(filepaths)} files")

    return np.array(X, dtype=np.float32), np.array(y)


def build_dataset_for_machine(dataset_dir, machine_id, sr, n_mfcc):
    """Convenience wrapper: find + extract features for a single machine ID."""
    files, labels, _ = find_wav_files(dataset_dir, [machine_id])
    X, y = build_dataset(files, labels, sr, n_mfcc)
    return X, y


# ============================================================
# Per-machine baseline normalization
# ============================================================
def normalize_by_own_normal_baseline(X, y):
    """z-scores X using the mean/std of ONLY the normal (y == 0) rows."""
    normal_mask = (y == 0)
    if normal_mask.sum() == 0:
        raise ValueError("No normal samples available to compute a baseline from.")

    mean = X[normal_mask].mean(axis=0)
    std = X[normal_mask].std(axis=0)
    std[std == 0] = 1e-8  # avoid divide-by-zero on any constant feature column

    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std


def load_machine_features(dataset_dir, machine_id, sr, n_mfcc, normalize):
    """Loads one machine's (X, y), optionally normalized against its own
    normal baseline. Returns (X, y)."""
    X, y = build_dataset_for_machine(dataset_dir, machine_id, sr, n_mfcc)
    if normalize and len(X) > 0:
        X, _, _ = normalize_by_own_normal_baseline(X, y)
    return X, y


# ============================================================
# Model backends — neural network (.h5) or RandomForest (.joblib)
# ============================================================
def build_nn_model(input_dim, cfg):
    units_a, units_b = cfg["NN_HIDDEN_UNITS"]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units_a, activation="relu"),
        keras.layers.Dropout(cfg["NN_DROPOUT"]),
        keras.layers.Dense(units_b, activation="relu"),
        keras.layers.Dropout(cfg["NN_DROPOUT"]),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(X_train, y_train, X_val, y_val, cfg):
    if cfg["MODEL_BACKEND"] == "neural_net":
        model = build_nn_model(X_train.shape[1], cfg)
        model.summary()

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): w for c, w in zip(classes, weights)}

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["NN_PATIENCE"], restore_best_weights=True
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["NN_EPOCHS"],
            batch_size=cfg["NN_BATCH_SIZE"],
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=2,
        )
        return model

    elif cfg["MODEL_BACKEND"] == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, random_state=cfg["RANDOM_STATE"], class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        return clf

    else:
        raise SystemExit(f"Unknown MODEL_BACKEND: {cfg['MODEL_BACKEND']!r}")


def get_proba(model, X, backend):
    """Returns P(abnormal) for each row, regardless of backend."""
    if backend == "neural_net":
        return model.predict(X, verbose=0).flatten()
    return model.predict_proba(X)[:, 1]


def save_model(model, chosen_threshold, model_out, backend):
    if backend == "neural_net":
        model.save(model_out)  # .h5 extension -> single-file HDF5 format
        # .h5 only stores architecture + weights, so the threshold needs
        # its own small file alongside it.
        threshold_path = os.path.splitext(model_out)[0] + "_threshold.txt"
        with open(threshold_path, "w") as f:
            f.write(str(chosen_threshold))
        print(f"  model saved to {model_out}")
        print(f"  threshold saved to {threshold_path} "
              f"(.h5 can't hold extra metadata the way .joblib can)")
    else:
        joblib.dump({"model": model, "threshold": chosen_threshold}, model_out)
        print(f"  saved to {model_out}")


def build_output_path(base_path, ext, suffix=None):
    """Swaps base_path's extension for ext, optionally inserting a suffix
    before it, e.g. build_output_path('model.h5', '.h5', 'loo_id_06')
    -> 'model_loo_id_06.h5'."""
    base_no_ext = os.path.splitext(base_path)[0]
    if suffix:
        return f"{base_no_ext}_{suffix}{ext}"
    return f"{base_no_ext}{ext}"


# ============================================================
# Decision-threshold tuning
# ============================================================
def tune_threshold_on_validation(model, X_val, y_val, thresholds, backend):
    """Sweeps candidate thresholds against the validation set's predicted
    probabilities and returns (best_result, full_sweep). Never touches
    the test set."""
    proba = get_proba(model, X_val, backend)

    sweep = []
    best = None
    for t in thresholds:
        preds = (proba >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        row = {"threshold": t, "precision": p, "recall": r, "f1": f1}
        sweep.append(row)
        if best is None or f1 > best["f1"]:
            best = row

    return best, sweep


def plot_threshold_curve(sweep, chosen_threshold, out_path):
    ts = [r["threshold"] for r in sweep]
    ps = [r["precision"] for r in sweep]
    rs = [r["recall"] for r in sweep]
    fs = [r["f1"] for r in sweep]

    plt.figure()
    plt.plot(ts, ps, label="precision (abnormal)")
    plt.plot(ts, rs, label="recall (abnormal)")
    plt.plot(ts, fs, label="f1 (abnormal)")
    plt.axvline(chosen_threshold, color="gray", linestyle="--",
                label=f"chosen = {chosen_threshold:.2f}")
    plt.xlabel("decision threshold")
    plt.ylabel("score")
    plt.title("Precision / Recall / F1 vs Threshold (validation set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


# ============================================================
# Shared training + evaluation + saving logic (used by all modes)
# ============================================================
def train_evaluate_and_save(X_train, y_train, X_val, y_val, X_test, y_test,
                             cfg, model_out, cm_out, cm_title, threshold_curve_out):
    backend = cfg["MODEL_BACKEND"]
    backend_label = "neural network" if backend == "neural_net" else "RandomForest"

    print(f"\nStep 4: Training the classifier ({backend_label})...")
    model = train_model(X_train, y_train, X_val, y_val, cfg)

    print("\nStep 5: Validation check at the default 0.5 threshold (sanity check)...")
    default_val_pred = (get_proba(model, X_val, backend) >= 0.5).astype(int)
    print(f"  val accuracy : {accuracy_score(y_val, default_val_pred):.4f}")
    print(f"  val F1-score : {f1_score(y_val, default_val_pred):.4f}")

    print("\nStep 5b: Tuning decision threshold on the validation set "
          "(the test machine is never used here)...")
    if cfg["THRESHOLD_MODE"] == "auto_f1":
        best, sweep = tune_threshold_on_validation(model, X_val, y_val, cfg["THRESHOLD_SWEEP"], backend)
        chosen_threshold = best["threshold"]
        print("  threshold | precision | recall  | f1")
        for row in sweep:
            marker = "  <- chosen" if row["threshold"] == chosen_threshold else ""
            print(f"    {row['threshold']:.2f}    |   {row['precision']:.3f}   | "
                  f"{row['recall']:.3f}  | {row['f1']:.3f}{marker}")
        print(f"  chosen threshold = {chosen_threshold:.2f} "
              f"(maximizes abnormal-class F1 on validation)")
        plot_threshold_curve(sweep, chosen_threshold, threshold_curve_out)
        print(f"  threshold curve saved to {threshold_curve_out}")
    else:
        chosen_threshold = cfg["FIXED_THRESHOLD"]
        print(f"  using fixed threshold = {chosen_threshold:.2f} (THRESHOLD_MODE='fixed')")

    print("\nStep 6: Final evaluation on the held-out test set...")
    test_proba = get_proba(model, X_test, backend)
    test_pred = (test_proba >= chosen_threshold).astype(int)
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    cm = confusion_matrix(y_test, test_pred)

    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1-score : {f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, test_pred, target_names=["normal", "abnormal"]))

    print("Step 7: Saving confusion matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "abnormal"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{cm_title}\n(threshold={chosen_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(cm_out)
    plt.show()
    print(f"  saved to {cm_out}")

    print("\nStep 8: Saving the trained model + chosen threshold...")
    save_model(model, chosen_threshold, model_out, backend)

    return {
        "model": model,
        "threshold": chosen_threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# Mode A — pooled: random split across all selected machines
# ============================================================
def run_pooled(cfg):
    machine_ids = cfg["MACHINE_IDS"] if cfg["MACHINE_IDS"] else list_machine_ids(cfg["DATASET_DIR"])
    print(f"Step 1: Machine IDs used: {machine_ids}")

    print("\nStep 2: Extracting features per machine"
          + (" (normalizing each against its own normal baseline)..." if cfg["NORMALIZE_PER_MACHINE"] else "..."))
    X_parts, y_parts = [], []
    for mid in machine_ids:
        X_m, y_m = load_machine_features(
            cfg["DATASET_DIR"], mid, cfg["SR"], cfg["N_MFCC"], cfg["NORMALIZE_PER_MACHINE"]
        )
        print(f"  {mid}: {len(X_m)} files (normal={int((y_m == 0).sum())}, abnormal={int((y_m == 1).sum())})")
        X_parts.append(X_m)
        y_parts.append(y_m)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"  combined feature matrix shape: {X.shape}")

    if len(X) == 0:
        raise SystemExit("No .wav files found — check CONFIG['DATASET_DIR'].")

    print("\nStep 3: Splitting into train / validation / test sets...")
    holdout_fraction = cfg["TEST_SIZE"] + cfg["VAL_SIZE"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=cfg["RANDOM_STATE"]
    )
    test_fraction_of_holdout = cfg["TEST_SIZE"] / holdout_fraction
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_fraction_of_holdout,
        stratify=y_holdout, random_state=cfg["RANDOM_STATE"]
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    model_ext = ".h5" if cfg["MODEL_BACKEND"] == "neural_net" else ".joblib"
    train_evaluate_and_save(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg,
        model_out=build_output_path(cfg["MODEL_OUT"], model_ext),
        cm_out=build_output_path(cfg["CONFUSION_MATRIX_OUT"], ".png"),
        cm_title="MIMII 6_dB_fan — Confusion Matrix (Test Set)",
        threshold_curve_out=build_output_path(cfg["CONFUSION_MATRIX_OUT"], ".png", "threshold_curve"),
    )


# ============================================================
# Mode B — leave_one_group_out: test on ONE machine never seen in training
# ============================================================
def run_single_leave_one_out(cfg, holdout_id, all_ids):
    if holdout_id not in all_ids:
        raise SystemExit(f"Holdout machine '{holdout_id}' not found among detected IDs: {all_ids}")

    train_ids = [m for m in all_ids if m != holdout_id]
    print(f"\n{'=' * 60}\nLeave-one-out run — held-out machine: {holdout_id}\n{'=' * 60}")
    print(f"Step 1: Training machines: {train_ids}   Held-out test machine: {holdout_id}")

    print("\nStep 2a: Extracting features for training machines"
          + (" (each normalized against its own normal baseline)..." if cfg["NORMALIZE_PER_MACHINE"] else "..."))
    X_parts, y_parts = [], []
    for mid in train_ids:
        X_m, y_m = load_machine_features(
            cfg["DATASET_DIR"], mid, cfg["SR"], cfg["N_MFCC"], cfg["NORMALIZE_PER_MACHINE"]
        )
        print(f"  {mid}: {len(X_m)} files (normal={int((y_m == 0).sum())}, abnormal={int((y_m == 1).sum())})")
        X_parts.append(X_m)
        y_parts.append(y_m)
    X_pool = np.vstack(X_parts)
    y_pool = np.concatenate(y_parts)

    print("\nStep 2b: Extracting features for the held-out machine"
          + (" (normalized against ITS OWN normal baseline)..." if cfg["NORMALIZE_PER_MACHINE"] else "..."))
    X_test, y_test = load_machine_features(
        cfg["DATASET_DIR"], holdout_id, cfg["SR"], cfg["N_MFCC"], cfg["NORMALIZE_PER_MACHINE"]
    )
    print(f"  {holdout_id}: {len(X_test)} files "
          f"(normal={int((y_test == 0).sum())}, abnormal={int((y_test == 1).sum())})")

    if len(X_pool) == 0 or len(X_test) == 0:
        raise SystemExit("No .wav files found for training pool or holdout machine — check CONFIG.")

    total_files = len(X_pool) + len(X_test)
    print(f"\n  overall split by file count: training machines={len(X_pool) / total_files * 100:.1f}%  "
          f"held-out machine={len(X_test) / total_files * 100:.1f}%")

    print("\nStep 3: Splitting training pool into train / validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=cfg["VAL_SIZE"], stratify=y_pool, random_state=cfg["RANDOM_STATE"]
    )
    print(f"  train={len(X_train)}  val={len(X_val)}  test(holdout machine)={len(X_test)}")

    model_ext = ".h5" if cfg["MODEL_BACKEND"] == "neural_net" else ".joblib"
    suffix = f"loo_{holdout_id}"
    model_out = build_output_path(cfg["MODEL_OUT"], model_ext, suffix)
    cm_out = build_output_path(cfg["CONFUSION_MATRIX_OUT"], ".png", suffix)
    threshold_curve_out = build_output_path(cfg["CONFUSION_MATRIX_OUT"], ".png", f"{suffix}_threshold_curve")
    cm_title = f"MIMII 6_dB_fan — Leave-One-Out Confusion Matrix (test={holdout_id})"

    result = train_evaluate_and_save(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg,
        model_out=model_out, cm_out=cm_out, cm_title=cm_title,
        threshold_curve_out=threshold_curve_out,
    )
    result["holdout_machine"] = holdout_id
    result["n_test_files"] = len(X_test)
    return result


def run_leave_one_group_out(cfg):
    all_ids = cfg["MACHINE_IDS"] if cfg["MACHINE_IDS"] else list_machine_ids(cfg["DATASET_DIR"])
    run_single_leave_one_out(cfg, cfg["HOLDOUT_MACHINE_ID"], all_ids)


# ============================================================
# Mode C — leave_one_group_out_all: loop every machine as holdout,
# then summarize
# ============================================================
def run_leave_one_group_out_all(cfg):
    all_ids = cfg["MACHINE_IDS"] if cfg["MACHINE_IDS"] else list_machine_ids(cfg["DATASET_DIR"])
    print(f"Running leave-one-out for all {len(all_ids)} machine IDs: {all_ids}")

    results = [run_single_leave_one_out(cfg, holdout_id, all_ids) for holdout_id in all_ids]
    summarize_leave_one_out_results(results, cfg)


def summarize_leave_one_out_results(results, cfg):
    print(f"\n{'=' * 70}\nSUMMARY — leave-one-out across all machine IDs\n{'=' * 70}")
    print(f"{'machine':<10}{'threshold':<11}{'accuracy':<10}{'precision':<11}{'recall':<9}{'f1':<8}{'n_test':<8}")
    for r in results:
        print(f"{r['holdout_machine']:<10}{r['threshold']:<11.2f}{r['accuracy']:<10.4f}"
              f"{r['precision']:<11.4f}{r['recall']:<9.4f}{r['f1']:<8.4f}{r['n_test_files']:<8}")

    n = len(results)
    avg_acc = sum(r["accuracy"] for r in results) / n
    avg_prec = sum(r["precision"] for r in results) / n
    avg_recall = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    print("-" * 70)
    print(f"{'MEAN':<10}{'':<11}{avg_acc:<10.4f}{avg_prec:<11.4f}{avg_recall:<9.4f}{avg_f1:<8.4f}")

    csv_path = os.path.join(os.path.dirname(cfg["MODEL_OUT"]) or ".", "leave_one_out_summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["holdout_machine", "threshold", "accuracy", "precision", "recall", "f1", "n_test_files"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSummary CSV saved to {csv_path}")

    machines = [r["holdout_machine"] for r in results]
    x = np.arange(len(machines))
    width = 0.2

    plt.figure(figsize=(8, 5))
    plt.bar(x - 1.5 * width, [r["accuracy"] for r in results], width, label="accuracy")
    plt.bar(x - 0.5 * width, [r["precision"] for r in results], width, label="precision (abnormal)")
    plt.bar(x + 0.5 * width, [r["recall"] for r in results], width, label="recall (abnormal)")
    plt.bar(x + 1.5 * width, [r["f1"] for r in results], width, label="f1 (abnormal)")
    plt.xticks(x, machines)
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Leave-One-Out Generalization Summary (per held-out machine)")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(cfg["MODEL_OUT"]) or ".", "leave_one_out_summary.png")
    plt.savefig(chart_path)
    plt.show()
    print(f"Summary chart saved to {chart_path}")


# ============================================================
# Main
# ============================================================
def main():
    cfg = CONFIG
    if cfg["EVAL_MODE"] == "leave_one_group_out_all":
        run_leave_one_group_out_all(cfg)
    elif cfg["EVAL_MODE"] == "leave_one_group_out":
        run_leave_one_group_out(cfg)
    elif cfg["EVAL_MODE"] == "pooled":
        run_pooled(cfg)
    else:
        raise SystemExit(f"Unknown EVAL_MODE: {cfg['EVAL_MODE']!r}")


if __name__ == "__main__":
    main()