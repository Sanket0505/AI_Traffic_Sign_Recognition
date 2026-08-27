"""
validate_qemu.py — Validate INT8 TFLite model on the full validation set.

Uses the TFLite interpreter with the EXACT same preprocessing as the QEMU
Zephyr app (uint8 RGB → int8 via pixel−128). Results are bit-identical to
running on QEMU because the same INT8 model and quantization are used.

Also reports model size and per-class breakdown.
"""
import os
import sys
import time
import numpy as np
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model", "quantized_model", "model_quant.tflite")
VAL_DIR    = os.path.join(ROOT_DIR, "data", "Validation")

# ── Constants (must match inference.h / class_mapping.h) ────────────────
IMG_H, IMG_W, IMG_C = 48, 48, 3
IMG_BYTES  = IMG_H * IMG_W * IMG_C          # 6 912
NUM_CLASSES = 17
CLASS_IDS  = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]

CLASS_NAMES = {
    0: "Speed 5",   2: "Speed 30",  3: "Speed 40",  4: "Speed 50",
    5: "Speed 60",  6: "Speed 70",  7: "Speed 80", 24: "Go Right",
   43: "Right/Straight", 69: "Height", 70: "Weight", 72: "Length",
   83: "Steep Down", 84: "Steep Up", 85: "Narrow Rd", 86: "Narrow Br",
   87: "Unknown",
}


def load_validation_set(val_dir):
    """Return list of (image_path, ground_truth_class_id)."""
    samples = []
    for folder in sorted(os.listdir(val_dir), key=lambda x: int(x)):
        cid = int(folder)
        if cid not in CLASS_IDS:
            continue
        folder_path = os.path.join(val_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            samples.append((os.path.join(folder_path, fname), cid))
    return samples


def preprocess(image_path):
    """Load → RGB → resize 48×48 → uint8 → int8 (pixel − 128).
    Exactly replicates the QEMU app's run_inference() quantisation."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    pixels = np.array(img, dtype=np.uint8)                # [48,48,3] uint8
    int8_pixels = (pixels.astype(np.int32) - 128).astype(np.int8)
    return int8_pixels.reshape(1, IMG_H, IMG_W, IMG_C)


def main():
    # ── Sanity checks ───────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"ERROR: model not found → {MODEL_PATH}")
    if not os.path.isdir(VAL_DIR):
        sys.exit(f"ERROR: validation dir not found → {VAL_DIR}")

    model_bytes = os.path.getsize(MODEL_PATH)
    print(f"Model : {MODEL_PATH}")
    print(f"Size  : {model_bytes:,} bytes ({model_bytes/1024:.1f} KB)")

    # ── Load TFLite interpreter ─────────────────────────────────────────
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()

    inp_det  = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]
    inp_idx  = inp_det["index"]
    out_idx  = out_det["index"]

    print(f"Input : shape={inp_det['shape']}  dtype={inp_det['dtype'].__name__}  "
          f"scale={inp_det['quantization'][0]:.6f}  zp={inp_det['quantization'][1]}")
    print(f"Output: shape={out_det['shape']}  dtype={out_det['dtype'].__name__}")

    # ── Load validation images ──────────────────────────────────────────
    samples = load_validation_set(VAL_DIR)
    print(f"\nValidation images: {len(samples)}")

    # ── Run inference ───────────────────────────────────────────────────
    correct = 0
    total   = 0
    per_cls = {c: {"tp": 0, "n": 0} for c in CLASS_IDS}
    times_ms = []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    t0_all = time.perf_counter()

    for i, (path, gt_cid) in enumerate(samples):
        input_data = preprocess(path)

        t0 = time.perf_counter()
        interp.set_tensor(inp_idx, input_data)
        interp.invoke()
        output = interp.get_tensor(out_idx)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000)

        pred_idx = int(np.argmax(output[0]))
        pred_cid = CLASS_IDS[pred_idx]

        gt_idx = CLASS_IDS.index(gt_cid)
        confusion[gt_idx, pred_idx] += 1

        per_cls[gt_cid]["n"] += 1
        total += 1
        if pred_cid == gt_cid:
            correct += 1
            per_cls[gt_cid]["tp"] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>4}/{len(samples)}]  running acc = "
                  f"{correct/total*100:.1f}%")

    t_total = time.perf_counter() - t0_all

    # ── Report ──────────────────────────────────────────────────────────
    accuracy = correct / total * 100 if total else 0
    print()
    print("=" * 64)
    print("  VALIDATION RESULTS")
    print("=" * 64)
    print(f"  Overall accuracy : {correct}/{total} = {accuracy:.1f}%")
    print(f"  Model size (INT8): {model_bytes:,} bytes ({model_bytes/1024:.1f} KB)")
    print(f"  Total wall time  : {t_total:.2f} s  "
          f"({t_total/len(samples)*1000:.1f} ms / image)")
    print(f"  Avg inference    : {np.mean(times_ms):.2f} ms  "
          f"(min {np.min(times_ms):.2f}, max {np.max(times_ms):.2f})")
    print()

    # Per-class table
    hdr = f"{'ID':>5}  {'Name':<16} {'Correct':>7} {'Total':>5} {'Acc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for cid in CLASS_IDS:
        n  = per_cls[cid]["n"]
        tp = per_cls[cid]["tp"]
        acc = tp / n * 100 if n else 0
        tag = CLASS_NAMES.get(cid, "")
        print(f"{cid:>5}  {tag:<16} {tp:>7} {n:>5} {acc:>6.1f}%")

    print("-" * len(hdr))
    print(f"{'':>5}  {'TOTAL':<16} {correct:>7} {total:>5} {accuracy:>6.1f}%")
    print()

    # ── Confusion matrix (compact) ──────────────────────────────────────
    print("Confusion matrix (rows=true, cols=predicted, class indices 0-16):")
    print("     " + " ".join(f"{i:>3}" for i in range(NUM_CLASSES)))
    for r in range(NUM_CLASSES):
        row_str = " ".join(f"{confusion[r,c]:>3}" for c in range(NUM_CLASSES))
        print(f" {r:>2}: {row_str}")

    print(f"\nDone. Model is {'UNTRAINED (random weights)' if accuracy < 15 else 'trained'}.")


if __name__ == "__main__":
    main()
