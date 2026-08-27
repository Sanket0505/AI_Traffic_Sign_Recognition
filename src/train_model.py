"""
train_model.py — Train the traffic sign recognition CNN, quantize to INT8,
and generate model_data.h for the Zephyr/QEMU application.

Architecture : 5-block depthwise-separable CNN + GAP + Dense(17)
Input        : 48×48×3 RGB, normalised to [0,1]
Quantisation : Full-integer INT8 post-training quantisation (PTQ)
Output       : model_quant.tflite  +  model_data.h (C header)
"""
import os
import sys
import time
import numpy as np
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
IMG_H, IMG_W, IMG_C = 48, 48, 3
NUM_CLASSES = 17
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 5e-4

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR   = os.path.join(ROOT_DIR, "data", "Train")
VAL_DIR     = os.path.join(ROOT_DIR, "data", "Validation")
MODEL_DIR   = os.path.join(ROOT_DIR, "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "quantized_model", "model_quant.tflite")
HEADER_PATH = os.path.join(SCRIPT_DIR, "qemu_app", "src", "model_data.h")
HISTORY_PNG = os.path.join(MODEL_DIR, "training_history.png")


# ═══════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_dataset(data_dir):
    """Load images from class sub-folders → numpy arrays.
    Returns (images [N,48,48,3] float32 [0,1]), (labels [N] int)."""
    images, labels = [], []
    for folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        cid = int(folder)
        if cid not in CLASS_IDS:
            continue
        idx = CLASS_IDS.index(cid)
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(idx)
            except Exception as e:
                print(f"  WARN: skip {fpath}: {e}")
    return np.array(images), np.array(labels)


# ═══════════════════════════════════════════════════════════════════════
#  2. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
def _dw_sep_block(x, filters, stride=1):
    """Depthwise 3×3 → Pointwise 1×1 → BN → ReLU."""
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               use_bias=False)(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_model():
    """5-block DS-CNN: Conv(16,s2)→DW24→DW32(s2)→DW48(s2)→DW64→GAP→Dense(17)."""
    inp = keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="image")

    # Block 1: standard Conv2D stride-2  →  24×24×16
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2-5: depthwise-separable
    x = _dw_sep_block(x, 24, stride=1)   # 24×24×24
    x = _dw_sep_block(x, 32, stride=2)   # 12×12×32
    x = _dw_sep_block(x, 48, stride=2)   #  6× 6×48
    x = _dw_sep_block(x, 64, stride=1)   #  6× 6×64

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)

    return keras.Model(inputs=inp, outputs=out, name="traffic_sign_cnn")


# ═══════════════════════════════════════════════════════════════════════
#  3. CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════
def compute_class_weights(labels, n_classes):
    """Balanced class weights: total / (n_classes × count_per_class)."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)          # avoid divide-by-zero
    weights = len(labels) / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ═══════════════════════════════════════════════════════════════════════
#  4. INT8 QUANTISATION
# ═══════════════════════════════════════════════════════════════════════
def convert_to_int8(keras_model, x_cal, output_path):
    """Full-integer INT8 PTQ with representative calibration data."""

    def _rep_gen():
        # Use 200 randomly-sampled training images for calibration
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(x_cal), min(200, len(x_cal)), replace=False)
        for i in idxs:
            yield [x_cal[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_data = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_data)

    size_kb = len(tflite_data) / 1024
    print(f"  INT8 model saved : {output_path}")
    print(f"  Model size       : {len(tflite_data):,} bytes ({size_kb:.1f} KB)")
    return tflite_data


# ═══════════════════════════════════════════════════════════════════════
#  5. C HEADER GENERATION
# ═══════════════════════════════════════════════════════════════════════
def tflite_to_c_header(tflite_data, header_path):
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_data)
    with open(header_path, "w") as f:
        f.write("/* Auto-generated by train_model.py — do not edit. */\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write(f"/* Model size: {len(tflite_data):,} bytes */\n")
        f.write(f"alignas(16) const unsigned char model_data[] = "
                f"{{\n  {hex_vals}\n}};\n\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_data)};\n\n")
        f.write("#endif /* MODEL_DATA_H */\n")
    print(f"  C header saved   : {header_path}")


# ═══════════════════════════════════════════════════════════════════════
#  6. INT8 EVALUATION  (replicates QEMU preprocessing exactly)
# ═══════════════════════════════════════════════════════════════════════
def evaluate_int8(tflite_path, x_val, y_val):
    """Run INT8 TFLite inference on validation set.
    Quantisation: float [0,1] → uint8 [0,255] → int8 (−128 offset)."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    correct = 0
    per_cls = np.zeros((NUM_CLASSES, 2))  # [correct, total]
    for i in range(len(x_val)):
        uint8_img = (x_val[i] * 255.0).astype(np.uint8)
        int8_img  = (uint8_img.astype(np.int32) - 128).astype(np.int8)
        interp.set_tensor(inp_idx, int8_img.reshape(1, IMG_H, IMG_W, IMG_C))
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out_idx)[0]))
        per_cls[y_val[i], 1] += 1
        if pred == y_val[i]:
            correct += 1
            per_cls[y_val[i], 0] += 1

    acc = correct / len(y_val) * 100
    return acc, per_cls


# ═══════════════════════════════════════════════════════════════════════
#  7. TRAINING HISTORY PLOT
# ═══════════════════════════════════════════════════════════════════════
def save_history_plot(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"],  label="Val")
    ax1.set_title("Accuracy");  ax1.set_xlabel("Epoch");  ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"],  label="Val")
    ax2.set_title("Loss");  ax2.set_xlabel("Epoch");  ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved       : {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    print("=" * 64)
    print("  TRAFFIC SIGN RECOGNITION — MODEL TRAINING")
    print("=" * 64)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/7] Loading datasets …")
    x_train, y_train = load_dataset(TRAIN_DIR)
    x_val,   y_val   = load_dataset(VAL_DIR)
    print(f"  Train : {len(x_train):,} images   Val : {len(x_val):,} images")
    print(f"  Shape : {x_train.shape[1:]}   dtype : {x_train.dtype}")

    train_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    val_counts   = np.bincount(y_val,   minlength=NUM_CLASSES)
    print(f"\n  {'Idx':>3}  {'CID':>4}  {'Train':>5}  {'Val':>4}")
    for i, cid in enumerate(CLASS_IDS):
        print(f"  {i:>3}  {cid:>4}  {train_counts[i]:>5}  {val_counts[i]:>4}")

    # ── 2. Build model ──────────────────────────────────────────────────
    print("\n[2/7] Building model …")
    model = build_model()
    model.summary(print_fn=lambda s: print(f"  {s}"))
    print(f"\n  Total parameters : {model.count_params():,}")

    # ── 3. Class weights ────────────────────────────────────────────────
    cw = compute_class_weights(y_train, NUM_CLASSES)
    print(f"\n[3/7] Class weights  "
          f"(min={min(cw.values()):.2f}  max={max(cw.values()):.2f})")

    # ── 4. Data augmentation (tf.data pipeline) ────────────────────────
    print("\n[4/7] Augmentation: rot ±10°, shift ±8%, zoom ±10%, "
          "brightness/contrast, NO h-flip")

    aug_pipeline = keras.Sequential([
        layers.RandomRotation(10 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        layers.RandomZoom((-0.1, 0.1), fill_mode="nearest"),
    ], name="augmentation")

    def augment_batch(images, labels):
        images = aug_pipeline(images, training=True)
        images = tf.image.random_brightness(images, 0.15)
        images = tf.image.random_contrast(images, 0.8, 1.2)
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels

    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES)

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

    # ── 5. Compile & train ──────────────────────────────────────────────
    print("\n[5/7] Training (up to {} epochs, early-stop patience 30) …"
          .format(EPOCHS))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Evaluate float model ─────────────────────────────────────────
    print("\n[6/7] Float32 evaluation …")
    val_loss, val_acc = model.evaluate(x_val, y_val_oh, verbose=0)
    print(f"  Float32 val accuracy : {val_acc * 100:.1f}%")
    print(f"  Float32 val loss     : {val_loss:.4f}")

    # Save float model
    float_keras = os.path.join(MODEL_DIR, "float_model.keras")
    model.save(float_keras)
    print(f"  Saved to             : {float_keras}")
    save_history_plot(history, HISTORY_PNG)

    # ── 7. INT8 quantisation + C header ─────────────────────────────────
    print("\n[7/7] INT8 quantisation …")
    tflite_data = convert_to_int8(model, x_train, TFLITE_PATH)
    tflite_to_c_header(tflite_data, HEADER_PATH)

    int8_acc, per_cls = evaluate_int8(TFLITE_PATH, x_val, y_val)
    print(f"  INT8 val accuracy    : {int8_acc:.1f}%")
    print(f"  Accuracy drop        : {val_acc * 100 - int8_acc:+.1f}%")

    # ── Per-class INT8 results ──────────────────────────────────────────
    print(f"\n  {'Idx':>3}  {'CID':>4}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print(f"  {'---':>3}  {'---':>4}  {'-------':>7}  {'-----':>5}  {'------':>6}")
    for i, cid in enumerate(CLASS_IDS):
        tp, n = int(per_cls[i, 0]), int(per_cls[i, 1])
        a = tp / n * 100 if n else 0
        print(f"  {i:>3}  {cid:>4}  {tp:>7}  {n:>5}  {a:>5.1f}%")

    # ── Final summary ───────────────────────────────────────────────────
    int8_size = os.path.getsize(TFLITE_PATH)
    float_size = os.path.getsize(float_keras)
    elapsed = time.time() - t_start

    print("\n" + "=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Float32 accuracy  : {val_acc * 100:.1f}%")
    print(f"  INT8 accuracy     : {int8_acc:.1f}%")
    print(f"  Float model size  : {float_size:,} bytes ({float_size / 1024:.1f} KB)")
    print(f"  INT8 model size   : {int8_size:,} bytes ({int8_size / 1024:.1f} KB)")
    print(f"  Compression       : {float_size / int8_size:.1f}x")
    print(f"  Parameters        : {model.count_params():,}")
    print(f"  Training time     : {elapsed:.0f} s")
    print(f"  C header ready    : {HEADER_PATH}")
    print(f"\n  Next: rebuild Zephyr app with the trained model:")
    print(f"    west build -b qemu_x86 "
          f"C:\\Data\\Sandbox\\DiagStds_QAC_IOHandler\\src\\qemu_app "
          f"-d C:\\Data\\Sandbox\\DiagStds_QAC_IOHandler\\src\\qemu_app\\build "
          f"--pristine")


if __name__ == "__main__":
    main()
