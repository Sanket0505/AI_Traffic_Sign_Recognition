"""
train_model_v7.py — Lift the non-steep classes toward 100%.

Strategy: DO NOT touch the steep see-saw (SteepUp/Down) or the architecture —
both were shown to backfire (v4). Instead keep the EXACT winning recipe
(QAT-on-v2: label-smoothed CE, mixup, cosine LR, proven DS-CNN) and only
extend weak-class oversampling to cover every non-steep class that is still
below 100% in the shipped model:

    CID3 (idx2), CID4 (idx3), CID5 (idx4), CID7 (idx6), CID87 (idx16)

Steep classes stay at the same 3× as the shipped model (no extra pressure,
so their fragile balance is not disturbed).

Outputs go to *_v5 files; ship only if it beats the shipped 91.5%.
Input 48×48×3 RGB, INT8 output, Zephyr TFLM. tf_keras for tfmot QAT.
"""
import os
import time
import math
import numpy as np
from PIL import Image

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
import tensorflow_model_optimization as tfmot

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
IMG_H, IMG_W, IMG_C = 48, 48, 3
NUM_CLASSES = 17
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]

BATCH_SIZE = 32
EPOCHS = 250
QAT_EPOCHS = 25
INITIAL_LR = 1e-3
QAT_LR = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

BASELINE_INT8_ACC = 92.9

# Weak classes to oversample 3× (identical factor to the shipped model).
#   Steep (unchanged):        idx 12=SteepDown, 13=SteepUp
#   Non-steep <100% (target): idx 2=CID3, 3=CID4, 4=CID5, 6=CID7, 16=CID87
WEAK_CLASS_IDXS = [2, 3, 4, 6, 12, 13, 16]
OVERSAMPLE_FACTOR = 3

# ═══════════════════════════════════════════════════════════════════════
#  Paths  (all *_v5 so the shipped model is untouched)
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR   = os.path.join(ROOT_DIR, "data", "Train")
VAL_DIR     = os.path.join(ROOT_DIR, "data", "Validation")
MODEL_DIR   = os.path.join(ROOT_DIR, "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "quantized_model", "model_quant_v7.tflite")
HEADER_PATH = os.path.join(SCRIPT_DIR, "qemu_app", "src", "model_data_v7.h")


# ═══════════════════════════════════════════════════════════════════════
#  1. DATA LOADING  +  WEAK-CLASS OVERSAMPLING
# ═══════════════════════════════════════════════════════════════════════
def load_dataset(data_dir, oversample_weak=False):
    images, labels = [], []
    for folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        cid = int(folder)
        if cid not in CLASS_IDS:
            continue
        idx = CLASS_IDS.index(cid)
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        class_imgs = []
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
                class_imgs.append(np.array(img, dtype=np.float32) / 255.0)
            except Exception as e:
                print(f"  WARN: skip {fpath}: {e}")

        repeat = OVERSAMPLE_FACTOR if (oversample_weak and idx in WEAK_CLASS_IDXS) else 1
        for _ in range(repeat):
            for img_arr in class_imgs:
                images.append(img_arr)
                labels.append(idx)

    return np.array(images), np.array(labels)


# ═══════════════════════════════════════════════════════════════════════
#  2. MODEL  —  proven v2 DS-CNN (unchanged; QAT-friendly)
# ═══════════════════════════════════════════════════════════════════════
def _dw_sep_block(x, filters, stride=1):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               use_bias=False)(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_model():
    inp = keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="image")

    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = _dw_sep_block(x, 24, stride=1)   # 24×24×24
    x = _dw_sep_block(x, 32, stride=2)   # 12×12×32
    x = _dw_sep_block(x, 48, stride=2)   #  6× 6×48
    x = _dw_sep_block(x, 64, stride=1)   #  6× 6×64

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)

    return keras.Model(inputs=inp, outputs=out, name="traffic_sign_cnn")


# ═══════════════════════════════════════════════════════════════════════
#  3. COSINE DECAY LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════
def cosine_decay_schedule(epoch, total_epochs, initial_lr, min_lr=1e-6,
                          warmup_epochs=5):
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
#  4. CLASS WEIGHTS  (+ attractor down-weight, confusion-matrix driven)
# ═══════════════════════════════════════════════════════════════════════
# The v5 confusion matrix shows 7/8 Unknown(87) errors are pulled into
# Right/Straight(43, idx 8) — an over-prediction "attractor".  Down-weight
# idx 8 so its decision boundary shrinks, releasing those borderline Unknown
# samples back to Unknown.  Nothing else is changed vs v5.
ATTRACTOR_IDX = 8          # CID43 Right/Straight
ATTRACTOR_DOWNWEIGHT = 0.6

def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (n_classes * counts)
    cw = {i: float(w) for i, w in enumerate(weights)}
    cw[ATTRACTOR_IDX] *= ATTRACTOR_DOWNWEIGHT
    return cw


# ═══════════════════════════════════════════════════════════════════════
#  5. INT8 QUANTISATION
# ═══════════════════════════════════════════════════════════════════════
def convert_to_int8(keras_model, x_cal, output_path, from_qat=False):
    def _rep_gen():
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(x_cal), min(300, len(x_cal)), replace=False)
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
    tag = "QAT" if from_qat else "PTQ"
    print(f"  INT8 model saved : {output_path}  ({tag})")
    print(f"  Model size       : {len(tflite_data):,} bytes ({size_kb:.1f} KB)")
    return tflite_data


# ═══════════════════════════════════════════════════════════════════════
#  6. C HEADER GENERATION
# ═══════════════════════════════════════════════════════════════════════
def tflite_to_c_header(tflite_data, header_path):
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_data)
    with open(header_path, "w") as f:
        f.write("/* Auto-generated by train_model_v7.py — do not edit. */\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write(f"/* Model size: {len(tflite_data):,} bytes */\n")
        f.write(f"alignas(16) const unsigned char model_data[] = "
                f"{{\n  {hex_vals}\n}};\n\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_data)};\n\n")
        f.write("#endif /* MODEL_DATA_H */\n")
    print(f"  C header saved   : {header_path}")


# ═══════════════════════════════════════════════════════════════════════
#  7. INT8 EVALUATION (replicates QEMU preprocessing: uint8 − 128)
# ═══════════════════════════════════════════════════════════════════════
def evaluate_int8(tflite_path, x_val, y_val):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    correct = 0
    per_cls = np.zeros((NUM_CLASSES, 2))
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
#  8. tf.data PIPELINE  —  proven v2 augment + mixup (no blur/noise)
# ═══════════════════════════════════════════════════════════════════════
def make_train_ds(x_train, y_train_oh, use_mixup=True):
    aug_pipeline = keras.Sequential([
        layers.RandomRotation(15 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.12, 0.12, fill_mode="nearest"),
        layers.RandomZoom((-0.15, 0.15), fill_mode="nearest"),
    ], name="augmentation")

    def augment_and_mixup(images, labels):
        images = aug_pipeline(images, training=True)
        images = tf.image.random_brightness(images, 0.2)
        images = tf.image.random_contrast(images, 0.7, 1.3)
        images = tf.clip_by_value(images, 0.0, 1.0)
        if use_mixup:
            bs = tf.shape(images)[0]
            lam = tf.random.uniform([bs, 1, 1, 1], 1.0 - MIXUP_ALPHA, 1.0)
            idx = tf.random.shuffle(tf.range(bs))
            images = lam * images + (1.0 - lam) * tf.gather(images, idx)
            lam1 = tf.reshape(lam[:, 0, 0, 0], [-1, 1])
            labels = lam1 * labels + (1.0 - lam1) * tf.gather(labels, idx)
        return images, labels

    return (tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .map(augment_and_mixup, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 64)
    print("  TRAFFIC SIGN RECOGNITION — MODEL TRAINING v7")
    print("  QAT-on-v2 + extended non-steep oversampling (steep untouched)")
    print("=" * 64)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/8] Loading datasets (3× oversampling of weak classes) …")
    x_train, y_train = load_dataset(TRAIN_DIR, oversample_weak=True)
    x_val,   y_val   = load_dataset(VAL_DIR,   oversample_weak=False)
    print(f"  Train : {len(x_train):,}  Val : {len(x_val):,}")

    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
    y_val_oh   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)

    # ── 2. Build model ──────────────────────────────────────────────────
    print("\n[2/8] Building proven DS-CNN (unchanged architecture) …")
    model = build_model()
    print(f"  Total parameters : {model.count_params():,}")

    cw = compute_class_weights(y_train, NUM_CLASSES)

    # ── 3. Loss ─────────────────────────────────────────────────────────
    print(f"\n[3/8] Loss: CategoricalCrossentropy "
          f"(label_smoothing={LABEL_SMOOTHING})")
    train_loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING)

    # ── 4. Float training ───────────────────────────────────────────────
    print(f"\n[4/8] Float training (up to {EPOCHS} epochs, cosine LR) …")
    train_ds = make_train_ds(x_train, y_train_oh, use_mixup=True)
    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model.compile(optimizer=keras.optimizers.Adam(INITIAL_LR),
                  loss=train_loss, metrics=["accuracy"])

    callbacks = [
        keras.callbacks.LearningRateScheduler(
            lambda e: cosine_decay_schedule(e, EPOCHS, INITIAL_LR)),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=40,
            restore_best_weights=True, verbose=1),
    ]
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
              class_weight=cw, callbacks=callbacks, verbose=2)

    float_loss, float_acc = model.evaluate(x_val, y_val_oh, verbose=0)
    print(f"  Float32 val accuracy : {float_acc * 100:.1f}%")
    model.save(os.path.join(MODEL_DIR, "float_model_v7.keras"))

    # ── 5. QAT fine-tuning ──────────────────────────────────────────────
    print(f"\n[5/8] QAT fine-tuning ({QAT_EPOCHS} epochs @ LR={QAT_LR}) …")
    qat_ok = False
    qat_model = None
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        qat_model.compile(optimizer=keras.optimizers.Adam(QAT_LR),
                          loss=train_loss, metrics=["accuracy"])
        qat_train_ds = make_train_ds(x_train, y_train_oh, use_mixup=False)
        qat_model.fit(qat_train_ds, epochs=QAT_EPOCHS, validation_data=val_ds,
                      class_weight=cw, verbose=2)
        qat_loss, qat_acc = qat_model.evaluate(x_val, y_val_oh, verbose=0)
        print(f"  QAT float val accuracy : {qat_acc * 100:.1f}%")
        qat_ok = True
    except Exception as e:
        print(f"  QAT FAILED ({type(e).__name__}: {e})")
        print("  Falling back to post-training quantization (PTQ).")

    # ── 6. INT8 export ──────────────────────────────────────────────────
    print("\n[6/8] INT8 quantisation …")
    if qat_ok:
        tflite_data = convert_to_int8(qat_model, x_train, TFLITE_PATH, from_qat=True)
    else:
        tflite_data = convert_to_int8(model, x_train, TFLITE_PATH, from_qat=False)
    tflite_to_c_header(tflite_data, HEADER_PATH)

    # ── 7. Evaluate INT8 ────────────────────────────────────────────────
    print("\n[7/8] INT8 evaluation …")
    int8_acc, per_cls = evaluate_int8(TFLITE_PATH, x_val, y_val)
    print(f"  INT8 val accuracy    : {int8_acc:.1f}%")

    print(f"\n  {'Idx':>3}  {'CID':>4}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print(f"  {'---':>3}  {'---':>4}  {'-------':>7}  {'-----':>5}  {'------':>6}")
    for i, cid in enumerate(CLASS_IDS):
        tp, n = int(per_cls[i, 0]), int(per_cls[i, 1])
        a = tp / n * 100 if n else 0
        print(f"  {i:>3}  {cid:>4}  {tp:>7}  {n:>5}  {a:>5.1f}%")

    # ── 8. Summary + ship decision ──────────────────────────────────────
    int8_size = os.path.getsize(TFLITE_PATH)
    elapsed = time.time() - t_start
    print("\n" + "=" * 64)
    print("  TRAINING v7 COMPLETE")
    print("=" * 64)
    print(f"  Quantization      : {'QAT' if qat_ok else 'PTQ (QAT fallback)'}")
    print(f"  Float32 accuracy  : {float_acc * 100:.1f}%")
    print(f"  INT8 accuracy     : {int8_acc:.1f}%  (baseline {BASELINE_INT8_ACC}%)")
    print(f"  INT8 size         : {int8_size:,} bytes ({int8_size/1024:.1f} KB)")
    # Non-steep target classes
    print("  Non-steep targets (want 100%):")
    for i, cid in [(2, 3), (3, 4), (4, 5), (6, 7), (16, 87)]:
        print(f"    CID{cid:<3} : {int(per_cls[i,0])}/{int(per_cls[i,1])}")
    if int8_acc > BASELINE_INT8_ACC:
        print(f"\n  ✅ BEATS baseline (+{int8_acc - BASELINE_INT8_ACC:.1f}%). "
              f"Promote model_data_v7.h → model_data.h to ship.")
    else:
        print(f"\n  ❌ Does NOT beat baseline. Keep the shipped 91.5% model.")
    print("=" * 64)


if __name__ == "__main__":
    main()
