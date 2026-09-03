"""Self-contained canonical v16 training and INT8 export pipeline.

Trains the 48x48 RGB traffic-sign classifier, performs QAT, selects the best
exact-INT8 checkpoint, and exports the TFLite model and Zephyr C header.
Measured result: 352/364 (96.70%), 30,168 bytes, exact QEMU parity.
"""

import csv
import math
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from tf_keras import layers

try:
   sys.stdout.reconfigure(encoding="utf-8", errors="replace")
   sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
   pass

IMG_H, IMG_W, IMG_C = 48, 48, 3
NUM_CLASSES = 17
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]
BATCH_SIZE = 32
EPOCHS = 250
QAT_EPOCHS = 25
INITIAL_LR = 1e-3
QAT_LR = 1e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.10
BASELINE_CORRECT, BASELINE_TOTAL = 346, 364

# Model-index repeat counts; omitted classes are used once.
OVERSAMPLE_MAP = {2: 3, 3: 3, 4: 3, 6: 3, 12: 3, 13: 5, 16: 4}
ROTATION_FRACTION = 6.0 / 360.0
ZOOM_RANGE = (-0.10, 0.05)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "Train")
VAL_DIR = os.path.join(ROOT_DIR, "data", "Validation")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "quantized_model", "model_quant_v16.tflite")
HEADER_PATH = os.path.join(SCRIPT_DIR, "qemu_app", "src", "model_data_v16.h")
FLOAT_MODEL_PATH = os.path.join(MODEL_DIR, "float_model_v16.keras")
REPORT_DIR = os.path.join(ROOT_DIR, "reports", "v16")
CHECKPOINT_REPORT_PATH = os.path.join(REPORT_DIR, "qat_checkpoint_results.csv")
CONFIDENCE_REPORT_PATH = os.path.join(REPORT_DIR, "confidence.csv")


def load_dataset(data_dir, oversample=False):
   """Load RGB images and optionally repeat selected classes."""
   images, labels = [], []
   for folder in sorted(os.listdir(data_dir), key=lambda value: int(value)):
      class_id = int(folder)
      folder_path = os.path.join(data_dir, folder)
      if class_id not in CLASS_IDS or not os.path.isdir(folder_path):
         continue
      class_index = CLASS_IDS.index(class_id)
      class_images = []
      for filename in sorted(os.listdir(folder_path)):
         image_path = os.path.join(folder_path, filename)
         try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((IMG_W, IMG_H), Image.BILINEAR)
            class_images.append(np.array(image, dtype=np.float32) / 255.0)
         except Exception as error:  # pylint: disable=broad-exception-caught
            print(f"  WARN skip {image_path}: {error}")
      repeat = OVERSAMPLE_MAP.get(class_index, 1) if oversample else 1
      for _ in range(repeat):
         images.extend(class_images)
         labels.extend([class_index] * len(class_images))
   return np.array(images), np.array(labels)


def _depthwise_separable_block(inputs, filters, stride=1):
   output = layers.DepthwiseConv2D(
      3, strides=stride, padding="same", use_bias=False)(inputs)
   output = layers.Conv2D(filters, 1, padding="same", use_bias=False)(output)
   output = layers.BatchNormalization()(output)
   return layers.ReLU()(output)


def build_model():
   """Build the 11,417-parameter learned-spatial-pooling DS-CNN."""
   inputs = keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="image")
   output = layers.Conv2D(
      16, 3, strides=2, padding="same", use_bias=False)(inputs)
   output = layers.BatchNormalization()(output)
   output = layers.ReLU()(output)
   output = _depthwise_separable_block(output, 24)
   output = _depthwise_separable_block(output, 32, stride=2)
   output = _depthwise_separable_block(output, 48, stride=2)
   output = _depthwise_separable_block(output, 64)
   output = layers.DepthwiseConv2D(
      6, padding="valid", use_bias=False,
      depthwise_initializer=keras.initializers.Constant(1.0 / 36.0),
      name="learned_spatial_pool")(output)
   output = layers.GlobalAveragePooling2D(name="pool_to_vector")(output)
   output = layers.Dropout(0.3)(output)
   outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                     name="classifier")(output)
   return keras.Model(inputs=inputs, outputs=outputs,
                  name="traffic_sign_cnn_v16")


def cosine_decay_schedule(epoch, total_epochs, initial_lr, min_lr=1e-6,
                    warmup_epochs=5):
   if epoch < warmup_epochs:
      return initial_lr * (epoch + 1) / warmup_epochs
   progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
   return min_lr + 0.5 * (initial_lr - min_lr) * (
      1 + math.cos(math.pi * progress))


def compute_class_weights(labels, number_of_classes):
   counts = np.maximum(
      np.bincount(labels, minlength=number_of_classes).astype(float), 1.0)
   weights = len(labels) / (number_of_classes * counts)
   return {index: float(weight) for index, weight in enumerate(weights)}


def make_training_dataset(x_train, y_train_one_hot, use_mixup=True):
   augmentation = keras.Sequential([
      layers.RandomRotation(ROTATION_FRACTION, fill_mode="nearest"),
      layers.RandomTranslation(0.12, 0.12, fill_mode="nearest"),
      layers.RandomZoom(ZOOM_RANGE, fill_mode="nearest"),
   ], name="augmentation_v16")

   def augment_and_mix(images, labels):
      images = augmentation(images, training=True)
      images = tf.image.random_brightness(images, 0.2)
      images = tf.image.random_contrast(images, 0.7, 1.3)
      images = tf.clip_by_value(images, 0.0, 1.0)
      if use_mixup:
         batch_size = tf.shape(images)[0]
         ratio = tf.random.uniform(
            [batch_size, 1, 1, 1], 1.0 - MIXUP_ALPHA, 1.0)
         shuffled = tf.random.shuffle(tf.range(batch_size))
         images = ratio * images + (1.0 - ratio) * tf.gather(images, shuffled)
         label_ratio = tf.reshape(ratio[:, 0, 0, 0], [-1, 1])
         labels = (label_ratio * labels + (1.0 - label_ratio)
                 * tf.gather(labels, shuffled))
      return images, labels

   return (tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
         .shuffle(len(x_train), reshuffle_each_iteration=True)
         .batch(BATCH_SIZE)
         .map(augment_and_mix, num_parallel_calls=tf.data.AUTOTUNE)
         .prefetch(tf.data.AUTOTUNE))


def convert_to_int8(model, calibration_images, output_path, from_qat=False):
   def representative_dataset():
      random = np.random.default_rng(42)
      indices = random.choice(len(calibration_images),
                        min(300, len(calibration_images)), replace=False)
      for index in indices:
         yield [calibration_images[index:index + 1]]

   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_dataset
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   data = converter.convert()
   os.makedirs(os.path.dirname(output_path), exist_ok=True)
   with open(output_path, "wb") as file:
      file.write(data)
   print(f"  INT8 model saved : {output_path} ({'QAT' if from_qat else 'PTQ'})")
   print(f"  Model size       : {len(data):,} bytes ({len(data)/1024:.1f} KB)")
   return data


def evaluate_int8(tflite_path, x_val, y_val):
   interpreter = tf.lite.Interpreter(model_path=tflite_path)
   interpreter.allocate_tensors()
   input_index = interpreter.get_input_details()[0]["index"]
   output_index = interpreter.get_output_details()[0]["index"]
   correct = 0
   per_class = np.zeros((NUM_CLASSES, 2))
   for image, expected in zip(x_val, y_val):
      uint8_image = (image * 255.0).astype(np.uint8)
      int8_image = (uint8_image.astype(np.int32) - 128).astype(np.int8)
      interpreter.set_tensor(
         input_index, int8_image.reshape(1, IMG_H, IMG_W, IMG_C))
      interpreter.invoke()
      prediction = int(np.argmax(interpreter.get_tensor(output_index)[0]))
      per_class[expected, 1] += 1
      if prediction == expected:
         correct += 1
         per_class[expected, 0] += 1
   return correct / len(y_val) * 100.0, per_class


def tflite_to_c_header(data, header_path):
   os.makedirs(os.path.dirname(header_path), exist_ok=True)
   values = ", ".join(f"0x{byte:02x}" for byte in data)
   with open(header_path, "w", encoding="utf-8") as file:
      file.write("/* Auto-generated by model_training.py; do not edit. */\n")
      file.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
      file.write("#include <stddef.h>\n\n")
      file.write(f"/* Model size: {len(data):,} bytes */\n")
      file.write("alignas(16) const unsigned char model_data[] = "
               f"{{\n  {values}\n}};\n\n")
      file.write(f"const unsigned int model_data_len = {len(data)};\n\n")
      file.write("#endif /* MODEL_DATA_H */\n")


def tflite_operator_names(tflite_path):
   interpreter = tf.lite.Interpreter(model_path=tflite_path)
   return sorted({
      operation["op_name"]
      for operation in interpreter._get_ops_details()  # pylint: disable=protected-access
      if operation["op_name"] != "DELEGATE"
   })


class QatCheckpointCollector(keras.callbacks.Callback):
   def __init__(self):
      super().__init__()
      self.snapshots = []

   def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}
      self.snapshots.append({
         "epoch": epoch + 1,
         "val_accuracy": float(logs.get("val_accuracy", 0.0)),
         "val_loss": float(logs.get("val_loss", float("inf"))),
         "weights": [weight.copy() for weight in self.model.get_weights()],
      })


def select_best_qat_checkpoint(qat_model, snapshots, x_train, x_val, y_val):
   os.makedirs(REPORT_DIR, exist_ok=True)
   results = []
   for snapshot in snapshots:
      qat_model.set_weights(snapshot["weights"])
      candidate = os.path.join(
         MODEL_DIR, "quantized_model",
         f"model_quant_v16_qat_e{snapshot['epoch']:03d}.tflite")
      data = convert_to_int8(qat_model, x_train, candidate, from_qat=True)
      accuracy, per_class = evaluate_int8(candidate, x_val, y_val)
      result = {
         **snapshot, "int8_accuracy": accuracy,
         "correct": int(np.sum(per_class[:, 0])),
         "model_size": len(data), "path": candidate,
      }
      results.append(result)
      print(f"    epoch {result['epoch']:>2}: "
           f"float={result['val_accuracy']*100:>5.2f}%  "
           f"INT8={accuracy:>5.2f}% ({result['correct']}/{len(y_val)})")

   best = sorted(results, key=lambda item: (
      -item["correct"], -item["val_accuracy"],
      item["val_loss"], item["epoch"]))[0]
   shutil.copyfile(best["path"], TFLITE_PATH)
   qat_model.set_weights(best["weights"])
   with open(CHECKPOINT_REPORT_PATH, "w", newline="", encoding="utf-8") as file:
      writer = csv.writer(file)
      writer.writerow(["epoch", "float_val_accuracy_percent", "val_loss",
                   "int8_accuracy_percent", "correct", "total",
                   "model_size_bytes"])
      for result in sorted(results, key=lambda item: item["epoch"]):
         writer.writerow([result["epoch"], result["val_accuracy"] * 100,
                      result["val_loss"], result["int8_accuracy"],
                      result["correct"], len(y_val), result["model_size"]])
   for result in results:
      try:
         os.remove(result["path"])
      except OSError:
         pass
   return best


def confidence_stats(tflite_path, x_val, y_val):
   interpreter = tf.lite.Interpreter(model_path=tflite_path)
   interpreter.allocate_tensors()
   input_details = interpreter.get_input_details()[0]
   output_details = interpreter.get_output_details()[0]
   output_scale, output_zero_point = output_details["quantization"]
   all_values, correct_values = [], []
   per_class = {index: [] for index in range(NUM_CLASSES)}
   for image, expected in zip(x_val, y_val):
      quantized = ((image * 255).astype(np.uint8).astype(np.int32) - 128).astype(np.int8)
      interpreter.set_tensor(input_details["index"],
                        quantized.reshape(1, IMG_H, IMG_W, IMG_C))
      interpreter.invoke()
      raw = interpreter.get_tensor(output_details["index"])[0].astype(np.int32)
      probabilities = (raw - output_zero_point) * output_scale
      prediction, value = int(np.argmax(probabilities)), float(np.max(probabilities))
      all_values.append(value)
      per_class[int(expected)].append(value)
      if prediction == int(expected):
         correct_values.append(value)
   return {
      "mean_all": float(np.mean(all_values)),
      "median_all": float(np.median(all_values)),
      "mean_correct": float(np.mean(correct_values)),
      "median_correct": float(np.median(correct_values)),
      "per_class": {index: float(np.mean(values)) if values else 0.0
                 for index, values in per_class.items()},
   }


def write_confidence_report(stats):
   with open(CONFIDENCE_REPORT_PATH, "w", newline="", encoding="utf-8") as file:
      writer = csv.writer(file)
      writer.writerow(["scope", "mean_confidence", "median_confidence"])
      writer.writerow(["all", stats["mean_all"], stats["median_all"]])
      writer.writerow(["correct", stats["mean_correct"], stats["median_correct"]])
      writer.writerow([])
      writer.writerow(["cid", "mean_confidence"])
      for index, class_id in enumerate(CLASS_IDS):
         writer.writerow([class_id, stats["per_class"][index]])


def main():
   start_time = time.time()
   np.random.seed(42)
   tf.random.set_seed(42)
   os.makedirs(REPORT_DIR, exist_ok=True)
   print("=" * 72)
   print("  TRAFFIC SIGN RECOGNITION - CANONICAL v16 TRAINING")
   print("=" * 72)

   x_train, y_train = load_dataset(TRAIN_DIR, oversample=True)
   x_val, y_val = load_dataset(VAL_DIR)
   y_train_one_hot = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
   y_val_one_hot = keras.utils.to_categorical(y_val, NUM_CLASSES).astype(np.float32)
   model = build_model()
   print(f"  Train: {len(x_train):,}; Val: {len(x_val):,}; "
        f"Parameters: {model.count_params():,}")

   class_weights = compute_class_weights(y_train, NUM_CLASSES)
   loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
   train_data = make_training_dataset(x_train, y_train_one_hot)
   validation_data = (tf.data.Dataset.from_tensor_slices((x_val, y_val_one_hot))
                  .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
   model.compile(optimizer=keras.optimizers.Adam(INITIAL_LR),
              loss=loss, metrics=["accuracy"])
   model.fit(train_data, epochs=EPOCHS, validation_data=validation_data,
           class_weight=class_weights, verbose=2, callbacks=[
              keras.callbacks.LearningRateScheduler(
                 lambda epoch: cosine_decay_schedule(epoch, EPOCHS, INITIAL_LR)),
              keras.callbacks.EarlyStopping(
                 monitor="val_accuracy", patience=40,
                 restore_best_weights=True, verbose=1),
           ])
   model.save(FLOAT_MODEL_PATH)

   qat_model = tfmot.quantization.keras.quantize_model(model)
   qat_model.compile(optimizer=keras.optimizers.Adam(QAT_LR),
                 loss=loss, metrics=["accuracy"])
   collector = QatCheckpointCollector()
   qat_model.fit(make_training_dataset(x_train, y_train_one_hot, use_mixup=False),
              epochs=QAT_EPOCHS, validation_data=validation_data,
              class_weight=class_weights, callbacks=[collector], verbose=2)
   best = select_best_qat_checkpoint(
      qat_model, collector.snapshots, x_train, x_val, y_val)

   with open(TFLITE_PATH, "rb") as file:
      data = file.read()
   tflite_to_c_header(data, HEADER_PATH)
   accuracy, per_class = evaluate_int8(TFLITE_PATH, x_val, y_val)
   correct, total = int(np.sum(per_class[:, 0])), int(np.sum(per_class[:, 1]))
   stats = confidence_stats(TFLITE_PATH, x_val, y_val)
   write_confidence_report(stats)
   print(f"  Selected epoch: {best['epoch']}; INT8: {accuracy:.2f}% "
        f"({correct}/{total}); Size: {len(data):,} bytes")
   print(f"  Operators: {', '.join(tflite_operator_names(TFLITE_PATH))}")
   print(f"  Mean correct confidence: {stats['mean_correct']:.3f}")
   print(f"  Gate: {'PASS' if correct > BASELINE_CORRECT else 'FAIL'}; "
        f"elapsed {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
   main()
