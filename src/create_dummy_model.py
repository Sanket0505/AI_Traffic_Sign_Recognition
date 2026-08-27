"""
Create a dummy tiny model for vertical prototype testing.
This model is NOT trained — it just proves the TFLM pipeline works.
Replace with the real trained model later.
"""
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- Constants ---
IMG_H, IMG_W, IMG_C = 48, 48, 3
NUM_CLASSES = 17
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "quantized_model", "model_quant.tflite")
HEADER_PATH = os.path.join(os.path.dirname(__file__), "qemu_app", "src", "model_data.h")


def build_dummy_model():
    """Build a minimal depthwise-separable CNN matching the target architecture."""
    inp = keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="image_input")

    # Block 1: Conv2D stride=2
    x = layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2: Depthwise separable
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.Conv2D(24, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3: Depthwise separable stride=2
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 4: Depthwise separable stride=2
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 5: Depthwise separable stride=1
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling + classifier
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


def representative_dataset_gen():
    """Generate random representative data for INT8 calibration."""
    for _ in range(100):
        yield [np.random.rand(1, IMG_H, IMG_W, IMG_C).astype(np.float32)]


def convert_to_int8_tflite(model, output_path):
    """Convert Keras model to full-integer INT8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model):,} bytes ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model


def tflite_to_c_header(tflite_data, header_path):
    """Convert TFLite binary to a C header file."""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    hex_values = ", ".join(f"0x{b:02x}" for b in tflite_data)

    with open(header_path, "w") as f:
        f.write("/* Auto-generated — do not edit manually. */\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write(f"/* Model size: {len(tflite_data):,} bytes */\n")
        f.write(f"alignas(16) const unsigned char model_data[] = {{\n  {hex_values}\n}};\n\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_data)};\n\n")
        f.write("#endif /* MODEL_DATA_H */\n")

    print(f"C header saved: {header_path}")


def verify_tflite(tflite_path):
    """Verify the TFLite model can run inference."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Input quantization params
    iq = input_details[0]["quantization_parameters"]
    print(f"Input quantization: scale={iq['scales']}, zero_point={iq['zero_points']}")

    oq = output_details[0]["quantization_parameters"]
    print(f"Output quantization: scale={oq['scales']}, zero_point={oq['zero_points']}")

    # Run a dummy inference
    test_input = np.zeros((1, IMG_H, IMG_W, IMG_C), dtype=np.int8)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    predicted_index = np.argmax(output[0])
    predicted_class_id = CLASS_IDS[predicted_index]
    print(f"\nDummy inference: index={predicted_index}, class_id={predicted_class_id}")
    print("Verification PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Building dummy model for vertical prototype")
    print("=" * 60)

    model = build_dummy_model()
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")
    print()

    tflite_data = convert_to_int8_tflite(model, TFLITE_PATH)
    tflite_to_c_header(tflite_data, HEADER_PATH)

    print()
    verify_tflite(TFLITE_PATH)
    print("\n✓ Dummy model pipeline complete. Ready for Zephyr integration.")
