# Traffic Sign Recognition — AI Arena 2026, Track 1

INT8, TinyML traffic-sign classifier for QEMU (`qemu_x86/atom`) and ESP32-S3,
built with TensorFlow Lite for Microcontrollers (TFLM) on Zephyr RTOS.

## Result summary

| Metric | Value |
| --- | --- |
| Classes | 17 (official IDs `0,2,3,4,5,6,7,24,43,69,70,72,83,84,85,86,87`) |
| Input | 48×48×3 RGB, int8 |
| Output | `[1,17]` int8 logits → argmax → official class ID |
| Parameters | 11,417 |
| Model size (INT8) | 30,168 bytes (~29.5 KB) |
| Validation accuracy (INT8) | 352/364 (96.70%) |
| TFLM tensor arena | 28,436 / 40,960 bytes used |
| Quantization | Quantization-Aware Training (QAT), full-integer int8 in/out |

## Model

Depthwise-separable CNN (5 blocks, 16→24→32→48→64) with a **learned
per-channel spatial-pooling head**: the final 6×6 feature map is reduced by a
`DepthwiseConv2D(6×6, valid, no bias)` initialized to `1/36` (equivalent to
global average pooling at init) followed by a trivial global average and a
`Dense(17)` classifier. Each channel can therefore learn *where* spatial
evidence matters instead of averaging uniformly.

TFLite operators used (all registered in the firmware op resolver):
`Conv2D, DepthwiseConv2D, FullyConnected, Mean, Softmax, Quantize, Dequantize`.

## Repository layout

```
src/
  model_training.py     Self-contained canonical v16 pipeline
                        (data loading + float training + QAT + INT8 export)
  model_optimize.py     Standalone INT8 re-export / verification utility
  qemu_app/             Zephyr application for QEMU / ESP32 (TFLM inference)
  qemu_uart_test.py     Host-side QEMU UART harness (sends images, reads IDs)
model/
  trained_model/        Float Keras model (traffic_sign_model.keras)
  quantized_model/       Deployed INT8 model (model_quant.tflite)
documentation/
  README.md             This file
  Solution_Documentation.pptx   Solution slide deck
```

## Prerequisites

- Python 3.13, TensorFlow 2.21, `tf_keras` 2.21, `tensorflow-model-optimization`
  0.8.1, Pillow, pyserial. **QAT requires `TF_USE_LEGACY_KERAS=1`.**
- Zephyr v4.4 + Zephyr SDK, with the TFLM module fetched:
  `python -m west config manifest.group-filter -- "+optional"`
  then `python -m west update tflite-micro`.
- A working `qemu-system-i386` (e.g. `C:\Program Files\qemu\`).

## Reproduce the model

```bat
set TF_USE_LEGACY_KERAS=1
python src\model_training.py
```

This trains the float model, runs QAT, converts every QAT checkpoint to INT8,
selects the checkpoint with the best exact INT8 accuracy, and writes:

- `model/float_model_v16.keras` (float)
- `model/quantized_model/model_quant_v16.tflite` (INT8)
- `src/qemu_app/src/model_data_v16.h` (C array for the firmware)

The canonical deployed artifacts (copies of the above) are:

- `model/trained_model/traffic_sign_model.keras`
- `model/quantized_model/model_quant.tflite`
- `src/qemu_app/src/model_data.h`

To re-export / verify INT8 from a saved model without retraining:

```bat
set TF_USE_LEGACY_KERAS=1
python src\model_optimize.py
```

## Build and run on QEMU

Build the Zephyr firmware (CMake + Ninja must be on PATH):

```bat
python -m west build -b qemu_x86 src\qemu_app -d src\qemu_app\build
```

Run the host harness, which boots QEMU, streams 48×48×3 images over UART
(marker `0xAA 0x55` + 6912 bytes) and reads back the predicted class ID:

```bat
python src\qemu_uart_test.py data\Validation\43\043_1_0013.png ^
    data\Validation\0\000_1_0031.png data\Validation\87\052_0001.png
```

Expected: TFLM initializes, input `[1,48,48,3]` int8, output `[1,17]` int8,
`Arena used: 28436 / 40960 bytes`, and correct official class IDs per image.

## Deployment notes (ESP32-S3)

The same TFLM model and inference code target ESP32-S3 WROOM. The int8 model
fits comfortably in flash; the 40 KB arena leaves margin on-device. The firmware
receives raw 48×48×3 frames over UART at 115200 8N1 and returns the class ID.
