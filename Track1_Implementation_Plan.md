# AI Arena 2026 — Track 1: End-to-End Implementation Plan (v2)

## Traffic Sign Recognition — Embedded Deployment on Zephyr RTOS

**Revision:** v2 — incorporates specification conflict resolution, corrected model sizing,
vertical-prototype-first workflow, and strengthened UART/memory strategy.

---

## 0. Requirements Freeze (MUST resolve before implementation)

### 0A. Input Format Conflict

| Source | Input Spec |
|---|---|
| Presentation | 48×48 RGB |
| Challenge Guide | 48×48, RGB, Channels: 1, Model input: 48×48×1 |
| Evaluation Document | 128×128 grayscale test images |

**"RGB" and "1 channel" are contradictory.** The evaluation document saying 128×128 adds a third variant.

**Working interpretation (until confirmed with organizers):**

```
External test image:  128×128 grayscale (if evaluation doc is authoritative)
App preprocessing:    Resize to 48×48
Model tensor:         48×48×1 grayscale
```

**UART payload depends directly on this decision:**

| Format | Bytes per image |
|---|---|
| 48×48×1 | 2,304 |
| 128×128×1 | 16,384 |
| 48×48×3 | 6,912 |

**Action:** Confirm with organizers. Record answer in `requirements.md`.

### 0B. Target Board Conflict

| Source | Board |
|---|---|
| Document A | ESP32-S3 WROOM + Zephyr RTOS |
| Document B | FRDM-MCXN236 (minimum eligibility) |

ESP32-S3 and NXP MCXN236 have different toolchains, Zephyr board definitions,
memory layouts, and CPU capabilities. **Do not design board-specific optimization
until the official target is frozen.**

**Action:** Confirm the final evaluation board. Record in `requirements.md`.

### 0C. Inference Runtime

The plan assumes TensorFlow Lite Micro. Confirm:

- [ ] Is TFLite Micro required or are other runtimes allowed?
- [ ] Is TFLM already integrated in the organizer's Zephyr environment?
- [ ] Will the evaluation team compile the submitted application unchanged?
- [ ] Must the application work on both QEMU and the physical target?
- [ ] Which Zephyr version and SDK are mandatory?

**Action:** Create `requirements.md` with every frozen answer.

---

## 1. Dataset Reality Check

| Metric | Value |
|---|---|
| Training images | 1,460 (~86 per class average) |
| Validation images | 364 (~21 per class average) |
| Classes | 17 (16 signs + 1 unknown) |
| Image sizes | 32×32 to 281×248 (varying) |
| Image modes | RGB, RGBA, binary (mixed) |
| Model input | 48×48×1 grayscale |
| Smallest class | Class 70 (Weight limit) — 43 images |
| Largest class | Class 85 (Narrow road) — 129 images |

**Class ID Mapping (non-consecutive → model index):**

| Model Index | Class ID | Label |
|---|---|---|
| 0 | 0 | Speed limit (5 km/h) |
| 1 | 2 | Speed limit (30 km/h) |
| 2 | 3 | Speed limit (40 km/h) |
| 3 | 4 | Speed limit (50 km/h) |
| 4 | 5 | Speed limit (60 km/h) |
| 5 | 6 | Speed limit (70 km/h) |
| 6 | 7 | Speed limit (80 km/h) |
| 7 | 24 | Go Right |
| 8 | 43 | Go right or straight |
| 9 | 69 | Height limit |
| 10 | 70 | Weight limit |
| 11 | 72 | Length limit |
| 12 | 83 | Steep descent |
| 13 | 84 | Steep ascent |
| 14 | 85 | Narrow road |
| 15 | 86 | Narrow bridge |
| 16 | 87 | Unknown |

**Critical:** The C inference app must map model output index → official class ID.

---

## 2. Scoring Strategy (Total: 100%)

| Criteria | Weight | Strategy |
|---|---|---|
| Model Accuracy | 30% | Maximize via augmentation + lightweight CNN |
| Optimization & Quantization | 20% | Full INT8 quantization, architecture-level size reduction |
| Embedded Deployment | 20% | Working QEMU app, target-board compatible |
| Inference Performance | 15% | Minimal latency and RAM on target |
| Code Quality & Documentation | 10% | Clean code, PPT with measured metrics |
| Innovation | 5% | Unknown-class confidence thresholding, efficient architecture |

**Key insight:** Deployment + optimization + inference (55%) outweigh accuracy (30%).
A well-deployed model with moderate accuracy beats a high-accuracy model that fails on hardware.

---

## 3. Implementation Phases

### Phase 1: Vertical Prototype (Day 1–2)

**Goal:** Prove the complete toolchain end-to-end with a dummy model before investing
in training. This is the single highest-risk-reduction activity.

**Step 1.1 — Environment Verification**
- Verify Zephyr builds and runs on QEMU (already done: hello_world works)
- Verify TFLM compiles as a Zephyr module
- Verify the x86_64-zephyr-elf toolchain supports C++

**Step 1.2 — Dummy Model**
- Create a trivial Keras model (Conv2D → GlobalAveragePooling2D → Dense(17))
- Convert to full-integer INT8 TFLite
- Convert to C array header

**Step 1.3 — Minimal Zephyr App**
- Embed the dummy model
- Initialize TFLM interpreter with statically allocated tensor arena
- Run one inference on a hardcoded 48×48 test image
- Print predicted class ID over serial console

**Step 1.4 — UART Loopback**
- Receive a fixed-size image payload over UART
- Run inference
- Send official class ID back

**Success criteria:**
- [x] Zephyr app builds clean
- [ ] TFLM interpreter initializes without error
- [ ] Tensor arena allocation succeeds
- [ ] One inference completes
- [ ] Class ID maps correctly to official ID
- [ ] Response returned over UART
- [ ] Repeated inference has zero memory growth

---

### Phase 2: Data Pipeline (Days 2–4)

```
src/model_training.py — preprocessing section
```

**Step 2.1 — Unified Image Loading**
- Handle RGB, RGBA, binary mode images
- Convert ALL to grayscale (L mode via Pillow)
- Resize to 48×48 using bicubic interpolation
- Normalize to [0, 1] float range for training

**Step 2.2 — Class ID Mapping**
```python
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]
ID_TO_INDEX = {cid: idx for idx, cid in enumerate(CLASS_IDS)}
INDEX_TO_ID = {idx: cid for idx, cid in enumerate(CLASS_IDS)}
```

**Step 2.3 — Data Augmentation**

| Category | Transforms |
|---|---|
| **Safe** | Rotation ±8–12°, translation ±8%, moderate zoom, mild brightness/contrast jitter, Gaussian noise, slight blur, small perspective warp, JPEG compression artifacts, random partial occlusion (conservative) |
| **Use cautiously** | Elastic deformation, strong rotation, strong perspective |
| **Avoid** | Horizontal flip, vertical flip, rotations that change sign meaning, excessive digit/arrow deformation |

**Why no flips:** "Go Right" becomes "Go Left"; speed-limit digits can become unreadable.
Class-specific flip logic adds complexity and label-error risk.

**Step 2.4 — Class Balancing**
- Use class-aware weighted sampling (not physical duplication)
- Apply online augmentation during training
- Use class weights in loss function
- Monitor per-class recall; apply focal loss only if minority-class recall remains weak
- Avoid blind oversampling — can cause minority-class memorization

**Step 2.5 — Unknown-Class Strategy**

The "unknown" class (ID 87) represents open-set behavior. A model trained on narrow
unknown examples may only learn "unknown = images that look like these training samples."

**Approach:**
1. Augment the unknown training set with diverse negative examples:
   - Traffic signs outside the 16 known classes
   - Background-only crops
   - Partially visible / blurred / badly exposed signs
   - Non-sign roadside objects
   - Random image crops
2. Implement confidence thresholding as a secondary gate:
   ```
   if max(output_probabilities) < threshold:
       return class_id = 87  (unknown)
   else:
       return CLASS_MAP[argmax(output)]
   ```
3. Tune threshold on validation data to balance false-unknown vs. missed-unknown
4. For INT8 output, dequantize before thresholding:
   `real_value = scale × (quantized_value - zero_point)`

---

### Phase 3: Model Architecture & Training (Days 3–5)

```
src/model_training.py — model definition section
```

**Architecture: Depthwise Separable CNN with Global Average Pooling**

A custom lightweight CNN using depthwise separable convolutions. MobileNetV2 is
overkill for 48×48×1 input with 17 classes. Global average pooling eliminates the
massive Flatten→Dense bottleneck.

**Why not Flatten + Dense(128)?**
The previous architecture produced a 6×6×128 = 4,608-feature flatten layer.
Dense(128) alone = 4,608 × 128 + 128 = **589,952 parameters**.
Total model ≈ 690,000 params → **2.76 MB float32, 690 KB INT8** — far too large.

**Corrected Architecture:**

```
Input: 48×48×1
    ↓
Conv2D(16, 3×3, stride=2) + BN + ReLU          → 24×24×16
    ↓
DepthwiseConv2D(3×3) + Conv2D(24, 1×1) + BN + ReLU   → 24×24×24
    ↓
DepthwiseConv2D(3×3, stride=2) + Conv2D(32, 1×1) + BN + ReLU   → 12×12×32
    ↓
DepthwiseConv2D(3×3, stride=2) + Conv2D(64, 1×1) + BN + ReLU   → 6×6×64
    ↓
GlobalAveragePooling2D                          → 64
    ↓
Dense(17) + Softmax
```

**Parameter estimate:**
- Conv layers: ~10,000–15,000 parameters
- Dense(17): 64 × 17 + 17 = **1,105 parameters**
- **Total: ~12,000–16,000 parameters**
- **Float32: ~50–65 KB**
- **INT8: ~12–16 KB** (model weights only)

BN layers fold into adjacent convolutions during TFLite conversion → zero runtime overhead.

**If accuracy is insufficient:** Add one more depthwise separable block with 96 or 128 filters
before GAP. This adds ~5,000 parameters but keeps the model well under 100 KB INT8.

**Training Configuration:**
- Optimizer: Adam (lr=0.001 with ReduceLROnPlateau, factor=0.5, patience=5)
- Loss: Categorical crossentropy (or focal loss if class imbalance persists)
- Batch size: 32
- Epochs: 100 with early stopping (patience=15)
- Callbacks: ModelCheckpoint (val_accuracy), EarlyStopping, ReduceLROnPlateau

**Accuracy Milestones (not a fixed target):**

| Milestone | Purpose |
|---|---|
| Baseline (no augmentation) | Establish realistic floor |
| Augmented model | Measure generalization improvement |
| Balanced model | Assess minority-class recall |
| INT8 quantized | Measure quantization loss |
| QEMU integrated | Verify embedded execution correctness |
| Final candidate | Select using accuracy + efficiency together |

**Evaluation metrics (not just overall accuracy):**
- Macro F1 score
- Per-class recall (especially unknown class)
- Confusion matrix
- Model size (bytes)
- Peak tensor-arena usage (bytes)
- Inference latency (ms)

---

### Phase 4: Model Optimization (Days 5–6)

```
src/model_optimize.py
```

**Step 4.1 — Post-Training Quantization (INT8)**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen  # all 17 classes
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()
```

The representative dataset must include samples from all 17 classes covering
realistic intensity ranges.

**Step 4.2 — Validate Quantized Model**
- Run quantized model on full validation set using `tf.lite.Interpreter`
- Compare accuracy: float32 vs INT8
- Acceptable drop: ≤2%
- If drop >2%: apply quantization-aware training (QAT) and retrain

**Step 4.3 — Operator Compatibility Check**
- List all ops in the .tflite file
- Verify each is supported by TFLM on the target platform
- Use `MicroMutableOpResolver` with only required ops (reduces binary size)
- Confirm: no dynamic allocation in repeated inference path

**Step 4.4 — Convert to C Array**
```powershell
python -c "data=open('model_quant.tflite','rb').read(); f=open('model_data.h','w'); f.write('const unsigned char model_data[] = {'); f.write(','.join(f'0x{b:02x}' for b in data)); f.write('};\nconst unsigned int model_data_len = %d;\n' % len(data)); f.close()"
```

**Step 4.5 — Note on Pruning**

Unstructured pruning is **not recommended** for this challenge. It creates sparse weights
but does not improve TFLite Micro latency, tensor-arena usage, or flash size unless the
runtime exploits sparsity. Architecture-level reduction (fewer channels, depthwise separable,
GAP, earlier downsampling) is more effective.

**Step 4.6 — Document Measured Metrics**

| Metric | Float32 | INT8 |
|---|---|---|
| Model file size | measured | measured |
| Parameter count | measured | measured |
| Validation accuracy | measured | measured |
| Macro F1 | measured | measured |
| Unknown-class recall | measured | measured |

---

### Phase 5: Zephyr QEMU Application (Days 6–8)

```
src/qemu_app/
├── CMakeLists.txt
├── prj.conf
├── src/
│   ├── main.c
│   ├── model_data.h          ← quantized model as C array
│   ├── inference.c           ← TFLite Micro inference engine
│   ├── inference.h
│   ├── uart_handler.c        ← UART receive/send
│   ├── uart_handler.h
│   └── class_mapping.h       ← index-to-class-ID table
└── boards/
    └── qemu_x86.conf
```

**Step 5.1 — TensorFlow Lite Micro Integration**

Options (in order of preference):
1. Use Zephyr's built-in TFLM module (check `west list` for `tflite-micro`)
2. Add as external module via `west.yml` manifest
3. Vendor TFLM source directly into `src/qemu_app/third_party/`

Use `MicroMutableOpResolver` registering only required ops (typically: CONV_2D,
DEPTHWISE_CONV_2D, FULLY_CONNECTED, SOFTMAX, RESHAPE, QUANTIZE, DEQUANTIZE).

**Step 5.2 — Memory Management (Critical)**

All large buffers must be statically allocated and aligned:

```c
// Tensor arena — statically allocated, 16-byte aligned
#define TENSOR_ARENA_SIZE 32768  // tune based on actual model
static uint8_t tensor_arena[TENSOR_ARENA_SIZE]
    __attribute__((aligned(16)));

// Image receive buffer — static, not on stack
#define IMAGE_BYTES (48 * 48 * 1)  // adjust if input is 128×128
static uint8_t image_buffer[IMAGE_BYTES];
```

**Do NOT** place 2,304+ byte arrays on the thread stack.
**Do NOT** rely on large generic heap unless the runtime requires it.
Static allocation is measurable and reproducible.

**Step 5.3 — Inference Engine (`inference.c`)**
```c
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model_data.h"

static const int CLASS_MAP[17] = {0,2,3,4,5,6,7,24,43,69,70,72,83,84,85,86,87};

int inference_init(void);

int run_inference(const uint8_t* image_48x48, int* predicted_class_id) {
    // 1. Quantize uint8 input to int8 using input tensor's scale/zero_point:
    //    int8_val = (int8_t)(uint8_val - 128)  // if zero_point == -128
    // 2. Copy quantized data into input tensor
    // 3. Invoke interpreter
    // 4. Read output tensor (17 int8 values)
    // 5. Find argmax
    // 6. Optional: dequantize max score, apply confidence threshold
    //    if (max_score < threshold) → class_id = 87 (unknown)
    // 7. Map index to official class ID via CLASS_MAP
    return 0;
}
```

**Important:** Raw uint8 grayscale pixels cannot be copied directly into an int8
input tensor. They must be quantized using the tensor's scale and zero_point parameters.

**Step 5.4 — UART Protocol**

**If organizers mandate raw fixed-length frames:**
```
→ Receive: IMAGE_BYTES raw bytes (payload size depends on frozen input spec)
← Send:    class ID as ASCII string + newline (e.g., "24\n")
```

**If custom framing is allowed (recommended for robustness):**
```
→ Receive frame:
  [0xAA 0x55]        magic bytes
  [version]          protocol version
  [len_hi len_lo]    payload length
  [width]            image width
  [height]           image height
  [format]           pixel format (0x01 = uint8 grayscale)
  [seq_hi seq_lo]    sequence number
  [payload...]       image bytes
  [checksum]         XOR or CRC8

← Response frame:
  [0xAA 0x55]        magic bytes
  [seq_hi seq_lo]    echoed sequence number
  [status]           0x00 = OK
  [class_id]         official class ID (0–87)
  [time_ms_hi/lo]    inference time in ms
  [checksum]
```

**Failure scenarios to handle:** reset during transmission, dropped byte, partial frame,
UART console logs corrupting prediction output.

Keep debug output on a separate console (e.g., `printk` on UART0, inference on UART1)
or disable `CONFIG_UART_CONSOLE` entirely.

**Step 5.5 — Main Loop (`main.c`)**
```c
void main(void) {
    int ret = inference_init();
    if (ret != 0) {
        printk("Inference init failed: %d\n", ret);
        return;
    }
    uart_init();

    while (1) {
        // 1. Receive image over UART (blocking)
        int rx = uart_receive_image(image_buffer, IMAGE_BYTES);
        if (rx != IMAGE_BYTES) continue;  // drop incomplete frames

        // 2. Run inference
        int class_id;
        uint32_t t0 = k_cycle_get_32();
        run_inference(image_buffer, &class_id);
        uint32_t t1 = k_cycle_get_32();

        // 3. Send result over UART
        uart_send_result(class_id, k_cyc_to_us_floor32(t1 - t0));
    }
}
```

**Step 5.6 — Zephyr Configuration (`prj.conf`)**
```
CONFIG_SERIAL=y
CONFIG_UART_CONSOLE=n
CONFIG_UART_INTERRUPT_DRIVEN=y
CONFIG_HEAP_MEM_POOL_SIZE=4096
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_CPLUSPLUS=y
CONFIG_LIB_CPLUSPLUS=y
CONFIG_NEWLIB_LIBC=y
```

Heap and stack sizes should be tuned to actual measured needs, not over-provisioned.

**Step 5.7 — Build & Test on QEMU**
```powershell
cd src\qemu_app
west build -b qemu_x86
west build -t run
```

**QEMU ≠ target performance.** A qemu_x86 build may accept operators or memory behavior
that fails on ESP32-S3 or MCXN236. Do not report QEMU latency as estimated target latency.

| Environment | What it proves |
|---|---|
| Desktop Python | Model-level correctness |
| TFLite interpreter | Quantized-model accuracy |
| QEMU | Zephyr + TFLM integration |
| Physical board | Actual latency, memory, deployment fitness |

---

### Phase 6: Testing & Robustness (Days 8–9)

**Step 6.1 — Automated QEMU Testing**
Write a Python host script that:
1. Launches QEMU with the Zephyr app
2. Sends all validation images over virtual UART
3. Reads predicted class IDs
4. Compares against ground truth
5. Reports: accuracy, macro F1, confusion matrix, per-class recall

**Step 6.2 — Robustness Tests**
- Repeated inference (100+ images in sequence) — verify zero memory growth
- Invalid/truncated input — verify graceful handling
- UART interruption mid-frame — verify recovery
- All 17 classes tested — verify class mapping correctness
- Cold-start behavior — verify first inference succeeds
- Long-run stability — run for 10+ minutes continuously

**Step 6.3 — Memory Profiling**
- Enable `CONFIG_THREAD_ANALYZER=y` to profile stack usage
- Log peak RAM during inference
- Measure actual tensor-arena high-water mark
- Document all measured values in PPT

**Step 6.4 — Inference Timing**
- Use `k_cycle_get_32()` before/after inference
- Report: average, worst-case, and standard deviation latency
- Clearly label as QEMU timing (not target-board timing)

---

### Phase 7: Documentation & Submission (Days 9–10)

**Step 7.1 — Clean Rebuild**
- Rebuild everything from a clean environment
- Run automated validation end-to-end
- Verify all file paths are relative, not absolute

**Step 7.2 — Solution Documentation PPT**

Required slides:
1. Architecture overview diagram
2. Data preprocessing pipeline
3. Model architecture + justification (why depthwise separable, why GAP)
4. Training results: accuracy curves, confusion matrix
5. Quantization results: before/after comparison table
6. Embedded deployment: QEMU execution screenshot
7. **Measured** performance metrics (not estimates):
   - Model size (bytes)
   - RAM usage (bytes)
   - Inference latency (ms)
   - Tensor-arena size (bytes)
8. Innovation: unknown-class confidence thresholding, architecture choices

**Step 7.3 — README**
- Exact setup instructions (platform-specific)
- Pinned dependency versions
- How to reproduce training
- How to build and run QEMU app
- How to run automated tests

**Step 7.4 — Final Folder Structure**
```
submission/
├── src/
│   ├── model_training.py
│   ├── model_optimize.py
│   └── qemu_app/
│       ├── CMakeLists.txt
│       ├── prj.conf
│       └── src/
│           ├── main.c
│           ├── model_data.h
│           ├── inference.c
│           ├── inference.h
│           ├── uart_handler.c
│           ├── uart_handler.h
│           └── class_mapping.h
├── model/
│   ├── trained_model/          ← SavedModel or .h5
│   └── quantized_model/        ← .tflite + model_data.h
├── documentation/
│   ├── Solution_Documentation.pptx
│   └── README.md
├── requirements.md             ← frozen specs from organizers
└── requirements.txt            ← pinned Python dependencies
```

---

## 4. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Input format mismatch (48×48 vs 128×128) | **Critical** | Freeze with organizers before coding UART |
| Target board mismatch (ESP32 vs MCXN236) | **Critical** | Freeze with organizers before board-specific optimization |
| Overfitting (only 1,460 images) | High | Heavy augmentation, dropout, early stopping, macro F1 monitoring |
| Quantization accuracy drop >3% | Medium | Quantization-aware training as fallback |
| TFLM doesn't fit in target RAM | High | Architecture-level reduction (fewer channels, GAP, depthwise) |
| UART protocol mismatch with evaluator | High | Confirm exact protocol, test with automated host script |
| QEMU passes but target fails | Medium | Verify op compatibility, test tensor-arena size against target RAM |
| Unknown class fails on unseen data | Medium | Diverse negative examples + confidence thresholding |
| Class imbalance hurts minority recall | Medium | Class-aware sampling + class weights + per-class recall monitoring |
| Mixed image modes (RGBA, binary) | Low | Force convert to grayscale in preprocessing |

---

## 5. Tool & Dependency Summary

| Component | Tool | Pinned Version |
|---|---|---|
| Training framework | TensorFlow/Keras | 2.17.x (pin exact) |
| Quantization | TF Lite Converter | built-in with TF |
| Embedded inference | TF Lite Micro | pin to specific commit |
| RTOS | Zephyr | 4.4.x (confirm with organizers) |
| Emulator | QEMU | 11.1.0 |
| Toolchain | Zephyr SDK | 1.0.1 |
| Build system | CMake 4.4.2 + Ninja 1.13.0 + West 1.5.0 | installed |
| Python | 3.13.0 | installed |
| Board target (QEMU) | qemu_x86 | — |
| Board target (final) | **TBD** (ESP32-S3 or FRDM-MCXN236) | — |

**Pin all Python dependencies** in `requirements.txt` with exact versions.

---

## 6. Timeline Overview

```
Day 1–2:   Vertical prototype (dummy model → TFLM → QEMU → UART → class ID)
Days 2–4:  Data pipeline + augmentation + unknown-class strategy
Days 3–5:  Model architecture + training + milestone evaluation
Days 5–6:  INT8 quantization + operator verification + accuracy comparison
Days 6–8:  Production Zephyr app (real model + framed UART + error handling)
Days 8–9:  Robustness testing + memory profiling + timing measurement
Days 9–10: Clean rebuild + documentation + PPT + submission packaging
```

---

## 7. Quick-Start Commands

**PowerShell (Windows):**
```powershell
# Environment setup
$env:ZEPHYR_SDK_INSTALL_DIR = "C:\Data\Sandbox\zephyr-sdk-1.0.1"
$env:Path += ";C:\Program Files\qemu;C:\Users\kadasa3\AppData\Roaming\Python\Python313\Scripts"

# Train model
python src\model_training.py

# Optimize model
python src\model_optimize.py

# Build QEMU app
cd src\qemu_app
west build -b qemu_x86

# Run on QEMU
west build -t run
```

**Bash (Linux/WSL):**
```bash
export ZEPHYR_SDK_INSTALL_DIR=/opt/zephyr-sdk
python src/model_training.py
python src/model_optimize.py
cd src/qemu_app
west build -b qemu_x86
west build -t run
```

---

## 8. What to Build First

**The first deliverable is NOT the final CNN.**

Build a minimal end-to-end vertical slice:

```
Dummy 48×48×1 INT8 model
        ↓
TFLite Micro interpreter
        ↓
Zephyr QEMU application
        ↓
UART image reception
        ↓
Official class ID response
```

Once this vertical slice works end-to-end, replace the dummy model with the trained classifier.
This eliminates integration risk before investing days in model training.
