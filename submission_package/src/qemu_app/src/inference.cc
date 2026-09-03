/* inference.cc — TFLite Micro inference engine */

#include "inference.h"
#include "model_data.h"
#include "class_mapping.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

/* Tensor arena — statically allocated, 16-byte aligned.
 * Measured: model activations peak ~23 KB + weights ~25 KB.
 * 40 KB provides safe margin while minimizing RAM. */
#define TENSOR_ARENA_SIZE (40 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

/* Input quantization parameters (from model) */
static float input_scale = 0.0f;
static int32_t input_zero_point = 0;

int inference_init(void)
{
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printk("Model schema version mismatch: got %u, expected %d\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    /* Register the operators used by the DS-CNN model.
     * Base: Conv2D, DepthwiseConv2D, FullyConnected, Softmax, Mean.
     * QAT INT8 export may wrap I/O with Quantize / Dequantize. */
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();         /* GlobalAveragePooling2D */
    resolver.AddQuantize();     /* QAT input quantize */
    resolver.AddDequantize();   /* QAT output dequantize */

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("AllocateTensors() failed\n");
        return -2;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    /* Cache input quantization params */
    input_scale = input_tensor->params.scale;
    input_zero_point = input_tensor->params.zero_point;

    printk("TFLM initialized OK\n");
    printk("  Input:  [%d,%d,%d,%d] type=%d\n",
           input_tensor->dims->data[0], input_tensor->dims->data[1],
           input_tensor->dims->data[2], input_tensor->dims->data[3],
           input_tensor->type);
    printk("  Output: [%d,%d] type=%d\n",
           output_tensor->dims->data[0], output_tensor->dims->data[1],
           output_tensor->type);
    printk("  Arena used: %zu / %d bytes\n",
           interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

    return 0;
}

int run_inference(const uint8_t *image_data, int *class_id)
{
    if (!interpreter || !input_tensor || !output_tensor) {
        return -1;
    }

    /* Quantize uint8 [0..255] → int8 using model's input params.
     * For scale≈0.00392156 and zero_point=-128:
     *   int8_val = round(uint8_val / scale) + zero_point
     *            ≈ uint8_val - 128
     * Input is RGB: R0,G0,B0,R1,G1,B1,... (6912 bytes) */
    int8_t *input_data = input_tensor->data.int8;
    for (int i = 0; i < IMAGE_BYTES; i++) {
        input_data[i] = (int8_t)((int)image_data[i] - 128);
    }

    /* Run inference */
    if (interpreter->Invoke() != kTfLiteOk) {
        printk("Invoke() failed\n");
        return -2;
    }

    /* Find argmax in output tensor */
    int8_t *output_data = output_tensor->data.int8;
    int best_idx = 0;
    int8_t best_val = output_data[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > best_val) {
            best_val = output_data[i];
            best_idx = i;
        }
    }

    /* Map model index → official class ID */
    *class_id = CLASS_ID_MAP[best_idx];
    return 0;
}
