/* inference.h — TFLite Micro inference interface */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMAGE_WIDTH   48
#define IMAGE_HEIGHT  48
#define IMAGE_CHANNELS 3
#define IMAGE_BYTES   (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)

/**
 * Initialize the TFLite Micro interpreter.
 * Must be called once before run_inference().
 * Returns 0 on success, negative on error.
 */
int inference_init(void);

/**
 * Run inference on a 48x48 RGB image.
 *
 * @param image_data  Raw uint8 RGB pixels, interleaved R,G,B per pixel
 *                    (IMAGE_BYTES length). Quantized to int8 internally.
 * @param class_id    Output: official class ID (from CLASS_ID_MAP).
 * @return 0 on success, negative on error.
 */
int run_inference(const uint8_t *image_data, int *class_id);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
