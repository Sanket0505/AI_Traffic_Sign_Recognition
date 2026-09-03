/* main.c — Traffic Sign Recognition QEMU Application
 * Vertical prototype: proves TFLM + UART pipeline on Zephyr/QEMU.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>

#include "inference.h"

/* Static image buffer — NOT on stack */
static uint8_t image_buffer[IMAGE_BYTES];

/* UART device */
static const struct device *uart_dev;

static int uart_receive_bytes(uint8_t *buf, int len)
{
    int received = 0;
    while (received < len) {
        unsigned char c;
        int ret = uart_poll_in(uart_dev, &c);
        if (ret == 0) {
            buf[received++] = c;
        } else {
            k_usleep(100);
        }
    }
    return received;
}

static void uart_send_string(const char *str)
{
    while (*str) {
        uart_poll_out(uart_dev, *str++);
    }
}

int main(void)
{
    printk("=== Traffic Sign Recognition ===\n");
    printk("Model input: %dx%dx%d (%d bytes)\n",
           IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, IMAGE_BYTES);

    /* Initialize inference engine */
    int ret = inference_init();
    if (ret != 0) {
        printk("ERROR: inference_init() failed: %d\n", ret);
        return ret;
    }

    /* Get UART device */
    uart_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
    if (!device_is_ready(uart_dev)) {
        printk("ERROR: UART device not ready\n");
        return -1;
    }

    printk("Ready. Waiting for images on UART (%d bytes each)...\n",
           IMAGE_BYTES);

    /* Main inference loop */
    while (1) {
        /* Wait for sync marker: 0xAA 0x55 */
        unsigned char c;
        while (1) {
            if (uart_poll_in(uart_dev, &c) == 0 && c == 0xAA) {
                if (uart_poll_in(uart_dev, &c) == 0 && c == 0x55) {
                    break;
                }
            }
            k_usleep(100);
        }

        /* Receive image data */
        int rx = uart_receive_bytes(image_buffer, IMAGE_BYTES);
        if (rx != IMAGE_BYTES) {
            printk("WARN: incomplete frame (%d/%d)\n", rx, IMAGE_BYTES);
            continue;
        }

        /* Run inference with timing */
        int class_id;
        uint32_t t0 = k_cycle_get_32();
        ret = run_inference(image_buffer, &class_id);
        uint32_t t1 = k_cycle_get_32();
        uint32_t elapsed_us = k_cyc_to_us_floor32(t1 - t0);

        if (ret != 0) {
            printk("ERROR: inference failed: %d\n", ret);
            uart_send_string("ERR\n");
            continue;
        }

        /* Send result: class_id as ASCII + newline */
        char result[16];
        snprintf(result, sizeof(result), "%d\n", class_id);
        uart_send_string(result);

        printk("Predicted: class_id=%d, time=%u us\n", class_id, elapsed_us);
    }

    return 0;
}
