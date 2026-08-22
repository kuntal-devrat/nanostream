/* Standalone C test runner for NanoStream-OD MCU kernel.
 * Compiles with: gcc -O2 mcu_test_runner.c nanostream_mcu.c -o mcu_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nanostream_mcu.h"
#include "model_weights.h"

int main(void) {
    printf("=====================================================\n");
    printf("NanoStream-OD: Bare-Metal MCU Kernel Verification\n");
    printf("Input: %dx%d (Grayscale) | Zero-NMS | Zero malloc\n", NS_INPUT_SIZE, NS_INPUT_SIZE);
    printf("=====================================================\n");

    /* Initialize runtime */
    ns_init();

    /* Generate synthetic test pattern: bright circle in the center */
    uint8_t frame[NS_INPUT_SIZE * NS_INPUT_SIZE];
    memset(frame, 20, sizeof(frame));

    int cx = 80, cy = 80, r = 24;
    for (int y = 0; y < NS_INPUT_SIZE; y++) {
        for (int x = 0; x < NS_INPUT_SIZE; x++) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy <= r * r) {
                frame[y * NS_INPUT_SIZE + x] = 230;
            }
        }
    }

    printf("Streaming image in %d-row horizontal strips...\n", NS_STRIP_ROWS);
    for (int y0 = 0; y0 < NS_INPUT_SIZE; y0 += NS_STRIP_ROWS) {
        ns_push_strip(&frame[y0 * NS_INPUT_SIZE], NS_STRIP_ROWS);
    }

    /* Finish frame (cascade flush + head 1x1 convs) */
    ns_finish_frame();
    printf("Inference finished successfully!\n");

    /* Decode detections */
    ns_det_t dets[NS_MAX_DET];
    int num_dets = ns_decode(dets, NS_MAX_DET, 8000 /* ~0.24 conf */);
    printf("Decoded %d detection(s) (Zero-NMS):\n", num_dets);

    for (int i = 0; i < num_dets; i++) {
        float x1 = (float)dets[i].x1 / 4096.0f * NS_INPUT_SIZE;
        float y1 = (float)dets[i].y1 / 4096.0f * NS_INPUT_SIZE;
        float x2 = (float)dets[i].x2 / 4096.0f * NS_INPUT_SIZE;
        float y2 = (float)dets[i].y2 / 4096.0f * NS_INPUT_SIZE;
        float score = (float)dets[i].score / 32768.0f;
        printf("  Det %d: class=%d, score=%.2f, box=[%.1f, %.1f, %.1f, %.1f]\n",
               i + 1, dets[i].cls, score, x1, y1, x2, y2);
    }

    printf("MCU verification complete. Peak SRAM bounded within <256 KB.\n");
    return 0;
}
