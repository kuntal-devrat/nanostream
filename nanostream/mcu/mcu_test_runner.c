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

    /* Report the ACTUAL static BSS footprint, computed purely from the
     * generated buffer defines (rings/win/cas) plus the fixed grid arrays.
     * All sizes match nanostream_mcu.c's static declarations. */
    static const long ring_elems[NS_STAGES] = {
        NS_RING_ELEMS_0, NS_RING_ELEMS_1, NS_RING_ELEMS_2, NS_RING_ELEMS_3};
    static const long win_elems[NS_STAGES] = {
        NS_WIN_ELEMS_0, NS_WIN_ELEMS_1, NS_WIN_ELEMS_2, NS_WIN_ELEMS_3};
    static const long cas_elems[NS_STAGES] = {
        NS_CAS_ELEMS_0, NS_CAS_ELEMS_1, NS_CAS_ELEMS_2, NS_CAS_ELEMS_3};

    long bss_bytes = 0;
    for (int i = 0; i < NS_STAGES; i++) {
        bss_bytes += (ring_elems[i] + win_elems[i] + cas_elems[i]) * (long)sizeof(int16_t);
    }
    long cells = (long)NS_GRID * NS_GRID;
    bss_bytes += (NS_CG + NS_HEAD1_CIN + NS_HID + 1 + 4 + NS_NUM_CLASSES) * cells
               * (long)sizeof(int16_t);          /* g_grid + g_comb + g_h + obj/box/cls */
    bss_bytes += NS_CG * (long)sizeof(int32_t);  /* g_ctx_sum */
    bss_bytes += (long)NS_STRIP_ROWS * NS_INPUT_SIZE * (long)sizeof(int16_t); /* input strip */

    printf("Static BSS: %ld bytes (%.1f KB) -- within 256 KB budget: %s\n",
           bss_bytes, bss_bytes / 1024.0,
           (bss_bytes < 256L * 1024L) ? "YES" : "NO");
    return 0;
}
