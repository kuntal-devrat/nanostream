/* NanoStream-OD MCU runtime: streaming shift-only inference, zero malloc.
 * Generated wrapper header -- include model_weights.h for weights.        */
#ifndef NANOSTREAM_MCU_H
#define NANOSTREAM_MCU_H

#include <stdint.h>

#define NS_TAP_OFF (-128)
/* NS_MAX_WIN_ROWS and all buffer sizing defines are generated into
   model_weights.h by export.stage_buffer_sizes() — do not hardcode here. */
#define NS_MAX_DET 24

typedef struct {
    int16_t x1, y1, x2, y2;   /* Q12 normalized coords in [0, 4096] */
    int16_t score;            /* Q15 probability in [0, 32767] */
    int8_t  cls;              /* Class ID */
} ns_det_t;

void ns_init(void);
void ns_push_strip(const uint8_t *pixels_u8, int nrows);
void ns_finish_frame(void);
int  ns_decode(ns_det_t *dets, int max_det, int conf_thr_q15);

#endif /* NANOSTREAM_MCU_H */
