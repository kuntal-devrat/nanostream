/* NanoStream-OD MCU runtime.
 * Patch-streaming, shift-only (zero-multiplier) inference kernel.
 * No dynamic allocation: every buffer below is static.
 *
 * Fixed-point contract (mirrored 1:1 by the Python simulator):
 *   - activations int16 in Q(frac); frac chain defined in model_weights.h
 *   - weights are signed powers of two: tap = sign * 2^exp -> shift-add
 *   - sigmoid via 512-entry LUT (Q15 in / Q15 out), same table as Python
 *   - windowed row processing reproduces full-frame convolution exactly
 */
#include <string.h>
#include <stdint.h>
#include "nanostream_mcu.h"
#include "model_weights.h"

typedef struct {
    int16_t *buf;
    int max_elems;  /* capacity in int16 elements (per-stage, exporter-sized) */
    int abs_base;   /* absolute input-row index of buf[0] */
    int count;      /* rows stored */
    int next_out;   /* absolute output row to produce next */
} ns_ring_t;

static ns_ring_t g_stage[NS_STAGES];

/* Per-stage ring buffers. Sizes are computed exactly by export.stage_buffer_sizes()
 * so the static footprint fits the 256 KB budget AND can never overflow — the old
 * flat 16384-element rings + 8192-element win/cas buffers overflowed on the first
 * strip (stem cas: 9*16*80 = 11520 > 8192). */
static int16_t g_ring_buf0[NS_RING_ELEMS_0];
static int16_t g_ring_buf1[NS_RING_ELEMS_1];
static int16_t g_ring_buf2[NS_RING_ELEMS_2];
static int16_t g_ring_buf3[NS_RING_ELEMS_3];
static int16_t *g_ring_buf[NS_STAGES] = {g_ring_buf0, g_ring_buf1, g_ring_buf2, g_ring_buf3};
static const int g_ring_elems[NS_STAGES] = {NS_RING_ELEMS_0, NS_RING_ELEMS_1, NS_RING_ELEMS_2, NS_RING_ELEMS_3};

static int16_t g_win0[NS_WIN_ELEMS_0];
static int16_t g_win1[NS_WIN_ELEMS_1];
static int16_t g_win2[NS_WIN_ELEMS_2];
static int16_t g_win3[NS_WIN_ELEMS_3];
static int16_t *g_win[NS_STAGES] = {g_win0, g_win1, g_win2, g_win3};
static const int g_win_elems[NS_STAGES] = {NS_WIN_ELEMS_0, NS_WIN_ELEMS_1, NS_WIN_ELEMS_2, NS_WIN_ELEMS_3};

static int16_t g_cas0[NS_CAS_ELEMS_0];
static int16_t g_cas1[NS_CAS_ELEMS_1];
static int16_t g_cas2[NS_CAS_ELEMS_2];
static int16_t g_cas3[NS_CAS_ELEMS_3];
static int16_t *g_cas[NS_STAGES] = {g_cas0, g_cas1, g_cas2, g_cas3};
static const int g_cas_elems[NS_STAGES] = {NS_CAS_ELEMS_0, NS_CAS_ELEMS_1, NS_CAS_ELEMS_2, NS_CAS_ELEMS_3};

static int16_t g_grid[NS_CG][NS_GRID * NS_GRID];
static int32_t g_ctx_sum[NS_CG];
static int g_grid_rows;
static int g_finished;

static int16_t g_comb[NS_HEAD1_CIN][NS_GRID * NS_GRID];
static int16_t g_h[NS_HID][NS_GRID * NS_GRID];
static int16_t g_obj[1][NS_GRID * NS_GRID];
static int16_t g_box[4][NS_GRID * NS_GRID];
static int16_t g_cls[NS_NUM_CLASSES][NS_GRID * NS_GRID];

static int16_t g_input_strip[NS_STRIP_ROWS * NS_INPUT_SIZE];

static inline int16_t ns_sig_q15(int32_t x) {
    int idx = (x + 32768) >> 7;
    if (idx < 0) idx = 0;
    if (idx > 511) idx = 511;
    return (int16_t)NS_SIG_LUT[idx];
}

/* ---------------------- shift-add convolution -------------------------- */

static void ns_conv_window(const ns_layer_t *L,
                           const int16_t *win, int Hw,
                           int16_t *out)
{
    int s = 1 << L->stride_sh, p = L->pad, k = L->k;
    int Win = L->w_in, Wout = L->w_out;
    int n_out = (Hw + 2 * p - k) / s + 1;
    for (int oy = 0; oy < n_out; oy++) {
        for (int ox = 0; ox < Wout; ox++) {
            for (int oc = 0; oc < L->cout; oc++) {
                int32_t acc = L->b[oc];
                for (int ic = 0; ic < L->cin; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        int iy = oy * s + ky - p;
                        if (iy < 0 || iy >= Hw) continue;
                        for (int kx = 0; kx < k; kx++) {
                            int ix = ox * s + kx - p;
                            if (ix < 0 || ix >= Win) continue;
                            int tap = ((oc * L->cin + ic) * k + ky) * k + kx;
                            int8_t e = L->exp[tap];
                            if (e == NS_TAP_OFF) continue;
                            int16_t x = win[(iy * L->cin + ic) * Win + ix];
                            int16_t t = (e >= 0) ? (int16_t)(x << e)
                                                 : (int16_t)(x >> (-e));
                            acc += (L->sgn[tap] > 0) ? t : -(int32_t)t;
                        }
                    }
                }
                int32_t y = acc >> L->out_shift;
                if (y > 32767) y = 32767;
                if (y < -32768) y = -32768;
                out[((oy * L->cout) + oc) * Wout + ox] = (int16_t)y;
            }
        }
    }
}

static void ns_conv1x1(const ns_layer_t *L,
                       const int16_t in_flat[][NS_GRID * NS_GRID],
                       int16_t out_flat[][NS_GRID * NS_GRID],
                       int relu)
{
    int num_cells = NS_GRID * NS_GRID;
    for (int cell = 0; cell < num_cells; cell++) {
        for (int oc = 0; oc < L->cout; oc++) {
            int32_t acc = L->b[oc];
            for (int ic = 0; ic < L->cin; ic++) {
                int tap = oc * L->cin + ic;
                int8_t e = L->exp[tap];
                if (e == NS_TAP_OFF) continue;
                int16_t x = in_flat[ic][cell];
                int16_t t = (e >= 0) ? (int16_t)(x << e)
                                     : (int16_t)(x >> (-e));
                acc += (L->sgn[tap] > 0) ? t : -(int32_t)t;
            }
            int32_t y = acc >> L->out_shift;
            if (relu && y < 0) y = 0;
            if (y > 32767) y = 32767;
            if (y < -32768) y = -32768;
            out_flat[oc][cell] = (int16_t)y;
        }
    }
}

/* --------------------------- stage plumbing ---------------------------- */

static void stage_feed(int si, const int16_t *rows, int n);

static void cascade_final(const int16_t *rows, int n) {
    const ns_layer_t *L = &ns_layers[NS_STAGES - 1];
    int w_out = L->w_out;
    for (int r = 0; r < n; r++) {
        int dst_row = g_grid_rows + r;
        if (dst_row >= NS_GRID) break;
        for (int c = 0; c < NS_CG; c++) {
            for (int x = 0; x < w_out; x++) {
                int16_t val = rows[(size_t)(r * NS_CG + c) * w_out + x];
                g_grid[c][dst_row * NS_GRID + x] = val;
                g_ctx_sum[c] += (int32_t)val;
            }
        }
    }
    g_grid_rows += n;
}

static void stage_emit_ready(int si) {
    while (si < NS_STAGES) {
        ns_ring_t *r = &g_stage[si];
        const ns_layer_t *L = &ns_layers[si];
        if (r->next_out >= L->total_out || r->count == 0) {
            return;
        }
        int s = 1 << L->stride_sh, p = L->pad, k = L->k;
        int last = r->abs_base + r->count - 1;
        int j_max = (last - p) >> L->stride_sh;
        if (j_max < r->next_out) return;
        if (j_max >= L->total_out) j_max = L->total_out - 1;

        int j_lo = r->next_out;

        /* FIX: clamp the emit span so the static window/output staging buffers
         * can never overflow (defense-in-depth; sizes are exact per-stage). */
        {
            int row_elems = L->cin * L->w_in;
            int out_row_elems = L->cout * L->w_out;
            int win_rows_cap = g_win_elems[si] / row_elems;       /* max Hw */
            int cas_rows_cap = g_cas_elems[si] / out_row_elems;   /* max n_out */
            int delta_cap = (win_rows_cap - 2 * p - 1) / s;       /* window bound */
            /* n_out = delta + (4p - k + 1)/s + 1  <=  cas_rows_cap */
            int cas_cap = cas_rows_cap - ((4 * p - k + 1) / s + 1);
            if (cas_cap < delta_cap) delta_cap = cas_cap;
            if (delta_cap < 0) delta_cap = 0;
            if (j_max - j_lo > delta_cap) j_max = j_lo + delta_cap;
        }

        int lo_raw = j_lo * s - p;
        int lo = ((lo_raw >> L->stride_sh) << L->stride_sh);
        int hi = j_max * s + p;
        int Hw = hi - lo + 1;

        int row_elems = L->cin * L->w_in;
        for (int a = lo; a <= hi; a++) {
            int16_t *dst = g_win[si] + (size_t)(a - lo) * row_elems;
            if (a < 0 || a < r->abs_base ||
                a >= r->abs_base + r->count) {
                memset(dst, 0, sizeof(int16_t) * row_elems);
            } else {
                memcpy(dst, r->buf + (size_t)(a - r->abs_base) * row_elems,
                       sizeof(int16_t) * row_elems);
            }
        }

        ns_conv_window(L, g_win[si], Hw, g_cas[si]);

        int n_new = j_max - j_lo + 1;
        int k_lo = (j_lo * s - lo) >> L->stride_sh;
        r->next_out += n_new;
        int keep_from = ((r->next_out * s - p) >> L->stride_sh) << L->stride_sh;
        if (keep_from > r->abs_base) {
            int drop = keep_from - r->abs_base;
            if (drop > r->count) drop = r->count;
            memmove(r->buf, r->buf + (size_t)drop * row_elems,
                    sizeof(int16_t) * (r->count - drop) * row_elems);
            r->abs_base += drop;
            r->count -= drop;
        }

        int out_row = L->cout * L->w_out;
        const int16_t *src = g_cas[si] + (size_t)k_lo * out_row;
        stage_feed(si + 1, src, n_new);
        return;
    }
}

static void stage_feed(int si, const int16_t *rows, int n) {
    if (n <= 0) return;
    if (si >= NS_STAGES) {
        cascade_final(rows, n);
        return;
    }
    ns_ring_t *r = &g_stage[si];
    const ns_layer_t *L = &ns_layers[si];
    int row_elems = L->cin * L->w_in;
    /* FIX: never write past the ring capacity (silent-corruption risk). */
    if ((size_t)(r->count + n) * row_elems > (size_t)r->max_elems) return;
    memcpy(r->buf + (size_t)r->count * row_elems, rows,
           sizeof(int16_t) * (size_t)n * row_elems);
    r->count += n;
    stage_emit_ready(si);
}

/* --------------------------- Public MCU API ---------------------------- */

void ns_init(void) {
    memset(g_stage, 0, sizeof(g_stage));
    for (int si = 0; si < NS_STAGES; si++) {
        g_stage[si].buf = g_ring_buf[si];
        g_stage[si].max_elems = g_ring_elems[si];
    }
    memset(g_grid, 0, sizeof(g_grid));
    memset(g_ctx_sum, 0, sizeof(g_ctx_sum));
    g_grid_rows = 0;
    g_finished = 0;
}

void ns_push_strip(const uint8_t *pixels_u8, int nrows) {
    if (nrows <= 0) return;
    int total_px = nrows * NS_INPUT_SIZE;
    for (int i = 0; i < total_px; i++) {
        int32_t u = (int32_t)pixels_u8[i];
        int32_t q = (u * 8192 + 127) / 255 - (1 << NS_INPUT_FRAC);
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        g_input_strip[i] = (int16_t)q;
    }
    stage_feed(0, g_input_strip, nrows);
}

void ns_finish_frame(void) {
    for (int si = 0; si < NS_STAGES; si++) {
        ns_ring_t *r = &g_stage[si];
        const ns_layer_t *L = &ns_layers[si];
        int s = 1 << L->stride_sh, p = L->pad;
        int row_elems = L->cin * L->w_in;
        while (r->next_out < L->total_out) {
            int need_hi = r->next_out * s + s - 1 + p;
            int last = r->abs_base + r->count - 1;
            if (need_hi > last) {
                int missing = need_hi - last;
                if ((size_t)(r->count + missing) * row_elems <= (size_t)r->max_elems) {
                    memset(r->buf + (size_t)r->count * row_elems, 0,
                           sizeof(int16_t) * (size_t)missing * row_elems);
                    r->count += missing;
                }
            }
            int prev_out = r->next_out;
            stage_emit_ready(si);
            if (r->next_out == prev_out) break;
        }
    }

    /* Aggregate context */
    int num_cells = NS_GRID * NS_GRID;
    for (int c = 0; c < NS_CG; c++) {
        int32_t avg = (int32_t)(((int64_t)g_ctx_sum[c] * NS_RECIP_M) >> NS_RECIP_S);
        int16_t avg16 = (avg > 32767) ? 32767 : ((avg < -32768) ? -32768 : (int16_t)avg);
        for (int cell = 0; cell < num_cells; cell++) {
            g_comb[c][cell] = g_grid[c][cell];
            g_comb[NS_CG + c][cell] = avg16;
        }
    }

    /* Head Convolutions */
    ns_conv1x1(&ns_head1, g_comb, g_h, 1);
    ns_conv1x1(&ns_head_obj, g_h, g_obj, 0);
    ns_conv1x1(&ns_head_box, g_h, g_box, 0);
    ns_conv1x1(&ns_head_cls, g_h, g_cls, 0);
    g_finished = 1;
}

static inline int32_t ns_q15_up(int32_t v, int up) {
    /* FIX: negative/oversized shift is UB in C; guard both directions. */
    if (up >= 0) return v << up;
    return v >> (-up);
}

int ns_decode(ns_det_t *dets, int max_det, int conf_thr_q15) {
    if (!g_finished || dets == NULL || max_det <= 0) return 0;
    int up = 15 - NS_HEAD_OUT_FRAC;
    if (up > 15) up = 15;   /* clamp: Q15 has 15 fraction bits */
    int count = 0;

    for (int cell = 0; cell < NS_GRID * NS_GRID; cell++) {
        int32_t obj_q15 = ns_q15_up((int32_t)g_obj[0][cell], up);
        int16_t prob = ns_sig_q15(obj_q15);
        if (prob > conf_thr_q15) {
            int row = cell / NS_GRID;
            int col = cell % NS_GRID;

            /* 3x3 local peak suppression (Zero-NMS direct spatial filtering) */
            int is_local_max = 1;
            for (int dr = -1; dr <= 1; dr++) {
                int nr = row + dr;
                if (nr < 0 || nr >= NS_GRID) continue;
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    int nc = col + dc;
                    if (nc < 0 || nc >= NS_GRID) continue;
                    int n_cell = nr * NS_GRID + nc;
                    int32_t n_obj_q15 = ns_q15_up((int32_t)g_obj[0][n_cell], up);
                    int16_t n_prob = ns_sig_q15(n_obj_q15);
                    if (n_prob > prob) {
                        is_local_max = 0;
                        break;
                    }
                }
                if (!is_local_max) break;
            }
            if (!is_local_max) continue;

            int16_t sx = ns_sig_q15(ns_q15_up((int32_t)g_box[0][cell], up));
            int16_t sy = ns_sig_q15(ns_q15_up((int32_t)g_box[1][cell], up));
            int16_t sw = ns_sig_q15(ns_q15_up((int32_t)g_box[2][cell], up));
            int16_t sh = ns_sig_q15(ns_q15_up((int32_t)g_box[3][cell], up));

            /* Q12 coordinates [0, 4096] with 2.5x box scale */
            int32_t cx = ((int32_t)(col << 12) + (sx >> 3)) / NS_GRID;
            int32_t cy = ((int32_t)(row << 12) + (sy >> 3)) / NS_GRID;
            int32_t w = ((int32_t)(sw >> 3) * NS_BOX_SCALE_NUM) / NS_BOX_SCALE_DEN;
            int32_t h = ((int32_t)(sh >> 3) * NS_BOX_SCALE_NUM) / NS_BOX_SCALE_DEN;
            if (w > 4096) w = 4096;
            if (h > 4096) h = 4096;

            int32_t x1 = cx - w / 2;
            int32_t y1 = cy - h / 2;
            int32_t x2 = cx + w / 2;
            int32_t y2 = cy + h / 2;

            if (x1 < 0) x1 = 0;
            if (y1 < 0) y1 = 0;
            if (x2 > 4096) x2 = 4096;
            if (y2 > 4096) y2 = 4096;

            int best_cls = 0;
            int16_t max_cls_val = g_cls[0][cell];
            for (int c = 1; c < NS_NUM_CLASSES; c++) {
                if (g_cls[c][cell] > max_cls_val) {
                    max_cls_val = g_cls[c][cell];
                    best_cls = c;
                }
            }

            int16_t cls_conf = ns_sig_q15(ns_q15_up((int32_t)max_cls_val, up));
            int32_t final_score = ((int32_t)prob * (16384 + (cls_conf >> 1))) >> 15;

            dets[count].x1 = (int16_t)x1;
            dets[count].y1 = (int16_t)y1;
            dets[count].x2 = (int16_t)x2;
            dets[count].y2 = (int16_t)y2;
            dets[count].score = (int16_t)final_score;
            dets[count].cls = (int8_t)best_cls;
            count++;
            if (count >= max_det) break;
        }
    }
    return count;
}
