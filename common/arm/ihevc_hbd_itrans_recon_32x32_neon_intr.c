/******************************************************************************
*
* Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
******************************************************************************/
/**
*******************************************************************************
* @file
*  ihevc_hbd_itrans_recon_32x32_neon_intr.c
*
* @brief
*  Contains function definitions for inverse transform and reconstruction 32x32
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_itrans_recon_32x32_neonintr()
*
* @remarks
*  None
*
*******************************************************************************
*/
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_trans_macros.h"
#include <arm_neon.h>
static inline void zero_tmp_16x32(WORD16 *tmp)
{
    int16x8_t z = vdupq_n_s16(0);
    WORD16 *dst = tmp + 16 * 32;
    for(WORD32 i = 0; i < 64; i++)
    {
        vst1q_s16(dst, z);
        dst += 8;
    }
}

/* Universal NEON Transpose 8x4 */
static inline void transpose8x4(const int16x4_t s[8],
                                int16x8_t *r0,
                                int16x8_t *r1,
                                int16x8_t *r2,
                                int16x8_t *r3)
{
    int16x4x2_t t0 = vtrn_s16(s[0], s[1]);
    int16x4x2_t t1 = vtrn_s16(s[2], s[3]);
    int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t0.val[0]),
                              vreinterpret_s32_s16(t1.val[0]));
    int32x2x2_t u1 = vtrn_s32(vreinterpret_s32_s16(t0.val[1]),
                              vreinterpret_s32_s16(t1.val[1]));

    int16x4x2_t t2 = vtrn_s16(s[4], s[5]);
    int16x4x2_t t3 = vtrn_s16(s[6], s[7]);
    int32x2x2_t u2 = vtrn_s32(vreinterpret_s32_s16(t2.val[0]),
                              vreinterpret_s32_s16(t3.val[0]));
    int32x2x2_t u3 = vtrn_s32(vreinterpret_s32_s16(t2.val[1]),
                              vreinterpret_s32_s16(t3.val[1]));

    *r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]), vreinterpret_s16_s32(u2.val[0]));
    *r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]), vreinterpret_s16_s32(u3.val[0]));
    *r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]), vreinterpret_s16_s32(u2.val[1]));
    *r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]), vreinterpret_s16_s32(u3.val[1]));
}

static inline void store_transposed_8x4(WORD16 *tmp,
                                        WORD32 cb,
                                        WORD32 col_offset,
                                        const int16x4_t s[8])
{
    int16x8_t r0, r1, r2, r3;
    transpose8x4(s, &r0, &r1, &r2, &r3);
    vst1q_s16(tmp + (cb + 0) * 32 + col_offset, r0);
    vst1q_s16(tmp + (cb + 1) * 32 + col_offset, r1);
    vst1q_s16(tmp + (cb + 2) * 32 + col_offset, r2);
    vst1q_s16(tmp + (cb + 3) * 32 + col_offset, r3);
}

static inline void recon_store_row_8(UWORD16 *dst,
                                     const UWORD16 *pred,
                                     int16x8_t res,
                                     int16x8_t clip_min,
                                     int16x8_t clip_max)
{
    int16x8_t pred_val = vreinterpretq_s16_u16(vld1q_u16(pred));
    int16x8_t acc = vqaddq_s16(res, pred_val);
    acc = vminq_s16(vmaxq_s16(acc, clip_min), clip_max);
    vst1q_u16(dst, vreinterpretq_u16_s16(acc));
}

static inline void recon_transposed_8x4(UWORD16 *dst,
                                        WORD32 dst_strd,
                                        const UWORD16 *pred,
                                        WORD32 pred_strd,
                                        WORD32 cb,
                                        WORD32 col_offset,
                                        const int16x4_t s[8],
                                        int16x8_t clip_min,
                                        int16x8_t clip_max)
{
    int16x8_t r0, r1, r2, r3;
    transpose8x4(s, &r0, &r1, &r2, &r3);
    recon_store_row_8(dst + (cb + 0) * dst_strd + col_offset,
                      pred + (cb + 0) * pred_strd + col_offset,
                      r0, clip_min, clip_max);
    recon_store_row_8(dst + (cb + 1) * dst_strd + col_offset,
                      pred + (cb + 1) * pred_strd + col_offset,
                      r1, clip_min, clip_max);
    recon_store_row_8(dst + (cb + 2) * dst_strd + col_offset,
                      pred + (cb + 2) * pred_strd + col_offset,
                      r2, clip_min, clip_max);
    recon_store_row_8(dst + (cb + 3) * dst_strd + col_offset,
                      pred + (cb + 3) * pred_strd + col_offset,
                      r3, clip_min, clip_max);
}

#define ADD_LANE(acc, reg, vec, lane) \
    ((acc) = vmlal_lane_s16((acc), (reg), (vec), (lane)))
#define SUB_LANE(acc, reg, vec, lane) \
    ((acc) = vmlsl_lane_s16((acc), (reg), (vec), (lane)))

static inline void idct32_4col_stage1_zero28(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t co0,
                                             int16x4_t co1,
                                             int16x4_t c_o32_0,
                                             int16x4_t c_o32_1,
                                             int16x4_t c_o32_2,
                                             int16x4_t c_o32_3)
{
    int16x4_t r0 = vld1_s16(src + 0 * src_strd + cb);
    int16x4_t r1 = vld1_s16(src + 1 * src_strd + cb);
    int16x4_t r2 = vld1_s16(src + 2 * src_strd + cb);
    int16x4_t r3 = vld1_s16(src + 3 * src_strd + cb);
    int32x4_t ee = vshll_n_s16(r0, 6);

    /* Phase 1: Chunk 0 (y[0..7]) and Chunk 3 (y[24..31]) */
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 0));
        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 0), r3, c_o32_0, 1);
        y0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        y3[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);
        int32x4_t e1 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 1));
        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 1), r3, c_o32_1, 0);
        y0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        y3[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);
        int32x4_t e2 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 2));
        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 2), r3, c_o32_1, 3);
        y0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        y3[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);
        int32x4_t e3 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 3));
        int32x4_t o3 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 3), r3, c_o32_2, 2);
        y0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        y3[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);
        int32x4_t e4 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 0));
        int32x4_t o4 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_1, 0), r3, c_o32_3, 1);
        y0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        y3[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);
        int32x4_t e5 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 1));
        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 1), r3, c_o32_3, 3);
        y0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        y3[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);
        int32x4_t e6 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 2));
        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 2), r3, c_o32_3, 0);
        y0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        y3[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);
        int32x4_t e7 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 3));
        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 3), r3, c_o32_2, 1);
        y0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        y3[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
        store_transposed_8x4(tmp, cb, 0, y0);

        store_transposed_8x4(tmp, cb, 24, y3);
    }

    /* Phase 2: Chunk 1 (y[8..15]) and Chunk 2 (y[16..23]) */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 0));
        int32x4_t o15 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 3), r3, c_o32_3, 2);
        y1[7] = vqrshrn_n_s32(vaddq_s32(e15, o15), 7);
        y2[0] = vqrshrn_n_s32(vsubq_s32(e15, o15), 7);
        int32x4_t e14 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 1));
        int32x4_t o14 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 2), r3, c_o32_2, 3);
        y1[6] = vqrshrn_n_s32(vaddq_s32(e14, o14), 7);
        y2[1] = vqrshrn_n_s32(vsubq_s32(e14, o14), 7);
        int32x4_t e13 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 2));
        int32x4_t o13 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 1), r3, c_o32_2, 0);
        y1[5] = vqrshrn_n_s32(vaddq_s32(e13, o13), 7);
        y2[2] = vqrshrn_n_s32(vsubq_s32(e13, o13), 7);
        int32x4_t e12 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 3));
        int32x4_t o12 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 0), r3, c_o32_1, 1);
        y1[4] = vqrshrn_n_s32(vaddq_s32(e12, o12), 7);
        y2[3] = vqrshrn_n_s32(vsubq_s32(e12, o12), 7);
        int32x4_t e11 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 0));
        int32x4_t o11 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 3), r3, c_o32_0, 2);
        y1[3] = vqrshrn_n_s32(vaddq_s32(e11, o11), 7);
        y2[4] = vqrshrn_n_s32(vsubq_s32(e11, o11), 7);
        int32x4_t e10 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 1));
        int32x4_t o10 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 2), r3, c_o32_0, 1);
        y1[2] = vqrshrn_n_s32(vaddq_s32(e10, o10), 7);
        y2[5] = vqrshrn_n_s32(vsubq_s32(e10, o10), 7);
        int32x4_t e9 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 2));
        int32x4_t o9 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 1), r3, c_o32_0, 3);
        y1[1] = vqrshrn_n_s32(vaddq_s32(e9, o9), 7);
        y2[6] = vqrshrn_n_s32(vsubq_s32(e9, o9), 7);
        int32x4_t e8 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 3));
        int32x4_t o8 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 0), r3, c_o32_1, 2);
        y1[0] = vqrshrn_n_s32(vaddq_s32(e8, o8), 7);
        y2[7] = vqrshrn_n_s32(vsubq_s32(e8, o8), 7);
        store_transposed_8x4(tmp, cb, 8, y1);

        store_transposed_8x4(tmp, cb, 16, y2);
    }
}

static inline void idct32_4col_stage1_zero24(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t ce1,
                                             int16x4_t co0,
                                             int16x4_t co1,
                                             int16x4_t c_o32_0,
                                             int16x4_t c_o32_1,
                                             int16x4_t c_o32_2,
                                             int16x4_t c_o32_3)
{
    int16x4_t r0 = vld1_s16(src + 0 * src_strd + cb);
    int16x4_t r1 = vld1_s16(src + 1 * src_strd + cb);
    int16x4_t r2 = vld1_s16(src + 2 * src_strd + cb);
    int16x4_t r3 = vld1_s16(src + 3 * src_strd + cb);
    int16x4_t r4 = vld1_s16(src + 4 * src_strd + cb);
    int16x4_t r5 = vld1_s16(src + 5 * src_strd + cb);
    int16x4_t r6 = vld1_s16(src + 6 * src_strd + cb);
    int16x4_t r7 = vld1_s16(src + 7 * src_strd + cb);
    int32x4_t eee = vshll_n_s16(r0, 6);
    int32x4_t ee[8];
    int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
    ee[0] = vaddq_s32(eee, eeo0);

    ee[7] = vsubq_s32(eee, eeo0);
    int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
    ee[1] = vaddq_s32(eee, eeo1);

    ee[6] = vsubq_s32(eee, eeo1);
    int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
    ee[2] = vaddq_s32(eee, eeo2);

    ee[5] = vsubq_s32(eee, eeo2);
    int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
    ee[3] = vaddq_s32(eee, eeo3);

    ee[4] = vsubq_s32(eee, eeo3);

    /* Phase 1: Chunk 0 & 3 */
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee[0],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 0), r6, co0, 1));
        int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
        ADD_LANE(o0, r3, c_o32_0, 1);
        ADD_LANE(o0, r5, c_o32_0, 2);
        ADD_LANE(o0, r7, c_o32_0, 3);
        y0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        y3[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);
        int32x4_t e1 = vaddq_s32(ee[1],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 1), r6, co1, 0));
        int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
        ADD_LANE(o1, r3, c_o32_1, 0);
        ADD_LANE(o1, r5, c_o32_1, 3);
        ADD_LANE(o1, r7, c_o32_2, 2);
        y0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        y3[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);
        int32x4_t e2 = vaddq_s32(ee[2],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 2), r6, co1, 3));
        int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
        ADD_LANE(o2, r3, c_o32_1, 3);
        ADD_LANE(o2, r5, c_o32_3, 0);
        SUB_LANE(o2, r7, c_o32_3, 2);
        y0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        y3[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);
        int32x4_t e3 = vaddq_s32(ee[3],
            vmlsl_lane_s16(vmull_lane_s16(r2, co0, 3), r6, co1, 1));
        int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
        ADD_LANE(o3, r3, c_o32_2, 2);
        SUB_LANE(o3, r5, c_o32_3, 2);
        SUB_LANE(o3, r7, c_o32_1, 3);
        y0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        y3[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);
        int32x4_t e4 = vaddq_s32(ee[4],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 0), r6, co0, 2));
        int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
        ADD_LANE(o4, r3, c_o32_3, 1);
        SUB_LANE(o4, r5, c_o32_2, 1);
        SUB_LANE(o4, r7, c_o32_0, 0);
        y0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        y3[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);
        int32x4_t e5 = vaddq_s32(ee[5],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 1), r6, co0, 0));
        int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
        SUB_LANE(o5, r3, c_o32_3, 3);
        SUB_LANE(o5, r5, c_o32_1, 0);
        SUB_LANE(o5, r7, c_o32_1, 2);
        y0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        y3[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);
        int32x4_t e6 = vaddq_s32(ee[6],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 2), r6, co0, 3));
        int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
        SUB_LANE(o6, r3, c_o32_3, 0);
        SUB_LANE(o6, r5, c_o32_0, 0);
        SUB_LANE(o6, r7, c_o32_3, 1);
        y0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        y3[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);
        int32x4_t e7 = vaddq_s32(ee[7],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 3), r6, co1, 2));
        int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
        SUB_LANE(o7, r3, c_o32_2, 1);
        SUB_LANE(o7, r5, c_o32_1, 1);
        ADD_LANE(o7, r7, c_o32_2, 3);
        y0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        y3[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
        store_transposed_8x4(tmp, cb, 0, y0);

        store_transposed_8x4(tmp, cb, 24, y3);
    }

    /* Phase 2: Chunk 1 & 2 */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee[0],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 0), r6, co0, 1));
        int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
        SUB_LANE(o15, r3, c_o32_3, 2);
        ADD_LANE(o15, r5, c_o32_3, 1);
        SUB_LANE(o15, r7, c_o32_3, 0);
        y1[7] = vqrshrn_n_s32(vaddq_s32(e15, o15), 7);
        y2[0] = vqrshrn_n_s32(vsubq_s32(e15, o15), 7);
        int32x4_t e14 = vsubq_s32(ee[1],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 1), r6, co1, 0));
        int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
        SUB_LANE(o14, r3, c_o32_2, 3);
        ADD_LANE(o14, r5, c_o32_2, 0);
        SUB_LANE(o14, r7, c_o32_1, 1);
        y1[6] = vqrshrn_n_s32(vaddq_s32(e14, o14), 7);
        y2[1] = vqrshrn_n_s32(vsubq_s32(e14, o14), 7);
        int32x4_t e13 = vsubq_s32(ee[2],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 2), r6, co1, 3));
        int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
        SUB_LANE(o13, r3, c_o32_2, 0);
        ADD_LANE(o13, r5, c_o32_0, 3);
        SUB_LANE(o13, r7, c_o32_0, 0);
        y1[5] = vqrshrn_n_s32(vaddq_s32(e13, o13), 7);
        y2[2] = vqrshrn_n_s32(vsubq_s32(e13, o13), 7);
        int32x4_t e12 = vsubq_s32(ee[3],
            vmlsl_lane_s16(vmull_lane_s16(r2, co0, 3), r6, co1, 1));
        int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
        SUB_LANE(o12, r3, c_o32_1, 1);
        ADD_LANE(o12, r5, c_o32_0, 0);
        SUB_LANE(o12, r7, c_o32_2, 0);
        y1[4] = vqrshrn_n_s32(vaddq_s32(e12, o12), 7);
        y2[3] = vqrshrn_n_s32(vsubq_s32(e12, o12), 7);
        int32x4_t e11 = vsubq_s32(ee[4],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 0), r6, co0, 2));
        int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
        SUB_LANE(o11, r3, c_o32_0, 2);
        ADD_LANE(o11, r5, c_o32_1, 2);
        SUB_LANE(o11, r7, c_o32_3, 3);
        y1[3] = vqrshrn_n_s32(vaddq_s32(e11, o11), 7);
        y2[4] = vqrshrn_n_s32(vsubq_s32(e11, o11), 7);
        int32x4_t e10 = vsubq_s32(ee[5],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 1), r6, co0, 0));
        int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
        SUB_LANE(o10, r3, c_o32_0, 1);
        ADD_LANE(o10, r5, c_o32_2, 3);
        ADD_LANE(o10, r7, c_o32_2, 1);
        y1[2] = vqrshrn_n_s32(vaddq_s32(e10, o10), 7);
        y2[5] = vqrshrn_n_s32(vsubq_s32(e10, o10), 7);
        int32x4_t e9 = vsubq_s32(ee[6],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 2), r6, co0, 3));
        int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
        SUB_LANE(o9, r3, c_o32_0, 3);
        SUB_LANE(o9, r5, c_o32_3, 3);
        ADD_LANE(o9, r7, c_o32_0, 2);
        y1[1] = vqrshrn_n_s32(vaddq_s32(e9, o9), 7);
        y2[6] = vqrshrn_n_s32(vsubq_s32(e9, o9), 7);
        int32x4_t e8 = vsubq_s32(ee[7],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 3), r6, co1, 2));
        int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
        SUB_LANE(o8, r3, c_o32_1, 2);
        SUB_LANE(o8, r5, c_o32_2, 2);
        ADD_LANE(o8, r7, c_o32_1, 0);
        y1[0] = vqrshrn_n_s32(vaddq_s32(e8, o8), 7);
        y2[7] = vqrshrn_n_s32(vsubq_s32(e8, o8), 7);
        store_transposed_8x4(tmp, cb, 8, y1);

        store_transposed_8x4(tmp, cb, 16, y2);
    }
}

static inline void idct32_4col_stage1_full(const WORD16 *src,
                                           WORD32 src_strd,
                                           WORD16 *tmp,
                                           WORD32 cb,
                                           int16x4_t ce0,
                                           int16x4_t ce1,
                                           int16x4_t co0,
                                           int16x4_t co1,
                                           int16x4_t c_o32_0,
                                           int16x4_t c_o32_1,
                                           int16x4_t c_o32_2,
                                           int16x4_t c_o32_3)
{
    int32x4_t ee[8];
    int32x4_t eo[8];
    {
        int16x4_t r0 = vld1_s16(src + 0 * src_strd + cb);
        int16x4_t r16 = vld1_s16(src + 16 * src_strd + cb);
        int16x4_t r8 = vld1_s16(src + 8 * src_strd + cb);
        int16x4_t r24 = vld1_s16(src + 24 * src_strd + cb);
        int16x4_t r4 = vld1_s16(src + 4 * src_strd + cb);
        int16x4_t r12 = vld1_s16(src + 12 * src_strd + cb);
        int16x4_t r20 = vld1_s16(src + 20 * src_strd + cb);
        int16x4_t r28 = vld1_s16(src + 28 * src_strd + cb);
        int16x4_t r2 = vld1_s16(src + 2 * src_strd + cb);
        int16x4_t r6 = vld1_s16(src + 6 * src_strd + cb);
        int16x4_t r10 = vld1_s16(src + 10 * src_strd + cb);
        int16x4_t r14 = vld1_s16(src + 14 * src_strd + cb);
        int16x4_t r18 = vld1_s16(src + 18 * src_strd + cb);
        int16x4_t r22 = vld1_s16(src + 22 * src_strd + cb);
        int16x4_t r26 = vld1_s16(src + 26 * src_strd + cb);
        int16x4_t r30 = vld1_s16(src + 30 * src_strd + cb);

        int32x4_t r0_shl = vshll_n_s16(r0, 6);
        int32x4_t r16_shl = vshll_n_s16(r16, 6);
        int32x4_t eeee0 = vaddq_s32(r0_shl, r16_shl);
        int32x4_t eeee1 = vsubq_s32(r0_shl, r16_shl);
        int32x4_t eeeo0 = vmlal_lane_s16(vmull_lane_s16(r8, ce0, 0), r24, ce0, 1);
        int32x4_t eeeo1 = vmlsl_lane_s16(vmull_lane_s16(r8, ce0, 1), r24, ce0, 0);
        int32x4_t eee0 = vaddq_s32(eeee0, eeeo0);
        int32x4_t eee3 = vsubq_s32(eeee0, eeeo0);
        int32x4_t eee1 = vaddq_s32(eeee1, eeeo1);
        int32x4_t eee2 = vsubq_s32(eeee1, eeeo1);

        int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
        ADD_LANE(eeo0, r12, ce1, 1);
        ADD_LANE(eeo0, r20, ce1, 2);
        ADD_LANE(eeo0, r28, ce1, 3);
        int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
        SUB_LANE(eeo1, r12, ce1, 3);
        SUB_LANE(eeo1, r20, ce1, 0);
        SUB_LANE(eeo1, r28, ce1, 2);
        int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
        SUB_LANE(eeo2, r12, ce1, 0);
        ADD_LANE(eeo2, r20, ce1, 3);
        ADD_LANE(eeo2, r28, ce1, 1);
        int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
        SUB_LANE(eeo3, r12, ce1, 2);
        ADD_LANE(eeo3, r20, ce1, 1);
        SUB_LANE(eeo3, r28, ce1, 0);

        ee[0] = vaddq_s32(eee0, eeo0);


        ee[7] = vsubq_s32(eee0, eeo0);
        ee[1] = vaddq_s32(eee1, eeo1);

        ee[6] = vsubq_s32(eee1, eeo1);
        ee[2] = vaddq_s32(eee2, eeo2);

        ee[5] = vsubq_s32(eee2, eeo2);
        ee[3] = vaddq_s32(eee3, eeo3);

        ee[4] = vsubq_s32(eee3, eeo3);

        eo[0] = vmull_lane_s16(r2, co0, 0);
        ADD_LANE(eo[0], r6, co0, 1);
        ADD_LANE(eo[0], r10, co0, 2);
        ADD_LANE(eo[0], r14, co0, 3);
        ADD_LANE(eo[0], r18, co1, 0);
        ADD_LANE(eo[0], r22, co1, 1);
        ADD_LANE(eo[0], r26, co1, 2);
        ADD_LANE(eo[0], r30, co1, 3);
        eo[1] = vmull_lane_s16(r2, co0, 1);
        ADD_LANE(eo[1], r6, co1, 0);
        ADD_LANE(eo[1], r10, co1, 3);
        SUB_LANE(eo[1], r14, co1, 1);
        SUB_LANE(eo[1], r18, co0, 2);
        SUB_LANE(eo[1], r22, co0, 0);
        SUB_LANE(eo[1], r26, co0, 3);
        SUB_LANE(eo[1], r30, co1, 2);
        eo[2] = vmull_lane_s16(r2, co0, 2);
        ADD_LANE(eo[2], r6, co1, 3);
        SUB_LANE(eo[2], r10, co0, 3);
        SUB_LANE(eo[2], r14, co0, 1);
        SUB_LANE(eo[2], r18, co1, 2);
        ADD_LANE(eo[2], r22, co1, 0);
        ADD_LANE(eo[2], r26, co0, 0);
        ADD_LANE(eo[2], r30, co1, 1);
        eo[3] = vmull_lane_s16(r2, co0, 3);
        SUB_LANE(eo[3], r6, co1, 1);
        SUB_LANE(eo[3], r10, co0, 1);
        ADD_LANE(eo[3], r14, co1, 3);
        ADD_LANE(eo[3], r18, co0, 0);
        ADD_LANE(eo[3], r22, co1, 2);
        SUB_LANE(eo[3], r26, co0, 2);
        SUB_LANE(eo[3], r30, co1, 0);
        eo[4] = vmull_lane_s16(r2, co1, 0);
        SUB_LANE(eo[4], r6, co0, 2);
        SUB_LANE(eo[4], r10, co1, 2);
        ADD_LANE(eo[4], r14, co0, 0);
        SUB_LANE(eo[4], r18, co1, 3);
        SUB_LANE(eo[4], r22, co0, 1);
        ADD_LANE(eo[4], r26, co1, 1);
        ADD_LANE(eo[4], r30, co0, 3);
        eo[5] = vmull_lane_s16(r2, co1, 1);
        SUB_LANE(eo[5], r6, co0, 0);
        ADD_LANE(eo[5], r10, co1, 0);
        ADD_LANE(eo[5], r14, co1, 2);
        SUB_LANE(eo[5], r18, co0, 1);
        ADD_LANE(eo[5], r22, co0, 3);
        ADD_LANE(eo[5], r26, co1, 3);
        SUB_LANE(eo[5], r30, co0, 2);
        eo[6] = vmull_lane_s16(r2, co1, 2);
        SUB_LANE(eo[6], r6, co0, 3);
        ADD_LANE(eo[6], r10, co0, 0);
        SUB_LANE(eo[6], r14, co0, 2);
        ADD_LANE(eo[6], r18, co1, 1);
        ADD_LANE(eo[6], r22, co1, 3);
        SUB_LANE(eo[6], r26, co1, 0);
        ADD_LANE(eo[6], r30, co0, 1);
        eo[7] = vmull_lane_s16(r2, co1, 3);
        SUB_LANE(eo[7], r6, co1, 2);
        ADD_LANE(eo[7], r10, co1, 1);
        SUB_LANE(eo[7], r14, co1, 0);
        ADD_LANE(eo[7], r18, co0, 3);
        SUB_LANE(eo[7], r22, co0, 2);
        ADD_LANE(eo[7], r26, co0, 1);
        SUB_LANE(eo[7], r30, co0, 0);
    }

    /* Odd stage */
    int16x4_t r1 = vld1_s16(src + 1 * src_strd + cb);
        int16x4_t r3 = vld1_s16(src + 3 * src_strd + cb);
        int16x4_t r5 = vld1_s16(src + 5 * src_strd + cb);
        int16x4_t r7 = vld1_s16(src + 7 * src_strd + cb);
        int16x4_t r9 = vld1_s16(src + 9 * src_strd + cb);
        int16x4_t r11 = vld1_s16(src + 11 * src_strd + cb);
        int16x4_t r13 = vld1_s16(src + 13 * src_strd + cb);
        int16x4_t r15 = vld1_s16(src + 15 * src_strd + cb);
        int16x4_t r17 = vld1_s16(src + 17 * src_strd + cb);
        int16x4_t r19 = vld1_s16(src + 19 * src_strd + cb);
        int16x4_t r21 = vld1_s16(src + 21 * src_strd + cb);
        int16x4_t r23 = vld1_s16(src + 23 * src_strd + cb);
        int16x4_t r25 = vld1_s16(src + 25 * src_strd + cb);
        int16x4_t r27 = vld1_s16(src + 27 * src_strd + cb);
        int16x4_t r29 = vld1_s16(src + 29 * src_strd + cb);
        int16x4_t r31 = vld1_s16(src + 31 * src_strd + cb);
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee[0], eo[0]);
        int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
        ADD_LANE(o0, r3, c_o32_0, 1);
        ADD_LANE(o0, r5, c_o32_0, 2);
        ADD_LANE(o0, r7, c_o32_0, 3);
        ADD_LANE(o0, r9, c_o32_1, 0);
        ADD_LANE(o0, r11, c_o32_1, 1);
        ADD_LANE(o0, r13, c_o32_1, 2);
        ADD_LANE(o0, r15, c_o32_1, 3);
        ADD_LANE(o0, r17, c_o32_2, 0);
        ADD_LANE(o0, r19, c_o32_2, 1);
        ADD_LANE(o0, r21, c_o32_2, 2);
        ADD_LANE(o0, r23, c_o32_2, 3);
        ADD_LANE(o0, r25, c_o32_3, 0);
        ADD_LANE(o0, r27, c_o32_3, 1);
        ADD_LANE(o0, r29, c_o32_3, 2);
        ADD_LANE(o0, r31, c_o32_3, 3);
        y0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        y3[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);
        int32x4_t e1 = vaddq_s32(ee[1], eo[1]);
        int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
        ADD_LANE(o1, r3, c_o32_1, 0);
        ADD_LANE(o1, r5, c_o32_1, 3);
        ADD_LANE(o1, r7, c_o32_2, 2);
        ADD_LANE(o1, r9, c_o32_3, 1);
        SUB_LANE(o1, r11, c_o32_3, 3);
        SUB_LANE(o1, r13, c_o32_3, 0);
        SUB_LANE(o1, r15, c_o32_2, 1);
        SUB_LANE(o1, r17, c_o32_1, 2);
        SUB_LANE(o1, r19, c_o32_0, 3);
        SUB_LANE(o1, r21, c_o32_0, 0);
        SUB_LANE(o1, r23, c_o32_0, 2);
        SUB_LANE(o1, r25, c_o32_1, 1);
        SUB_LANE(o1, r27, c_o32_2, 0);
        SUB_LANE(o1, r29, c_o32_2, 3);
        SUB_LANE(o1, r31, c_o32_3, 2);
        y0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        y3[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);
        int32x4_t e2 = vaddq_s32(ee[2], eo[2]);
        int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
        ADD_LANE(o2, r3, c_o32_1, 3);
        ADD_LANE(o2, r5, c_o32_3, 0);
        SUB_LANE(o2, r7, c_o32_3, 2);
        SUB_LANE(o2, r9, c_o32_2, 1);
        SUB_LANE(o2, r11, c_o32_1, 0);
        SUB_LANE(o2, r13, c_o32_0, 0);
        SUB_LANE(o2, r15, c_o32_1, 1);
        SUB_LANE(o2, r17, c_o32_2, 2);
        SUB_LANE(o2, r19, c_o32_3, 3);
        ADD_LANE(o2, r21, c_o32_2, 3);
        ADD_LANE(o2, r23, c_o32_1, 2);
        ADD_LANE(o2, r25, c_o32_0, 0);
        ADD_LANE(o2, r27, c_o32_0, 3);
        ADD_LANE(o2, r29, c_o32_2, 0);
        ADD_LANE(o2, r31, c_o32_3, 1);
        y0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        y3[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);
        int32x4_t e3 = vaddq_s32(ee[3], eo[3]);
        int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
        ADD_LANE(o3, r3, c_o32_2, 2);
        SUB_LANE(o3, r5, c_o32_3, 2);
        SUB_LANE(o3, r7, c_o32_1, 3);
        SUB_LANE(o3, r9, c_o32_0, 0);
        SUB_LANE(o3, r11, c_o32_1, 2);
        SUB_LANE(o3, r13, c_o32_3, 1);
        ADD_LANE(o3, r15, c_o32_2, 3);
        ADD_LANE(o3, r17, c_o32_1, 0);
        ADD_LANE(o3, r19, c_o32_0, 2);
        ADD_LANE(o3, r21, c_o32_2, 1);
        SUB_LANE(o3, r23, c_o32_3, 3);
        SUB_LANE(o3, r25, c_o32_2, 0);
        SUB_LANE(o3, r27, c_o32_0, 0);
        SUB_LANE(o3, r29, c_o32_1, 1);
        SUB_LANE(o3, r31, c_o32_3, 0);
        y0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        y3[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);
        int32x4_t e4 = vaddq_s32(ee[4], eo[4]);
        int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
        ADD_LANE(o4, r3, c_o32_3, 1);
        SUB_LANE(o4, r5, c_o32_2, 1);
        SUB_LANE(o4, r7, c_o32_0, 0);
        SUB_LANE(o4, r9, c_o32_2, 0);
        ADD_LANE(o4, r11, c_o32_3, 2);
        ADD_LANE(o4, r13, c_o32_1, 1);
        ADD_LANE(o4, r15, c_o32_0, 3);
        ADD_LANE(o4, r17, c_o32_3, 0);
        SUB_LANE(o4, r19, c_o32_2, 2);
        SUB_LANE(o4, r21, c_o32_0, 0);
        SUB_LANE(o4, r23, c_o32_1, 3);
        ADD_LANE(o4, r25, c_o32_3, 3);
        ADD_LANE(o4, r27, c_o32_1, 2);
        ADD_LANE(o4, r29, c_o32_0, 2);
        ADD_LANE(o4, r31, c_o32_2, 3);
        y0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        y3[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);
        int32x4_t e5 = vaddq_s32(ee[5], eo[5]);
        int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
        SUB_LANE(o5, r3, c_o32_3, 3);
        SUB_LANE(o5, r5, c_o32_1, 0);
        SUB_LANE(o5, r7, c_o32_1, 2);
        ADD_LANE(o5, r9, c_o32_3, 2);
        ADD_LANE(o5, r11, c_o32_0, 3);
        ADD_LANE(o5, r13, c_o32_1, 3);
        SUB_LANE(o5, r15, c_o32_3, 1);
        SUB_LANE(o5, r17, c_o32_0, 2);
        SUB_LANE(o5, r19, c_o32_2, 0);
        ADD_LANE(o5, r21, c_o32_3, 0);
        ADD_LANE(o5, r23, c_o32_0, 0);
        ADD_LANE(o5, r25, c_o32_2, 1);
        SUB_LANE(o5, r27, c_o32_2, 3);
        SUB_LANE(o5, r29, c_o32_0, 0);
        SUB_LANE(o5, r31, c_o32_2, 2);
        y0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        y3[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);
        int32x4_t e6 = vaddq_s32(ee[6], eo[6]);
        int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
        SUB_LANE(o6, r3, c_o32_3, 0);
        SUB_LANE(o6, r5, c_o32_0, 0);
        SUB_LANE(o6, r7, c_o32_3, 1);
        ADD_LANE(o6, r9, c_o32_1, 1);
        ADD_LANE(o6, r11, c_o32_1, 3);
        SUB_LANE(o6, r13, c_o32_2, 3);
        SUB_LANE(o6, r15, c_o32_0, 0);
        SUB_LANE(o6, r17, c_o32_3, 2);
        ADD_LANE(o6, r19, c_o32_1, 0);
        ADD_LANE(o6, r21, c_o32_2, 0);
        SUB_LANE(o6, r23, c_o32_2, 2);
        SUB_LANE(o6, r25, c_o32_0, 2);
        SUB_LANE(o6, r27, c_o32_3, 3);
        ADD_LANE(o6, r29, c_o32_0, 3);
        ADD_LANE(o6, r31, c_o32_2, 1);
        y0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        y3[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);
        int32x4_t e7 = vaddq_s32(ee[7], eo[7]);
        int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
        SUB_LANE(o7, r3, c_o32_2, 1);
        SUB_LANE(o7, r5, c_o32_1, 1);
        ADD_LANE(o7, r7, c_o32_2, 3);
        ADD_LANE(o7, r9, c_o32_0, 3);
        SUB_LANE(o7, r11, c_o32_3, 1);
        SUB_LANE(o7, r13, c_o32_0, 0);
        ADD_LANE(o7, r15, c_o32_3, 3);
        ADD_LANE(o7, r17, c_o32_0, 0);
        ADD_LANE(o7, r19, c_o32_3, 2);
        SUB_LANE(o7, r21, c_o32_0, 2);
        SUB_LANE(o7, r23, c_o32_3, 0);
        ADD_LANE(o7, r25, c_o32_1, 0);
        ADD_LANE(o7, r27, c_o32_2, 2);
        SUB_LANE(o7, r29, c_o32_1, 2);
        SUB_LANE(o7, r31, c_o32_2, 0);
        y0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        y3[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
        store_transposed_8x4(tmp, cb, 0, y0);

        store_transposed_8x4(tmp, cb, 24, y3);
    }

    /* Phase 2: Chunk 1 & 2 */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee[0], eo[0]);
        int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
        SUB_LANE(o15, r3, c_o32_3, 2);
        ADD_LANE(o15, r5, c_o32_3, 1);
        SUB_LANE(o15, r7, c_o32_3, 0);
        ADD_LANE(o15, r9, c_o32_2, 3);
        SUB_LANE(o15, r11, c_o32_2, 2);
        ADD_LANE(o15, r13, c_o32_2, 1);
        SUB_LANE(o15, r15, c_o32_2, 0);
        ADD_LANE(o15, r17, c_o32_1, 3);
        SUB_LANE(o15, r19, c_o32_1, 2);
        ADD_LANE(o15, r21, c_o32_1, 1);
        SUB_LANE(o15, r23, c_o32_1, 0);
        ADD_LANE(o15, r25, c_o32_0, 3);
        SUB_LANE(o15, r27, c_o32_0, 2);
        ADD_LANE(o15, r29, c_o32_0, 0);
        SUB_LANE(o15, r31, c_o32_0, 0);
        y1[7] = vqrshrn_n_s32(vaddq_s32(e15, o15), 7);
        y2[0] = vqrshrn_n_s32(vsubq_s32(e15, o15), 7);
        int32x4_t e14 = vsubq_s32(ee[1], eo[1]);
        int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
        SUB_LANE(o14, r3, c_o32_2, 3);
        ADD_LANE(o14, r5, c_o32_2, 0);
        SUB_LANE(o14, r7, c_o32_1, 1);
        ADD_LANE(o14, r9, c_o32_0, 2);
        SUB_LANE(o14, r11, c_o32_0, 0);
        ADD_LANE(o14, r13, c_o32_0, 3);
        SUB_LANE(o14, r15, c_o32_1, 2);
        ADD_LANE(o14, r17, c_o32_2, 1);
        SUB_LANE(o14, r19, c_o32_3, 0);
        ADD_LANE(o14, r21, c_o32_3, 3);
        ADD_LANE(o14, r23, c_o32_3, 1);
        SUB_LANE(o14, r25, c_o32_2, 2);
        ADD_LANE(o14, r27, c_o32_1, 3);
        SUB_LANE(o14, r29, c_o32_1, 0);
        ADD_LANE(o14, r31, c_o32_0, 0);
        y1[6] = vqrshrn_n_s32(vaddq_s32(e14, o14), 7);
        y2[1] = vqrshrn_n_s32(vsubq_s32(e14, o14), 7);
        int32x4_t e13 = vsubq_s32(ee[2], eo[2]);
        int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
        SUB_LANE(o13, r3, c_o32_2, 0);
        ADD_LANE(o13, r5, c_o32_0, 3);
        SUB_LANE(o13, r7, c_o32_0, 0);
        ADD_LANE(o13, r9, c_o32_1, 2);
        SUB_LANE(o13, r11, c_o32_2, 3);
        SUB_LANE(o13, r13, c_o32_3, 3);
        ADD_LANE(o13, r15, c_o32_2, 2);
        SUB_LANE(o13, r17, c_o32_1, 1);
        ADD_LANE(o13, r19, c_o32_0, 0);
        SUB_LANE(o13, r21, c_o32_1, 0);
        ADD_LANE(o13, r23, c_o32_2, 1);
        SUB_LANE(o13, r25, c_o32_3, 2);
        SUB_LANE(o13, r27, c_o32_3, 0);
        ADD_LANE(o13, r29, c_o32_1, 3);
        SUB_LANE(o13, r31, c_o32_0, 2);
        y1[5] = vqrshrn_n_s32(vaddq_s32(e13, o13), 7);
        y2[2] = vqrshrn_n_s32(vsubq_s32(e13, o13), 7);
        int32x4_t e12 = vsubq_s32(ee[3], eo[3]);
        int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
        SUB_LANE(o12, r3, c_o32_1, 1);
        ADD_LANE(o12, r5, c_o32_0, 0);
        SUB_LANE(o12, r7, c_o32_2, 0);
        ADD_LANE(o12, r9, c_o32_3, 3);
        ADD_LANE(o12, r11, c_o32_2, 1);
        SUB_LANE(o12, r13, c_o32_0, 2);
        ADD_LANE(o12, r15, c_o32_1, 0);
        SUB_LANE(o12, r17, c_o32_2, 3);
        SUB_LANE(o12, r19, c_o32_3, 1);
        ADD_LANE(o12, r21, c_o32_1, 2);
        SUB_LANE(o12, r23, c_o32_0, 0);
        ADD_LANE(o12, r25, c_o32_1, 3);
        SUB_LANE(o12, r27, c_o32_3, 2);
        SUB_LANE(o12, r29, c_o32_2, 2);
        ADD_LANE(o12, r31, c_o32_0, 3);
        y1[4] = vqrshrn_n_s32(vaddq_s32(e12, o12), 7);
        y2[3] = vqrshrn_n_s32(vsubq_s32(e12, o12), 7);
        int32x4_t e11 = vsubq_s32(ee[4], eo[4]);
        int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
        SUB_LANE(o11, r3, c_o32_0, 2);
        ADD_LANE(o11, r5, c_o32_1, 2);
        SUB_LANE(o11, r7, c_o32_3, 3);
        SUB_LANE(o11, r9, c_o32_1, 3);
        ADD_LANE(o11, r11, c_o32_0, 0);
        SUB_LANE(o11, r13, c_o32_2, 2);
        SUB_LANE(o11, r15, c_o32_3, 0);
        ADD_LANE(o11, r17, c_o32_0, 3);
        SUB_LANE(o11, r19, c_o32_1, 1);
        ADD_LANE(o11, r21, c_o32_3, 2);
        ADD_LANE(o11, r23, c_o32_2, 0);
        SUB_LANE(o11, r25, c_o32_0, 0);
        ADD_LANE(o11, r27, c_o32_2, 1);
        ADD_LANE(o11, r29, c_o32_3, 1);
        SUB_LANE(o11, r31, c_o32_1, 0);
        y1[3] = vqrshrn_n_s32(vaddq_s32(e11, o11), 7);
        y2[4] = vqrshrn_n_s32(vsubq_s32(e11, o11), 7);
        int32x4_t e10 = vsubq_s32(ee[5], eo[5]);
        int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
        SUB_LANE(o10, r3, c_o32_0, 1);
        ADD_LANE(o10, r5, c_o32_2, 3);
        ADD_LANE(o10, r7, c_o32_2, 1);
        SUB_LANE(o10, r9, c_o32_0, 0);
        ADD_LANE(o10, r11, c_o32_3, 0);
        ADD_LANE(o10, r13, c_o32_2, 0);
        SUB_LANE(o10, r15, c_o32_0, 2);
        ADD_LANE(o10, r17, c_o32_3, 1);
        ADD_LANE(o10, r19, c_o32_1, 3);
        SUB_LANE(o10, r21, c_o32_0, 3);
        ADD_LANE(o10, r23, c_o32_3, 2);
        ADD_LANE(o10, r25, c_o32_1, 2);
        SUB_LANE(o10, r27, c_o32_1, 0);
        ADD_LANE(o10, r29, c_o32_3, 3);
        ADD_LANE(o10, r31, c_o32_1, 1);
        y1[2] = vqrshrn_n_s32(vaddq_s32(e10, o10), 7);
        y2[5] = vqrshrn_n_s32(vsubq_s32(e10, o10), 7);
        int32x4_t e9 = vsubq_s32(ee[6], eo[6]);
        int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
        SUB_LANE(o9, r3, c_o32_0, 3);
        SUB_LANE(o9, r5, c_o32_3, 3);
        ADD_LANE(o9, r7, c_o32_0, 2);
        SUB_LANE(o9, r9, c_o32_2, 2);
        SUB_LANE(o9, r11, c_o32_2, 0);
        ADD_LANE(o9, r13, c_o32_1, 0);
        ADD_LANE(o9, r15, c_o32_3, 2);
        SUB_LANE(o9, r17, c_o32_0, 0);
        ADD_LANE(o9, r19, c_o32_2, 3);
        ADD_LANE(o9, r21, c_o32_1, 3);
        SUB_LANE(o9, r23, c_o32_1, 1);
        SUB_LANE(o9, r25, c_o32_3, 1);
        ADD_LANE(o9, r27, c_o32_0, 0);
        SUB_LANE(o9, r29, c_o32_3, 0);
        SUB_LANE(o9, r31, c_o32_1, 2);
        y1[1] = vqrshrn_n_s32(vaddq_s32(e9, o9), 7);
        y2[6] = vqrshrn_n_s32(vsubq_s32(e9, o9), 7);
        int32x4_t e8 = vsubq_s32(ee[7], eo[7]);
        int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
        SUB_LANE(o8, r3, c_o32_1, 2);
        SUB_LANE(o8, r5, c_o32_2, 2);
        ADD_LANE(o8, r7, c_o32_1, 0);
        ADD_LANE(o8, r9, c_o32_3, 0);
        SUB_LANE(o8, r11, c_o32_0, 2);
        SUB_LANE(o8, r13, c_o32_3, 2);
        ADD_LANE(o8, r15, c_o32_0, 0);
        SUB_LANE(o8, r17, c_o32_3, 3);
        SUB_LANE(o8, r19, c_o32_0, 0);
        ADD_LANE(o8, r21, c_o32_3, 1);
        ADD_LANE(o8, r23, c_o32_0, 3);
        SUB_LANE(o8, r25, c_o32_2, 3);
        SUB_LANE(o8, r27, c_o32_1, 1);
        ADD_LANE(o8, r29, c_o32_2, 1);
        ADD_LANE(o8, r31, c_o32_1, 3);
        y1[0] = vqrshrn_n_s32(vaddq_s32(e8, o8), 7);
        y2[7] = vqrshrn_n_s32(vsubq_s32(e8, o8), 7);
        store_transposed_8x4(tmp, cb, 8, y1);

        store_transposed_8x4(tmp, cb, 16, y2);
    }
}

static inline void idct32_4col_stage2_recon_zero28(const WORD16 *tmp,
                                                   const UWORD16 *pred,
                                                   UWORD16 *dst,
                                                   WORD32 pred_strd,
                                                   WORD32 dst_strd,
                                                   WORD32 cb,
                                                   int32x4_t neg_shift,
                                                   int16x8_t clip_min,
                                                   int16x8_t clip_max,
                                                   int16x4_t co0,
                                                   int16x4_t co1,
                                                   int16x4_t c_o32_0,
                                                   int16x4_t c_o32_1,
                                                   int16x4_t c_o32_2,
                                                   int16x4_t c_o32_3)
{
    int16x4_t r0 = vld1_s16(tmp + 0 * 32 + cb);
    int16x4_t r1 = vld1_s16(tmp + 1 * 32 + cb);
    int16x4_t r2 = vld1_s16(tmp + 2 * 32 + cb);
    int16x4_t r3 = vld1_s16(tmp + 3 * 32 + cb);
    int32x4_t ee = vshll_n_s16(r0, 6);

    /* Phase 1: Chunk 0 & Chunk 3 */
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 0));
        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 0), r3, c_o32_0, 1);
        y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));
        int32x4_t e1 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 1));
        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 1), r3, c_o32_1, 0);
        y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));
        int32x4_t e2 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 2));
        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 2), r3, c_o32_1, 3);
        y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));
        int32x4_t e3 = vaddq_s32(ee, vmull_lane_s16(r2, co0, 3));
        int32x4_t o3 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_0, 3), r3, c_o32_2, 2);
        y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));
        int32x4_t e4 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 0));
        int32x4_t o4 = vmlal_lane_s16(vmull_lane_s16(r1, c_o32_1, 0), r3, c_o32_3, 1);
        y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));
        int32x4_t e5 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 1));
        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 1), r3, c_o32_3, 3);
        y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));
        int32x4_t e6 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 2));
        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 2), r3, c_o32_3, 0);
        y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));
        int32x4_t e7 = vaddq_s32(ee, vmull_lane_s16(r2, co1, 3));
        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_1, 3), r3, c_o32_2, 1);
        y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 0, y0, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 24, y3, clip_min, clip_max);
    }

    /* Phase 2: Chunk 1 & Chunk 2 */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 0));
        int32x4_t o15 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 3), r3, c_o32_3, 2);
        y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
        y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        int32x4_t e14 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 1));
        int32x4_t o14 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 2), r3, c_o32_2, 3);
        y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
        y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));
        int32x4_t e13 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 2));
        int32x4_t o13 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 1), r3, c_o32_2, 0);
        y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
        y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));
        int32x4_t e12 = vsubq_s32(ee, vmull_lane_s16(r2, co0, 3));
        int32x4_t o12 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_3, 0), r3, c_o32_1, 1);
        y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
        y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));
        int32x4_t e11 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 0));
        int32x4_t o11 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 3), r3, c_o32_0, 2);
        y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
        y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));
        int32x4_t e10 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 1));
        int32x4_t o10 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 2), r3, c_o32_0, 1);
        y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
        y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));
        int32x4_t e9 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 2));
        int32x4_t o9 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 1), r3, c_o32_0, 3);
        y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
        y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));
        int32x4_t e8 = vsubq_s32(ee, vmull_lane_s16(r2, co1, 3));
        int32x4_t o8 = vmlsl_lane_s16(vmull_lane_s16(r1, c_o32_2, 0), r3, c_o32_1, 2);
        y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
        y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 8, y1, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 16, y2, clip_min, clip_max);
    }
}

static inline void idct32_4col_stage2_recon_zero24(const WORD16 *tmp,
                                                   const UWORD16 *pred,
                                                   UWORD16 *dst,
                                                   WORD32 pred_strd,
                                                   WORD32 dst_strd,
                                                   WORD32 cb,
                                                   int32x4_t neg_shift,
                                                   int16x8_t clip_min,
                                                   int16x8_t clip_max,
                                                   int16x4_t ce1,
                                                   int16x4_t co0,
                                                   int16x4_t co1,
                                                   int16x4_t c_o32_0,
                                                   int16x4_t c_o32_1,
                                                   int16x4_t c_o32_2,
                                                   int16x4_t c_o32_3)
{
    int16x4_t r0 = vld1_s16(tmp + 0 * 32 + cb);
    int16x4_t r1 = vld1_s16(tmp + 1 * 32 + cb);
    int16x4_t r2 = vld1_s16(tmp + 2 * 32 + cb);
    int16x4_t r3 = vld1_s16(tmp + 3 * 32 + cb);
    int16x4_t r4 = vld1_s16(tmp + 4 * 32 + cb);
    int16x4_t r5 = vld1_s16(tmp + 5 * 32 + cb);
    int16x4_t r6 = vld1_s16(tmp + 6 * 32 + cb);
    int16x4_t r7 = vld1_s16(tmp + 7 * 32 + cb);
    int32x4_t eee = vshll_n_s16(r0, 6);
    int32x4_t ee[8];
    int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
    ee[0] = vaddq_s32(eee, eeo0);

    ee[7] = vsubq_s32(eee, eeo0);
    int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
    ee[1] = vaddq_s32(eee, eeo1);

    ee[6] = vsubq_s32(eee, eeo1);
    int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
    ee[2] = vaddq_s32(eee, eeo2);

    ee[5] = vsubq_s32(eee, eeo2);
    int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
    ee[3] = vaddq_s32(eee, eeo3);

    ee[4] = vsubq_s32(eee, eeo3);

    /* Phase 1: Chunk 0 & Chunk 3 */
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee[0],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 0), r6, co0, 1));
        int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
        ADD_LANE(o0, r3, c_o32_0, 1);
        ADD_LANE(o0, r5, c_o32_0, 2);
        ADD_LANE(o0, r7, c_o32_0, 3);
        y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));
        int32x4_t e1 = vaddq_s32(ee[1],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 1), r6, co1, 0));
        int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
        ADD_LANE(o1, r3, c_o32_1, 0);
        ADD_LANE(o1, r5, c_o32_1, 3);
        ADD_LANE(o1, r7, c_o32_2, 2);
        y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));
        int32x4_t e2 = vaddq_s32(ee[2],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 2), r6, co1, 3));
        int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
        ADD_LANE(o2, r3, c_o32_1, 3);
        ADD_LANE(o2, r5, c_o32_3, 0);
        SUB_LANE(o2, r7, c_o32_3, 2);
        y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));
        int32x4_t e3 = vaddq_s32(ee[3],
            vmlsl_lane_s16(vmull_lane_s16(r2, co0, 3), r6, co1, 1));
        int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
        ADD_LANE(o3, r3, c_o32_2, 2);
        SUB_LANE(o3, r5, c_o32_3, 2);
        SUB_LANE(o3, r7, c_o32_1, 3);
        y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));
        int32x4_t e4 = vaddq_s32(ee[4],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 0), r6, co0, 2));
        int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
        ADD_LANE(o4, r3, c_o32_3, 1);
        SUB_LANE(o4, r5, c_o32_2, 1);
        SUB_LANE(o4, r7, c_o32_0, 0);
        y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));
        int32x4_t e5 = vaddq_s32(ee[5],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 1), r6, co0, 0));
        int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
        SUB_LANE(o5, r3, c_o32_3, 3);
        SUB_LANE(o5, r5, c_o32_1, 0);
        SUB_LANE(o5, r7, c_o32_1, 2);
        y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));
        int32x4_t e6 = vaddq_s32(ee[6],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 2), r6, co0, 3));
        int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
        SUB_LANE(o6, r3, c_o32_3, 0);
        SUB_LANE(o6, r5, c_o32_0, 0);
        SUB_LANE(o6, r7, c_o32_3, 1);
        y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));
        int32x4_t e7 = vaddq_s32(ee[7],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 3), r6, co1, 2));
        int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
        SUB_LANE(o7, r3, c_o32_2, 1);
        SUB_LANE(o7, r5, c_o32_1, 1);
        ADD_LANE(o7, r7, c_o32_2, 3);
        y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 0, y0, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 24, y3, clip_min, clip_max);
    }

    /* Phase 2: Chunk 1 & Chunk 2 */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee[0],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 0), r6, co0, 1));
        int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
        SUB_LANE(o15, r3, c_o32_3, 2);
        ADD_LANE(o15, r5, c_o32_3, 1);
        SUB_LANE(o15, r7, c_o32_3, 0);
        y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
        y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        int32x4_t e14 = vsubq_s32(ee[1],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 1), r6, co1, 0));
        int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
        SUB_LANE(o14, r3, c_o32_2, 3);
        ADD_LANE(o14, r5, c_o32_2, 0);
        SUB_LANE(o14, r7, c_o32_1, 1);
        y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
        y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));
        int32x4_t e13 = vsubq_s32(ee[2],
            vmlal_lane_s16(vmull_lane_s16(r2, co0, 2), r6, co1, 3));
        int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
        SUB_LANE(o13, r3, c_o32_2, 0);
        ADD_LANE(o13, r5, c_o32_0, 3);
        SUB_LANE(o13, r7, c_o32_0, 0);
        y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
        y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));
        int32x4_t e12 = vsubq_s32(ee[3],
            vmlsl_lane_s16(vmull_lane_s16(r2, co0, 3), r6, co1, 1));
        int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
        SUB_LANE(o12, r3, c_o32_1, 1);
        ADD_LANE(o12, r5, c_o32_0, 0);
        SUB_LANE(o12, r7, c_o32_2, 0);
        y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
        y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));
        int32x4_t e11 = vsubq_s32(ee[4],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 0), r6, co0, 2));
        int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
        SUB_LANE(o11, r3, c_o32_0, 2);
        ADD_LANE(o11, r5, c_o32_1, 2);
        SUB_LANE(o11, r7, c_o32_3, 3);
        y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
        y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));
        int32x4_t e10 = vsubq_s32(ee[5],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 1), r6, co0, 0));
        int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
        SUB_LANE(o10, r3, c_o32_0, 1);
        ADD_LANE(o10, r5, c_o32_2, 3);
        ADD_LANE(o10, r7, c_o32_2, 1);
        y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
        y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));
        int32x4_t e9 = vsubq_s32(ee[6],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 2), r6, co0, 3));
        int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
        SUB_LANE(o9, r3, c_o32_0, 3);
        SUB_LANE(o9, r5, c_o32_3, 3);
        ADD_LANE(o9, r7, c_o32_0, 2);
        y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
        y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));
        int32x4_t e8 = vsubq_s32(ee[7],
            vmlsl_lane_s16(vmull_lane_s16(r2, co1, 3), r6, co1, 2));
        int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
        SUB_LANE(o8, r3, c_o32_1, 2);
        SUB_LANE(o8, r5, c_o32_2, 2);
        ADD_LANE(o8, r7, c_o32_1, 0);
        y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
        y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 8, y1, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 16, y2, clip_min, clip_max);
    }
}

static inline void idct32_4col_stage2_recon_full(const WORD16 *tmp,
                                                 const UWORD16 *pred,
                                                 UWORD16 *dst,
                                                 WORD32 pred_strd,
                                                 WORD32 dst_strd,
                                                 WORD32 cb,
                                                 int32x4_t neg_shift,
                                                 int16x8_t clip_min,
                                                 int16x8_t clip_max,
                                                 int16x4_t ce0,
                                                 int16x4_t ce1,
                                                 int16x4_t co0,
                                                 int16x4_t co1,
                                                 int16x4_t c_o32_0,
                                                 int16x4_t c_o32_1,
                                                 int16x4_t c_o32_2,
                                                 int16x4_t c_o32_3)
{
    int32x4_t ee[8];
    int32x4_t eo[8];
    {
        int16x4_t r0 = vld1_s16(tmp + 0 * 32 + cb);
        int16x4_t r16 = vld1_s16(tmp + 16 * 32 + cb);
        int16x4_t r8 = vld1_s16(tmp + 8 * 32 + cb);
        int16x4_t r24 = vld1_s16(tmp + 24 * 32 + cb);
        int16x4_t r4 = vld1_s16(tmp + 4 * 32 + cb);
        int16x4_t r12 = vld1_s16(tmp + 12 * 32 + cb);
        int16x4_t r20 = vld1_s16(tmp + 20 * 32 + cb);
        int16x4_t r28 = vld1_s16(tmp + 28 * 32 + cb);
        int16x4_t r2 = vld1_s16(tmp + 2 * 32 + cb);
        int16x4_t r6 = vld1_s16(tmp + 6 * 32 + cb);
        int16x4_t r10 = vld1_s16(tmp + 10 * 32 + cb);
        int16x4_t r14 = vld1_s16(tmp + 14 * 32 + cb);
        int16x4_t r18 = vld1_s16(tmp + 18 * 32 + cb);
        int16x4_t r22 = vld1_s16(tmp + 22 * 32 + cb);
        int16x4_t r26 = vld1_s16(tmp + 26 * 32 + cb);
        int16x4_t r30 = vld1_s16(tmp + 30 * 32 + cb);

        int32x4_t r0_shl = vshll_n_s16(r0, 6);
        int32x4_t r16_shl = vshll_n_s16(r16, 6);
        int32x4_t eeee0 = vaddq_s32(r0_shl, r16_shl);
        int32x4_t eeee1 = vsubq_s32(r0_shl, r16_shl);
        int32x4_t eeeo0 = vmlal_lane_s16(vmull_lane_s16(r8, ce0, 0), r24, ce0, 1);
        int32x4_t eeeo1 = vmlsl_lane_s16(vmull_lane_s16(r8, ce0, 1), r24, ce0, 0);
        int32x4_t eee0 = vaddq_s32(eeee0, eeeo0);
        int32x4_t eee3 = vsubq_s32(eeee0, eeeo0);
        int32x4_t eee1 = vaddq_s32(eeee1, eeeo1);
        int32x4_t eee2 = vsubq_s32(eeee1, eeeo1);

        int32x4_t eeo0 = vmull_lane_s16(r4, ce1, 0);
        ADD_LANE(eeo0, r12, ce1, 1);
        ADD_LANE(eeo0, r20, ce1, 2);
        ADD_LANE(eeo0, r28, ce1, 3);
        int32x4_t eeo1 = vmull_lane_s16(r4, ce1, 1);
        SUB_LANE(eeo1, r12, ce1, 3);
        SUB_LANE(eeo1, r20, ce1, 0);
        SUB_LANE(eeo1, r28, ce1, 2);
        int32x4_t eeo2 = vmull_lane_s16(r4, ce1, 2);
        SUB_LANE(eeo2, r12, ce1, 0);
        ADD_LANE(eeo2, r20, ce1, 3);
        ADD_LANE(eeo2, r28, ce1, 1);
        int32x4_t eeo3 = vmull_lane_s16(r4, ce1, 3);
        SUB_LANE(eeo3, r12, ce1, 2);
        ADD_LANE(eeo3, r20, ce1, 1);
        SUB_LANE(eeo3, r28, ce1, 0);

        ee[0] = vaddq_s32(eee0, eeo0);


        ee[7] = vsubq_s32(eee0, eeo0);
        ee[1] = vaddq_s32(eee1, eeo1);

        ee[6] = vsubq_s32(eee1, eeo1);
        ee[2] = vaddq_s32(eee2, eeo2);

        ee[5] = vsubq_s32(eee2, eeo2);
        ee[3] = vaddq_s32(eee3, eeo3);

        ee[4] = vsubq_s32(eee3, eeo3);

        eo[0] = vmull_lane_s16(r2, co0, 0);
        ADD_LANE(eo[0], r6, co0, 1);
        ADD_LANE(eo[0], r10, co0, 2);
        ADD_LANE(eo[0], r14, co0, 3);
        ADD_LANE(eo[0], r18, co1, 0);
        ADD_LANE(eo[0], r22, co1, 1);
        ADD_LANE(eo[0], r26, co1, 2);
        ADD_LANE(eo[0], r30, co1, 3);
        eo[1] = vmull_lane_s16(r2, co0, 1);
        ADD_LANE(eo[1], r6, co1, 0);
        ADD_LANE(eo[1], r10, co1, 3);
        SUB_LANE(eo[1], r14, co1, 1);
        SUB_LANE(eo[1], r18, co0, 2);
        SUB_LANE(eo[1], r22, co0, 0);
        SUB_LANE(eo[1], r26, co0, 3);
        SUB_LANE(eo[1], r30, co1, 2);
        eo[2] = vmull_lane_s16(r2, co0, 2);
        ADD_LANE(eo[2], r6, co1, 3);
        SUB_LANE(eo[2], r10, co0, 3);
        SUB_LANE(eo[2], r14, co0, 1);
        SUB_LANE(eo[2], r18, co1, 2);
        ADD_LANE(eo[2], r22, co1, 0);
        ADD_LANE(eo[2], r26, co0, 0);
        ADD_LANE(eo[2], r30, co1, 1);
        eo[3] = vmull_lane_s16(r2, co0, 3);
        SUB_LANE(eo[3], r6, co1, 1);
        SUB_LANE(eo[3], r10, co0, 1);
        ADD_LANE(eo[3], r14, co1, 3);
        ADD_LANE(eo[3], r18, co0, 0);
        ADD_LANE(eo[3], r22, co1, 2);
        SUB_LANE(eo[3], r26, co0, 2);
        SUB_LANE(eo[3], r30, co1, 0);
        eo[4] = vmull_lane_s16(r2, co1, 0);
        SUB_LANE(eo[4], r6, co0, 2);
        SUB_LANE(eo[4], r10, co1, 2);
        ADD_LANE(eo[4], r14, co0, 0);
        SUB_LANE(eo[4], r18, co1, 3);
        SUB_LANE(eo[4], r22, co0, 1);
        ADD_LANE(eo[4], r26, co1, 1);
        ADD_LANE(eo[4], r30, co0, 3);
        eo[5] = vmull_lane_s16(r2, co1, 1);
        SUB_LANE(eo[5], r6, co0, 0);
        ADD_LANE(eo[5], r10, co1, 0);
        ADD_LANE(eo[5], r14, co1, 2);
        SUB_LANE(eo[5], r18, co0, 1);
        ADD_LANE(eo[5], r22, co0, 3);
        ADD_LANE(eo[5], r26, co1, 3);
        SUB_LANE(eo[5], r30, co0, 2);
        eo[6] = vmull_lane_s16(r2, co1, 2);
        SUB_LANE(eo[6], r6, co0, 3);
        ADD_LANE(eo[6], r10, co0, 0);
        SUB_LANE(eo[6], r14, co0, 2);
        ADD_LANE(eo[6], r18, co1, 1);
        ADD_LANE(eo[6], r22, co1, 3);
        SUB_LANE(eo[6], r26, co1, 0);
        ADD_LANE(eo[6], r30, co0, 1);
        eo[7] = vmull_lane_s16(r2, co1, 3);
        SUB_LANE(eo[7], r6, co1, 2);
        ADD_LANE(eo[7], r10, co1, 1);
        SUB_LANE(eo[7], r14, co1, 0);
        ADD_LANE(eo[7], r18, co0, 3);
        SUB_LANE(eo[7], r22, co0, 2);
        ADD_LANE(eo[7], r26, co0, 1);
        SUB_LANE(eo[7], r30, co0, 0);
    }

    /* Odd stage */
    int16x4_t r1 = vld1_s16(tmp + 1 * 32 + cb);
        int16x4_t r3 = vld1_s16(tmp + 3 * 32 + cb);
        int16x4_t r5 = vld1_s16(tmp + 5 * 32 + cb);
        int16x4_t r7 = vld1_s16(tmp + 7 * 32 + cb);
        int16x4_t r9 = vld1_s16(tmp + 9 * 32 + cb);
        int16x4_t r11 = vld1_s16(tmp + 11 * 32 + cb);
        int16x4_t r13 = vld1_s16(tmp + 13 * 32 + cb);
        int16x4_t r15 = vld1_s16(tmp + 15 * 32 + cb);
        int16x4_t r17 = vld1_s16(tmp + 17 * 32 + cb);
        int16x4_t r19 = vld1_s16(tmp + 19 * 32 + cb);
        int16x4_t r21 = vld1_s16(tmp + 21 * 32 + cb);
        int16x4_t r23 = vld1_s16(tmp + 23 * 32 + cb);
        int16x4_t r25 = vld1_s16(tmp + 25 * 32 + cb);
        int16x4_t r27 = vld1_s16(tmp + 27 * 32 + cb);
        int16x4_t r29 = vld1_s16(tmp + 29 * 32 + cb);
        int16x4_t r31 = vld1_s16(tmp + 31 * 32 + cb);
    {
        int16x4_t y0[8], y3[8];
        int32x4_t e0 = vaddq_s32(ee[0], eo[0]);
        int32x4_t o0 = vmull_lane_s16(r1, c_o32_0, 0);
        ADD_LANE(o0, r3, c_o32_0, 1);
        ADD_LANE(o0, r5, c_o32_0, 2);
        ADD_LANE(o0, r7, c_o32_0, 3);
        ADD_LANE(o0, r9, c_o32_1, 0);
        ADD_LANE(o0, r11, c_o32_1, 1);
        ADD_LANE(o0, r13, c_o32_1, 2);
        ADD_LANE(o0, r15, c_o32_1, 3);
        ADD_LANE(o0, r17, c_o32_2, 0);
        ADD_LANE(o0, r19, c_o32_2, 1);
        ADD_LANE(o0, r21, c_o32_2, 2);
        ADD_LANE(o0, r23, c_o32_2, 3);
        ADD_LANE(o0, r25, c_o32_3, 0);
        ADD_LANE(o0, r27, c_o32_3, 1);
        ADD_LANE(o0, r29, c_o32_3, 2);
        ADD_LANE(o0, r31, c_o32_3, 3);
        y0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        y3[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));
        int32x4_t e1 = vaddq_s32(ee[1], eo[1]);
        int32x4_t o1 = vmull_lane_s16(r1, c_o32_0, 1);
        ADD_LANE(o1, r3, c_o32_1, 0);
        ADD_LANE(o1, r5, c_o32_1, 3);
        ADD_LANE(o1, r7, c_o32_2, 2);
        ADD_LANE(o1, r9, c_o32_3, 1);
        SUB_LANE(o1, r11, c_o32_3, 3);
        SUB_LANE(o1, r13, c_o32_3, 0);
        SUB_LANE(o1, r15, c_o32_2, 1);
        SUB_LANE(o1, r17, c_o32_1, 2);
        SUB_LANE(o1, r19, c_o32_0, 3);
        SUB_LANE(o1, r21, c_o32_0, 0);
        SUB_LANE(o1, r23, c_o32_0, 2);
        SUB_LANE(o1, r25, c_o32_1, 1);
        SUB_LANE(o1, r27, c_o32_2, 0);
        SUB_LANE(o1, r29, c_o32_2, 3);
        SUB_LANE(o1, r31, c_o32_3, 2);
        y0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        y3[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));
        int32x4_t e2 = vaddq_s32(ee[2], eo[2]);
        int32x4_t o2 = vmull_lane_s16(r1, c_o32_0, 2);
        ADD_LANE(o2, r3, c_o32_1, 3);
        ADD_LANE(o2, r5, c_o32_3, 0);
        SUB_LANE(o2, r7, c_o32_3, 2);
        SUB_LANE(o2, r9, c_o32_2, 1);
        SUB_LANE(o2, r11, c_o32_1, 0);
        SUB_LANE(o2, r13, c_o32_0, 0);
        SUB_LANE(o2, r15, c_o32_1, 1);
        SUB_LANE(o2, r17, c_o32_2, 2);
        SUB_LANE(o2, r19, c_o32_3, 3);
        ADD_LANE(o2, r21, c_o32_2, 3);
        ADD_LANE(o2, r23, c_o32_1, 2);
        ADD_LANE(o2, r25, c_o32_0, 0);
        ADD_LANE(o2, r27, c_o32_0, 3);
        ADD_LANE(o2, r29, c_o32_2, 0);
        ADD_LANE(o2, r31, c_o32_3, 1);
        y0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        y3[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));
        int32x4_t e3 = vaddq_s32(ee[3], eo[3]);
        int32x4_t o3 = vmull_lane_s16(r1, c_o32_0, 3);
        ADD_LANE(o3, r3, c_o32_2, 2);
        SUB_LANE(o3, r5, c_o32_3, 2);
        SUB_LANE(o3, r7, c_o32_1, 3);
        SUB_LANE(o3, r9, c_o32_0, 0);
        SUB_LANE(o3, r11, c_o32_1, 2);
        SUB_LANE(o3, r13, c_o32_3, 1);
        ADD_LANE(o3, r15, c_o32_2, 3);
        ADD_LANE(o3, r17, c_o32_1, 0);
        ADD_LANE(o3, r19, c_o32_0, 2);
        ADD_LANE(o3, r21, c_o32_2, 1);
        SUB_LANE(o3, r23, c_o32_3, 3);
        SUB_LANE(o3, r25, c_o32_2, 0);
        SUB_LANE(o3, r27, c_o32_0, 0);
        SUB_LANE(o3, r29, c_o32_1, 1);
        SUB_LANE(o3, r31, c_o32_3, 0);
        y0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        y3[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));
        int32x4_t e4 = vaddq_s32(ee[4], eo[4]);
        int32x4_t o4 = vmull_lane_s16(r1, c_o32_1, 0);
        ADD_LANE(o4, r3, c_o32_3, 1);
        SUB_LANE(o4, r5, c_o32_2, 1);
        SUB_LANE(o4, r7, c_o32_0, 0);
        SUB_LANE(o4, r9, c_o32_2, 0);
        ADD_LANE(o4, r11, c_o32_3, 2);
        ADD_LANE(o4, r13, c_o32_1, 1);
        ADD_LANE(o4, r15, c_o32_0, 3);
        ADD_LANE(o4, r17, c_o32_3, 0);
        SUB_LANE(o4, r19, c_o32_2, 2);
        SUB_LANE(o4, r21, c_o32_0, 0);
        SUB_LANE(o4, r23, c_o32_1, 3);
        ADD_LANE(o4, r25, c_o32_3, 3);
        ADD_LANE(o4, r27, c_o32_1, 2);
        ADD_LANE(o4, r29, c_o32_0, 2);
        ADD_LANE(o4, r31, c_o32_2, 3);
        y0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        y3[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));
        int32x4_t e5 = vaddq_s32(ee[5], eo[5]);
        int32x4_t o5 = vmull_lane_s16(r1, c_o32_1, 1);
        SUB_LANE(o5, r3, c_o32_3, 3);
        SUB_LANE(o5, r5, c_o32_1, 0);
        SUB_LANE(o5, r7, c_o32_1, 2);
        ADD_LANE(o5, r9, c_o32_3, 2);
        ADD_LANE(o5, r11, c_o32_0, 3);
        ADD_LANE(o5, r13, c_o32_1, 3);
        SUB_LANE(o5, r15, c_o32_3, 1);
        SUB_LANE(o5, r17, c_o32_0, 2);
        SUB_LANE(o5, r19, c_o32_2, 0);
        ADD_LANE(o5, r21, c_o32_3, 0);
        ADD_LANE(o5, r23, c_o32_0, 0);
        ADD_LANE(o5, r25, c_o32_2, 1);
        SUB_LANE(o5, r27, c_o32_2, 3);
        SUB_LANE(o5, r29, c_o32_0, 0);
        SUB_LANE(o5, r31, c_o32_2, 2);
        y0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        y3[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));
        int32x4_t e6 = vaddq_s32(ee[6], eo[6]);
        int32x4_t o6 = vmull_lane_s16(r1, c_o32_1, 2);
        SUB_LANE(o6, r3, c_o32_3, 0);
        SUB_LANE(o6, r5, c_o32_0, 0);
        SUB_LANE(o6, r7, c_o32_3, 1);
        ADD_LANE(o6, r9, c_o32_1, 1);
        ADD_LANE(o6, r11, c_o32_1, 3);
        SUB_LANE(o6, r13, c_o32_2, 3);
        SUB_LANE(o6, r15, c_o32_0, 0);
        SUB_LANE(o6, r17, c_o32_3, 2);
        ADD_LANE(o6, r19, c_o32_1, 0);
        ADD_LANE(o6, r21, c_o32_2, 0);
        SUB_LANE(o6, r23, c_o32_2, 2);
        SUB_LANE(o6, r25, c_o32_0, 2);
        SUB_LANE(o6, r27, c_o32_3, 3);
        ADD_LANE(o6, r29, c_o32_0, 3);
        ADD_LANE(o6, r31, c_o32_2, 1);
        y0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        y3[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));
        int32x4_t e7 = vaddq_s32(ee[7], eo[7]);
        int32x4_t o7 = vmull_lane_s16(r1, c_o32_1, 3);
        SUB_LANE(o7, r3, c_o32_2, 1);
        SUB_LANE(o7, r5, c_o32_1, 1);
        ADD_LANE(o7, r7, c_o32_2, 3);
        ADD_LANE(o7, r9, c_o32_0, 3);
        SUB_LANE(o7, r11, c_o32_3, 1);
        SUB_LANE(o7, r13, c_o32_0, 0);
        ADD_LANE(o7, r15, c_o32_3, 3);
        ADD_LANE(o7, r17, c_o32_0, 0);
        ADD_LANE(o7, r19, c_o32_3, 2);
        SUB_LANE(o7, r21, c_o32_0, 2);
        SUB_LANE(o7, r23, c_o32_3, 0);
        ADD_LANE(o7, r25, c_o32_1, 0);
        ADD_LANE(o7, r27, c_o32_2, 2);
        SUB_LANE(o7, r29, c_o32_1, 2);
        SUB_LANE(o7, r31, c_o32_2, 0);
        y0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        y3[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 0, y0, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 24, y3, clip_min, clip_max);
    }

    /* Phase 2: Chunk 1 & Chunk 2 */
    {
        int16x4_t y1[8], y2[8];
        int32x4_t e15 = vsubq_s32(ee[0], eo[0]);
        int32x4_t o15 = vmull_lane_s16(r1, c_o32_3, 3);
        SUB_LANE(o15, r3, c_o32_3, 2);
        ADD_LANE(o15, r5, c_o32_3, 1);
        SUB_LANE(o15, r7, c_o32_3, 0);
        ADD_LANE(o15, r9, c_o32_2, 3);
        SUB_LANE(o15, r11, c_o32_2, 2);
        ADD_LANE(o15, r13, c_o32_2, 1);
        SUB_LANE(o15, r15, c_o32_2, 0);
        ADD_LANE(o15, r17, c_o32_1, 3);
        SUB_LANE(o15, r19, c_o32_1, 2);
        ADD_LANE(o15, r21, c_o32_1, 1);
        SUB_LANE(o15, r23, c_o32_1, 0);
        ADD_LANE(o15, r25, c_o32_0, 3);
        SUB_LANE(o15, r27, c_o32_0, 2);
        ADD_LANE(o15, r29, c_o32_0, 0);
        SUB_LANE(o15, r31, c_o32_0, 0);
        y1[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e15, o15), neg_shift));
        y2[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e15, o15), neg_shift));
        int32x4_t e14 = vsubq_s32(ee[1], eo[1]);
        int32x4_t o14 = vmull_lane_s16(r1, c_o32_3, 2);
        SUB_LANE(o14, r3, c_o32_2, 3);
        ADD_LANE(o14, r5, c_o32_2, 0);
        SUB_LANE(o14, r7, c_o32_1, 1);
        ADD_LANE(o14, r9, c_o32_0, 2);
        SUB_LANE(o14, r11, c_o32_0, 0);
        ADD_LANE(o14, r13, c_o32_0, 3);
        SUB_LANE(o14, r15, c_o32_1, 2);
        ADD_LANE(o14, r17, c_o32_2, 1);
        SUB_LANE(o14, r19, c_o32_3, 0);
        ADD_LANE(o14, r21, c_o32_3, 3);
        ADD_LANE(o14, r23, c_o32_3, 1);
        SUB_LANE(o14, r25, c_o32_2, 2);
        ADD_LANE(o14, r27, c_o32_1, 3);
        SUB_LANE(o14, r29, c_o32_1, 0);
        ADD_LANE(o14, r31, c_o32_0, 0);
        y1[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e14, o14), neg_shift));
        y2[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e14, o14), neg_shift));
        int32x4_t e13 = vsubq_s32(ee[2], eo[2]);
        int32x4_t o13 = vmull_lane_s16(r1, c_o32_3, 1);
        SUB_LANE(o13, r3, c_o32_2, 0);
        ADD_LANE(o13, r5, c_o32_0, 3);
        SUB_LANE(o13, r7, c_o32_0, 0);
        ADD_LANE(o13, r9, c_o32_1, 2);
        SUB_LANE(o13, r11, c_o32_2, 3);
        SUB_LANE(o13, r13, c_o32_3, 3);
        ADD_LANE(o13, r15, c_o32_2, 2);
        SUB_LANE(o13, r17, c_o32_1, 1);
        ADD_LANE(o13, r19, c_o32_0, 0);
        SUB_LANE(o13, r21, c_o32_1, 0);
        ADD_LANE(o13, r23, c_o32_2, 1);
        SUB_LANE(o13, r25, c_o32_3, 2);
        SUB_LANE(o13, r27, c_o32_3, 0);
        ADD_LANE(o13, r29, c_o32_1, 3);
        SUB_LANE(o13, r31, c_o32_0, 2);
        y1[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e13, o13), neg_shift));
        y2[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e13, o13), neg_shift));
        int32x4_t e12 = vsubq_s32(ee[3], eo[3]);
        int32x4_t o12 = vmull_lane_s16(r1, c_o32_3, 0);
        SUB_LANE(o12, r3, c_o32_1, 1);
        ADD_LANE(o12, r5, c_o32_0, 0);
        SUB_LANE(o12, r7, c_o32_2, 0);
        ADD_LANE(o12, r9, c_o32_3, 3);
        ADD_LANE(o12, r11, c_o32_2, 1);
        SUB_LANE(o12, r13, c_o32_0, 2);
        ADD_LANE(o12, r15, c_o32_1, 0);
        SUB_LANE(o12, r17, c_o32_2, 3);
        SUB_LANE(o12, r19, c_o32_3, 1);
        ADD_LANE(o12, r21, c_o32_1, 2);
        SUB_LANE(o12, r23, c_o32_0, 0);
        ADD_LANE(o12, r25, c_o32_1, 3);
        SUB_LANE(o12, r27, c_o32_3, 2);
        SUB_LANE(o12, r29, c_o32_2, 2);
        ADD_LANE(o12, r31, c_o32_0, 3);
        y1[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e12, o12), neg_shift));
        y2[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e12, o12), neg_shift));
        int32x4_t e11 = vsubq_s32(ee[4], eo[4]);
        int32x4_t o11 = vmull_lane_s16(r1, c_o32_2, 3);
        SUB_LANE(o11, r3, c_o32_0, 2);
        ADD_LANE(o11, r5, c_o32_1, 2);
        SUB_LANE(o11, r7, c_o32_3, 3);
        SUB_LANE(o11, r9, c_o32_1, 3);
        ADD_LANE(o11, r11, c_o32_0, 0);
        SUB_LANE(o11, r13, c_o32_2, 2);
        SUB_LANE(o11, r15, c_o32_3, 0);
        ADD_LANE(o11, r17, c_o32_0, 3);
        SUB_LANE(o11, r19, c_o32_1, 1);
        ADD_LANE(o11, r21, c_o32_3, 2);
        ADD_LANE(o11, r23, c_o32_2, 0);
        SUB_LANE(o11, r25, c_o32_0, 0);
        ADD_LANE(o11, r27, c_o32_2, 1);
        ADD_LANE(o11, r29, c_o32_3, 1);
        SUB_LANE(o11, r31, c_o32_1, 0);
        y1[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e11, o11), neg_shift));
        y2[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e11, o11), neg_shift));
        int32x4_t e10 = vsubq_s32(ee[5], eo[5]);
        int32x4_t o10 = vmull_lane_s16(r1, c_o32_2, 2);
        SUB_LANE(o10, r3, c_o32_0, 1);
        ADD_LANE(o10, r5, c_o32_2, 3);
        ADD_LANE(o10, r7, c_o32_2, 1);
        SUB_LANE(o10, r9, c_o32_0, 0);
        ADD_LANE(o10, r11, c_o32_3, 0);
        ADD_LANE(o10, r13, c_o32_2, 0);
        SUB_LANE(o10, r15, c_o32_0, 2);
        ADD_LANE(o10, r17, c_o32_3, 1);
        ADD_LANE(o10, r19, c_o32_1, 3);
        SUB_LANE(o10, r21, c_o32_0, 3);
        ADD_LANE(o10, r23, c_o32_3, 2);
        ADD_LANE(o10, r25, c_o32_1, 2);
        SUB_LANE(o10, r27, c_o32_1, 0);
        ADD_LANE(o10, r29, c_o32_3, 3);
        ADD_LANE(o10, r31, c_o32_1, 1);
        y1[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e10, o10), neg_shift));
        y2[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e10, o10), neg_shift));
        int32x4_t e9 = vsubq_s32(ee[6], eo[6]);
        int32x4_t o9 = vmull_lane_s16(r1, c_o32_2, 1);
        SUB_LANE(o9, r3, c_o32_0, 3);
        SUB_LANE(o9, r5, c_o32_3, 3);
        ADD_LANE(o9, r7, c_o32_0, 2);
        SUB_LANE(o9, r9, c_o32_2, 2);
        SUB_LANE(o9, r11, c_o32_2, 0);
        ADD_LANE(o9, r13, c_o32_1, 0);
        ADD_LANE(o9, r15, c_o32_3, 2);
        SUB_LANE(o9, r17, c_o32_0, 0);
        ADD_LANE(o9, r19, c_o32_2, 3);
        ADD_LANE(o9, r21, c_o32_1, 3);
        SUB_LANE(o9, r23, c_o32_1, 1);
        SUB_LANE(o9, r25, c_o32_3, 1);
        ADD_LANE(o9, r27, c_o32_0, 0);
        SUB_LANE(o9, r29, c_o32_3, 0);
        SUB_LANE(o9, r31, c_o32_1, 2);
        y1[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e9, o9), neg_shift));
        y2[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e9, o9), neg_shift));
        int32x4_t e8 = vsubq_s32(ee[7], eo[7]);
        int32x4_t o8 = vmull_lane_s16(r1, c_o32_2, 0);
        SUB_LANE(o8, r3, c_o32_1, 2);
        SUB_LANE(o8, r5, c_o32_2, 2);
        ADD_LANE(o8, r7, c_o32_1, 0);
        ADD_LANE(o8, r9, c_o32_3, 0);
        SUB_LANE(o8, r11, c_o32_0, 2);
        SUB_LANE(o8, r13, c_o32_3, 2);
        ADD_LANE(o8, r15, c_o32_0, 0);
        SUB_LANE(o8, r17, c_o32_3, 3);
        SUB_LANE(o8, r19, c_o32_0, 0);
        ADD_LANE(o8, r21, c_o32_3, 1);
        ADD_LANE(o8, r23, c_o32_0, 3);
        SUB_LANE(o8, r25, c_o32_2, 3);
        SUB_LANE(o8, r27, c_o32_1, 1);
        ADD_LANE(o8, r29, c_o32_2, 1);
        ADD_LANE(o8, r31, c_o32_1, 3);
        y1[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e8, o8), neg_shift));
        y2[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e8, o8), neg_shift));
        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 8, y1, clip_min, clip_max);

        recon_transposed_8x4(dst, dst_strd, pred, pred_strd, cb, 16, y2, clip_min, clip_max);
    }
}

/* 8-column wrappers for Stage 1 */
static inline void idct32_8col_stage1_zero28(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t co0,
                                             int16x4_t co1,
                                             int16x4_t c_o32_0,
                                             int16x4_t c_o32_1,
                                             int16x4_t c_o32_2,
                                             int16x4_t c_o32_3)
{
    idct32_4col_stage1_zero28(src, src_strd, tmp, cb + 0,
                              co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage1_zero28(src, src_strd, tmp, cb + 4,
                              co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

static inline void idct32_8col_stage1_zero24(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t ce1,
                                             int16x4_t co0,
                                             int16x4_t co1,
                                             int16x4_t c_o32_0,
                                             int16x4_t c_o32_1,
                                             int16x4_t c_o32_2,
                                             int16x4_t c_o32_3)
{
    idct32_4col_stage1_zero24(src, src_strd, tmp, cb + 0,
                              ce1, co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage1_zero24(src, src_strd, tmp, cb + 4,
                              ce1, co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

static inline void idct32_8col_stage1_full(const WORD16 *src,
                                           WORD32 src_strd,
                                           WORD16 *tmp,
                                           WORD32 cb,
                                           int16x4_t ce0,
                                           int16x4_t ce1,
                                           int16x4_t co0,
                                           int16x4_t co1,
                                           int16x4_t c_o32_0,
                                           int16x4_t c_o32_1,
                                           int16x4_t c_o32_2,
                                           int16x4_t c_o32_3)
{
    idct32_4col_stage1_full(src, src_strd, tmp, cb + 0,
                            ce0, ce1, co0, co1,
                            c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage1_full(src, src_strd, tmp, cb + 4,
                            ce0, ce1, co0, co1,
                            c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

/* 8-column wrappers for Stage 2 & Recon */
static inline void idct32_8col_stage2_recon_zero28(const WORD16 *tmp,
                                                   const UWORD16 *pred,
                                                   UWORD16 *dst,
                                                   WORD32 pred_strd,
                                                   WORD32 dst_strd,
                                                   WORD32 cb,
                                                   int32x4_t neg_shift,
                                                   int16x8_t clip_min,
                                                   int16x8_t clip_max,
                                                   int16x4_t co0,
                                                   int16x4_t co1,
                                                   int16x4_t c_o32_0,
                                                   int16x4_t c_o32_1,
                                                   int16x4_t c_o32_2,
                                                   int16x4_t c_o32_3)
{
    idct32_4col_stage2_recon_zero28(tmp, pred, dst, pred_strd, dst_strd, cb + 0,
                                   neg_shift, clip_min, clip_max,
                                   co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage2_recon_zero28(tmp, pred, dst, pred_strd, dst_strd, cb + 4,
                                   neg_shift, clip_min, clip_max,
                                   co0, co1, c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

static inline void idct32_8col_stage2_recon_zero24(const WORD16 *tmp,
                                                   const UWORD16 *pred,
                                                   UWORD16 *dst,
                                                   WORD32 pred_strd,
                                                   WORD32 dst_strd,
                                                   WORD32 cb,
                                                   int32x4_t neg_shift,
                                                   int16x8_t clip_min,
                                                   int16x8_t clip_max,
                                                   int16x4_t ce1,
                                                   int16x4_t co0,
                                                   int16x4_t co1,
                                                   int16x4_t c_o32_0,
                                                   int16x4_t c_o32_1,
                                                   int16x4_t c_o32_2,
                                                   int16x4_t c_o32_3)
{
    idct32_4col_stage2_recon_zero24(tmp, pred, dst, pred_strd, dst_strd, cb + 0,
                                   neg_shift, clip_min, clip_max,
                                   ce1, co0, co1,
                                   c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage2_recon_zero24(tmp, pred, dst, pred_strd, dst_strd, cb + 4,
                                   neg_shift, clip_min, clip_max,
                                   ce1, co0, co1,
                                   c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

static inline void idct32_8col_stage2_recon_full(const WORD16 *tmp,
                                                 const UWORD16 *pred,
                                                 UWORD16 *dst,
                                                 WORD32 pred_strd,
                                                 WORD32 dst_strd,
                                                 WORD32 cb,
                                                 int32x4_t neg_shift,
                                                 int16x8_t clip_min,
                                                 int16x8_t clip_max,
                                                 int16x4_t ce0,
                                                 int16x4_t ce1,
                                                 int16x4_t co0,
                                                 int16x4_t co1,
                                                 int16x4_t c_o32_0,
                                                 int16x4_t c_o32_1,
                                                 int16x4_t c_o32_2,
                                                 int16x4_t c_o32_3)
{
    idct32_4col_stage2_recon_full(tmp, pred, dst, pred_strd, dst_strd, cb + 0,
                                  neg_shift, clip_min, clip_max,
                                  ce0, ce1, co0, co1,
                                  c_o32_0, c_o32_1, c_o32_2, c_o32_3);
    idct32_4col_stage2_recon_full(tmp, pred, dst, pred_strd, dst_strd, cb + 4,
                                  neg_shift, clip_min, clip_max,
                                  ce0, ce1, co0, co1,
                                  c_o32_0, c_o32_1, c_o32_2, c_o32_3);
}

void ihevc_hbd_itrans_recon_32x32_neonintr(WORD16 *pi2_src,
                     WORD16 *pi2_tmp,
                     UWORD16 *pu2_pred,
                     UWORD16 *pu2_dst,
                     WORD32 src_strd,
                     WORD32 pred_strd,
                     WORD32 dst_strd,
                     WORD32 zero_cols,
                     WORD32 zero_rows,
                     UWORD8 u1_bit_depth)
{
    /* Fast path: If either all rows or all columns are zero, copy pred to dst */
    if(((UWORD32)zero_cols == 0xFFFFFFFFU) || ((UWORD32)zero_rows == 0xFFFFFFFFU))
    {
        for(WORD32 i = 0; i < 32; i++)
        {
            uint16x8_t p0 = vld1q_u16(pu2_pred + 0);
            uint16x8_t p1 = vld1q_u16(pu2_pred + 8);
            uint16x8_t p2 = vld1q_u16(pu2_pred + 16);
            uint16x8_t p3 = vld1q_u16(pu2_pred + 24);
            vst1q_u16(pu2_dst + 0, p0);
            vst1q_u16(pu2_dst + 8, p1);
            vst1q_u16(pu2_dst + 16, p2);
            vst1q_u16(pu2_dst + 24, p3);
            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
        return;
    }

    int16x4_t ce0      = vld1_s16(&g_ai2_ihevc_trans_16[4][0]);
    int16x4_t ce1      = vld1_s16(&g_ai2_ihevc_trans_16[2][0]);
    int16x4_t co0      = vld1_s16(&g_ai2_ihevc_trans_16[1][0]);
    int16x4_t co1      = vld1_s16(&g_ai2_ihevc_trans_16[1][4]);
    int16x4_t c_o32_0  = vld1_s16(&g_ai2_ihevc_trans_32[1][0]);
    int16x4_t c_o32_1  = vld1_s16(&g_ai2_ihevc_trans_32[1][4]);
    int16x4_t c_o32_2  = vld1_s16(&g_ai2_ihevc_trans_32[1][8]);
    int16x4_t c_o32_3  = vld1_s16(&g_ai2_ihevc_trans_32[1][12]);

    WORD32 zero_last28_rows_stg1 = ((zero_rows & 0xFFFFFFF0) == 0xFFFFFFF0);
    WORD32 zero_last24_rows_stg1 = ((zero_rows & 0xFFFFFF00) == 0xFFFFFF00);

    WORD32 zero_last28_rows_stg2 = ((zero_cols & 0xFFFFFFF0) == 0xFFFFFFF0);
    WORD32 zero_last24_rows_stg2 = ((zero_cols & 0xFFFFFF00) == 0xFFFFFF00);

    /* Stage 1: Vector across columns, store transposing into pi2_tmp */
    if(zero_last28_rows_stg2)
    {
        if(zero_last28_rows_stg1)
        {
            idct32_4col_stage1_zero28(pi2_src, src_strd, pi2_tmp, 0,
                                      co0, co1,
                                      c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
        else if(zero_last24_rows_stg1)
        {
            idct32_4col_stage1_zero24(pi2_src, src_strd, pi2_tmp, 0,
                                      ce1, co0, co1,
                                      c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
        else
        {
            idct32_4col_stage1_full(pi2_src, src_strd, pi2_tmp, 0,
                                    ce0, ce1, co0, co1,
                                    c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
    }
    else
    {
        WORD32 zero_last16_rows_stg2 = ((zero_cols & 0xFFFF0000) == 0xFFFF0000);
        WORD32 num_blocks_8col = zero_last24_rows_stg2 ? 1 : (zero_last16_rows_stg2 ? 2 : 4);
        if(zero_last16_rows_stg2 && !zero_last24_rows_stg2)
        {
            zero_tmp_16x32(pi2_tmp);
        }
        if(zero_last28_rows_stg1)
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct32_8col_stage1_zero28(pi2_src, src_strd, pi2_tmp, i << 3,
                                          co0, co1,
                                          c_o32_0, c_o32_1, c_o32_2, c_o32_3);
            }
        }
        else if(zero_last24_rows_stg1)
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct32_8col_stage1_zero24(pi2_src, src_strd, pi2_tmp, i << 3,
                                          ce1, co0, co1,
                                          c_o32_0, c_o32_1, c_o32_2, c_o32_3);
            }
        }
        else
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct32_8col_stage1_full(pi2_src, src_strd, pi2_tmp, i << 3,
                                        ce0, ce1, co0, co1,
                                        c_o32_0, c_o32_1, c_o32_2, c_o32_3);
            }
        }
    }

    /* Stage 2 & In-line Reconstruction (zero intermediate transposition overhead) */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - u1_bit_depth));
    int16x8_t clip_min = vdupq_n_s16(0);
    int16x8_t clip_max = vdupq_n_s16((1 << u1_bit_depth) - 1);

    if(zero_last28_rows_stg2)
    {
        for(WORD32 i = 0; i < 4; i++)
        {
            idct32_8col_stage2_recon_zero28(pi2_tmp, pu2_pred, pu2_dst,
                                            pred_strd, dst_strd, i << 3,
                                            neg_shift, clip_min, clip_max,
                                            co0, co1,
                                            c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
    }
    else if(zero_last24_rows_stg2)
    {
        for(WORD32 i = 0; i < 4; i++)
        {
            idct32_8col_stage2_recon_zero24(pi2_tmp, pu2_pred, pu2_dst,
                                            pred_strd, dst_strd, i << 3,
                                            neg_shift, clip_min, clip_max,
                                            ce1, co0, co1,
                                            c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
    }
    else
    {
        for(WORD32 i = 0; i < 4; i++)
        {
            idct32_8col_stage2_recon_full(pi2_tmp, pu2_pred, pu2_dst,
                                          pred_strd, dst_strd, i << 3,
                                          neg_shift, clip_min, clip_max,
                                          ce0, ce1, co0, co1,
                                          c_o32_0, c_o32_1, c_o32_2, c_o32_3);
        }
    }
}
