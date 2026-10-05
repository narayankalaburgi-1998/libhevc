/******************************************************************************
*
* Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore
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
*  ihevc_hbd_chroma_itrans_recon_16x16_neon_intr.c
*
* @brief
*  Contains function definitions for HBD Chroma inverse transform and reconstruction 16x16
*  using ARM NEON intrinsics
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_chroma_itrans_recon_16x16_neonintr()
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
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_trans_macros.h"
#include <arm_neon.h>

/* Universal NEON Transpose 8x4 for Chroma 16x16 */
static inline void chroma_transpose8x4_16x16(const int16x4_t s[8],
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

    *r0 = vcombine_s16(vreinterpret_s16_s32(u0.val[0]),
                       vreinterpret_s16_s32(u2.val[0]));
    *r1 = vcombine_s16(vreinterpret_s16_s32(u1.val[0]),
                       vreinterpret_s16_s32(u3.val[0]));
    *r2 = vcombine_s16(vreinterpret_s16_s32(u0.val[1]),
                       vreinterpret_s16_s32(u2.val[1]));
    *r3 = vcombine_s16(vreinterpret_s16_s32(u1.val[1]),
                       vreinterpret_s16_s32(u3.val[1]));
}

static inline void chroma_store_transposed_8x4_16x16(WORD16 *tmp,
                                                     WORD32 cb,
                                                     WORD32 col_offset,
                                                     const int16x4_t s[8])
{
    int16x8_t r0, r1, r2, r3;
    chroma_transpose8x4_16x16(s, &r0, &r1, &r2, &r3);
    vst1q_s16(tmp + (cb + 0) * 16 + col_offset, r0);
    vst1q_s16(tmp + (cb + 1) * 16 + col_offset, r1);
    vst1q_s16(tmp + (cb + 2) * 16 + col_offset, r2);
    vst1q_s16(tmp + (cb + 3) * 16 + col_offset, r3);
}

/* Chroma interleaved reconstruction store */
static inline void chroma_recon_store_8_16x16(int16x8_t recon_row,
                                              const UWORD16 *pu2_pred,
                                              UWORD16 *pu2_dst,
                                              int16x8_t clip_min,
                                              int16x8_t clip_max)
{
    uint16x8x2_t p = vld2q_u16(pu2_pred);
    int16x8_t a = vqaddq_s16(recon_row, vreinterpretq_s16_u16(p.val[0]));
    int16x8_t c = vminq_s16(vmaxq_s16(a, clip_min), clip_max);
    uint16x8x2_t d = vld2q_u16(pu2_dst);
    d.val[0] = vreinterpretq_u16_s16(c);
    vst2q_u16(pu2_dst, d);
}

static inline void chroma_recon_transposed_8x4_16x16(UWORD16 *dst,
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
    chroma_transpose8x4_16x16(s, &r0, &r1, &r2, &r3);
    chroma_recon_store_8_16x16(r0, pred + (cb + 0) * pred_strd + col_offset,
                               dst + (cb + 0) * dst_strd + col_offset,
                               clip_min, clip_max);
    chroma_recon_store_8_16x16(r1, pred + (cb + 1) * pred_strd + col_offset,
                               dst + (cb + 1) * dst_strd + col_offset,
                               clip_min, clip_max);
    chroma_recon_store_8_16x16(r2, pred + (cb + 2) * pred_strd + col_offset,
                               dst + (cb + 2) * dst_strd + col_offset,
                               clip_min, clip_max);
    chroma_recon_store_8_16x16(r3, pred + (cb + 3) * pred_strd + col_offset,
                               dst + (cb + 3) * dst_strd + col_offset,
                               clip_min, clip_max);
}

static inline void idct16_4col_stage1_zero12(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t ce1,
                                             int16x4_t co0,
                                             int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0 = vld1_s16(src + 0 * src_strd + cb);
        int16x4_t r2 = vld1_s16(src + 2 * src_strd + cb);

        int32x4_t ee = vshll_n_s16(r0, 6);

        int32x4_t eo0 = vmull_lane_s16(r2, ce1, 0);
        int32x4_t eo1 = vmull_lane_s16(r2, ce1, 1);
        int32x4_t eo2 = vmull_lane_s16(r2, ce1, 2);
        int32x4_t eo3 = vmull_lane_s16(r2, ce1, 3);

        e0 = vaddq_s32(ee, eo0);
        e1 = vaddq_s32(ee, eo1);
        e2 = vaddq_s32(ee, eo2);
        e3 = vaddq_s32(ee, eo3);
        e4 = vsubq_s32(ee, eo3);
        e5 = vsubq_s32(ee, eo2);
        e6 = vsubq_s32(ee, eo1);
        e7 = vsubq_s32(ee, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1 = vld1_s16(src + 1 * src_strd + cb);
        int16x4_t r3 = vld1_s16(src + 3 * src_strd + cb);

        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 0), r3, co0, 1);
        s0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        s1[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);

        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 1), r3, co1, 0);
        s0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        s1[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);

        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 2), r3, co1, 3);
        s0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        s1[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);

        int32x4_t o3 = vmlsl_lane_s16(vmull_lane_s16(r1, co0, 3), r3, co1, 1);
        s0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        s1[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);

        int32x4_t o4 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 0), r3, co0, 2);
        s0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        s1[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);

        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 1), r3, co0, 0);
        s0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        s1[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);

        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 2), r3, co0, 3);
        s0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        s1[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);

        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 3), r3, co1, 2);
        s0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        s1[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
    }

    chroma_store_transposed_8x4_16x16(tmp, cb, 0, s0);
    chroma_store_transposed_8x4_16x16(tmp, cb, 8, s1);
}

static inline void idct16_4col_stage1_zero8(const WORD16 *src,
                                            WORD32 src_strd,
                                            WORD16 *tmp,
                                            WORD32 cb,
                                            int16x4_t ce0,
                                            int16x4_t ce1,
                                            int16x4_t co0,
                                            int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0 = vld1_s16(src + 0 * src_strd + cb);
        int16x4_t r2 = vld1_s16(src + 2 * src_strd + cb);
        int16x4_t r4 = vld1_s16(src + 4 * src_strd + cb);
        int16x4_t r6 = vld1_s16(src + 6 * src_strd + cb);

        int32x4_t eee = vshll_n_s16(r0, 6);
        int32x4_t eeo0 = vmull_lane_s16(r4, ce0, 0);
        int32x4_t eeo1 = vmull_lane_s16(r4, ce0, 1);
        int32x4_t ee0 = vaddq_s32(eee, eeo0);
        int32x4_t ee1 = vaddq_s32(eee, eeo1);
        int32x4_t ee2 = vsubq_s32(eee, eeo1);
        int32x4_t ee3 = vsubq_s32(eee, eeo0);

        int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2, ce1, 0), r6, ce1, 1);
        int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 1), r6, ce1, 3);
        int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 2), r6, ce1, 0);
        int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 3), r6, ce1, 2);

        e0 = vaddq_s32(ee0, eo0);
        e1 = vaddq_s32(ee1, eo1);
        e2 = vaddq_s32(ee2, eo2);
        e3 = vaddq_s32(ee3, eo3);
        e4 = vsubq_s32(ee3, eo3);
        e5 = vsubq_s32(ee2, eo2);
        e6 = vsubq_s32(ee1, eo1);
        e7 = vsubq_s32(ee0, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1 = vld1_s16(src + 1 * src_strd + cb);
        int16x4_t r3 = vld1_s16(src + 3 * src_strd + cb);
        int16x4_t r5 = vld1_s16(src + 5 * src_strd + cb);
        int16x4_t r7 = vld1_s16(src + 7 * src_strd + cb);

        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
        o0 = vmlal_lane_s16(o0, r3, co0, 1);
        o0 = vmlal_lane_s16(o0, r5, co0, 2);
        o0 = vmlal_lane_s16(o0, r7, co0, 3);
        s0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        s1[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);

        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
        o1 = vmlal_lane_s16(o1, r3, co1, 0);
        o1 = vmlal_lane_s16(o1, r5, co1, 3);
        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
        s0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        s1[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);

        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
        o2 = vmlal_lane_s16(o2, r3, co1, 3);
        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
        s0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        s1[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);

        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
        o3 = vmlal_lane_s16(o3, r7, co1, 3);
        s0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        s1[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);

        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
        o4 = vmlal_lane_s16(o4, r7, co0, 0);
        s0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        s1[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);

        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
        o5 = vmlal_lane_s16(o5, r5, co1, 0);
        o5 = vmlal_lane_s16(o5, r7, co1, 2);
        s0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        s1[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);

        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
        o6 = vmlal_lane_s16(o6, r5, co0, 0);
        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
        s0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        s1[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);

        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
        o7 = vmlal_lane_s16(o7, r5, co1, 1);
        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
        s0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        s1[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
    }

    chroma_store_transposed_8x4_16x16(tmp, cb, 0, s0);
    chroma_store_transposed_8x4_16x16(tmp, cb, 8, s1);
}

static inline void idct16_4col_stage1_full(const WORD16 *src,
                                           WORD32 src_strd,
                                           WORD16 *tmp,
                                           WORD32 cb,
                                           int16x4_t ce0,
                                           int16x4_t ce1,
                                           int16x4_t co0,
                                           int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0  = vld1_s16(src +  0 * src_strd + cb);
        int16x4_t r2  = vld1_s16(src +  2 * src_strd + cb);
        int16x4_t r4  = vld1_s16(src +  4 * src_strd + cb);
        int16x4_t r6  = vld1_s16(src +  6 * src_strd + cb);
        int16x4_t r8  = vld1_s16(src +  8 * src_strd + cb);
        int16x4_t r10 = vld1_s16(src + 10 * src_strd + cb);
        int16x4_t r12 = vld1_s16(src + 12 * src_strd + cb);
        int16x4_t r14 = vld1_s16(src + 14 * src_strd + cb);

        int32x4_t eee0 = vshlq_n_s32(vaddl_s16(r0, r8), 6);
        int32x4_t eee1 = vshlq_n_s32(vsubl_s16(r0, r8), 6);
        int32x4_t eeo0 = vmlal_lane_s16(vmull_lane_s16(r4, ce0, 0), r12, ce0, 1);
        int32x4_t eeo1 = vmlsl_lane_s16(vmull_lane_s16(r4, ce0, 1), r12, ce0, 0);
        int32x4_t ee0 = vaddq_s32(eee0, eeo0);
        int32x4_t ee1 = vaddq_s32(eee1, eeo1);
        int32x4_t ee2 = vsubq_s32(eee1, eeo1);
        int32x4_t ee3 = vsubq_s32(eee0, eeo0);

        int32x4_t eo0 = vmull_lane_s16(r2, ce1, 0);
        eo0 = vmlal_lane_s16(eo0, r6, ce1, 1);
        eo0 = vmlal_lane_s16(eo0, r10, ce1, 2);
        eo0 = vmlal_lane_s16(eo0, r14, ce1, 3);

        int32x4_t eo1 = vmull_lane_s16(r2, ce1, 1);
        eo1 = vmlsl_lane_s16(eo1, r6, ce1, 3);
        eo1 = vmlsl_lane_s16(eo1, r10, ce1, 0);
        eo1 = vmlsl_lane_s16(eo1, r14, ce1, 2);

        int32x4_t eo2 = vmull_lane_s16(r2, ce1, 2);
        eo2 = vmlsl_lane_s16(eo2, r6, ce1, 0);
        eo2 = vmlal_lane_s16(eo2, r10, ce1, 3);
        eo2 = vmlal_lane_s16(eo2, r14, ce1, 1);

        int32x4_t eo3 = vmull_lane_s16(r2, ce1, 3);
        eo3 = vmlsl_lane_s16(eo3, r6, ce1, 2);
        eo3 = vmlal_lane_s16(eo3, r10, ce1, 1);
        eo3 = vmlsl_lane_s16(eo3, r14, ce1, 0);

        e0 = vaddq_s32(ee0, eo0);
        e1 = vaddq_s32(ee1, eo1);
        e2 = vaddq_s32(ee2, eo2);
        e3 = vaddq_s32(ee3, eo3);
        e4 = vsubq_s32(ee3, eo3);
        e5 = vsubq_s32(ee2, eo2);
        e6 = vsubq_s32(ee1, eo1);
        e7 = vsubq_s32(ee0, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1  = vld1_s16(src +  1 * src_strd + cb);
        int16x4_t r3  = vld1_s16(src +  3 * src_strd + cb);
        int16x4_t r5  = vld1_s16(src +  5 * src_strd + cb);
        int16x4_t r7  = vld1_s16(src +  7 * src_strd + cb);
        int16x4_t r9  = vld1_s16(src +  9 * src_strd + cb);
        int16x4_t r11 = vld1_s16(src + 11 * src_strd + cb);
        int16x4_t r13 = vld1_s16(src + 13 * src_strd + cb);
        int16x4_t r15 = vld1_s16(src + 15 * src_strd + cb);

        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
        o0 = vmlal_lane_s16(o0, r3, co0, 1);
        o0 = vmlal_lane_s16(o0, r5, co0, 2);
        o0 = vmlal_lane_s16(o0, r7, co0, 3);
        o0 = vmlal_lane_s16(o0, r9, co1, 0);
        o0 = vmlal_lane_s16(o0, r11, co1, 1);
        o0 = vmlal_lane_s16(o0, r13, co1, 2);
        o0 = vmlal_lane_s16(o0, r15, co1, 3);
        s0[0] = vqrshrn_n_s32(vaddq_s32(e0, o0), 7);
        s1[7] = vqrshrn_n_s32(vsubq_s32(e0, o0), 7);

        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
        o1 = vmlal_lane_s16(o1, r3, co1, 0);
        o1 = vmlal_lane_s16(o1, r5, co1, 3);
        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
        o1 = vmlsl_lane_s16(o1, r9, co0, 2);
        o1 = vmlsl_lane_s16(o1, r11, co0, 0);
        o1 = vmlsl_lane_s16(o1, r13, co0, 3);
        o1 = vmlsl_lane_s16(o1, r15, co1, 2);
        s0[1] = vqrshrn_n_s32(vaddq_s32(e1, o1), 7);
        s1[6] = vqrshrn_n_s32(vsubq_s32(e1, o1), 7);

        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
        o2 = vmlal_lane_s16(o2, r3, co1, 3);
        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
        o2 = vmlsl_lane_s16(o2, r9, co1, 2);
        o2 = vmlal_lane_s16(o2, r11, co1, 0);
        o2 = vmlal_lane_s16(o2, r13, co0, 0);
        o2 = vmlal_lane_s16(o2, r15, co1, 1);
        s0[2] = vqrshrn_n_s32(vaddq_s32(e2, o2), 7);
        s1[5] = vqrshrn_n_s32(vsubq_s32(e2, o2), 7);

        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
        o3 = vmlal_lane_s16(o3, r7, co1, 3);
        o3 = vmlal_lane_s16(o3, r9, co0, 0);
        o3 = vmlal_lane_s16(o3, r11, co1, 2);
        o3 = vmlsl_lane_s16(o3, r13, co0, 2);
        o3 = vmlsl_lane_s16(o3, r15, co1, 0);
        s0[3] = vqrshrn_n_s32(vaddq_s32(e3, o3), 7);
        s1[4] = vqrshrn_n_s32(vsubq_s32(e3, o3), 7);

        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
        o4 = vmlal_lane_s16(o4, r7, co0, 0);
        o4 = vmlsl_lane_s16(o4, r9, co1, 3);
        o4 = vmlsl_lane_s16(o4, r11, co0, 1);
        o4 = vmlal_lane_s16(o4, r13, co1, 1);
        o4 = vmlal_lane_s16(o4, r15, co0, 3);
        s0[4] = vqrshrn_n_s32(vaddq_s32(e4, o4), 7);
        s1[3] = vqrshrn_n_s32(vsubq_s32(e4, o4), 7);

        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
        o5 = vmlal_lane_s16(o5, r5, co1, 0);
        o5 = vmlal_lane_s16(o5, r7, co1, 2);
        o5 = vmlsl_lane_s16(o5, r9, co0, 1);
        o5 = vmlal_lane_s16(o5, r11, co0, 3);
        o5 = vmlal_lane_s16(o5, r13, co1, 3);
        o5 = vmlsl_lane_s16(o5, r15, co0, 2);
        s0[5] = vqrshrn_n_s32(vaddq_s32(e5, o5), 7);
        s1[2] = vqrshrn_n_s32(vsubq_s32(e5, o5), 7);

        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
        o6 = vmlal_lane_s16(o6, r5, co0, 0);
        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
        o6 = vmlal_lane_s16(o6, r9, co1, 1);
        o6 = vmlal_lane_s16(o6, r11, co1, 3);
        o6 = vmlsl_lane_s16(o6, r13, co1, 0);
        o6 = vmlal_lane_s16(o6, r15, co0, 1);
        s0[6] = vqrshrn_n_s32(vaddq_s32(e6, o6), 7);
        s1[1] = vqrshrn_n_s32(vsubq_s32(e6, o6), 7);

        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
        o7 = vmlal_lane_s16(o7, r5, co1, 1);
        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
        o7 = vmlal_lane_s16(o7, r9, co0, 3);
        o7 = vmlsl_lane_s16(o7, r11, co0, 2);
        o7 = vmlal_lane_s16(o7, r13, co0, 1);
        o7 = vmlsl_lane_s16(o7, r15, co0, 0);
        s0[7] = vqrshrn_n_s32(vaddq_s32(e7, o7), 7);
        s1[0] = vqrshrn_n_s32(vsubq_s32(e7, o7), 7);
    }

    chroma_store_transposed_8x4_16x16(tmp, cb, 0, s0);
    chroma_store_transposed_8x4_16x16(tmp, cb, 8, s1);
}

static inline void idct16_4col_stage2_recon_zero12(const WORD16 *tmp,
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
                                                   int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0 = vld1_s16(tmp + 0 * 16 + cb);
        int16x4_t r2 = vld1_s16(tmp + 2 * 16 + cb);

        int32x4_t ee = vshll_n_s16(r0, 6);

        int32x4_t eo0 = vmull_lane_s16(r2, ce1, 0);
        int32x4_t eo1 = vmull_lane_s16(r2, ce1, 1);
        int32x4_t eo2 = vmull_lane_s16(r2, ce1, 2);
        int32x4_t eo3 = vmull_lane_s16(r2, ce1, 3);

        e0 = vaddq_s32(ee, eo0);
        e1 = vaddq_s32(ee, eo1);
        e2 = vaddq_s32(ee, eo2);
        e3 = vaddq_s32(ee, eo3);
        e4 = vsubq_s32(ee, eo3);
        e5 = vsubq_s32(ee, eo2);
        e6 = vsubq_s32(ee, eo1);
        e7 = vsubq_s32(ee, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1 = vld1_s16(tmp + 1 * 16 + cb);
        int16x4_t r3 = vld1_s16(tmp + 3 * 16 + cb);

        int32x4_t o0 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 0), r3, co0, 1);
        s0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        s1[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

        int32x4_t o1 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 1), r3, co1, 0);
        s0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        s1[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

        int32x4_t o2 = vmlal_lane_s16(vmull_lane_s16(r1, co0, 2), r3, co1, 3);
        s0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        s1[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

        int32x4_t o3 = vmlsl_lane_s16(vmull_lane_s16(r1, co0, 3), r3, co1, 1);
        s0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        s1[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

        int32x4_t o4 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 0), r3, co0, 2);
        s0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        s1[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

        int32x4_t o5 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 1), r3, co0, 0);
        s0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        s1[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

        int32x4_t o6 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 2), r3, co0, 3);
        s0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        s1[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

        int32x4_t o7 = vmlsl_lane_s16(vmull_lane_s16(r1, co1, 3), r3, co1, 2);
        s0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        s1[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
    }

    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 0, s0, clip_min, clip_max);
    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 16, s1, clip_min, clip_max);
}

static inline void idct16_4col_stage2_recon_zero8(const WORD16 *tmp,
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
                                                  int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0 = vld1_s16(tmp + 0 * 16 + cb);
        int16x4_t r2 = vld1_s16(tmp + 2 * 16 + cb);
        int16x4_t r4 = vld1_s16(tmp + 4 * 16 + cb);
        int16x4_t r6 = vld1_s16(tmp + 6 * 16 + cb);

        int32x4_t eee = vshll_n_s16(r0, 6);
        int32x4_t eeo0 = vmull_lane_s16(r4, ce0, 0);
        int32x4_t eeo1 = vmull_lane_s16(r4, ce0, 1);
        int32x4_t ee0 = vaddq_s32(eee, eeo0);
        int32x4_t ee1 = vaddq_s32(eee, eeo1);
        int32x4_t ee2 = vsubq_s32(eee, eeo1);
        int32x4_t ee3 = vsubq_s32(eee, eeo0);

        int32x4_t eo0 = vmlal_lane_s16(vmull_lane_s16(r2, ce1, 0), r6, ce1, 1);
        int32x4_t eo1 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 1), r6, ce1, 3);
        int32x4_t eo2 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 2), r6, ce1, 0);
        int32x4_t eo3 = vmlsl_lane_s16(vmull_lane_s16(r2, ce1, 3), r6, ce1, 2);

        e0 = vaddq_s32(ee0, eo0);
        e1 = vaddq_s32(ee1, eo1);
        e2 = vaddq_s32(ee2, eo2);
        e3 = vaddq_s32(ee3, eo3);
        e4 = vsubq_s32(ee3, eo3);
        e5 = vsubq_s32(ee2, eo2);
        e6 = vsubq_s32(ee1, eo1);
        e7 = vsubq_s32(ee0, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1 = vld1_s16(tmp + 1 * 16 + cb);
        int16x4_t r3 = vld1_s16(tmp + 3 * 16 + cb);
        int16x4_t r5 = vld1_s16(tmp + 5 * 16 + cb);
        int16x4_t r7 = vld1_s16(tmp + 7 * 16 + cb);

        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
        o0 = vmlal_lane_s16(o0, r3, co0, 1);
        o0 = vmlal_lane_s16(o0, r5, co0, 2);
        o0 = vmlal_lane_s16(o0, r7, co0, 3);
        s0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        s1[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
        o1 = vmlal_lane_s16(o1, r3, co1, 0);
        o1 = vmlal_lane_s16(o1, r5, co1, 3);
        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
        s0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        s1[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
        o2 = vmlal_lane_s16(o2, r3, co1, 3);
        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
        s0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        s1[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
        o3 = vmlal_lane_s16(o3, r7, co1, 3);
        s0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        s1[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
        o4 = vmlal_lane_s16(o4, r7, co0, 0);
        s0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        s1[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
        o5 = vmlal_lane_s16(o5, r5, co1, 0);
        o5 = vmlal_lane_s16(o5, r7, co1, 2);
        s0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        s1[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
        o6 = vmlal_lane_s16(o6, r5, co0, 0);
        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
        s0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        s1[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
        o7 = vmlal_lane_s16(o7, r5, co1, 1);
        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
        s0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        s1[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
    }

    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 0, s0, clip_min, clip_max);
    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 16, s1, clip_min, clip_max);
}

static inline void idct16_4col_stage2_recon_full(const WORD16 *tmp,
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
                                                 int16x4_t co1)
{
    int16x4_t s0[8], s1[8];
    int32x4_t e0, e1, e2, e3, e4, e5, e6, e7;

    /* Phase 1: Even terms */
    {
        int16x4_t r0  = vld1_s16(tmp +  0 * 16 + cb);
        int16x4_t r2  = vld1_s16(tmp +  2 * 16 + cb);
        int16x4_t r4  = vld1_s16(tmp +  4 * 16 + cb);
        int16x4_t r6  = vld1_s16(tmp +  6 * 16 + cb);
        int16x4_t r8  = vld1_s16(tmp +  8 * 16 + cb);
        int16x4_t r10 = vld1_s16(tmp + 10 * 16 + cb);
        int16x4_t r12 = vld1_s16(tmp + 12 * 16 + cb);
        int16x4_t r14 = vld1_s16(tmp + 14 * 16 + cb);

        int32x4_t eee0 = vshlq_n_s32(vaddl_s16(r0, r8), 6);
        int32x4_t eee1 = vshlq_n_s32(vsubl_s16(r0, r8), 6);
        int32x4_t eeo0 = vmlal_lane_s16(vmull_lane_s16(r4, ce0, 0), r12, ce0, 1);
        int32x4_t eeo1 = vmlsl_lane_s16(vmull_lane_s16(r4, ce0, 1), r12, ce0, 0);
        int32x4_t ee0 = vaddq_s32(eee0, eeo0);
        int32x4_t ee1 = vaddq_s32(eee1, eeo1);
        int32x4_t ee2 = vsubq_s32(eee1, eeo1);
        int32x4_t ee3 = vsubq_s32(eee0, eeo0);

        int32x4_t eo0 = vmull_lane_s16(r2, ce1, 0);
        eo0 = vmlal_lane_s16(eo0, r6, ce1, 1);
        eo0 = vmlal_lane_s16(eo0, r10, ce1, 2);
        eo0 = vmlal_lane_s16(eo0, r14, ce1, 3);

        int32x4_t eo1 = vmull_lane_s16(r2, ce1, 1);
        eo1 = vmlsl_lane_s16(eo1, r6, ce1, 3);
        eo1 = vmlsl_lane_s16(eo1, r10, ce1, 0);
        eo1 = vmlsl_lane_s16(eo1, r14, ce1, 2);

        int32x4_t eo2 = vmull_lane_s16(r2, ce1, 2);
        eo2 = vmlsl_lane_s16(eo2, r6, ce1, 0);
        eo2 = vmlal_lane_s16(eo2, r10, ce1, 3);
        eo2 = vmlal_lane_s16(eo2, r14, ce1, 1);

        int32x4_t eo3 = vmull_lane_s16(r2, ce1, 3);
        eo3 = vmlsl_lane_s16(eo3, r6, ce1, 2);
        eo3 = vmlal_lane_s16(eo3, r10, ce1, 1);
        eo3 = vmlsl_lane_s16(eo3, r14, ce1, 0);

        e0 = vaddq_s32(ee0, eo0);
        e1 = vaddq_s32(ee1, eo1);
        e2 = vaddq_s32(ee2, eo2);
        e3 = vaddq_s32(ee3, eo3);
        e4 = vsubq_s32(ee3, eo3);
        e5 = vsubq_s32(ee2, eo2);
        e6 = vsubq_s32(ee1, eo1);
        e7 = vsubq_s32(ee0, eo0);
    }

    /* Phase 2: Odd terms & butterfly combination */
    {
        int16x4_t r1  = vld1_s16(tmp +  1 * 16 + cb);
        int16x4_t r3  = vld1_s16(tmp +  3 * 16 + cb);
        int16x4_t r5  = vld1_s16(tmp +  5 * 16 + cb);
        int16x4_t r7  = vld1_s16(tmp +  7 * 16 + cb);
        int16x4_t r9  = vld1_s16(tmp +  9 * 16 + cb);
        int16x4_t r11 = vld1_s16(tmp + 11 * 16 + cb);
        int16x4_t r13 = vld1_s16(tmp + 13 * 16 + cb);
        int16x4_t r15 = vld1_s16(tmp + 15 * 16 + cb);

        int32x4_t o0 = vmull_lane_s16(r1, co0, 0);
        o0 = vmlal_lane_s16(o0, r3, co0, 1);
        o0 = vmlal_lane_s16(o0, r5, co0, 2);
        o0 = vmlal_lane_s16(o0, r7, co0, 3);
        o0 = vmlal_lane_s16(o0, r9, co1, 0);
        o0 = vmlal_lane_s16(o0, r11, co1, 1);
        o0 = vmlal_lane_s16(o0, r13, co1, 2);
        o0 = vmlal_lane_s16(o0, r15, co1, 3);
        s0[0] = vqmovn_s32(vrshlq_s32(vaddq_s32(e0, o0), neg_shift));
        s1[7] = vqmovn_s32(vrshlq_s32(vsubq_s32(e0, o0), neg_shift));

        int32x4_t o1 = vmull_lane_s16(r1, co0, 1);
        o1 = vmlal_lane_s16(o1, r3, co1, 0);
        o1 = vmlal_lane_s16(o1, r5, co1, 3);
        o1 = vmlsl_lane_s16(o1, r7, co1, 1);
        o1 = vmlsl_lane_s16(o1, r9, co0, 2);
        o1 = vmlsl_lane_s16(o1, r11, co0, 0);
        o1 = vmlsl_lane_s16(o1, r13, co0, 3);
        o1 = vmlsl_lane_s16(o1, r15, co1, 2);
        s0[1] = vqmovn_s32(vrshlq_s32(vaddq_s32(e1, o1), neg_shift));
        s1[6] = vqmovn_s32(vrshlq_s32(vsubq_s32(e1, o1), neg_shift));

        int32x4_t o2 = vmull_lane_s16(r1, co0, 2);
        o2 = vmlal_lane_s16(o2, r3, co1, 3);
        o2 = vmlsl_lane_s16(o2, r5, co0, 3);
        o2 = vmlsl_lane_s16(o2, r7, co0, 1);
        o2 = vmlsl_lane_s16(o2, r9, co1, 2);
        o2 = vmlal_lane_s16(o2, r11, co1, 0);
        o2 = vmlal_lane_s16(o2, r13, co0, 0);
        o2 = vmlal_lane_s16(o2, r15, co1, 1);
        s0[2] = vqmovn_s32(vrshlq_s32(vaddq_s32(e2, o2), neg_shift));
        s1[5] = vqmovn_s32(vrshlq_s32(vsubq_s32(e2, o2), neg_shift));

        int32x4_t o3 = vmull_lane_s16(r1, co0, 3);
        o3 = vmlsl_lane_s16(o3, r3, co1, 1);
        o3 = vmlsl_lane_s16(o3, r5, co0, 1);
        o3 = vmlal_lane_s16(o3, r7, co1, 3);
        o3 = vmlal_lane_s16(o3, r9, co0, 0);
        o3 = vmlal_lane_s16(o3, r11, co1, 2);
        o3 = vmlsl_lane_s16(o3, r13, co0, 2);
        o3 = vmlsl_lane_s16(o3, r15, co1, 0);
        s0[3] = vqmovn_s32(vrshlq_s32(vaddq_s32(e3, o3), neg_shift));
        s1[4] = vqmovn_s32(vrshlq_s32(vsubq_s32(e3, o3), neg_shift));

        int32x4_t o4 = vmull_lane_s16(r1, co1, 0);
        o4 = vmlsl_lane_s16(o4, r3, co0, 2);
        o4 = vmlsl_lane_s16(o4, r5, co1, 2);
        o4 = vmlal_lane_s16(o4, r7, co0, 0);
        o4 = vmlsl_lane_s16(o4, r9, co1, 3);
        o4 = vmlsl_lane_s16(o4, r11, co0, 1);
        o4 = vmlal_lane_s16(o4, r13, co1, 1);
        o4 = vmlal_lane_s16(o4, r15, co0, 3);
        s0[4] = vqmovn_s32(vrshlq_s32(vaddq_s32(e4, o4), neg_shift));
        s1[3] = vqmovn_s32(vrshlq_s32(vsubq_s32(e4, o4), neg_shift));

        int32x4_t o5 = vmull_lane_s16(r1, co1, 1);
        o5 = vmlsl_lane_s16(o5, r3, co0, 0);
        o5 = vmlal_lane_s16(o5, r5, co1, 0);
        o5 = vmlal_lane_s16(o5, r7, co1, 2);
        o5 = vmlsl_lane_s16(o5, r9, co0, 1);
        o5 = vmlal_lane_s16(o5, r11, co0, 3);
        o5 = vmlal_lane_s16(o5, r13, co1, 3);
        o5 = vmlsl_lane_s16(o5, r15, co0, 2);
        s0[5] = vqmovn_s32(vrshlq_s32(vaddq_s32(e5, o5), neg_shift));
        s1[2] = vqmovn_s32(vrshlq_s32(vsubq_s32(e5, o5), neg_shift));

        int32x4_t o6 = vmull_lane_s16(r1, co1, 2);
        o6 = vmlsl_lane_s16(o6, r3, co0, 3);
        o6 = vmlal_lane_s16(o6, r5, co0, 0);
        o6 = vmlsl_lane_s16(o6, r7, co0, 2);
        o6 = vmlal_lane_s16(o6, r9, co1, 1);
        o6 = vmlal_lane_s16(o6, r11, co1, 3);
        o6 = vmlsl_lane_s16(o6, r13, co1, 0);
        o6 = vmlal_lane_s16(o6, r15, co0, 1);
        s0[6] = vqmovn_s32(vrshlq_s32(vaddq_s32(e6, o6), neg_shift));
        s1[1] = vqmovn_s32(vrshlq_s32(vsubq_s32(e6, o6), neg_shift));

        int32x4_t o7 = vmull_lane_s16(r1, co1, 3);
        o7 = vmlsl_lane_s16(o7, r3, co1, 2);
        o7 = vmlal_lane_s16(o7, r5, co1, 1);
        o7 = vmlsl_lane_s16(o7, r7, co1, 0);
        o7 = vmlal_lane_s16(o7, r9, co0, 3);
        o7 = vmlsl_lane_s16(o7, r11, co0, 2);
        o7 = vmlal_lane_s16(o7, r13, co0, 1);
        o7 = vmlsl_lane_s16(o7, r15, co0, 0);
        s0[7] = vqmovn_s32(vrshlq_s32(vaddq_s32(e7, o7), neg_shift));
        s1[0] = vqmovn_s32(vrshlq_s32(vsubq_s32(e7, o7), neg_shift));
    }

    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 0, s0, clip_min, clip_max);
    chroma_recon_transposed_8x4_16x16(dst, dst_strd, pred, pred_strd,
                                      cb, 16, s1, clip_min, clip_max);
}

/* 8-column wrappers for Stage 1 */
static inline void idct16_8col_stage1_zero12(const WORD16 *src,
                                             WORD32 src_strd,
                                             WORD16 *tmp,
                                             WORD32 cb,
                                             int16x4_t ce1,
                                             int16x4_t co0,
                                             int16x4_t co1)
{
    idct16_4col_stage1_zero12(src, src_strd, tmp, cb + 0, ce1, co0, co1);
    idct16_4col_stage1_zero12(src, src_strd, tmp, cb + 4, ce1, co0, co1);
}

static inline void idct16_8col_stage1_zero8(const WORD16 *src,
                                            WORD32 src_strd,
                                            WORD16 *tmp,
                                            WORD32 cb,
                                            int16x4_t ce0,
                                            int16x4_t ce1,
                                            int16x4_t co0,
                                            int16x4_t co1)
{
    idct16_4col_stage1_zero8(src, src_strd, tmp, cb + 0, ce0, ce1, co0, co1);
    idct16_4col_stage1_zero8(src, src_strd, tmp, cb + 4, ce0, ce1, co0, co1);
}

static inline void idct16_8col_stage1_full(const WORD16 *src,
                                           WORD32 src_strd,
                                           WORD16 *tmp,
                                           WORD32 cb,
                                           int16x4_t ce0,
                                           int16x4_t ce1,
                                           int16x4_t co0,
                                           int16x4_t co1)
{
    idct16_4col_stage1_full(src, src_strd, tmp, cb + 0, ce0, ce1, co0, co1);
    idct16_4col_stage1_full(src, src_strd, tmp, cb + 4, ce0, ce1, co0, co1);
}

/* 8-column wrappers for Stage 2 & Recon */
static inline void idct16_8col_stage2_recon_zero12(const WORD16 *tmp,
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
                                                   int16x4_t co1)
{
    idct16_4col_stage2_recon_zero12(tmp, pred, dst, pred_strd, dst_strd,
                                   cb + 0, neg_shift, clip_min, clip_max,
                                   ce1, co0, co1);
    idct16_4col_stage2_recon_zero12(tmp, pred, dst, pred_strd, dst_strd,
                                   cb + 4, neg_shift, clip_min, clip_max,
                                   ce1, co0, co1);
}

static inline void idct16_8col_stage2_recon_zero8(const WORD16 *tmp,
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
                                                  int16x4_t co1)
{
    idct16_4col_stage2_recon_zero8(tmp, pred, dst, pred_strd, dst_strd,
                                  cb + 0, neg_shift, clip_min, clip_max,
                                  ce0, ce1, co0, co1);
    idct16_4col_stage2_recon_zero8(tmp, pred, dst, pred_strd, dst_strd,
                                  cb + 4, neg_shift, clip_min, clip_max,
                                  ce0, ce1, co0, co1);
}

static inline void idct16_8col_stage2_recon_full(const WORD16 *tmp,
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
                                                 int16x4_t co1)
{
    idct16_4col_stage2_recon_full(tmp, pred, dst, pred_strd, dst_strd,
                                 cb + 0, neg_shift, clip_min, clip_max,
                                 ce0, ce1, co0, co1);
    idct16_4col_stage2_recon_full(tmp, pred, dst, pred_strd, dst_strd,
                                 cb + 4, neg_shift, clip_min, clip_max,
                                 ce0, ce1, co0, co1);
}

void ihevc_hbd_chroma_itrans_recon_16x16_neonintr(WORD16 *pi2_src,
                                                  WORD16 *pi2_tmp,
                                                  UWORD16 *pu2_pred,
                                                  UWORD16 *pu2_dst,
                                                  WORD32 src_strd,
                                                  WORD32 pred_strd,
                                                  WORD32 dst_strd,
                                                  WORD32 zero_cols,
                                                  WORD32 zero_rows,
                                                  UWORD8 bit_depth)
{
    /* Full zero block fast-path bypass checking both zero_cols and zero_rows */
    if(((UWORD32)(zero_cols & 0xFFFF) == 0xFFFFU) || ((UWORD32)(zero_rows & 0xFFFF) == 0xFFFFU))
    {
        for(WORD32 i = 0; i < 16; i++)
        {
            uint16x8x2_t p0 = vld2q_u16(pu2_pred + 0);
            uint16x8x2_t d0 = vld2q_u16(pu2_dst + 0);
            d0.val[0] = p0.val[0];
            vst2q_u16(pu2_dst + 0, d0);

            uint16x8x2_t p1 = vld2q_u16(pu2_pred + 16);
            uint16x8x2_t d1 = vld2q_u16(pu2_dst + 16);
            d1.val[0] = p1.val[0];
            vst2q_u16(pu2_dst + 16, d1);

            pu2_pred += pred_strd;
            pu2_dst += dst_strd;
        }
        return;
    }

    int16x4_t ce0 = vld1_s16(&g_ai2_ihevc_trans_16[4][0]);
    int16x4_t ce1 = vld1_s16(&g_ai2_ihevc_trans_16[2][0]);
    int16x4_t co0 = vld1_s16(&g_ai2_ihevc_trans_16[1][0]);
    int16x4_t co1 = vld1_s16(&g_ai2_ihevc_trans_16[1][4]);

    WORD32 zero_last12_rows_stg1 = ((zero_rows & 0xFFF0) == 0xFFF0);
    WORD32 zero_last8_rows_stg1  = ((zero_rows & 0xFF00) == 0xFF00);

    WORD32 zero_last12_rows_stg2 = ((zero_cols & 0xFFF0) == 0xFFF0);
    WORD32 zero_last8_rows_stg2  = ((zero_cols & 0xFF00) == 0xFF00);

    /* Stage 1: Vector across columns, store transposing into pi2_tmp */
    if(zero_last12_rows_stg2)
    {
        if(zero_last12_rows_stg1)
        {
            idct16_4col_stage1_zero12(pi2_src, src_strd, pi2_tmp, 0,
                                      ce1, co0, co1);
        }
        else if(zero_last8_rows_stg1)
        {
            idct16_4col_stage1_zero8(pi2_src, src_strd, pi2_tmp, 0,
                                     ce0, ce1, co0, co1);
        }
        else
        {
            idct16_4col_stage1_full(pi2_src, src_strd, pi2_tmp, 0,
                                    ce0, ce1, co0, co1);
        }
    }
    else
    {
        WORD32 num_blocks_8col = zero_last8_rows_stg2 ? 1 : 2;
        if(zero_last12_rows_stg1)
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct16_8col_stage1_zero12(pi2_src, src_strd, pi2_tmp, i << 3,
                                          ce1, co0, co1);
            }
        }
        else if(zero_last8_rows_stg1)
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct16_8col_stage1_zero8(pi2_src, src_strd, pi2_tmp, i << 3,
                                         ce0, ce1, co0, co1);
            }
        }
        else
        {
            for(WORD32 i = 0; i < num_blocks_8col; i++)
            {
                idct16_8col_stage1_full(pi2_src, src_strd, pi2_tmp, i << 3,
                                        ce0, ce1, co0, co1);
            }
        }
    }

    /* Stage 2 & In-line Recon (zero intermediate transposition overhead) */
    int32x4_t neg_shift = vdupq_n_s32(-(20 - bit_depth));
    int16x8_t clip_min = vdupq_n_s16(0);
    int16x8_t clip_max = vdupq_n_s16((1 << bit_depth) - 1);

    if(zero_last12_rows_stg2)
    {
        for(WORD32 i = 0; i < 2; i++)
        {
            idct16_8col_stage2_recon_zero12(pi2_tmp, pu2_pred, pu2_dst,
                                            pred_strd, dst_strd, i << 3,
                                            neg_shift, clip_min, clip_max,
                                            ce1, co0, co1);
        }
    }
    else if(zero_last8_rows_stg2)
    {
        for(WORD32 i = 0; i < 2; i++)
        {
            idct16_8col_stage2_recon_zero8(pi2_tmp, pu2_pred, pu2_dst,
                                           pred_strd, dst_strd, i << 3,
                                           neg_shift, clip_min, clip_max,
                                           ce0, ce1, co0, co1);
        }
    }
    else
    {
        for(WORD32 i = 0; i < 2; i++)
        {
            idct16_8col_stage2_recon_full(pi2_tmp, pu2_pred, pu2_dst,
                                          pred_strd, dst_strd, i << 3,
                                          neg_shift, clip_min, clip_max,
                                          ce0, ce1, co0, co1);
        }
    }
}
