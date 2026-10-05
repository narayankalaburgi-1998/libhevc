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
 *  ihevc_hbd_chroma_itrans_recon_neon_intr.c
 *
 * @brief
 *  Contains function definitions for high bit depth inverse transform and
 *  reconstruction of chroma interleaved data using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_chroma_itrans_recon_4x4_neonintr()
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
#include "ihevc_chroma_itrans_recon.h"
#include "ihevc_trans_macros.h"
#include "arm_neon.h"

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform and reconstruction for 4x4
 *  chroma interleaved input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 4x4 inverse DCT transform, adds the 16-bit interleaved chroma
 *  prediction data (U or V component at stride 2), and clips output to the
 *  configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 interleaved chroma block
 *
 * @param[out] pu2_dst
 *  Output 4x4 interleaved chroma block
 *
 * @param[in] src_strd
 *  Input stride
 *
 * @param[in] pred_strd
 *  Prediction stride
 *
 * @param[in] dst_strd
 *  Output Stride
 *
 * @param[in] zero_cols
 *  Zero columns in pi2_src
 *
 * @param[in] zero_rows
 *  Zero rows in pi2_src
 *
 * @param[in] u1_bit_depth
 *  Bit depth of the pixel data
 *
 * @returns  Void
 *
 * @remarks
 *  None
 *
 *******************************************************************************
 */
void ihevc_hbd_chroma_itrans_recon_4x4_neonintr(WORD16 *pi2_src,
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
    WORD32 shift;
    WORD16 clip_limit;

    UNUSED(pi2_tmp);
    UNUSED(zero_rows);

    /* Inverse Transform 1st stage */
    const WORD32 *g_ai2_ihevc_trans_4_1 = (const WORD32 *)&g_ai2_ihevc_trans_4[1][0];
    const WORD32 *g_ai2_ihevc_trans_4_0 = (const WORD32 *)&g_ai2_ihevc_trans_4[0][0];
    const WORD32 *g_ai2_ihevc_trans_4_3 = (const WORD32 *)&g_ai2_ihevc_trans_4[3][0];
    const WORD32 *g_ai2_ihevc_trans_4_2 = (const WORD32 *)&g_ai2_ihevc_trans_4[2][0];

    int32x2_t g_ai2_ihevc_trans_4_val_1 = vdup_n_s32(0);
    g_ai2_ihevc_trans_4_val_1 = vld1_lane_s32(g_ai2_ihevc_trans_4_0, g_ai2_ihevc_trans_4_val_1, 0);
    g_ai2_ihevc_trans_4_val_1 = vld1_lane_s32(g_ai2_ihevc_trans_4_1, g_ai2_ihevc_trans_4_val_1, 1);

    int32x2_t g_ai2_ihevc_trans_4_val_2 = vdup_n_s32(0);
    g_ai2_ihevc_trans_4_val_2 = vld1_lane_s32(g_ai2_ihevc_trans_4_2, g_ai2_ihevc_trans_4_val_2, 0);
    g_ai2_ihevc_trans_4_val_2 = vld1_lane_s32(g_ai2_ihevc_trans_4_3, g_ai2_ihevc_trans_4_val_2, 1);

    WORD16 *pi2_src_tmp_0_src_strd = pi2_src;
    WORD16 *pi2_src_tmp_1_src_strd = pi2_src_tmp_0_src_strd + src_strd;
    WORD16 *pi2_src_tmp_2_src_strd = pi2_src_tmp_1_src_strd + src_strd;
    WORD16 *pi2_src_tmp_3_src_strd = pi2_src_tmp_2_src_strd + src_strd;

    int16x4_t pi2_src_0_src_strd_val;
    int16x4_t pi2_src_1_src_strd_val;
    int16x4_t pi2_src_2_src_strd_val;
    int16x4_t pi2_src_3_src_strd_val;

    UWORD16 *pu2_pred_tmp_0_pred_strd = pu2_pred;
    uint16x4_t pu2_pred_0_pred_strd_val = vdup_n_u16(0);
    uint16x4_t pu2_pred_1_pred_strd_val = vdup_n_u16(0);
    uint16x4_t pu2_pred_2_pred_strd_val = vdup_n_u16(0);
    uint16x4_t pu2_pred_3_pred_strd_val = vdup_n_u16(0);

    int32x4_t o_0_val;
    int32x4_t o_1_val;
    int32x4_t e_0_val;
    int32x4_t e_1_val;
    int32x4_t e_0_add_o_0;
    int32x4_t e_1_add_o_1;
    int32x4_t e_1_sub_o_1;
    int32x4_t e_0_sub_o_0;
    int16x4_t shift_res_0;
    int16x4_t shift_res_1;
    int16x4_t shift_res_2;
    int16x4_t shift_res_3;
    int32x4_t shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
    int32x4_t shift_val_neg = vnegq_s32(shift_val);
    UWORD16 *pu2_dst_tmp_0_dst_strd = pu2_dst;
    int16x4x2_t transpose_val_16_1;
    int16x4x2_t transpose_val_16_2;
    int32x2x2_t transpose_val_32_1;
    int32x2x2_t transpose_val_32_2;
    int16x4_t const_0 = vdup_n_s16(0);
    int16x4_t dup_const_clip_limit;
    int16x4_t pred_add_val_t;
    int16x4_t clip_res;
    uint16x4_t u2_clip_res;

    /* Checking for Zero Cols */
    if((zero_cols & 0xF) == 0xF)
    {
        transpose_val_32_1.val[0] = vdup_n_s32(0);
        transpose_val_32_1.val[1] = vdup_n_s32(0);
        transpose_val_32_2.val[0] = vdup_n_s32(0);
        transpose_val_32_2.val[1] = vdup_n_s32(0);
    }
    else
    {
        /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
        /* If individual columns are flagged in zero_cols, zero out those lanes in the loaded row vectors */
        int16x4_t col_mask = vdup_n_s16(-1);
        if((zero_cols & 0x1) == 0x1) col_mask = vset_lane_s16(0, col_mask, 0);
        if((zero_cols & 0x2) == 0x2) col_mask = vset_lane_s16(0, col_mask, 1);
        if((zero_cols & 0x4) == 0x4) col_mask = vset_lane_s16(0, col_mask, 2);
        if((zero_cols & 0x8) == 0x8) col_mask = vset_lane_s16(0, col_mask, 3);

        pi2_src_1_src_strd_val = vand_s16(vld1_s16(pi2_src_tmp_1_src_strd), col_mask);
        pi2_src_3_src_strd_val = vand_s16(vld1_s16(pi2_src_tmp_3_src_strd), col_mask);

        o_0_val = vmull_lane_s16(pi2_src_1_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 2);
        pi2_src_0_src_strd_val = vand_s16(vld1_s16(pi2_src_tmp_0_src_strd), col_mask);
        o_0_val = vmlal_lane_s16(o_0_val, pi2_src_3_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 2);

        o_1_val = vmull_lane_s16(pi2_src_1_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 3);
        pi2_src_2_src_strd_val = vand_s16(vld1_s16(pi2_src_tmp_2_src_strd), col_mask);
        o_1_val = vmlal_lane_s16(o_1_val, pi2_src_3_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 3);

        e_0_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 0);
        e_0_val = vmlal_lane_s16(e_0_val, pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 0);

        e_1_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 1);
        e_0_add_o_0 = vaddq_s32(e_0_val, o_0_val);

        e_1_val = vmlal_lane_s16(e_1_val, pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 1);
        e_0_add_o_0 = vrshlq_s32(e_0_add_o_0, shift_val_neg);

        e_1_add_o_1 = vaddq_s32(e_1_val, o_1_val);

        e_1_add_o_1 = vrshlq_s32(e_1_add_o_1, shift_val_neg);
        shift_res_0 = vqmovn_s32(e_0_add_o_0);
        e_1_sub_o_1 = vsubq_s32(e_1_val, o_1_val);

        shift_res_1 = vqmovn_s32(e_1_add_o_1);
        e_1_sub_o_1 = vrshlq_s32(e_1_sub_o_1, shift_val_neg);

        e_0_sub_o_0 = vsubq_s32(e_0_val, o_0_val);
        shift_res_2 = vqmovn_s32(e_1_sub_o_1);

        e_0_sub_o_0 = vrshlq_s32(e_0_sub_o_0, shift_val_neg);
        shift_res_3 = vqmovn_s32(e_0_sub_o_0);

        transpose_val_16_1 = vtrn_s16(shift_res_0, shift_res_1);
        transpose_val_16_2 = vtrn_s16(shift_res_2, shift_res_3);

        transpose_val_32_1 = vtrn_s32(vreinterpret_s32_s16(transpose_val_16_1.val[0]),
                                      vreinterpret_s32_s16(transpose_val_16_2.val[0]));
        transpose_val_32_2 = vtrn_s32(vreinterpret_s32_s16(transpose_val_16_1.val[1]),
                                      vreinterpret_s32_s16(transpose_val_16_2.val[1]));
    }

    /* Inverse Transform 2nd stage */
    shift = 20 - u1_bit_depth;
    clip_limit = (1 << u1_bit_depth) - 1;
    dup_const_clip_limit = vdup_n_s16(clip_limit);

    shift_val = vdupq_n_s32(shift);
    shift_val_neg = vnegq_s32(shift_val);

    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
    o_0_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_2.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 2);

    pu2_pred_0_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[0 * 2], pu2_pred_0_pred_strd_val, 0);
    pu2_pred_0_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[1 * 2], pu2_pred_0_pred_strd_val, 1);
    pu2_pred_0_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[2 * 2], pu2_pred_0_pred_strd_val, 2);
    pu2_pred_0_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[3 * 2], pu2_pred_0_pred_strd_val, 3);
    pu2_pred_tmp_0_pred_strd += pred_strd;

    o_0_val = vmlal_lane_s16(o_0_val, vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 2);

    o_1_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_2.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 3);

    pu2_pred_1_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[0 * 2], pu2_pred_1_pred_strd_val, 0);
    pu2_pred_1_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[1 * 2], pu2_pred_1_pred_strd_val, 1);
    pu2_pred_1_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[2 * 2], pu2_pred_1_pred_strd_val, 2);
    pu2_pred_1_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[3 * 2], pu2_pred_1_pred_strd_val, 3);
    pu2_pred_tmp_0_pred_strd += pred_strd;

    o_1_val = vmlal_lane_s16(o_1_val, vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 3);

    e_0_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_1.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 0);

    pu2_pred_2_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[0 * 2], pu2_pred_2_pred_strd_val, 0);
    pu2_pred_2_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[1 * 2], pu2_pred_2_pred_strd_val, 1);
    pu2_pred_2_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[2 * 2], pu2_pred_2_pred_strd_val, 2);
    pu2_pred_2_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[3 * 2], pu2_pred_2_pred_strd_val, 3);
    pu2_pred_tmp_0_pred_strd += pred_strd;

    e_0_val = vmlal_lane_s16(e_0_val, vreinterpret_s16_s32(transpose_val_32_1.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 0);

    e_1_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_1.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 1);

    pu2_pred_3_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[0 * 2], pu2_pred_3_pred_strd_val, 0);
    pu2_pred_3_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[1 * 2], pu2_pred_3_pred_strd_val, 1);
    pu2_pred_3_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[2 * 2], pu2_pred_3_pred_strd_val, 2);
    pu2_pred_3_pred_strd_val = vld1_lane_u16(&pu2_pred_tmp_0_pred_strd[3 * 2], pu2_pred_3_pred_strd_val, 3);

    e_1_val = vmlal_lane_s16(e_1_val, vreinterpret_s16_s32(transpose_val_32_1.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 1);

    e_0_add_o_0 = vaddq_s32(e_0_val, o_0_val);
    e_0_add_o_0 = vrshlq_s32(e_0_add_o_0, shift_val_neg);

    e_1_add_o_1 = vaddq_s32(e_1_val, o_1_val);
    shift_res_0 = vqmovn_s32(e_0_add_o_0);
    e_1_add_o_1 = vrshlq_s32(e_1_add_o_1, shift_val_neg);

    e_1_sub_o_1 = vsubq_s32(e_1_val, o_1_val);
    shift_res_1 = vqmovn_s32(e_1_add_o_1);
    e_1_sub_o_1 = vrshlq_s32(e_1_sub_o_1, shift_val_neg);

    e_0_sub_o_0 = vsubq_s32(e_0_val, o_0_val);
    shift_res_2 = vqmovn_s32(e_1_sub_o_1);
    e_0_sub_o_0 = vrshlq_s32(e_0_sub_o_0, shift_val_neg);

    shift_res_3 = vqmovn_s32(e_0_sub_o_0);

    transpose_val_16_1 = vtrn_s16(shift_res_0, shift_res_1);
    transpose_val_16_2 = vtrn_s16(shift_res_2, shift_res_3);

    transpose_val_32_1 = vtrn_s32(vreinterpret_s32_s16(transpose_val_16_1.val[0]),
                                  vreinterpret_s32_s16(transpose_val_16_2.val[0]));
    transpose_val_32_2 = vtrn_s32(vreinterpret_s32_s16(transpose_val_16_1.val[1]),
                                  vreinterpret_s32_s16(transpose_val_16_2.val[1]));

    /* Row 0 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_1.val[0]),
                               vreinterpret_s16_u16(pu2_pred_0_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    u2_clip_res = vreinterpret_u16_s16(clip_res);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[0 * 2], u2_clip_res, 0);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[1 * 2], u2_clip_res, 1);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[2 * 2], u2_clip_res, 2);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[3 * 2], u2_clip_res, 3);
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 1 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_2.val[0]),
                               vreinterpret_s16_u16(pu2_pred_1_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    u2_clip_res = vreinterpret_u16_s16(clip_res);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[0 * 2], u2_clip_res, 0);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[1 * 2], u2_clip_res, 1);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[2 * 2], u2_clip_res, 2);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[3 * 2], u2_clip_res, 3);
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 2 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_1.val[1]),
                               vreinterpret_s16_u16(pu2_pred_2_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    u2_clip_res = vreinterpret_u16_s16(clip_res);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[0 * 2], u2_clip_res, 0);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[1 * 2], u2_clip_res, 1);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[2 * 2], u2_clip_res, 2);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[3 * 2], u2_clip_res, 3);
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 3 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                               vreinterpret_s16_u16(pu2_pred_3_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    u2_clip_res = vreinterpret_u16_s16(clip_res);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[0 * 2], u2_clip_res, 0);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[1 * 2], u2_clip_res, 1);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[2 * 2], u2_clip_res, 2);
    vst1_lane_u16(&pu2_dst_tmp_0_dst_strd[3 * 2], u2_clip_res, 3);
}
