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
 ******************************************************************************
 * @file
 *  ihevc_hbd_itrans_recon_neon_intr.c
 *
 * @brief
 *  Contains function definitions for HBD inverse transform and reconstruction
 *  using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_itrans_recon_4x4_ttype1_neonintr()
 *  - ihevc_hbd_itrans_recon_4x4_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
#include <stdio.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_defs.h"
#include "ihevc_trans_tables.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_function_selector.h"
#include "ihevc_trans_macros.h"
#include "arm_neon.h"

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform type 1 (DST) and reconstruction
 *  for 4x4 input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs inverse transform and adds the prediction data and clips output
 *  to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 block
 *
 * @param[out] pu2_dst
 *  Output 4x4 block
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
void ihevc_hbd_itrans_recon_4x4_ttype1_neonintr(WORD16 *pi2_src,
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
    WORD32 i;
    WORD32 trans_size;
    WORD32 shift;
    WORD16 clip_limit;

    UNUSED(zero_rows);
    trans_size = TRANS_SIZE_4;
    WORD16 *pi2_src_tmp1;
    WORD16 *pi2_src_tmp2;
    WORD16 *pi2_src_tmp3;
    WORD16 *pi2_src_tmp4;

    WORD16 *pi2_tmp_0 = pi2_tmp;
    UWORD16 *pu2_dst_tmp1 = pu2_dst;

    int16x4_t pi2_src_tmp1_t;
    int16x4_t pi2_src_tmp2_t;
    int16x4_t pi2_src_tmp3_t;
    int16x4_t pi2_src_tmp4_t;
    int32x4_t c0;
    int32x4_t c1;
    int32x4_t c2;
    int32x4_t c3;
    int32x4_t dup_const_29;
    int32x4_t dup_const_55;
    int32x4_t dup_const_shift;
    int32x4_t shift_res1;
    int32x4_t shift_res2;
    int32x4_t shift_res3;
    int32x4_t shift_res4;
    int16x4_t sto_res1;
    int16x4_t sto_res2;
    int16x4_t sto_res3;
    int16x4_t sto_res4;
    int32x4_t neg_var_i4_shift;
    int32x4_t sub_res1;
    int32x4_t add_res2;
    int32x4_t sub_res2;
    int16x4x2_t trans_load1;
    int16x4x2_t trans_load2;
    int32x2x2_t trans_load3;
    int32x2x2_t trans_load4;
    int16x4_t const_0;
    int16_t const_74 = 74;
    int32x4_t delta_tmp_val1;
    int32x4_t delta_tmp_val2;
    int32x4_t delta_tmp_val3;
    int32x4_t delta_tmp_val4;
    int32x4_t delta_tmp_val5;
    int32x4_t delta_tmp_val6;
    int32x4_t delta_tmp_val7;

    shift = IT_SHIFT_STAGE_1;

    dup_const_shift = vdupq_n_s32(shift);
    neg_var_i4_shift = vnegq_s32(dup_const_shift);

    dup_const_29 = vdupq_n_s32(29);
    dup_const_55 = vdupq_n_s32(55);

    for(i = trans_size; i > 0; i -= 4)
    {
        /* Checking for Zero Cols */
        if((zero_cols & 0xF) == 0xF)
        {
            const_0 = vdup_n_s16(0);
            vst1_s16(pi2_tmp_0, const_0);
            pi2_tmp_0 += trans_size;
            vst1_s16(pi2_tmp_0, const_0);
            pi2_tmp_0 += trans_size;
            vst1_s16(pi2_tmp_0, const_0);
            pi2_tmp_0 += trans_size;
            vst1_s16(pi2_tmp_0, const_0);
            pi2_tmp_0 += trans_size;
        }
        else
        {
            pi2_src_tmp1 = pi2_src;
            pi2_src_tmp4 = pi2_src + src_strd;
            pi2_src_tmp2 = pi2_src_tmp4 + src_strd;
            pi2_src_tmp3 = pi2_src_tmp2 + src_strd;

            pi2_src_tmp4_t = vld1_s16(pi2_src_tmp4);
            pi2_src_tmp2_t = vld1_s16(pi2_src_tmp2);

            c3 = vmull_n_s16(pi2_src_tmp4_t, const_74);

            pi2_src_tmp1_t = vld1_s16(pi2_src_tmp1);
            c0 = vaddl_s16(pi2_src_tmp1_t, pi2_src_tmp2_t);

            pi2_src_tmp3_t = vld1_s16(pi2_src_tmp3);

            add_res2 = vaddl_s16(pi2_src_tmp1_t, pi2_src_tmp3_t);
            sub_res2 = vsubw_s16(add_res2, pi2_src_tmp2_t);
            delta_tmp_val5 = vmulq_n_s32(sub_res2, (int32_t)const_74);
            shift_res3 = vrshlq_s32(delta_tmp_val5, neg_var_i4_shift);
            sto_res3 = vqmovn_s32(shift_res3);

            c2 = vsubl_s16(pi2_src_tmp1_t, pi2_src_tmp3_t);

            delta_tmp_val6 = vmulq_s32(dup_const_55, c0);
            delta_tmp_val7 = vmlaq_s32(delta_tmp_val6, dup_const_29, c2);
            sub_res1 = vsubq_s32(delta_tmp_val7, c3);
            shift_res4 = vrshlq_s32(sub_res1, neg_var_i4_shift);
            sto_res4 = vqmovn_s32(shift_res4);

            c1 = vaddl_s16(pi2_src_tmp2_t, pi2_src_tmp3_t);

            delta_tmp_val1 = vmlaq_s32(c3, dup_const_29, c0);
            trans_load2 = vtrn_s16(sto_res3, sto_res4);
            delta_tmp_val2 = vmlaq_s32(delta_tmp_val1, dup_const_55, c1);
            shift_res1 = vrshlq_s32(delta_tmp_val2, neg_var_i4_shift);
            sto_res1 = vqmovn_s32(shift_res1);

            delta_tmp_val3 = vmlaq_s32(c3, dup_const_55, c2);
            delta_tmp_val4 = vmlsq_s32(delta_tmp_val3, dup_const_29, c1);
            shift_res2 = vrshlq_s32(delta_tmp_val4, neg_var_i4_shift);
            sto_res2 = vqmovn_s32(shift_res2);
            trans_load1 = vtrn_s16(sto_res1, sto_res2);

            trans_load3 = vtrn_s32(vreinterpret_s32_s16(trans_load1.val[0]),
                                   vreinterpret_s32_s16(trans_load2.val[0]));
            trans_load4 = vtrn_s32(vreinterpret_s32_s16(trans_load1.val[1]),
                                   vreinterpret_s32_s16(trans_load2.val[1]));

            vst1_s16(pi2_tmp_0, vreinterpret_s16_s32(trans_load3.val[0]));
            pi2_tmp_0 += trans_size;

            vst1_s16(pi2_tmp_0, vreinterpret_s16_s32(trans_load4.val[0]));
            pi2_tmp_0 += trans_size;

            vst1_s16(pi2_tmp_0, vreinterpret_s16_s32(trans_load3.val[1]));
            pi2_tmp_0 += trans_size;

            vst1_s16(pi2_tmp_0, vreinterpret_s16_s32(trans_load4.val[1]));
            pi2_tmp_0 += trans_size;
        }
    }

    int16x4_t pi2_tmp1_t, pi2_tmp2_t, pi2_tmp0_t, pi2_tmp3_t;
    int16x4_t s_itrans_out_1, s_itrans_out_2, s_itrans_out_3, s_itrans_out_4;

    uint16x4_t vec_1, vec_2, vec_3, vec_4;

    WORD16 *pi2_tmp_zero = pi2_tmp;
    WORD16 *pi2_tmp_one = pi2_tmp + trans_size;
    WORD16 *pi2_tmp_two = pi2_tmp_one + trans_size;
    WORD16 *pi2_tmp_three = pi2_tmp_two + trans_size;

    UWORD16 *pu2_pred_tmp1 = pu2_pred;
    int16x4_t clip_res1, clip_res2, clip_res3, clip_res4;
    int16x4_t pred_add_val_t;
    int16x4_t dup_const_clip_limit;

    shift = 20 - u1_bit_depth;
    clip_limit = (1 << u1_bit_depth) - 1;

    dup_const_shift = vdupq_n_s32(shift);
    neg_var_i4_shift = vnegq_s32(dup_const_shift);
    dup_const_clip_limit = vdup_n_s16(clip_limit);
    const_0 = vdup_n_s16(0);

    for(i = trans_size; i > 0; i -= 4)
    {
        pi2_tmp0_t = vld1_s16(pi2_tmp_zero);
        pi2_tmp3_t = vld1_s16(pi2_tmp_three);

        add_res2 = vaddl_s16(pi2_tmp0_t, pi2_tmp3_t);
        pi2_tmp2_t = vld1_s16(pi2_tmp_two);
        sub_res2 = vsubw_s16(add_res2, pi2_tmp2_t);
        delta_tmp_val5 = vmulq_n_s32(sub_res2, (int32_t)const_74);
        shift_res3 = vrshlq_s32(delta_tmp_val5, neg_var_i4_shift);
        s_itrans_out_3 = vqmovn_s32(shift_res3);

        c0 = vaddl_s16(pi2_tmp0_t, pi2_tmp2_t);
        delta_tmp_val6 = vmulq_s32(dup_const_55, c0);
        c2 = vsubl_s16(pi2_tmp0_t, pi2_tmp3_t);
        delta_tmp_val7 = vmlaq_s32(delta_tmp_val6, dup_const_29, c2);
        pi2_tmp1_t = vld1_s16(pi2_tmp_one);
        c3 = vmull_n_s16(pi2_tmp1_t, const_74);
        sub_res1 = vsubq_s32(delta_tmp_val7, c3);
        shift_res4 = vrshlq_s32(sub_res1, neg_var_i4_shift);
        s_itrans_out_4 = vqmovn_s32(shift_res4);

        c1 = vaddl_s16(pi2_tmp2_t, pi2_tmp3_t);
        delta_tmp_val1 = vmlaq_s32(c3, dup_const_29, c0);
        delta_tmp_val2 = vmlaq_s32(delta_tmp_val1, dup_const_55, c1);
        shift_res1 = vrshlq_s32(delta_tmp_val2, neg_var_i4_shift);
        s_itrans_out_1 = vqmovn_s32(shift_res1);

        trans_load2 = vtrn_s16(s_itrans_out_3, s_itrans_out_4);
        delta_tmp_val3 = vmlaq_s32(c3, dup_const_55, c2);
        delta_tmp_val4 = vmlsq_s32(delta_tmp_val3, dup_const_29, c1);
        shift_res2 = vrshlq_s32(delta_tmp_val4, neg_var_i4_shift);
        s_itrans_out_2 = vqmovn_s32(shift_res2);

        trans_load1 = vtrn_s16(s_itrans_out_1, s_itrans_out_2);

        trans_load3 = vtrn_s32(vreinterpret_s32_s16(trans_load1.val[0]),
                               vreinterpret_s32_s16(trans_load2.val[0]));
        trans_load4 = vtrn_s32(vreinterpret_s32_s16(trans_load1.val[1]),
                               vreinterpret_s32_s16(trans_load2.val[1]));

        vec_1 = vld1_u16(pu2_pred_tmp1);
        pu2_pred_tmp1 += pred_strd;
        vec_2 = vld1_u16(pu2_pred_tmp1);
        pu2_pred_tmp1 += pred_strd;
        vec_3 = vld1_u16(pu2_pred_tmp1);
        pu2_pred_tmp1 += pred_strd;
        vec_4 = vld1_u16(pu2_pred_tmp1);
        pu2_pred_tmp1 += pred_strd;

        pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(trans_load3.val[0]),
                                   vreinterpret_s16_u16(vec_1));
        pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
        clip_res1 = vmin_s16(pred_add_val_t, dup_const_clip_limit);
        vst1_u16(pu2_dst_tmp1, vreinterpret_u16_s16(clip_res1));
        pu2_dst_tmp1 += dst_strd;

        pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(trans_load4.val[0]),
                                   vreinterpret_s16_u16(vec_2));
        pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
        clip_res2 = vmin_s16(pred_add_val_t, dup_const_clip_limit);
        vst1_u16(pu2_dst_tmp1, vreinterpret_u16_s16(clip_res2));
        pu2_dst_tmp1 += dst_strd;

        pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(trans_load3.val[1]),
                                   vreinterpret_s16_u16(vec_3));
        pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
        clip_res3 = vmin_s16(pred_add_val_t, dup_const_clip_limit);
        vst1_u16(pu2_dst_tmp1, vreinterpret_u16_s16(clip_res3));
        pu2_dst_tmp1 += dst_strd;

        pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(trans_load4.val[1]),
                                   vreinterpret_s16_u16(vec_4));
        pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
        clip_res4 = vmin_s16(pred_add_val_t, dup_const_clip_limit);
        vst1_u16(pu2_dst_tmp1, vreinterpret_u16_s16(clip_res4));
        pu2_dst_tmp1 += dst_strd;
    }
}

/**
 *******************************************************************************
 *
 * @brief
 *  This function performs HBD Inverse transform and reconstruction for 4x4
 *  input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 4x4 inverse DCT transform, adds the 16-bit prediction data, and
 *  clips output to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 4x4 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 4x4 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 4x4 block
 *
 * @param[out] pu2_dst
 *  Output 4x4 block
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
void ihevc_hbd_itrans_recon_4x4_neonintr(WORD16 *pi2_src,
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
    uint16x4_t pu2_pred_0_pred_strd_val;
    uint16x4_t pu2_pred_1_pred_strd_val;
    uint16x4_t pu2_pred_2_pred_strd_val;
    uint16x4_t pu2_pred_3_pred_strd_val;

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
        pi2_src_1_src_strd_val = vld1_s16(pi2_src_tmp_1_src_strd);
        pi2_src_3_src_strd_val = vld1_s16(pi2_src_tmp_3_src_strd);

        o_0_val = vmull_lane_s16(pi2_src_1_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 2);
        pi2_src_0_src_strd_val = vld1_s16(pi2_src_tmp_0_src_strd);
        o_0_val = vmlal_lane_s16(o_0_val, pi2_src_3_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 2);

        o_1_val = vmull_lane_s16(pi2_src_1_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 3);
        pi2_src_2_src_strd_val = vld1_s16(pi2_src_tmp_2_src_strd);
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

    pu2_pred_0_pred_strd_val = vld1_u16(pu2_pred_tmp_0_pred_strd);
    pu2_pred_tmp_0_pred_strd += pred_strd;
    o_0_val = vmlal_lane_s16(o_0_val, vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 2);

    o_1_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_2.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 3);

    pu2_pred_1_pred_strd_val = vld1_u16(pu2_pred_tmp_0_pred_strd);
    pu2_pred_tmp_0_pred_strd += pred_strd;
    o_1_val = vmlal_lane_s16(o_1_val, vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 3);

    e_0_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_1.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 0);

    pu2_pred_2_pred_strd_val = vld1_u16(pu2_pred_tmp_0_pred_strd);
    pu2_pred_tmp_0_pred_strd += pred_strd;
    e_0_val = vmlal_lane_s16(e_0_val, vreinterpret_s16_s32(transpose_val_32_1.val[1]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_2), 0);

    e_1_val = vmull_lane_s16(vreinterpret_s16_s32(transpose_val_32_1.val[0]),
                             vreinterpret_s16_s32(g_ai2_ihevc_trans_4_val_1), 1);

    pu2_pred_3_pred_strd_val = vld1_u16(pu2_pred_tmp_0_pred_strd);
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
    vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(clip_res));
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 1 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_2.val[0]),
                               vreinterpret_s16_u16(pu2_pred_1_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(clip_res));
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 2 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_1.val[1]),
                               vreinterpret_s16_u16(pu2_pred_2_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(clip_res));
    pu2_dst_tmp_0_dst_strd += dst_strd;

    /* Row 3 */
    pred_add_val_t = vqadd_s16(vreinterpret_s16_s32(transpose_val_32_2.val[1]),
                               vreinterpret_s16_u16(pu2_pred_3_pred_strd_val));
    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
    vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(clip_res));
}

