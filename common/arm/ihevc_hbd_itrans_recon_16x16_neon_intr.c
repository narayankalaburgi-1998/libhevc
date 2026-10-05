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
 ******************************************************************************
 * @file
 *  ihevc_hbd_itrans_recon_16x16_neon_intr.c
 *
 * @brief
 *  Contains function definitions for HBD 16x16 inverse transform and
 *  reconstruction using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_itrans_recon_16x16_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
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
 *  This function performs HBD Inverse transform and reconstruction for 16x16
 *  input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs 16x16 inverse DCT transform, adds the 16-bit prediction data, and
 *  clips output to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 16x16 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 16x16 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 16x16 block
 *
 * @param[out] pu2_dst
 *  Output 16x16 block
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
void ihevc_hbd_itrans_recon_16x16_neonintr(WORD16 *pi2_src,
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
    WORD32 j, k;
    WORD32 ee[4];
    WORD32 eee[2], eeo[2];
    WORD16 *pi2_tmp_orig;
    WORD32 trans_size;
    WORD32 zero_rows_2nd_stage = zero_cols;
    WORD32 row_limit_2nd_stage;
    WORD32 shift;
    WORD16 clip_limit;
    int16x4_t dup_const_clip_limit;
    int16x4_t pred_add_val_t;
    int16x4_t clip_res;

    trans_size = TRANS_SIZE_16;
    pi2_tmp_orig = pi2_tmp;
    shift = 20 - u1_bit_depth;
    clip_limit = (1 << u1_bit_depth) - 1;
    dup_const_clip_limit = vdup_n_s16(clip_limit);

    if((zero_cols & 0xFFF0) == 0xFFF0)
        row_limit_2nd_stage = 4;
    else if((zero_cols & 0xFF00) == 0xFF00)
        row_limit_2nd_stage = 8;
    else
        row_limit_2nd_stage = TRANS_SIZE_16;

    if((zero_rows & 0xFFF0) == 0xFFF0)  /* First 4 rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */
        const WORD16 *g_ai2_ihevc_trans_16_1 = (const WORD16 *)&g_ai2_ihevc_trans_16[1][0];

        int16x8_t g_ai2_ihevc_trans_16_val_1;
        int16x8_t g_ai2_ihevc_trans_16_val_2;
        int16x8_t g_ai2_ihevc_trans_16_val_3;

        g_ai2_ihevc_trans_16_val_1 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_2 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_3 = vld1q_s16(g_ai2_ihevc_trans_16_1);

        WORD16 *pi2_src_tmp_0_src_strd;
        WORD16 *pi2_src_tmp_1_src_strd;

        WORD16 *pi2_tmp_0_trans_size;
        WORD16 *pi2_tmp_1_trans_size;

        WORD16 pi2_src_0_src_strd_val;
        WORD16 pi2_src_1_src_strd_val;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_2_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;
        int16x4_t pi2_src_5_src_strd_val_t;
        int16x4_t pi2_src_6_src_strd_val_t;
        int16x4_t pi2_src_7_src_strd_val_t;
        int16x4_t pi2_src_9_src_strd_val_t;
        int16x4_t pi2_src_10_src_strd_val_t;
        int16x4_t pi2_src_11_src_strd_val_t;
        int16x4_t pi2_src_13_src_strd_val_t;
        int16x4_t pi2_src_14_src_strd_val_t;
        int16x4_t pi2_src_15_src_strd_val_t;

        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        WORD32 *ee_ptr;
        int32x4_t eo_val_0;
        int32x4_t ee_val_0;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        ee_ptr = &ee[0];
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int16x8_t constq_0 = vdupq_n_s16(0);;
        int16x4_t const_0;
        UWORD16 *pu2_pred_tmp_0_pred_strd;
        uint16x8_t pu2_pred_0_pred_strd_val;
        const_0 = vdup_n_s16(0);

        for(j = 0; j < row_limit_2nd_stage; j++)
        {
            /* Checking for Zero Cols */
            pi2_tmp_str = pi2_tmp;

            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_0_src_strd = pi2_src_tmp_1_src_strd + src_strd;

            pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
            pi2_src_tmp_1_src_strd += 2 * src_strd;
            pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                }
                {
                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                }
                eeo[0] = 0;
                eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_src[0];
                eeo[1] = 0;
                eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_src[0];

                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                for(k = 0; k < 2; k++)
                {
                    ee[k] = eee[k] + eeo[k];
                    ee[k + 2] = eee[1 - k] - eeo[1 - k];
                }
                ee_val_0 = vld1q_s32(ee_ptr);
                e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                e_val_1 = vrev64q_s32(e_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                shift_res_1 = vqmovn_s32(e_add_o_val);
                rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);
                vst1_s16(pi2_tmp_str, shift_res_1);
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */

        shift_val = vdupq_n_s32(shift);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFF0) == 0xFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                    }
                    eeo[0] = 0;
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = 0;
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFF00) == 0xFF00)
        {
            int16x8_t g_ai2_ihevc_trans_16_val_5;
            int16x8_t g_ai2_ihevc_trans_16_val_6;
            int16x8_t g_ai2_ihevc_trans_16_val_7;

            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_5 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_6 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_7 = vld1q_s16(g_ai2_ihevc_trans_16_1);

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);

                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else /* All rows of output of 1st stage are non-zero */
        {
            int16x8_t g_ai2_ihevc_trans_16_val_5;
            int16x8_t g_ai2_ihevc_trans_16_val_6;
            int16x8_t g_ai2_ihevc_trans_16_val_7;
            int16x8_t g_ai2_ihevc_trans_16_val_9;
            int16x8_t g_ai2_ihevc_trans_16_val_10;
            int16x8_t g_ai2_ihevc_trans_16_val_11;
            int16x8_t g_ai2_ihevc_trans_16_val_13;
            int16x8_t g_ai2_ihevc_trans_16_val_14;
            int16x8_t g_ai2_ihevc_trans_16_val_15;

            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_5 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_6 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_7 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_9 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_10 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_11 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_13 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_14 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_15 = vld1q_s16(g_ai2_ihevc_trans_16_1);

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2* trans_size;
                        pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_14), pi2_src_14_src_strd_val_t);
                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][0]
                    * pi2_tmp[12 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][0]
                    * pi2_tmp[8 * trans_size];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][1]
                    * pi2_tmp[12 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][1]
                    * pi2_tmp[8* trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
    else if((zero_rows & 0xFF00) == 0xFF00)   /* First 8 rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */
        const WORD16 *g_ai2_ihevc_trans_16_1 = (const WORD16 *)&g_ai2_ihevc_trans_16[1][0];

        int16x8_t g_ai2_ihevc_trans_16_val_1;
        int16x8_t g_ai2_ihevc_trans_16_val_2;
        int16x8_t g_ai2_ihevc_trans_16_val_3;
        int16x8_t g_ai2_ihevc_trans_16_val_5;
        int16x8_t g_ai2_ihevc_trans_16_val_6;
        int16x8_t g_ai2_ihevc_trans_16_val_7;

        g_ai2_ihevc_trans_16_val_1 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_2 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_3 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 32;
        g_ai2_ihevc_trans_16_val_5 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_6 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_7 = vld1q_s16(g_ai2_ihevc_trans_16_1);

        WORD16 *pi2_src_tmp_0_src_strd;
        WORD16 *pi2_src_tmp_1_src_strd;

        WORD16 *pi2_tmp_0_trans_size;
        WORD16 *pi2_tmp_1_trans_size;

        WORD16 pi2_src_0_src_strd_val;
        WORD16 pi2_src_1_src_strd_val;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_2_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;
        int16x4_t pi2_src_5_src_strd_val_t;
        int16x4_t pi2_src_6_src_strd_val_t;
        int16x4_t pi2_src_7_src_strd_val_t;

        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        WORD32 *ee_ptr;
        int32x4_t eo_val_0;
        int32x4_t ee_val_0;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        ee_ptr = &ee[0];
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int16x8_t constq_0 = vdupq_n_s16(0);;
        int16x4_t const_0;
        UWORD16 *pu2_pred_tmp_0_pred_strd;
        uint16x8_t pu2_pred_0_pred_strd_val;
        const_0 = vdup_n_s16(0);

        for(j = 0; j < row_limit_2nd_stage; j++)
        {
            /* Checking for Zero Cols */
            pi2_tmp_str = pi2_tmp;

            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_0_src_strd = pi2_src_tmp_1_src_strd + src_strd;

            pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
            pi2_src_tmp_1_src_strd += 2 * src_strd;
            pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);
                }
                {
                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);
                }
                eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_src[4 * src_strd];
                eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_src[0];
                eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_src[4 * src_strd];
                eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_src[0];
                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                for(k = 0; k < 2; k++)
                {
                    ee[k] = eee[k] + eeo[k];
                    ee[k + 2] = eee[1 - k] - eeo[1 - k];
                }
                ee_val_0 = vld1q_s32(ee_ptr);
                e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                e_val_1 = vrev64q_s32(e_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                shift_res_1 = vqmovn_s32(e_add_o_val);
                rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);
                vst1_s16(pi2_tmp_str, shift_res_1);
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */

        shift_val = vdupq_n_s32(shift);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFF0) == 0xFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                    }
                    eeo[0] = 0;
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = 0;
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFF00) == 0xFF00) /* First 8 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);

                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else /* All rows of output of 1st stage are non-zero */
        {
            int16x8_t g_ai2_ihevc_trans_16_val_9;
            int16x8_t g_ai2_ihevc_trans_16_val_10;
            int16x8_t g_ai2_ihevc_trans_16_val_11;
            int16x8_t g_ai2_ihevc_trans_16_val_13;
            int16x8_t g_ai2_ihevc_trans_16_val_14;
            int16x8_t g_ai2_ihevc_trans_16_val_15;

            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_9 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_10 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_11 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 32;
            g_ai2_ihevc_trans_16_val_13 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_14 = vld1q_s16(g_ai2_ihevc_trans_16_1);
            g_ai2_ihevc_trans_16_1 += 16;
            g_ai2_ihevc_trans_16_val_15 = vld1q_s16(g_ai2_ihevc_trans_16_1);

            int16x4_t pi2_src_9_src_strd_val_t;
            int16x4_t pi2_src_10_src_strd_val_t;
            int16x4_t pi2_src_11_src_strd_val_t;
            int16x4_t pi2_src_13_src_strd_val_t;
            int16x4_t pi2_src_14_src_strd_val_t;
            int16x4_t pi2_src_15_src_strd_val_t;

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2* trans_size;
                        pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_14), pi2_src_14_src_strd_val_t);
                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][0]
                    * pi2_tmp[12 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][0]
                    * pi2_tmp[8 * trans_size];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][1]
                    * pi2_tmp[12 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][1]
                    * pi2_tmp[8* trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
    else /* All rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */
        const WORD16 *g_ai2_ihevc_trans_16_1 = (const WORD16 *)&g_ai2_ihevc_trans_16[1][0];

        int16x8_t g_ai2_ihevc_trans_16_val_1;
        int16x8_t g_ai2_ihevc_trans_16_val_2;
        int16x8_t g_ai2_ihevc_trans_16_val_3;
        int16x8_t g_ai2_ihevc_trans_16_val_5;
        int16x8_t g_ai2_ihevc_trans_16_val_6;
        int16x8_t g_ai2_ihevc_trans_16_val_7;
        int16x8_t g_ai2_ihevc_trans_16_val_9;
        int16x8_t g_ai2_ihevc_trans_16_val_10;
        int16x8_t g_ai2_ihevc_trans_16_val_11;
        int16x8_t g_ai2_ihevc_trans_16_val_13;
        int16x8_t g_ai2_ihevc_trans_16_val_14;
        int16x8_t g_ai2_ihevc_trans_16_val_15;

        g_ai2_ihevc_trans_16_val_1 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_2 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_3 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 32;
        g_ai2_ihevc_trans_16_val_5 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_6 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_7 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 32;
        g_ai2_ihevc_trans_16_val_9 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_10 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_11 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 32;
        g_ai2_ihevc_trans_16_val_13 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_14 = vld1q_s16(g_ai2_ihevc_trans_16_1);
        g_ai2_ihevc_trans_16_1 += 16;
        g_ai2_ihevc_trans_16_val_15 = vld1q_s16(g_ai2_ihevc_trans_16_1);

        WORD16 *pi2_src_tmp_0_src_strd;
        WORD16 *pi2_src_tmp_1_src_strd;

        WORD16 *pi2_tmp_0_trans_size;
        WORD16 *pi2_tmp_1_trans_size;

        WORD16 pi2_src_0_src_strd_val;
        WORD16 pi2_src_1_src_strd_val;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_2_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;
        int16x4_t pi2_src_5_src_strd_val_t;
        int16x4_t pi2_src_6_src_strd_val_t;
        int16x4_t pi2_src_7_src_strd_val_t;
        int16x4_t pi2_src_9_src_strd_val_t;
        int16x4_t pi2_src_10_src_strd_val_t;
        int16x4_t pi2_src_11_src_strd_val_t;
        int16x4_t pi2_src_13_src_strd_val_t;
        int16x4_t pi2_src_14_src_strd_val_t;
        int16x4_t pi2_src_15_src_strd_val_t;

        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        WORD32 *ee_ptr;
        int32x4_t eo_val_0;
        int32x4_t ee_val_0;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        ee_ptr = &ee[0];
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int16x8_t constq_0;
        int16x4_t const_0;
        UWORD16 *pu2_pred_tmp_0_pred_strd;
        uint16x8_t pu2_pred_0_pred_strd_val;
        const_0 = vdup_n_s16(0);

        for(j = 0; j < trans_size; j++)
        {
            /* Checking for Zero Cols */
            pi2_tmp_str = pi2_tmp;

            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_0_src_strd = pi2_src_tmp_1_src_strd + src_strd;

            pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
            pi2_src_tmp_1_src_strd += 2 * src_strd;
            pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                constq_0 = vdupq_n_s16(0);
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * src_strd;
                    pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2* src_strd;
                    pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);

                    o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_tmp_0_src_strd += 4 * src_strd;
                    pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                    pi2_src_0_src_strd_val = pi2_src_tmp_0_src_strd[0];
                    pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);
                    o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);
                }
                {
                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_10), pi2_src_10_src_strd_val_t);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_14), pi2_src_14_src_strd_val_t);
                }
                eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_src[4 * src_strd]
                                + g_ai2_ihevc_trans_16[12][0]
                                                * pi2_src[12 * src_strd];
                eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_src[0]
                                + g_ai2_ihevc_trans_16[8][0]
                                                * pi2_src[8 * src_strd];
                eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_src[4 * src_strd]
                                + g_ai2_ihevc_trans_16[12][1]
                                                * pi2_src[12 * src_strd];
                eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_src[0]
                                + g_ai2_ihevc_trans_16[8][1]
                                                * pi2_src[8* src_strd];

                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                for(k = 0; k < 2; k++)
                {
                    ee[k] = eee[k] + eeo[k];
                    ee[k + 2] = eee[1 - k] - eeo[1 - k];
                }
                ee_val_0 = vld1q_s32(ee_ptr);
                e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                e_val_1 = vrev64q_s32(e_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                shift_res_1 = vqmovn_s32(e_add_o_val);
                rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);

                vst1_s16(pi2_tmp_str, shift_res_1);
                pi2_tmp_str += 4;
                e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val);
                shift_res_1 = vrev64_s16(shift_res_1);
                vst1_s16(pi2_tmp_str, shift_res_1);
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */

        shift_val = vdupq_n_s32(shift);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFF0) == 0xFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                    }
                    eeo[0] = 0;
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = 0;
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFF00) == 0xFF00) /* First 8 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);

                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else /* All rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;

                pi2_tmp_1_trans_size = pi2_tmp + trans_size;
                pi2_tmp_0_trans_size = pi2_tmp_1_trans_size + trans_size;

                pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                pi2_tmp_1_trans_size += 2 * trans_size;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                {
                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    {
                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2 * trans_size;
                        pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_tmp_1_trans_size += 2* trans_size;
                        pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);

                        pi2_src_1_src_strd_val = pi2_tmp_1_trans_size[0];
                        pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);
                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_16_val_1), pi2_src_1_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_3), pi2_src_3_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_tmp_0_trans_size += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_5), pi2_src_5_src_strd_val_t);

                        pi2_src_0_src_strd_val = pi2_tmp_0_trans_size[0];
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_0_src_strd_val);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_7), pi2_src_7_src_strd_val_t);

                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_9), pi2_src_9_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_11), pi2_src_11_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_13), pi2_src_13_src_strd_val_t);
                        o_val_1 = vmlal_s16(o_val_1, vget_high_s16(g_ai2_ihevc_trans_16_val_15), pi2_src_15_src_strd_val_t);
                    }
                    {
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_16_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_16_val_14), pi2_src_14_src_strd_val_t);
                    }
                    eeo[0] = g_ai2_ihevc_trans_16[4][0] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][0]
                    * pi2_tmp[12 * trans_size];
                    eee[0] = g_ai2_ihevc_trans_16[0][0] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][0]
                    * pi2_tmp[8 * trans_size];
                    eeo[1] = g_ai2_ihevc_trans_16[4][1] * pi2_tmp[4 * trans_size]
                    + g_ai2_ihevc_trans_16[12][1]
                    * pi2_tmp[12 * trans_size];
                    eee[1] = g_ai2_ihevc_trans_16[0][1] * pi2_tmp[0]
                    + g_ai2_ihevc_trans_16[8][1]
                    * pi2_tmp[8* trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    for(k = 0; k < 2; k++)
                    {
                        ee[k] = eee[k] + eeo[k];
                        ee[k + 2] = eee[1 - k] - eeo[1 - k];
                    }
                    ee_val_0 = vld1q_s32(ee_ptr);
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);

                    e_val_1 = vsubq_s32(ee_val_0, eo_val_0);

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    e_val_1 = vrev64q_s32(e_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 0);

                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_1), 1);

                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_1), 1));
                    e_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_1), 0));

                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                    pu2_pred_tmp_0_pred_strd += 8;
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                    pu2_dst_tmp += 4;
                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);
                    pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                    pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                    clip_res = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                    vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(clip_res));
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
}
