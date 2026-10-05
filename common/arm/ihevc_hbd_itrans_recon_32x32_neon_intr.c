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
 *  ihevc_hbd_itrans_recon_32x32_neon_intr.c
 *
 * @brief
 *  Contains function definitions for high bit depth inverse transform and
 *  reconstruction 32x32 using ARM NEON intrinsics
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
#include "arm_neon.h"


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
    WORD32 j, k;
    WORD32 o[16];
    WORD32 eee[4];
    WORD32 eeee[2], eeeo[2];
    WORD16 *pi2_tmp_orig;
    WORD32 trans_size;
    WORD32 zero_rows_2nd_stage = zero_cols;
    WORD32 row_limit_2nd_stage;
    int16x4_t v_clip_limit = vdup_n_s16((1 << u1_bit_depth) - 1);

    trans_size = TRANS_SIZE_32;
    pi2_tmp_orig = pi2_tmp;

    if((zero_cols & 0xFFFFFFF0) == 0xFFFFFFF0)
        row_limit_2nd_stage = 4;
    else if((zero_cols & 0xFFFFFF00) == 0xFFFFFF00)
        row_limit_2nd_stage = 8;
    else
        row_limit_2nd_stage = TRANS_SIZE_32;

    if((zero_rows & 0xFFFFFFF0) == 0xFFFFFFF0)  /* First 4 rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */

        WORD16 *g_ai2_ihevc_trans_32_1;
        WORD16 *g_ai2_ihevc_trans_32_2;
        WORD16 *g_ai2_ihevc_trans_32_3;

        int16x4_t g_ai2_ihevc_trans_32_val_1;
        int16x4_t g_ai2_ihevc_trans_32_val_3;

        WORD16 *pi2_src_tmp_1_src_strd;
        WORD16 *pi2_src_tmp_2_src_strd;

        WORD16 pi2_src_1_src_strd_val;
        WORD16 pi2_src_2_src_strd_val;

        WORD32 eeo_val_0_init;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;

        int16x4_t pi2_src_2_src_strd_val_t;

        int16x8_t g_ai2_ihevc_trans_32_val_2;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t o_val_2;
        int32x4_t o_val_3;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        int32x4_t e_val_2;
        int32x4_t e_val_3;
        int32x4_t eeo_val_0;
        int32x4_t eee_val_0 = vdupq_n_s32(0);
        int32x4_t eo_val_0;
        int32x4_t eo_val_1;
        int32x4_t ee_val_0;
        int32x4_t ee_val_1;
        WORD32 *o_val_ptr_1;
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;


        int16x8_t constq_0 = vdupq_n_s16(0);
        int16x4_t const_0 = vdup_n_s16(0);
        int16x8_t wide_pu2_pred_val;
        int16x4_t res_s16;

        UWORD16 *pu2_pred_tmp_0_pred_strd;

        o_val_ptr_1 = &o[0];
        g_ai2_ihevc_trans_32_2 = (WORD16 *)&g_ai2_ihevc_trans_32[2][0];

        g_ai2_ihevc_trans_32_val_2 = vld1q_s16(g_ai2_ihevc_trans_32_2);

        for(j = 0; j < row_limit_2nd_stage; j++)
        {
            pi2_tmp_str = pi2_tmp;
            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_2_src_strd = pi2_src + 2 * src_strd;

            g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
            g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                for(k = 16; k > 0; k -= 4)
                {
                    g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                    g_ai2_ihevc_trans_32_1 += 4;

                    g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                    g_ai2_ihevc_trans_32_3 += 4;
                    o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                    vst1q_s32(o_val_ptr_1, o_val_0);
                    o_val_ptr_1 += 4;
                }
                {
                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);
                }
                {
                    eeo_val_0_init = 0;
                    eeo_val_0 = vdupq_n_s32(eeo_val_0_init);
                }

                eeeo[0] = 0;
                eeeo[1] = 0;
                eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_src[0];
                eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_src[0];

                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                eee[0] = eeee[0] + eeeo[0];
                eee[3] = eeee[0] - eeeo[0];
                eee[1] = eeee[1] + eeeo[1];
                eee[2] = eeee[1] - eeeo[1];

                eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                o_val_ptr_1 -= 16;

                {
                    ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vrev64q_s32(ee_val_1);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                }
                {
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                    o_val_0 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    o_val_1 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_1 = vaddq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_2 = vsubq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_val_2 = vrev64q_s32(e_val_2);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                    o_val_2 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                    e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                    e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    e_val_3 = vrev64q_s32(e_val_3);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                    o_val_3 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                }
                {
                    e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    vst1_s16(pi2_tmp_str, shift_res_1);
                    o_val_ptr_1 -= 16;
                }
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */
        shift_val = vdupq_n_s32(20 - u1_bit_depth);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFFFFFF0) == 0xFFFFFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;

                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;

                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    }
                    {
                        eeo_val_0_init = 0;
                        eeo_val_0 = vdupq_n_s32(eeo_val_0_init);

                    }
                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFFFFFF00) == 0xFFFFFF00) /* First 8 rows of output of 1st stage are non-zero */
        {
            WORD16 *g_ai2_ihevc_trans_32_4;
            WORD16 *g_ai2_ihevc_trans_32_5;
            WORD16 *g_ai2_ihevc_trans_32_7;

            int16x4_t g_ai2_ihevc_trans_32_val_5;
            int16x4_t g_ai2_ihevc_trans_32_val_7;

            WORD16 *pi2_src_tmp_4_src_strd;

            WORD16 pi2_src_4_src_strd_val;

            int16x4_t pi2_src_5_src_strd_val_t;
            int16x4_t pi2_src_7_src_strd_val_t;

            int16x4_t pi2_src_4_src_strd_val_t;
            int16x4_t pi2_src_6_src_strd_val_t;

            int16x4_t g_ai2_ihevc_trans_32_val_4;
            int16x8_t g_ai2_ihevc_trans_32_val_6;

            g_ai2_ihevc_trans_32_4 = (WORD16 *)&g_ai2_ihevc_trans_32[4][0];

            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_6 = vld1q_s16(g_ai2_ihevc_trans_32_2);

            g_ai2_ihevc_trans_32_val_4 = vld1_s16(g_ai2_ihevc_trans_32_4);

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);

                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                    }
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);
                    }

                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else  /* All rows of output of 1st stage are non-zero */
        {
            WORD16 *g_ai2_ihevc_trans_32_4;
            WORD16 *g_ai2_ihevc_trans_32_5;
            WORD16 *g_ai2_ihevc_trans_32_7;
            WORD16 *g_ai2_ihevc_trans_32_9;
            WORD16 *g_ai2_ihevc_trans_32_11;
            WORD16 *g_ai2_ihevc_trans_32_13;
            WORD16 *g_ai2_ihevc_trans_32_15;
            WORD16 *g_ai2_ihevc_trans_32_17;
            WORD16 *g_ai2_ihevc_trans_32_19;
            WORD16 *g_ai2_ihevc_trans_32_21;
            WORD16 *g_ai2_ihevc_trans_32_23;
            WORD16 *g_ai2_ihevc_trans_32_25;
            WORD16 *g_ai2_ihevc_trans_32_27;
            WORD16 *g_ai2_ihevc_trans_32_29;
            WORD16 *g_ai2_ihevc_trans_32_31;

            int16x4_t g_ai2_ihevc_trans_32_val_5;
            int16x4_t g_ai2_ihevc_trans_32_val_7;
            int16x4_t g_ai2_ihevc_trans_32_val_9;
            int16x4_t g_ai2_ihevc_trans_32_val_11;
            int16x4_t g_ai2_ihevc_trans_32_val_13;
            int16x4_t g_ai2_ihevc_trans_32_val_15;
            int16x4_t g_ai2_ihevc_trans_32_val_17;
            int16x4_t g_ai2_ihevc_trans_32_val_19;
            int16x4_t g_ai2_ihevc_trans_32_val_21;
            int16x4_t g_ai2_ihevc_trans_32_val_23;
            int16x4_t g_ai2_ihevc_trans_32_val_25;
            int16x4_t g_ai2_ihevc_trans_32_val_27;
            int16x4_t g_ai2_ihevc_trans_32_val_29;
            int16x4_t g_ai2_ihevc_trans_32_val_31;

            WORD16 *pi2_src_tmp_4_src_strd;

            WORD16 pi2_src_4_src_strd_val;

            int16x4_t pi2_src_5_src_strd_val_t;
            int16x4_t pi2_src_7_src_strd_val_t;
            int16x4_t pi2_src_9_src_strd_val_t;
            int16x4_t pi2_src_11_src_strd_val_t;
            int16x4_t pi2_src_13_src_strd_val_t;
            int16x4_t pi2_src_15_src_strd_val_t;
            int16x4_t pi2_src_17_src_strd_val_t;
            int16x4_t pi2_src_19_src_strd_val_t;
            int16x4_t pi2_src_21_src_strd_val_t;
            int16x4_t pi2_src_23_src_strd_val_t;
            int16x4_t pi2_src_25_src_strd_val_t;
            int16x4_t pi2_src_27_src_strd_val_t;
            int16x4_t pi2_src_29_src_strd_val_t;
            int16x4_t pi2_src_31_src_strd_val_t;

            int16x4_t pi2_src_4_src_strd_val_t;
            int16x4_t pi2_src_6_src_strd_val_t;
            int16x4_t pi2_src_10_src_strd_val_t;
            int16x4_t pi2_src_12_src_strd_val_t;
            int16x4_t pi2_src_14_src_strd_val_t;
            int16x4_t pi2_src_18_src_strd_val_t;
            int16x4_t pi2_src_20_src_strd_val_t;
            int16x4_t pi2_src_22_src_strd_val_t;
            int16x4_t pi2_src_26_src_strd_val_t;
            int16x4_t pi2_src_28_src_strd_val_t;
            int16x4_t pi2_src_30_src_strd_val_t;

            int16x4_t g_ai2_ihevc_trans_32_val_4;
            int16x8_t g_ai2_ihevc_trans_32_val_6;
            int16x8_t g_ai2_ihevc_trans_32_val_10;
            int16x4_t g_ai2_ihevc_trans_32_val_12;
            int16x8_t g_ai2_ihevc_trans_32_val_14;
            int16x8_t g_ai2_ihevc_trans_32_val_18;
            int16x4_t g_ai2_ihevc_trans_32_val_20;
            int16x8_t g_ai2_ihevc_trans_32_val_22;
            int16x8_t g_ai2_ihevc_trans_32_val_26;
            int16x4_t g_ai2_ihevc_trans_32_val_28;
            int16x8_t g_ai2_ihevc_trans_32_val_30;


            g_ai2_ihevc_trans_32_4 = (WORD16 *)&g_ai2_ihevc_trans_32[4][0];

            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_6 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_10 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_14 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_18 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_22 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_26 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_30 = vld1q_s16(g_ai2_ihevc_trans_32_2);

            g_ai2_ihevc_trans_32_val_4 = vld1_s16(g_ai2_ihevc_trans_32_4);
            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_12 = vld1_s16(g_ai2_ihevc_trans_32_4);
            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_20 = vld1_s16(g_ai2_ihevc_trans_32_4);
            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_28 = vld1_s16(g_ai2_ihevc_trans_32_4);

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];
                g_ai2_ihevc_trans_32_9 = (WORD16 *)&g_ai2_ihevc_trans_32[9][0];
                g_ai2_ihevc_trans_32_11 = (WORD16 *)&g_ai2_ihevc_trans_32[11][0];
                g_ai2_ihevc_trans_32_13 = (WORD16 *)&g_ai2_ihevc_trans_32[13][0];
                g_ai2_ihevc_trans_32_15 = (WORD16 *)&g_ai2_ihevc_trans_32[15][0];
                g_ai2_ihevc_trans_32_17 = (WORD16 *)&g_ai2_ihevc_trans_32[17][0];
                g_ai2_ihevc_trans_32_19 = (WORD16 *)&g_ai2_ihevc_trans_32[19][0];
                g_ai2_ihevc_trans_32_21 = (WORD16 *)&g_ai2_ihevc_trans_32[21][0];
                g_ai2_ihevc_trans_32_23 = (WORD16 *)&g_ai2_ihevc_trans_32[23][0];
                g_ai2_ihevc_trans_32_25 = (WORD16 *)&g_ai2_ihevc_trans_32[25][0];
                g_ai2_ihevc_trans_32_27 = (WORD16 *)&g_ai2_ihevc_trans_32[27][0];
                g_ai2_ihevc_trans_32_29 = (WORD16 *)&g_ai2_ihevc_trans_32[29][0];
                g_ai2_ihevc_trans_32_31 = (WORD16 *)&g_ai2_ihevc_trans_32[31][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_17_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_19_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_21_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_23_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_25_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_27_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_29_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_31_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_9 = vld1_s16(g_ai2_ihevc_trans_32_9);
                        g_ai2_ihevc_trans_32_9 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_11 = vld1_s16(g_ai2_ihevc_trans_32_11);
                        g_ai2_ihevc_trans_32_11 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_9, pi2_src_9_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_13 = vld1_s16(g_ai2_ihevc_trans_32_13);
                        g_ai2_ihevc_trans_32_13 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_11, pi2_src_11_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_15 = vld1_s16(g_ai2_ihevc_trans_32_15);
                        g_ai2_ihevc_trans_32_15 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_13, pi2_src_13_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_17 = vld1_s16(g_ai2_ihevc_trans_32_17);
                        g_ai2_ihevc_trans_32_17 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_15, pi2_src_15_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_19 = vld1_s16(g_ai2_ihevc_trans_32_19);
                        g_ai2_ihevc_trans_32_19 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_17, pi2_src_17_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_21 = vld1_s16(g_ai2_ihevc_trans_32_21);
                        g_ai2_ihevc_trans_32_21 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_19, pi2_src_19_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_23 = vld1_s16(g_ai2_ihevc_trans_32_23);
                        g_ai2_ihevc_trans_32_23 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_21, pi2_src_21_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_25 = vld1_s16(g_ai2_ihevc_trans_32_25);
                        g_ai2_ihevc_trans_32_25 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_23, pi2_src_23_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_27 = vld1_s16(g_ai2_ihevc_trans_32_27);
                        g_ai2_ihevc_trans_32_27 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_25, pi2_src_25_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_29 = vld1_s16(g_ai2_ihevc_trans_32_29);
                        g_ai2_ihevc_trans_32_29 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_27, pi2_src_27_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_31 = vld1_s16(g_ai2_ihevc_trans_32_31);
                        g_ai2_ihevc_trans_32_31 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_29, pi2_src_29_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_31, pi2_src_31_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_18_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_22_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_26_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_30_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_12_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                    }
        //            for(k = 0; k < 4; k++)
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_20_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_12, pi2_src_12_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_28_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_20, pi2_src_20_src_strd_val_t);

                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_28, pi2_src_28_src_strd_val_t);
                    }

                    eeeo[0] = g_ai2_ihevc_trans_32[8][0] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][0]
                                                    * pi2_tmp[24 * trans_size];
                    eeeo[1] = g_ai2_ihevc_trans_32[8][1] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][1]
                                                    * pi2_tmp[24 * trans_size];
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][0]
                                                    * pi2_tmp[16 * trans_size];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][1]
                                                    * pi2_tmp[16 * trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
    else if((zero_rows & 0xFFFFFF00) == 0xFFFFFF00)  /* First 8 rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */

        WORD16 *g_ai2_ihevc_trans_32_1;
        WORD16 *g_ai2_ihevc_trans_32_2;
        WORD16 *g_ai2_ihevc_trans_32_3;
        WORD16 *g_ai2_ihevc_trans_32_4;
        WORD16 *g_ai2_ihevc_trans_32_5;
        WORD16 *g_ai2_ihevc_trans_32_7;

        int16x4_t g_ai2_ihevc_trans_32_val_1;
        int16x4_t g_ai2_ihevc_trans_32_val_3;
        int16x4_t g_ai2_ihevc_trans_32_val_5;
        int16x4_t g_ai2_ihevc_trans_32_val_7;

        WORD16 *pi2_src_tmp_1_src_strd;
        WORD16 *pi2_src_tmp_2_src_strd;
        WORD16 *pi2_src_tmp_4_src_strd;

        WORD16 pi2_src_1_src_strd_val;
        WORD16 pi2_src_2_src_strd_val;
        WORD16 pi2_src_4_src_strd_val;

        WORD32 eeo_val_0_init;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;
        int16x4_t pi2_src_5_src_strd_val_t;
        int16x4_t pi2_src_7_src_strd_val_t;

        int16x4_t pi2_src_2_src_strd_val_t;
        int16x4_t pi2_src_4_src_strd_val_t;
        int16x4_t pi2_src_6_src_strd_val_t;

        int16x8_t g_ai2_ihevc_trans_32_val_2;
        int16x4_t g_ai2_ihevc_trans_32_val_4;
        int16x8_t g_ai2_ihevc_trans_32_val_6;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t o_val_2;
        int32x4_t o_val_3;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        int32x4_t e_val_2;
        int32x4_t e_val_3;
        int32x4_t eeo_val_0;
        int32x4_t eee_val_0 = vdupq_n_s32(0);
        int32x4_t eo_val_0;
        int32x4_t eo_val_1;
        int32x4_t ee_val_0;
        int32x4_t ee_val_1;
        WORD32 *o_val_ptr_1;
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;


        int16x8_t constq_0 = vdupq_n_s16(0);
        int16x4_t const_0 = vdup_n_s16(0);
        int16x8_t wide_pu2_pred_val;
        int16x4_t res_s16;

        UWORD16 *pu2_pred_tmp_0_pred_strd;

        o_val_ptr_1 = &o[0];
        g_ai2_ihevc_trans_32_2 = (WORD16 *)&g_ai2_ihevc_trans_32[2][0];
        g_ai2_ihevc_trans_32_4 = (WORD16 *)&g_ai2_ihevc_trans_32[4][0];

        g_ai2_ihevc_trans_32_val_2 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_6 = vld1q_s16(g_ai2_ihevc_trans_32_2);

        g_ai2_ihevc_trans_32_val_4 = vld1_s16(g_ai2_ihevc_trans_32_4);

        for(j = 0; j < row_limit_2nd_stage; j++)
        {
            pi2_tmp_str = pi2_tmp;
            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_2_src_strd = pi2_src + 2 * src_strd;
            pi2_src_tmp_4_src_strd = pi2_src + 4 * src_strd;

            g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
            g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
            g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
            g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                for(k = 16; k > 0; k -= 4)
                {
                    g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                    g_ai2_ihevc_trans_32_1 += 4;

                    g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                    g_ai2_ihevc_trans_32_3 += 4;
                    o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                    g_ai2_ihevc_trans_32_5 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                    g_ai2_ihevc_trans_32_7 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                    vst1q_s32(o_val_ptr_1, o_val_0);
                    o_val_ptr_1 += 4;
                }
                {
                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);


                    eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                    pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);

                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                }
                {
                    eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);

                }

                eeeo[0] = 0;
                eeeo[1] = 0;
                eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_src[0];
                eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_src[0];

                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                eee[0] = eeee[0] + eeeo[0];
                eee[3] = eeee[0] - eeeo[0];
                eee[1] = eeee[1] + eeeo[1];
                eee[2] = eeee[1] - eeeo[1];

                eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                o_val_ptr_1 -= 16;

                {
                    ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vrev64q_s32(ee_val_1);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                }
                {
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                    o_val_0 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    o_val_1 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_1 = vaddq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_2 = vsubq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_val_2 = vrev64q_s32(e_val_2);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                    o_val_2 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                    e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                    e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    e_val_3 = vrev64q_s32(e_val_3);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                    o_val_3 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                }
                {
                    e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    vst1_s16(pi2_tmp_str, shift_res_1);
                    o_val_ptr_1 -= 16;
                }
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */
        shift_val = vdupq_n_s32(20 - u1_bit_depth);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFFFFFF0) == 0xFFFFFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;

                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    }
                    {
                        eeo_val_0_init = 0;
                        eeo_val_0 = vdupq_n_s32(eeo_val_0_init);

                    }
                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFFFFFF00) == 0xFFFFFF00) /* First 8 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);

                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                    }
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);
                    }

                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else /* All rows of output of 1st stage are non-zero */
        {
            WORD16 *g_ai2_ihevc_trans_32_9;
            WORD16 *g_ai2_ihevc_trans_32_11;
            WORD16 *g_ai2_ihevc_trans_32_13;
            WORD16 *g_ai2_ihevc_trans_32_15;
            WORD16 *g_ai2_ihevc_trans_32_17;
            WORD16 *g_ai2_ihevc_trans_32_19;
            WORD16 *g_ai2_ihevc_trans_32_21;
            WORD16 *g_ai2_ihevc_trans_32_23;
            WORD16 *g_ai2_ihevc_trans_32_25;
            WORD16 *g_ai2_ihevc_trans_32_27;
            WORD16 *g_ai2_ihevc_trans_32_29;
            WORD16 *g_ai2_ihevc_trans_32_31;

            int16x4_t g_ai2_ihevc_trans_32_val_9;
            int16x4_t g_ai2_ihevc_trans_32_val_11;
            int16x4_t g_ai2_ihevc_trans_32_val_13;
            int16x4_t g_ai2_ihevc_trans_32_val_15;
            int16x4_t g_ai2_ihevc_trans_32_val_17;
            int16x4_t g_ai2_ihevc_trans_32_val_19;
            int16x4_t g_ai2_ihevc_trans_32_val_21;
            int16x4_t g_ai2_ihevc_trans_32_val_23;
            int16x4_t g_ai2_ihevc_trans_32_val_25;
            int16x4_t g_ai2_ihevc_trans_32_val_27;
            int16x4_t g_ai2_ihevc_trans_32_val_29;
            int16x4_t g_ai2_ihevc_trans_32_val_31;

            int16x4_t pi2_src_9_src_strd_val_t;
            int16x4_t pi2_src_11_src_strd_val_t;
            int16x4_t pi2_src_13_src_strd_val_t;
            int16x4_t pi2_src_15_src_strd_val_t;
            int16x4_t pi2_src_17_src_strd_val_t;
            int16x4_t pi2_src_19_src_strd_val_t;
            int16x4_t pi2_src_21_src_strd_val_t;
            int16x4_t pi2_src_23_src_strd_val_t;
            int16x4_t pi2_src_25_src_strd_val_t;
            int16x4_t pi2_src_27_src_strd_val_t;
            int16x4_t pi2_src_29_src_strd_val_t;
            int16x4_t pi2_src_31_src_strd_val_t;

            int16x4_t pi2_src_10_src_strd_val_t;
            int16x4_t pi2_src_12_src_strd_val_t;
            int16x4_t pi2_src_14_src_strd_val_t;
            int16x4_t pi2_src_18_src_strd_val_t;
            int16x4_t pi2_src_20_src_strd_val_t;
            int16x4_t pi2_src_22_src_strd_val_t;
            int16x4_t pi2_src_26_src_strd_val_t;
            int16x4_t pi2_src_28_src_strd_val_t;
            int16x4_t pi2_src_30_src_strd_val_t;

            int16x8_t g_ai2_ihevc_trans_32_val_10;
            int16x4_t g_ai2_ihevc_trans_32_val_12;
            int16x8_t g_ai2_ihevc_trans_32_val_14;
            int16x8_t g_ai2_ihevc_trans_32_val_18;
            int16x4_t g_ai2_ihevc_trans_32_val_20;
            int16x8_t g_ai2_ihevc_trans_32_val_22;
            int16x8_t g_ai2_ihevc_trans_32_val_26;
            int16x4_t g_ai2_ihevc_trans_32_val_28;
            int16x8_t g_ai2_ihevc_trans_32_val_30;

            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_10 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_14 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_18 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_22 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_26 = vld1q_s16(g_ai2_ihevc_trans_32_2);
            g_ai2_ihevc_trans_32_2 += 128;
            g_ai2_ihevc_trans_32_val_30 = vld1q_s16(g_ai2_ihevc_trans_32_2);

            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_12 = vld1_s16(g_ai2_ihevc_trans_32_4);
            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_20 = vld1_s16(g_ai2_ihevc_trans_32_4);
            g_ai2_ihevc_trans_32_4 += 256;
            g_ai2_ihevc_trans_32_val_28 = vld1_s16(g_ai2_ihevc_trans_32_4);

            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];
                g_ai2_ihevc_trans_32_9 = (WORD16 *)&g_ai2_ihevc_trans_32[9][0];
                g_ai2_ihevc_trans_32_11 = (WORD16 *)&g_ai2_ihevc_trans_32[11][0];
                g_ai2_ihevc_trans_32_13 = (WORD16 *)&g_ai2_ihevc_trans_32[13][0];
                g_ai2_ihevc_trans_32_15 = (WORD16 *)&g_ai2_ihevc_trans_32[15][0];
                g_ai2_ihevc_trans_32_17 = (WORD16 *)&g_ai2_ihevc_trans_32[17][0];
                g_ai2_ihevc_trans_32_19 = (WORD16 *)&g_ai2_ihevc_trans_32[19][0];
                g_ai2_ihevc_trans_32_21 = (WORD16 *)&g_ai2_ihevc_trans_32[21][0];
                g_ai2_ihevc_trans_32_23 = (WORD16 *)&g_ai2_ihevc_trans_32[23][0];
                g_ai2_ihevc_trans_32_25 = (WORD16 *)&g_ai2_ihevc_trans_32[25][0];
                g_ai2_ihevc_trans_32_27 = (WORD16 *)&g_ai2_ihevc_trans_32[27][0];
                g_ai2_ihevc_trans_32_29 = (WORD16 *)&g_ai2_ihevc_trans_32[29][0];
                g_ai2_ihevc_trans_32_31 = (WORD16 *)&g_ai2_ihevc_trans_32[31][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_17_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_19_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_21_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_23_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_25_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_27_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_29_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_31_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_9 = vld1_s16(g_ai2_ihevc_trans_32_9);
                        g_ai2_ihevc_trans_32_9 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_11 = vld1_s16(g_ai2_ihevc_trans_32_11);
                        g_ai2_ihevc_trans_32_11 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_9, pi2_src_9_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_13 = vld1_s16(g_ai2_ihevc_trans_32_13);
                        g_ai2_ihevc_trans_32_13 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_11, pi2_src_11_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_15 = vld1_s16(g_ai2_ihevc_trans_32_15);
                        g_ai2_ihevc_trans_32_15 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_13, pi2_src_13_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_17 = vld1_s16(g_ai2_ihevc_trans_32_17);
                        g_ai2_ihevc_trans_32_17 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_15, pi2_src_15_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_19 = vld1_s16(g_ai2_ihevc_trans_32_19);
                        g_ai2_ihevc_trans_32_19 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_17, pi2_src_17_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_21 = vld1_s16(g_ai2_ihevc_trans_32_21);
                        g_ai2_ihevc_trans_32_21 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_19, pi2_src_19_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_23 = vld1_s16(g_ai2_ihevc_trans_32_23);
                        g_ai2_ihevc_trans_32_23 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_21, pi2_src_21_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_25 = vld1_s16(g_ai2_ihevc_trans_32_25);
                        g_ai2_ihevc_trans_32_25 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_23, pi2_src_23_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_27 = vld1_s16(g_ai2_ihevc_trans_32_27);
                        g_ai2_ihevc_trans_32_27 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_25, pi2_src_25_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_29 = vld1_s16(g_ai2_ihevc_trans_32_29);
                        g_ai2_ihevc_trans_32_29 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_27, pi2_src_27_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_31 = vld1_s16(g_ai2_ihevc_trans_32_31);
                        g_ai2_ihevc_trans_32_31 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_29, pi2_src_29_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_31, pi2_src_31_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
        //            for(k = 0; k < 8; k++)
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_18_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_22_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_26_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_30_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_12_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                    }
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_20_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_12, pi2_src_12_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_28_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_20, pi2_src_20_src_strd_val_t);

                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_28, pi2_src_28_src_strd_val_t);
                    }

                    eeeo[0] = g_ai2_ihevc_trans_32[8][0] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][0]
                                                    * pi2_tmp[24 * trans_size];
                    eeeo[1] = g_ai2_ihevc_trans_32[8][1] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][1]
                                                    * pi2_tmp[24 * trans_size];
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][0]
                                                    * pi2_tmp[16 * trans_size];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][1]
                                                    * pi2_tmp[16 * trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }

                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
    else  /* All rows of input are non-zero */
    {
        /* Inverse Transform 1st stage */

        WORD16 *g_ai2_ihevc_trans_32_1;
        WORD16 *g_ai2_ihevc_trans_32_2;
        WORD16 *g_ai2_ihevc_trans_32_3;
        WORD16 *g_ai2_ihevc_trans_32_4;
        WORD16 *g_ai2_ihevc_trans_32_5;
        WORD16 *g_ai2_ihevc_trans_32_7;
        WORD16 *g_ai2_ihevc_trans_32_9;
        WORD16 *g_ai2_ihevc_trans_32_11;
        WORD16 *g_ai2_ihevc_trans_32_13;
        WORD16 *g_ai2_ihevc_trans_32_15;
        WORD16 *g_ai2_ihevc_trans_32_17;
        WORD16 *g_ai2_ihevc_trans_32_19;
        WORD16 *g_ai2_ihevc_trans_32_21;
        WORD16 *g_ai2_ihevc_trans_32_23;
        WORD16 *g_ai2_ihevc_trans_32_25;
        WORD16 *g_ai2_ihevc_trans_32_27;
        WORD16 *g_ai2_ihevc_trans_32_29;
        WORD16 *g_ai2_ihevc_trans_32_31;

        int16x4_t g_ai2_ihevc_trans_32_val_1;
        int16x4_t g_ai2_ihevc_trans_32_val_3;
        int16x4_t g_ai2_ihevc_trans_32_val_5;
        int16x4_t g_ai2_ihevc_trans_32_val_7;
        int16x4_t g_ai2_ihevc_trans_32_val_9;
        int16x4_t g_ai2_ihevc_trans_32_val_11;
        int16x4_t g_ai2_ihevc_trans_32_val_13;
        int16x4_t g_ai2_ihevc_trans_32_val_15;
        int16x4_t g_ai2_ihevc_trans_32_val_17;
        int16x4_t g_ai2_ihevc_trans_32_val_19;
        int16x4_t g_ai2_ihevc_trans_32_val_21;
        int16x4_t g_ai2_ihevc_trans_32_val_23;
        int16x4_t g_ai2_ihevc_trans_32_val_25;
        int16x4_t g_ai2_ihevc_trans_32_val_27;
        int16x4_t g_ai2_ihevc_trans_32_val_29;
        int16x4_t g_ai2_ihevc_trans_32_val_31;

        WORD16 *pi2_src_tmp_1_src_strd;
        WORD16 *pi2_src_tmp_2_src_strd;
        WORD16 *pi2_src_tmp_4_src_strd;

        WORD16 pi2_src_1_src_strd_val;
        WORD16 pi2_src_2_src_strd_val;
        WORD16 pi2_src_4_src_strd_val;

        WORD32 eeo_val_0_init;

        int16x4_t pi2_src_1_src_strd_val_t;
        int16x4_t pi2_src_3_src_strd_val_t;
        int16x4_t pi2_src_5_src_strd_val_t;
        int16x4_t pi2_src_7_src_strd_val_t;
        int16x4_t pi2_src_9_src_strd_val_t;
        int16x4_t pi2_src_11_src_strd_val_t;
        int16x4_t pi2_src_13_src_strd_val_t;
        int16x4_t pi2_src_15_src_strd_val_t;
        int16x4_t pi2_src_17_src_strd_val_t;
        int16x4_t pi2_src_19_src_strd_val_t;
        int16x4_t pi2_src_21_src_strd_val_t;
        int16x4_t pi2_src_23_src_strd_val_t;
        int16x4_t pi2_src_25_src_strd_val_t;
        int16x4_t pi2_src_27_src_strd_val_t;
        int16x4_t pi2_src_29_src_strd_val_t;
        int16x4_t pi2_src_31_src_strd_val_t;

        int16x4_t pi2_src_2_src_strd_val_t;
        int16x4_t pi2_src_4_src_strd_val_t;
        int16x4_t pi2_src_6_src_strd_val_t;
        int16x4_t pi2_src_10_src_strd_val_t;
        int16x4_t pi2_src_12_src_strd_val_t;
        int16x4_t pi2_src_14_src_strd_val_t;
        int16x4_t pi2_src_18_src_strd_val_t;
        int16x4_t pi2_src_20_src_strd_val_t;
        int16x4_t pi2_src_22_src_strd_val_t;
        int16x4_t pi2_src_26_src_strd_val_t;
        int16x4_t pi2_src_28_src_strd_val_t;
        int16x4_t pi2_src_30_src_strd_val_t;

        int16x8_t g_ai2_ihevc_trans_32_val_2;
        int16x4_t g_ai2_ihevc_trans_32_val_4;
        int16x8_t g_ai2_ihevc_trans_32_val_6;
        int16x8_t g_ai2_ihevc_trans_32_val_10;
        int16x4_t g_ai2_ihevc_trans_32_val_12;
        int16x8_t g_ai2_ihevc_trans_32_val_14;
        int16x8_t g_ai2_ihevc_trans_32_val_18;
        int16x4_t g_ai2_ihevc_trans_32_val_20;
        int16x8_t g_ai2_ihevc_trans_32_val_22;
        int16x8_t g_ai2_ihevc_trans_32_val_26;
        int16x4_t g_ai2_ihevc_trans_32_val_28;
        int16x8_t g_ai2_ihevc_trans_32_val_30;

        int32x4_t o_val_0;
        int32x4_t o_val_1;
        int32x4_t o_val_2;
        int32x4_t o_val_3;
        int32x4_t e_val_0;
        int32x4_t e_val_1;
        int32x4_t e_val_2;
        int32x4_t e_val_3;
        int32x4_t eeo_val_0;
        int32x4_t eee_val_0 = vdupq_n_s32(0);
        int32x4_t eo_val_0;
        int32x4_t eo_val_1;
        int32x4_t ee_val_0;
        int32x4_t ee_val_1;
        WORD32 *o_val_ptr_1;
        int64_t rev_val_temp0;
        int64_t rev_val_temp1;
        int32x4_t e_add_o_val;
        int32x4_t e_sub_o_val;
        int32x4_t shift_val;
        int32x4_t shift_val_neg;
        shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
        shift_val_neg = vnegq_s32(shift_val);
        int16x4_t shift_res_1;
        UWORD16 *pu2_dst_tmp;
        WORD16 *pi2_tmp_str;


        int16x8_t constq_0 = vdupq_n_s16(0);
        int16x4_t const_0 = vdup_n_s16(0);
        int16x8_t wide_pu2_pred_val;
        int16x4_t res_s16;

        UWORD16 *pu2_pred_tmp_0_pred_strd;

        o_val_ptr_1 = &o[0];
        g_ai2_ihevc_trans_32_2 = (WORD16 *)&g_ai2_ihevc_trans_32[2][0];
        g_ai2_ihevc_trans_32_4 = (WORD16 *)&g_ai2_ihevc_trans_32[4][0];

        g_ai2_ihevc_trans_32_val_2 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_6 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_10 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_14 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_18 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_22 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_26 = vld1q_s16(g_ai2_ihevc_trans_32_2);
        g_ai2_ihevc_trans_32_2 += 128;
        g_ai2_ihevc_trans_32_val_30 = vld1q_s16(g_ai2_ihevc_trans_32_2);

        g_ai2_ihevc_trans_32_val_4 = vld1_s16(g_ai2_ihevc_trans_32_4);
        g_ai2_ihevc_trans_32_4 += 256;
        g_ai2_ihevc_trans_32_val_12 = vld1_s16(g_ai2_ihevc_trans_32_4);
        g_ai2_ihevc_trans_32_4 += 256;
        g_ai2_ihevc_trans_32_val_20 = vld1_s16(g_ai2_ihevc_trans_32_4);
        g_ai2_ihevc_trans_32_4 += 256;
        g_ai2_ihevc_trans_32_val_28 = vld1_s16(g_ai2_ihevc_trans_32_4);

        for(j = 0; j < row_limit_2nd_stage; j++)
        {
            pi2_tmp_str = pi2_tmp;
            pi2_src_tmp_1_src_strd = pi2_src + src_strd;
            pi2_src_tmp_2_src_strd = pi2_src + 2 * src_strd;
            pi2_src_tmp_4_src_strd = pi2_src + 4 * src_strd;

            g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
            g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
            g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
            g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];
            g_ai2_ihevc_trans_32_9 = (WORD16 *)&g_ai2_ihevc_trans_32[9][0];
            g_ai2_ihevc_trans_32_11 = (WORD16 *)&g_ai2_ihevc_trans_32[11][0];
            g_ai2_ihevc_trans_32_13 = (WORD16 *)&g_ai2_ihevc_trans_32[13][0];
            g_ai2_ihevc_trans_32_15 = (WORD16 *)&g_ai2_ihevc_trans_32[15][0];
            g_ai2_ihevc_trans_32_17 = (WORD16 *)&g_ai2_ihevc_trans_32[17][0];
            g_ai2_ihevc_trans_32_19 = (WORD16 *)&g_ai2_ihevc_trans_32[19][0];
            g_ai2_ihevc_trans_32_21 = (WORD16 *)&g_ai2_ihevc_trans_32[21][0];
            g_ai2_ihevc_trans_32_23 = (WORD16 *)&g_ai2_ihevc_trans_32[23][0];
            g_ai2_ihevc_trans_32_25 = (WORD16 *)&g_ai2_ihevc_trans_32[25][0];
            g_ai2_ihevc_trans_32_27 = (WORD16 *)&g_ai2_ihevc_trans_32[27][0];
            g_ai2_ihevc_trans_32_29 = (WORD16 *)&g_ai2_ihevc_trans_32[29][0];
            g_ai2_ihevc_trans_32_31 = (WORD16 *)&g_ai2_ihevc_trans_32[31][0];

            /* Checking for Zero Cols */
            if((zero_cols & 0x1) == 0x1)
            {
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
                pi2_tmp_str += 8;
                vst1q_s16(pi2_tmp_str, constq_0);
            }
            else
            {
                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_17_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_19_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_21_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_23_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_25_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_27_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_tmp_1_src_strd += 2 * src_strd;
                pi2_src_29_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                pi2_src_31_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                for(k = 16; k > 0; k -= 4)
                {
                    g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                    g_ai2_ihevc_trans_32_1 += 4;

                    g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                    g_ai2_ihevc_trans_32_3 += 4;
                    o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                    g_ai2_ihevc_trans_32_5 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                    g_ai2_ihevc_trans_32_7 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_9 = vld1_s16(g_ai2_ihevc_trans_32_9);
                    g_ai2_ihevc_trans_32_9 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_11 = vld1_s16(g_ai2_ihevc_trans_32_11);
                    g_ai2_ihevc_trans_32_11 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_9, pi2_src_9_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_13 = vld1_s16(g_ai2_ihevc_trans_32_13);
                    g_ai2_ihevc_trans_32_13 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_11, pi2_src_11_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_15 = vld1_s16(g_ai2_ihevc_trans_32_15);
                    g_ai2_ihevc_trans_32_15 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_13, pi2_src_13_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_17 = vld1_s16(g_ai2_ihevc_trans_32_17);
                    g_ai2_ihevc_trans_32_17 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_15, pi2_src_15_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_19 = vld1_s16(g_ai2_ihevc_trans_32_19);
                    g_ai2_ihevc_trans_32_19 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_17, pi2_src_17_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_21 = vld1_s16(g_ai2_ihevc_trans_32_21);
                    g_ai2_ihevc_trans_32_21 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_19, pi2_src_19_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_23 = vld1_s16(g_ai2_ihevc_trans_32_23);
                    g_ai2_ihevc_trans_32_23 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_21, pi2_src_21_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_25 = vld1_s16(g_ai2_ihevc_trans_32_25);
                    g_ai2_ihevc_trans_32_25 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_23, pi2_src_23_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_27 = vld1_s16(g_ai2_ihevc_trans_32_27);
                    g_ai2_ihevc_trans_32_27 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_25, pi2_src_25_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_29 = vld1_s16(g_ai2_ihevc_trans_32_29);
                    g_ai2_ihevc_trans_32_29 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_27, pi2_src_27_src_strd_val_t);

                    g_ai2_ihevc_trans_32_val_31 = vld1_s16(g_ai2_ihevc_trans_32_31);
                    g_ai2_ihevc_trans_32_31 += 4;
                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_29, pi2_src_29_src_strd_val_t);

                    o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_31, pi2_src_31_src_strd_val_t);

                    vst1q_s32(o_val_ptr_1, o_val_0);
                    o_val_ptr_1 += 4;
                }
                {
                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_18_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_22_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_tmp_2_src_strd += 4 * src_strd;
                    pi2_src_26_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                    pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                    pi2_src_30_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);
                    eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                    eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                    pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                    pi2_src_tmp_4_src_strd += 8 * src_strd;
                    pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);

                    pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                    pi2_src_tmp_4_src_strd += 8 * src_strd;
                    pi2_src_12_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                    eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                }
                {
                    eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);

                    pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                    pi2_src_tmp_4_src_strd += 8 * src_strd;
                    pi2_src_20_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                    eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_12, pi2_src_12_src_strd_val_t);

                    pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                    pi2_src_28_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                    eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_20, pi2_src_20_src_strd_val_t);

                    eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_28, pi2_src_28_src_strd_val_t);
                }

                eeeo[0] = g_ai2_ihevc_trans_32[8][0] * pi2_src[8 * src_strd]
                                + g_ai2_ihevc_trans_32[24][0]
                                                * pi2_src[24 * src_strd];
                eeeo[1] = g_ai2_ihevc_trans_32[8][1] * pi2_src[8 * src_strd]
                                + g_ai2_ihevc_trans_32[24][1]
                                                * pi2_src[24 * src_strd];
                eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_src[0]
                                + g_ai2_ihevc_trans_32[16][0]
                                                * pi2_src[16 * src_strd];
                eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_src[0]
                                + g_ai2_ihevc_trans_32[16][1]
                                                * pi2_src[16 * src_strd];

                /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                eee[0] = eeee[0] + eeeo[0];
                eee[3] = eeee[0] - eeeo[0];
                eee[1] = eeee[1] + eeeo[1];
                eee[2] = eeee[1] - eeeo[1];

                eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                o_val_ptr_1 -= 16;

                {
                    ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                    ee_val_1 = vrev64q_s32(ee_val_1);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                    ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                }
                {
                    e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                    o_val_0 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                    o_val_1 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_1 = vaddq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;
                    e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_val_2 = vsubq_s32(ee_val_1, eo_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_val_2 = vrev64q_s32(e_val_2);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                    o_val_2 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                    e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                    e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                    e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);
                    e_val_3 = vrev64q_s32(e_val_3);

                    rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                    o_val_3 = vld1q_s32(o_val_ptr_1);
                    o_val_ptr_1 += 4;
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                    e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                }
                {
                    e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                    e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_add_o_val);

                    e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                    vst1_s16(pi2_tmp_str, shift_res_1);
                    pi2_tmp_str += 4;

                    e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                    shift_res_1 = vqmovn_s32(e_sub_o_val);
                    shift_res_1 = vrev64_s16(shift_res_1);

                    vst1_s16(pi2_tmp_str, shift_res_1);
                    o_val_ptr_1 -= 16;
                }
            }
            pi2_src++;
            pi2_tmp += trans_size;
            zero_cols = zero_cols >> 1;
        }

        pi2_tmp = pi2_tmp_orig;

        /* Inverse Transform 2nd stage */
        shift_val = vdupq_n_s32(20 - u1_bit_depth);
        shift_val_neg = vnegq_s32(shift_val);

        if((zero_rows_2nd_stage & 0xFFFFFFF0) == 0xFFFFFFF0) /* First 4 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;

                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                    }
                    {
                        eeo_val_0_init = 0;
                        eeo_val_0 = vdupq_n_s32(eeo_val_0_init);

                    }
                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else if((zero_rows_2nd_stage & 0xFFFFFF00) == 0xFFFFFF00) /* First 8 rows of output of 1st stage are non-zero */
        {
            for(j = 0; j < trans_size; j++)
            {
                /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                pu2_dst_tmp = pu2_dst;
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);

                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                    }
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);
                    }

                    eeeo[0] = 0;
                    eeeo[1] = 0;
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
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
                pi2_src_tmp_1_src_strd = pi2_tmp + trans_size;
                pi2_src_tmp_2_src_strd = pi2_tmp + 2 * trans_size;
                pi2_src_tmp_4_src_strd = pi2_tmp + 4 * trans_size;
                pu2_pred_tmp_0_pred_strd = pu2_pred;

                g_ai2_ihevc_trans_32_1 = (WORD16 *)&g_ai2_ihevc_trans_32[1][0];
                g_ai2_ihevc_trans_32_3 = (WORD16 *)&g_ai2_ihevc_trans_32[3][0];
                g_ai2_ihevc_trans_32_5 = (WORD16 *)&g_ai2_ihevc_trans_32[5][0];
                g_ai2_ihevc_trans_32_7 = (WORD16 *)&g_ai2_ihevc_trans_32[7][0];
                g_ai2_ihevc_trans_32_9 = (WORD16 *)&g_ai2_ihevc_trans_32[9][0];
                g_ai2_ihevc_trans_32_11 = (WORD16 *)&g_ai2_ihevc_trans_32[11][0];
                g_ai2_ihevc_trans_32_13 = (WORD16 *)&g_ai2_ihevc_trans_32[13][0];
                g_ai2_ihevc_trans_32_15 = (WORD16 *)&g_ai2_ihevc_trans_32[15][0];
                g_ai2_ihevc_trans_32_17 = (WORD16 *)&g_ai2_ihevc_trans_32[17][0];
                g_ai2_ihevc_trans_32_19 = (WORD16 *)&g_ai2_ihevc_trans_32[19][0];
                g_ai2_ihevc_trans_32_21 = (WORD16 *)&g_ai2_ihevc_trans_32[21][0];
                g_ai2_ihevc_trans_32_23 = (WORD16 *)&g_ai2_ihevc_trans_32[23][0];
                g_ai2_ihevc_trans_32_25 = (WORD16 *)&g_ai2_ihevc_trans_32[25][0];
                g_ai2_ihevc_trans_32_27 = (WORD16 *)&g_ai2_ihevc_trans_32[27][0];
                g_ai2_ihevc_trans_32_29 = (WORD16 *)&g_ai2_ihevc_trans_32[29][0];
                g_ai2_ihevc_trans_32_31 = (WORD16 *)&g_ai2_ihevc_trans_32[31][0];

                {
                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_1_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_3_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_5_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_7_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_9_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_11_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_13_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_15_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_17_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_19_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_21_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_23_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_25_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_27_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_tmp_1_src_strd += 2 * trans_size;
                    pi2_src_29_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    pi2_src_1_src_strd_val = pi2_src_tmp_1_src_strd[0];
                    pi2_src_31_src_strd_val_t = vdup_n_s16(pi2_src_1_src_strd_val);

                    /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
                    for(k = 16; k > 0; k -= 4)
                    {
                        g_ai2_ihevc_trans_32_val_1 = vld1_s16(g_ai2_ihevc_trans_32_1);
                        g_ai2_ihevc_trans_32_1 += 4;

                        g_ai2_ihevc_trans_32_val_3 = vld1_s16(g_ai2_ihevc_trans_32_3);
                        g_ai2_ihevc_trans_32_3 += 4;
                        o_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_1, pi2_src_1_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_5 = vld1_s16(g_ai2_ihevc_trans_32_5);
                        g_ai2_ihevc_trans_32_5 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_3, pi2_src_3_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_7 = vld1_s16(g_ai2_ihevc_trans_32_7);
                        g_ai2_ihevc_trans_32_7 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_5, pi2_src_5_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_9 = vld1_s16(g_ai2_ihevc_trans_32_9);
                        g_ai2_ihevc_trans_32_9 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_7, pi2_src_7_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_11 = vld1_s16(g_ai2_ihevc_trans_32_11);
                        g_ai2_ihevc_trans_32_11 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_9, pi2_src_9_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_13 = vld1_s16(g_ai2_ihevc_trans_32_13);
                        g_ai2_ihevc_trans_32_13 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_11, pi2_src_11_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_15 = vld1_s16(g_ai2_ihevc_trans_32_15);
                        g_ai2_ihevc_trans_32_15 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_13, pi2_src_13_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_17 = vld1_s16(g_ai2_ihevc_trans_32_17);
                        g_ai2_ihevc_trans_32_17 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_15, pi2_src_15_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_19 = vld1_s16(g_ai2_ihevc_trans_32_19);
                        g_ai2_ihevc_trans_32_19 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_17, pi2_src_17_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_21 = vld1_s16(g_ai2_ihevc_trans_32_21);
                        g_ai2_ihevc_trans_32_21 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_19, pi2_src_19_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_23 = vld1_s16(g_ai2_ihevc_trans_32_23);
                        g_ai2_ihevc_trans_32_23 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_21, pi2_src_21_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_25 = vld1_s16(g_ai2_ihevc_trans_32_25);
                        g_ai2_ihevc_trans_32_25 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_23, pi2_src_23_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_27 = vld1_s16(g_ai2_ihevc_trans_32_27);
                        g_ai2_ihevc_trans_32_27 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_25, pi2_src_25_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_29 = vld1_s16(g_ai2_ihevc_trans_32_29);
                        g_ai2_ihevc_trans_32_29 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_27, pi2_src_27_src_strd_val_t);

                        g_ai2_ihevc_trans_32_val_31 = vld1_s16(g_ai2_ihevc_trans_32_31);
                        g_ai2_ihevc_trans_32_31 += 4;
                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_29, pi2_src_29_src_strd_val_t);

                        o_val_0 = vmlal_s16(o_val_0, g_ai2_ihevc_trans_32_val_31, pi2_src_31_src_strd_val_t);

                        vst1q_s32(o_val_ptr_1, o_val_0);
                        o_val_ptr_1 += 4;
                    }
        //            for(k = 0; k < 8; k++)
                    {
                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_2_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_6_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmull_s16(vget_low_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_10_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_14_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_18_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_22_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_tmp_2_src_strd += 4 * trans_size;
                        pi2_src_26_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_2_src_strd_val = pi2_src_tmp_2_src_strd[0];
                        pi2_src_30_src_strd_val_t = vdup_n_s16(pi2_src_2_src_strd_val);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);
                        eo_val_0 = vmlal_s16(eo_val_0, vget_low_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                        eo_val_1 = vmull_s16(vget_high_s16(g_ai2_ihevc_trans_32_val_2), pi2_src_2_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_6), pi2_src_6_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_10), pi2_src_10_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_14), pi2_src_14_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_18), pi2_src_18_src_strd_val_t);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_22), pi2_src_22_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_4_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_26), pi2_src_26_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_12_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eo_val_1 = vmlal_s16(eo_val_1, vget_high_s16(g_ai2_ihevc_trans_32_val_30), pi2_src_30_src_strd_val_t);

                    }
                    {
                        eeo_val_0 = vmull_s16(g_ai2_ihevc_trans_32_val_4, pi2_src_4_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_tmp_4_src_strd += 8 * trans_size;
                        pi2_src_20_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_12, pi2_src_12_src_strd_val_t);

                        pi2_src_4_src_strd_val = pi2_src_tmp_4_src_strd[0];
                        pi2_src_28_src_strd_val_t = vdup_n_s16(pi2_src_4_src_strd_val);
                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_20, pi2_src_20_src_strd_val_t);

                        eeo_val_0 = vmlal_s16(eeo_val_0, g_ai2_ihevc_trans_32_val_28, pi2_src_28_src_strd_val_t);
                    }

                    eeeo[0] = g_ai2_ihevc_trans_32[8][0] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][0]
                                                    * pi2_tmp[24 * trans_size];
                    eeeo[1] = g_ai2_ihevc_trans_32[8][1] * pi2_tmp[8 * trans_size]
                                    + g_ai2_ihevc_trans_32[24][1]
                                                    * pi2_tmp[24 * trans_size];
                    eeee[0] = g_ai2_ihevc_trans_32[0][0] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][0]
                                                    * pi2_tmp[16 * trans_size];
                    eeee[1] = g_ai2_ihevc_trans_32[0][1] * pi2_tmp[0]
                                    + g_ai2_ihevc_trans_32[16][1]
                                                    * pi2_tmp[16 * trans_size];

                    /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
                    eee[0] = eeee[0] + eeeo[0];
                    eee[3] = eeee[0] - eeeo[0];
                    eee[1] = eeee[1] + eeeo[1];
                    eee[2] = eeee[1] - eeeo[1];

                    eee_val_0 = vsetq_lane_s32(eee[0], eee_val_0, 0);
                    eee_val_0 = vsetq_lane_s32(eee[1], eee_val_0, 1);
                    eee_val_0 = vsetq_lane_s32(eee[2], eee_val_0, 2);
                    eee_val_0 = vsetq_lane_s32(eee[3], eee_val_0, 3);
                    o_val_ptr_1 -= 16;

                    {
                        ee_val_0 = vaddq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vsubq_s32(eee_val_0, eeo_val_0);
                        ee_val_1 = vrev64q_s32(ee_val_1);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(ee_val_1), 1);

                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(ee_val_1), 1));
                        ee_val_1 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(ee_val_1), 0));
                    }
                    {
                        e_val_0 = vaddq_s32(ee_val_0, eo_val_0);
                        o_val_0 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vaddq_s32(e_val_0, o_val_0);
                        o_val_1 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_1 = vaddq_s32(ee_val_1, eo_val_1);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;
                        e_add_o_val = vaddq_s32(e_val_1, o_val_1);

                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);

                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_val_2 = vsubq_s32(ee_val_1, eo_val_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_val_2 = vrev64q_s32(e_val_2);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 0);
                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_2), 1);

                        o_val_2 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_2), 1));
                        e_val_2 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_2), 0));

                        e_val_3 = vsubq_s32(ee_val_0, eo_val_0);
                        e_add_o_val = vaddq_s32(e_val_2, o_val_2);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);
                        e_val_3 = vrev64q_s32(e_val_3);

                        rev_val_temp0 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 0);

                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        rev_val_temp1 = vgetq_lane_s64(vreinterpretq_s64_s32(e_val_3), 1);

                        o_val_3 = vld1q_s32(o_val_ptr_1);
                        o_val_ptr_1 += 4;
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp0, vreinterpretq_s64_s32(e_val_3), 1));
                        e_val_3 = vreinterpretq_s32_s64(vsetq_lane_s64(rev_val_temp1, vreinterpretq_s64_s32(e_val_3), 0));
                    }
                    {
                        e_add_o_val = vaddq_s32(e_val_3, o_val_3);
                        e_add_o_val = vrshlq_s32(e_add_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_add_o_val);

                        e_sub_o_val = vsubq_s32(e_val_3, o_val_3);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_2, o_val_2);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_1, o_val_1);
                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        e_sub_o_val = vsubq_s32(e_val_0, o_val_0);
                        wide_pu2_pred_val = vreinterpretq_s16_u16(vld1q_u16(pu2_pred_tmp_0_pred_strd));
                        pu2_pred_tmp_0_pred_strd += 8;
                        res_s16 = vqadd_s16(shift_res_1, vget_low_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        pu2_dst_tmp += 4;

                        e_sub_o_val = vrshlq_s32(e_sub_o_val, shift_val_neg);
                        shift_res_1 = vqmovn_s32(e_sub_o_val);
                        shift_res_1 = vrev64_s16(shift_res_1);

                        res_s16 = vqadd_s16(shift_res_1, vget_high_s16(wide_pu2_pred_val));
                        res_s16 = vmin_s16(vmax_s16(res_s16, const_0), v_clip_limit);
                        vst1_u16(pu2_dst_tmp, vreinterpret_u16_s16(res_s16));
                        o_val_ptr_1 -= 16;
                    }
                }
                pi2_tmp++;
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
    }
}
