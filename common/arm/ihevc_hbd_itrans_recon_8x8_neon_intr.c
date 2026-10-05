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
 *  ihevc_hbd_itrans_recon_8x8_neon_intr.c
 *
 * @brief
 *  Contains function definitions for HBD 8x8 inverse transform and
 *  reconstruction using ARM NEON intrinsics
 *
 * @author
 *  Ittiam
 *
 * @par List of Functions:
 *  - ihevc_hbd_itrans_recon_8x8_neonintr()
 *
 * @remarks
 *  None
 *
 ******************************************************************************
 */
#include <stdio.h>
#include <string.h>
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
 *  This function performs HBD Inverse transform and reconstruction for 8x8
 *  input block using ARM NEON intrinsics
 *
 * @par Description:
 *  Performs inverse transform and adds the prediction data and clips output
 *  to the configured bit_depth
 *
 * @param[in] pi2_src
 *  Input 8x8 coefficients
 *
 * @param[in] pi2_tmp
 *  Temporary 8x8 buffer for storing inverse transform 1st stage output
 *
 * @param[in] pu2_pred
 *  Prediction 8x8 block
 *
 * @param[out] pu2_dst
 *  Output 8x8 block
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
void ihevc_hbd_itrans_recon_8x8_neonintr(WORD16 *pi2_src,
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
    WORD32 j;
    WORD16 *pi2_tmp_orig;
    WORD32 trans_size;
    WORD32 shift;
    WORD16 clip_limit;
    WORD32 zero_rows_2nd_stage = zero_cols;

    trans_size = TRANS_SIZE_8;
    pi2_tmp_orig = pi2_tmp;

    /* Inverse Transform 1st stage */
    WORD32 *g_ai2_ihevc_trans_8_0 = (WORD32 *)&g_ai2_ihevc_trans_8[0][0];
    WORD16 *g_ai2_ihevc_trans_8_1 = (WORD16 *)&g_ai2_ihevc_trans_8[1][0];
    WORD32 *g_ai2_ihevc_trans_8_2 = (WORD32 *)&g_ai2_ihevc_trans_8[2][0];
    WORD16 *g_ai2_ihevc_trans_8_3 = (WORD16 *)&g_ai2_ihevc_trans_8[3][0];
    WORD32 *g_ai2_ihevc_trans_8_4 = (WORD32 *)&g_ai2_ihevc_trans_8[4][0];
    WORD16 *g_ai2_ihevc_trans_8_5 = (WORD16 *)&g_ai2_ihevc_trans_8[5][0];
    WORD32 *g_ai2_ihevc_trans_8_6 = (WORD32 *)&g_ai2_ihevc_trans_8[6][0];
    WORD16 *g_ai2_ihevc_trans_8_7 = (WORD16 *)&g_ai2_ihevc_trans_8[7][0];

    int32x2_t g_ai2_ihevc_trans_8_val_0_2 = vdup_n_s32(0);
    g_ai2_ihevc_trans_8_val_0_2 = vld1_lane_s32(g_ai2_ihevc_trans_8_0, g_ai2_ihevc_trans_8_val_0_2, 0);
    g_ai2_ihevc_trans_8_val_0_2 = vld1_lane_s32(g_ai2_ihevc_trans_8_2, g_ai2_ihevc_trans_8_val_0_2, 1);

    int32x2_t g_ai2_ihevc_trans_8_val_4_6 = vdup_n_s32(0);
    g_ai2_ihevc_trans_8_val_4_6 = vld1_lane_s32(g_ai2_ihevc_trans_8_4, g_ai2_ihevc_trans_8_val_4_6, 0);
    g_ai2_ihevc_trans_8_val_4_6 = vld1_lane_s32(g_ai2_ihevc_trans_8_6, g_ai2_ihevc_trans_8_val_4_6, 1);

    int16x4_t g_ai2_ihevc_trans_8_val_1 = vld1_s16(g_ai2_ihevc_trans_8_1);
    int16x4_t g_ai2_ihevc_trans_8_val_3 = vld1_s16(g_ai2_ihevc_trans_8_3);
    int16x4_t g_ai2_ihevc_trans_8_val_5 = vld1_s16(g_ai2_ihevc_trans_8_5);
    int16x4_t g_ai2_ihevc_trans_8_val_7 = vld1_s16(g_ai2_ihevc_trans_8_7);

    WORD16 *pi2_src_tmp_0_src_strd = pi2_src;
    int16x4_t pi2_src_0_src_strd_val;
    WORD16 *pi2_src_tmp_1_src_strd;
    int16x4_t pi2_src_1_src_strd_val;
    WORD16 *pi2_src_tmp_2_src_strd;
    int16x4_t pi2_src_2_src_strd_val;
    WORD16 *pi2_src_tmp_3_src_strd;
    int16x4_t pi2_src_3_src_strd_val;
    WORD16 *pi2_src_tmp_4_src_strd;
    int16x4_t pi2_src_4_src_strd_val;
    WORD16 *pi2_src_tmp_5_src_strd;
    int16x4_t pi2_src_5_src_strd_val;
    WORD16 *pi2_src_tmp_6_src_strd;
    int16x4_t pi2_src_6_src_strd_val;
    WORD16 *pi2_src_tmp_7_src_strd;
    int16x4_t pi2_src_7_src_strd_val;

    int32x4_t o_0_val;
    int32x4_t o_1_val;
    int32x4_t o_2_val;
    int32x4_t o_3_val;
    int32x4_t eo_0_val;
    int32x4_t eo_1_val;
    int32x4_t ee_0_val;
    int32x4_t ee_1_val;
    int32x4_t e_0_val;
    int32x4_t e_1_val;
    int32x4_t e_2_val;
    int32x4_t e_3_val;
    int32x4_t e_add_o_val_0;
    int32x4_t e_sub_o_val_0;
    int32x4_t e_add_o_val_1;
    int32x4_t e_sub_o_val_1;
    int32x4_t e_add_o_val_2;
    int32x4_t e_sub_o_val_2;
    int32x4_t e_add_o_val_3;
    int32x4_t e_sub_o_val_3;
    int32x4_t shift_val = vdupq_n_s32(IT_SHIFT_STAGE_1);
    int32x4_t shift_val_neg = vnegq_s32(shift_val);
    WORD16 *pi2_tmp_0;
    WORD16 *pi2_tmp_4;
    UWORD16 *pu2_dst_tmp_0_dst_strd = pu2_dst;
    UWORD16 *pu2_dst_tmp_0_4_dst_strd = pu2_dst + 4;
    int16x4_t shift_res_0;
    int16x4_t shift_res_1;
    int16x4_t shift_res_2;
    int16x4_t shift_res_3;
    int16x8_t constq_0;
    int16x4_t const_0;
    int32x4x2_t temp_unzip_1;
    int32x4x2_t temp_unzip_2;
    int32x4x2_t transpose_val_32q_1;
    int32x4x2_t transpose_val_32q_2;

    WORD16 *pi2_tmp_0_trans_size;
    WORD16 *pi2_tmp_1_trans_size;
    WORD16 *pi2_tmp_2_trans_size;
    WORD16 *pi2_tmp_3_trans_size;
    WORD16 *pi2_tmp_4_trans_size;
    WORD16 *pi2_tmp_5_trans_size;
    WORD16 *pi2_tmp_6_trans_size;
    WORD16 *pi2_tmp_7_trans_size;

    UWORD16 *pu2_pred_tmp_0_pred_strd = pu2_pred;
    uint16x8_t pu2_pred_0_pred_strd_val;
    uint16x8_t pu2_pred_1_pred_strd_val;
    uint16x8_t pu2_pred_2_pred_strd_val;
    uint16x8_t pu2_pred_3_pred_strd_val;
    int16x4_t pred_add_val_t;
    int16x4_t dup_const_clip_limit;

    for(j = TRANS_SIZE_8; j > 0; j -= 4)
    {
        pi2_src_tmp_1_src_strd = pi2_src_tmp_0_src_strd + src_strd;
        pi2_src_tmp_2_src_strd = pi2_src_tmp_1_src_strd + src_strd;
        pi2_src_tmp_3_src_strd = pi2_src_tmp_2_src_strd + src_strd;
        pi2_src_tmp_4_src_strd = pi2_src_tmp_3_src_strd + src_strd;
        pi2_src_tmp_5_src_strd = pi2_src_tmp_4_src_strd + src_strd;
        pi2_src_tmp_6_src_strd = pi2_src_tmp_5_src_strd + src_strd;
        pi2_src_tmp_7_src_strd = pi2_src_tmp_6_src_strd + src_strd;

        pi2_tmp_0 = pi2_tmp;
        pi2_tmp_4 = pi2_tmp + 4;
        /* Checking for Zero Cols */
        if((zero_cols & 0xF) == 0xF)
        {
            constq_0 = vdupq_n_s16(0);
            vst1q_s16(pi2_tmp_0, constq_0);
            pi2_tmp_0 += trans_size;
            vst1q_s16(pi2_tmp_0, constq_0);
            pi2_tmp_0 += trans_size;
            vst1q_s16(pi2_tmp_0, constq_0);
            pi2_tmp_0 += trans_size;
            vst1q_s16(pi2_tmp_0, constq_0);
        }
        else if((zero_rows & 0xF0) == 0xF0)
        {
            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            pi2_src_1_src_strd_val = vld1_s16(pi2_src_tmp_1_src_strd);
            pi2_src_3_src_strd_val = vld1_s16(pi2_src_tmp_3_src_strd);
            o_0_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 0);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 0);

            pi2_src_2_src_strd_val = vld1_s16(pi2_src_tmp_2_src_strd);

            o_1_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 1);
            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 1);
            pi2_src_0_src_strd_val = vld1_s16(pi2_src_tmp_0_src_strd);

            o_2_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 2);

            o_3_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 3);

            eo_0_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 2);
            eo_1_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 3);

            ee_0_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 0);
            ee_1_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 1);

            /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
            e_0_val = vaddq_s32(ee_0_val, eo_0_val);
            e_2_val = vsubq_s32(ee_1_val, eo_1_val);

            e_1_val = vaddq_s32(ee_1_val, eo_1_val);
            temp_unzip_1 = vuzpq_s32(e_0_val, e_2_val);

            e_3_val = vsubq_s32(ee_0_val, eo_0_val);
            {
                temp_unzip_2 = vuzpq_s32(e_1_val, e_3_val);
                transpose_val_32q_1 = vtrnq_s32(temp_unzip_1.val[0], temp_unzip_2.val[0]);

                e_add_o_val_0 = vaddq_s32(transpose_val_32q_1.val[0], o_0_val);
                transpose_val_32q_2 = vtrnq_s32(temp_unzip_1.val[1], temp_unzip_2.val[1]);

                e_add_o_val_0 = vrshlq_s32(e_add_o_val_0, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_add_o_val_0);

                e_add_o_val_1 = vaddq_s32(transpose_val_32q_2.val[0], o_1_val);
                vst1_s16(pi2_tmp_0, shift_res_0);
                pi2_tmp_0 += trans_size;

                e_add_o_val_1 = vrshlq_s32(e_add_o_val_1, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val_1);

                e_add_o_val_2 = vaddq_s32(transpose_val_32q_1.val[1], o_2_val);
                vst1_s16(pi2_tmp_0, shift_res_1);
                pi2_tmp_0 += trans_size;

                e_add_o_val_2 = vrshlq_s32(e_add_o_val_2, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_add_o_val_2);

                e_add_o_val_3 = vaddq_s32(transpose_val_32q_2.val[1], o_3_val);
                vst1_s16(pi2_tmp_0, shift_res_2);
                pi2_tmp_0 += trans_size;

                e_add_o_val_3 = vrshlq_s32(e_add_o_val_3, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_add_o_val_3);

                e_sub_o_val_3 = vsubq_s32(transpose_val_32q_1.val[0], o_0_val);
                vst1_s16(pi2_tmp_0, shift_res_3);

                e_sub_o_val_3 = vrshlq_s32(e_sub_o_val_3, shift_val_neg);

                shift_res_0 = vqmovn_s32(e_sub_o_val_3);
                e_sub_o_val_2 = vsubq_s32(transpose_val_32q_2.val[0], o_1_val);

                shift_res_0 = vrev64_s16(shift_res_0);
                e_sub_o_val_2 = vrshlq_s32(e_sub_o_val_2, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_0);
                pi2_tmp_4 += trans_size;

                shift_res_1 = vqmovn_s32(e_sub_o_val_2);

                e_sub_o_val_1 = vsubq_s32(transpose_val_32q_1.val[1], o_2_val);
                shift_res_1 = vrev64_s16(shift_res_1);

                e_sub_o_val_1 = vrshlq_s32(e_sub_o_val_1, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_1);
                pi2_tmp_4 += trans_size;

                shift_res_2 = vqmovn_s32(e_sub_o_val_1);

                e_sub_o_val_0 = vsubq_s32(transpose_val_32q_2.val[1], o_3_val);
                shift_res_2 = vrev64_s16(shift_res_2);

                e_sub_o_val_0 = vrshlq_s32(e_sub_o_val_0, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_2);
                pi2_tmp_4 += trans_size;

                shift_res_3 = vqmovn_s32(e_sub_o_val_0);
                shift_res_3 = vrev64_s16(shift_res_3);

                vst1_s16(pi2_tmp_4, shift_res_3);
            }
        }
        else
        {
            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            pi2_src_1_src_strd_val = vld1_s16(pi2_src_tmp_1_src_strd);
            pi2_src_3_src_strd_val = vld1_s16(pi2_src_tmp_3_src_strd);
            o_0_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 0);

            pi2_src_5_src_strd_val = vld1_s16(pi2_src_tmp_5_src_strd);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 0);

            pi2_src_7_src_strd_val = vld1_s16(pi2_src_tmp_7_src_strd);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 0);

            pi2_src_2_src_strd_val = vld1_s16(pi2_src_tmp_2_src_strd);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 0);

            o_1_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 1);
            pi2_src_6_src_strd_val = vld1_s16(pi2_src_tmp_6_src_strd);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 1);
            pi2_src_0_src_strd_val = vld1_s16(pi2_src_tmp_0_src_strd);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 1);
            pi2_src_4_src_strd_val = vld1_s16(pi2_src_tmp_4_src_strd);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 1);

            o_2_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 2);

            o_3_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 3);

            eo_0_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 2);
            eo_0_val = vmlal_lane_s16(eo_0_val, pi2_src_6_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 2);

            eo_1_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 3);
            eo_1_val = vmlal_lane_s16(eo_1_val, pi2_src_6_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 3);

            ee_0_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 0);
            ee_0_val = vmlal_lane_s16(ee_0_val, pi2_src_4_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 0);

            ee_1_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 1);
            ee_1_val = vmlal_lane_s16(ee_1_val, pi2_src_4_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 1);

            /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
            e_0_val = vaddq_s32(ee_0_val, eo_0_val);
            e_2_val = vsubq_s32(ee_1_val, eo_1_val);

            e_1_val = vaddq_s32(ee_1_val, eo_1_val);
            temp_unzip_1 = vuzpq_s32(e_0_val, e_2_val);

            e_3_val = vsubq_s32(ee_0_val, eo_0_val);
            {
                temp_unzip_2 = vuzpq_s32(e_1_val, e_3_val);
                transpose_val_32q_1 = vtrnq_s32(temp_unzip_1.val[0], temp_unzip_2.val[0]);

                e_add_o_val_0 = vaddq_s32(transpose_val_32q_1.val[0], o_0_val);
                transpose_val_32q_2 = vtrnq_s32(temp_unzip_1.val[1], temp_unzip_2.val[1]);

                e_add_o_val_0 = vrshlq_s32(e_add_o_val_0, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_add_o_val_0);

                e_add_o_val_1 = vaddq_s32(transpose_val_32q_2.val[0], o_1_val);
                vst1_s16(pi2_tmp_0, shift_res_0);
                pi2_tmp_0 += trans_size;

                e_add_o_val_1 = vrshlq_s32(e_add_o_val_1, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val_1);

                e_add_o_val_2 = vaddq_s32(transpose_val_32q_1.val[1], o_2_val);
                vst1_s16(pi2_tmp_0, shift_res_1);
                pi2_tmp_0 += trans_size;

                e_add_o_val_2 = vrshlq_s32(e_add_o_val_2, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_add_o_val_2);

                e_add_o_val_3 = vaddq_s32(transpose_val_32q_2.val[1], o_3_val);
                vst1_s16(pi2_tmp_0, shift_res_2);
                pi2_tmp_0 += trans_size;

                e_add_o_val_3 = vrshlq_s32(e_add_o_val_3, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_add_o_val_3);

                e_sub_o_val_3 = vsubq_s32(transpose_val_32q_1.val[0], o_0_val);
                vst1_s16(pi2_tmp_0, shift_res_3);

                e_sub_o_val_3 = vrshlq_s32(e_sub_o_val_3, shift_val_neg);

                shift_res_0 = vqmovn_s32(e_sub_o_val_3);
                e_sub_o_val_2 = vsubq_s32(transpose_val_32q_2.val[0], o_1_val);

                shift_res_0 = vrev64_s16(shift_res_0);
                e_sub_o_val_2 = vrshlq_s32(e_sub_o_val_2, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_0);
                pi2_tmp_4 += trans_size;

                shift_res_1 = vqmovn_s32(e_sub_o_val_2);

                e_sub_o_val_1 = vsubq_s32(transpose_val_32q_1.val[1], o_2_val);
                shift_res_1 = vrev64_s16(shift_res_1);

                e_sub_o_val_1 = vrshlq_s32(e_sub_o_val_1, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_1);
                pi2_tmp_4 += trans_size;

                shift_res_2 = vqmovn_s32(e_sub_o_val_1);

                e_sub_o_val_0 = vsubq_s32(transpose_val_32q_2.val[1], o_3_val);
                shift_res_2 = vrev64_s16(shift_res_2);

                e_sub_o_val_0 = vrshlq_s32(e_sub_o_val_0, shift_val_neg);
                vst1_s16(pi2_tmp_4, shift_res_2);
                pi2_tmp_4 += trans_size;

                shift_res_3 = vqmovn_s32(e_sub_o_val_0);
                shift_res_3 = vrev64_s16(shift_res_3);

                vst1_s16(pi2_tmp_4, shift_res_3);
            }
        }
        pi2_tmp += trans_size << 2;
        pi2_src_tmp_0_src_strd += 4;
        zero_cols = zero_cols >> 4;
    }

    pi2_tmp = pi2_tmp_orig;

    /* Inverse Transform 2nd stage */
    shift = 20 - u1_bit_depth;
    clip_limit = (1 << u1_bit_depth) - 1;
    shift_val = vdupq_n_s32(shift);
    shift_val_neg = vnegq_s32(shift_val);
    dup_const_clip_limit = vdup_n_s16(clip_limit);
    const_0 = vdup_n_s16(0);

    if((zero_rows_2nd_stage & 0xF0) == 0xF0) /* First 4 rows of output of 1st stage are non-zero */
    {
        for(j = trans_size; j > 0; j -= 4)
        {
            pi2_tmp_0_trans_size = pi2_tmp;
            pi2_tmp_1_trans_size = pi2_tmp + trans_size;
            pi2_tmp_2_trans_size = pi2_tmp + 2 * trans_size;
            pi2_tmp_3_trans_size = pi2_tmp_1_trans_size + 2 * trans_size;

            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            pi2_src_1_src_strd_val = vld1_s16(pi2_tmp_1_trans_size);
            pi2_src_3_src_strd_val = vld1_s16(pi2_tmp_3_trans_size);
            o_0_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 0);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 0);

            pi2_src_2_src_strd_val = vld1_s16(pi2_tmp_2_trans_size);

            o_1_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 1);
            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 1);
            pi2_src_0_src_strd_val = vld1_s16(pi2_tmp_0_trans_size);

            o_2_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 2);

            o_3_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 3);

            eo_0_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 2);
            eo_1_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 3);

            ee_0_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 0);
            ee_1_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 1);

            /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
            e_0_val = vaddq_s32(ee_0_val, eo_0_val);
            e_2_val = vsubq_s32(ee_1_val, eo_1_val);

            e_1_val = vaddq_s32(ee_1_val, eo_1_val);
            e_3_val = vsubq_s32(ee_0_val, eo_0_val);
            {
                temp_unzip_1 = vuzpq_s32(e_0_val, e_2_val);
                temp_unzip_2 = vuzpq_s32(e_1_val, e_3_val);
                transpose_val_32q_1 = vtrnq_s32(temp_unzip_1.val[0], temp_unzip_2.val[0]);
                transpose_val_32q_2 = vtrnq_s32(temp_unzip_1.val[1], temp_unzip_2.val[1]);

                e_add_o_val_0 = vaddq_s32(transpose_val_32q_1.val[0], o_0_val);
                e_add_o_val_0 = vrshlq_s32(e_add_o_val_0, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_add_o_val_0);

                pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_0, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_1 = vaddq_s32(transpose_val_32q_2.val[0], o_1_val);
                e_add_o_val_1 = vrshlq_s32(e_add_o_val_1, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val_1);

                pu2_pred_1_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_1_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_2 = vaddq_s32(transpose_val_32q_1.val[1], o_2_val);
                e_add_o_val_2 = vrshlq_s32(e_add_o_val_2, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_add_o_val_2);

                pu2_pred_2_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_2, vreinterpret_s16_u16(vget_low_u16(pu2_pred_2_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_3 = vaddq_s32(transpose_val_32q_2.val[1], o_3_val);
                e_add_o_val_3 = vrshlq_s32(e_add_o_val_3, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_add_o_val_3);

                pu2_pred_3_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_3, vreinterpret_s16_u16(vget_low_u16(pu2_pred_3_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_sub_o_val_3 = vsubq_s32(transpose_val_32q_1.val[0], o_0_val);
                e_sub_o_val_3 = vrshlq_s32(e_sub_o_val_3, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_sub_o_val_3);
                shift_res_0 = vrev64_s16(shift_res_0);

                pred_add_val_t = vqadd_s16(shift_res_0, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_2 = vsubq_s32(transpose_val_32q_2.val[0], o_1_val);
                e_sub_o_val_2 = vrshlq_s32(e_sub_o_val_2, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val_2);
                shift_res_1 = vrev64_s16(shift_res_1);

                pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_1_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_1 = vsubq_s32(transpose_val_32q_1.val[1], o_2_val);
                e_sub_o_val_1 = vrshlq_s32(e_sub_o_val_1, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_sub_o_val_1);
                shift_res_2 = vrev64_s16(shift_res_2);

                pred_add_val_t = vqadd_s16(shift_res_2, vreinterpret_s16_u16(vget_high_u16(pu2_pred_2_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_0 = vsubq_s32(transpose_val_32q_2.val[1], o_3_val);
                e_sub_o_val_0 = vrshlq_s32(e_sub_o_val_0, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_sub_o_val_0);
                shift_res_3 = vrev64_s16(shift_res_3);

                pred_add_val_t = vqadd_s16(shift_res_3, vreinterpret_s16_u16(vget_high_u16(pu2_pred_3_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;
            }
            pi2_tmp += 4;
        }
    }
    else /* All rows of output of 1st stage are non-zero */
    {
        for(j = trans_size; j > 0; j -= 4)
        {
            pi2_tmp_0_trans_size = pi2_tmp;
            pi2_tmp_1_trans_size = pi2_tmp + trans_size;
            pi2_tmp_2_trans_size = pi2_tmp + 2 * trans_size;
            pi2_tmp_3_trans_size = pi2_tmp_1_trans_size + 2 * trans_size;
            pi2_tmp_4_trans_size = pi2_tmp_2_trans_size + 2 * trans_size;
            pi2_tmp_5_trans_size = pi2_tmp_1_trans_size + 4 * trans_size;
            pi2_tmp_6_trans_size = pi2_tmp_2_trans_size + 4 * trans_size;
            pi2_tmp_7_trans_size = pi2_tmp_5_trans_size + 2 * trans_size;

            /* Utilizing symmetry properties to the maximum to minimize the number of multiplications */
            pi2_src_1_src_strd_val = vld1_s16(pi2_tmp_1_trans_size);
            pi2_src_3_src_strd_val = vld1_s16(pi2_tmp_3_trans_size);
            o_0_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 0);

            pi2_src_5_src_strd_val = vld1_s16(pi2_tmp_5_trans_size);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 0);

            pi2_src_7_src_strd_val = vld1_s16(pi2_tmp_7_trans_size);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 0);

            pi2_src_2_src_strd_val = vld1_s16(pi2_tmp_2_trans_size);
            o_0_val = vmlal_lane_s16(o_0_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 0);

            o_1_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 1);
            pi2_src_6_src_strd_val = vld1_s16(pi2_tmp_6_trans_size);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 1);
            pi2_src_0_src_strd_val = vld1_s16(pi2_tmp_0_trans_size);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 1);
            pi2_src_4_src_strd_val = vld1_s16(pi2_tmp_4_trans_size);

            o_1_val = vmlal_lane_s16(o_1_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 1);

            o_2_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 2);
            o_2_val = vmlal_lane_s16(o_2_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 2);

            o_3_val = vmull_lane_s16(g_ai2_ihevc_trans_8_val_1, pi2_src_1_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_3, pi2_src_3_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_5, pi2_src_5_src_strd_val, 3);
            o_3_val = vmlal_lane_s16(o_3_val, g_ai2_ihevc_trans_8_val_7, pi2_src_7_src_strd_val, 3);

            eo_0_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 2);
            eo_0_val = vmlal_lane_s16(eo_0_val, pi2_src_6_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 2);

            eo_1_val = vmull_lane_s16(pi2_src_2_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 3);
            eo_1_val = vmlal_lane_s16(eo_1_val, pi2_src_6_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 3);

            ee_0_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 0);
            ee_0_val = vmlal_lane_s16(ee_0_val, pi2_src_4_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 0);

            ee_1_val = vmull_lane_s16(pi2_src_0_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_0_2), 1);
            ee_1_val = vmlal_lane_s16(ee_1_val, pi2_src_4_src_strd_val, vreinterpret_s16_s32(g_ai2_ihevc_trans_8_val_4_6), 1);

            /* Combining e and o terms at each hierarchy levels to calculate the final spatial domain vector */
            e_0_val = vaddq_s32(ee_0_val, eo_0_val);
            e_2_val = vsubq_s32(ee_1_val, eo_1_val);

            e_1_val = vaddq_s32(ee_1_val, eo_1_val);
            e_3_val = vsubq_s32(ee_0_val, eo_0_val);
            {
                temp_unzip_1 = vuzpq_s32(e_0_val, e_2_val);
                temp_unzip_2 = vuzpq_s32(e_1_val, e_3_val);
                transpose_val_32q_1 = vtrnq_s32(temp_unzip_1.val[0], temp_unzip_2.val[0]);
                transpose_val_32q_2 = vtrnq_s32(temp_unzip_1.val[1], temp_unzip_2.val[1]);

                e_add_o_val_0 = vaddq_s32(transpose_val_32q_1.val[0], o_0_val);
                e_add_o_val_0 = vrshlq_s32(e_add_o_val_0, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_add_o_val_0);

                pu2_pred_0_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_0, vreinterpret_s16_u16(vget_low_u16(pu2_pred_0_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_1 = vaddq_s32(transpose_val_32q_2.val[0], o_1_val);
                e_add_o_val_1 = vrshlq_s32(e_add_o_val_1, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_add_o_val_1);

                pu2_pred_1_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_low_u16(pu2_pred_1_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_2 = vaddq_s32(transpose_val_32q_1.val[1], o_2_val);
                e_add_o_val_2 = vrshlq_s32(e_add_o_val_2, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_add_o_val_2);

                pu2_pred_2_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_2, vreinterpret_s16_u16(vget_low_u16(pu2_pred_2_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_add_o_val_3 = vaddq_s32(transpose_val_32q_2.val[1], o_3_val);
                e_add_o_val_3 = vrshlq_s32(e_add_o_val_3, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_add_o_val_3);

                pu2_pred_3_pred_strd_val = vld1q_u16(pu2_pred_tmp_0_pred_strd);
                pu2_pred_tmp_0_pred_strd += pred_strd;
                pred_add_val_t = vqadd_s16(shift_res_3, vreinterpret_s16_u16(vget_low_u16(pu2_pred_3_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_dst_strd += dst_strd;

                e_sub_o_val_3 = vsubq_s32(transpose_val_32q_1.val[0], o_0_val);
                e_sub_o_val_3 = vrshlq_s32(e_sub_o_val_3, shift_val_neg);
                shift_res_0 = vqmovn_s32(e_sub_o_val_3);
                shift_res_0 = vrev64_s16(shift_res_0);

                pred_add_val_t = vqadd_s16(shift_res_0, vreinterpret_s16_u16(vget_high_u16(pu2_pred_0_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_2 = vsubq_s32(transpose_val_32q_2.val[0], o_1_val);
                e_sub_o_val_2 = vrshlq_s32(e_sub_o_val_2, shift_val_neg);
                shift_res_1 = vqmovn_s32(e_sub_o_val_2);
                shift_res_1 = vrev64_s16(shift_res_1);

                pred_add_val_t = vqadd_s16(shift_res_1, vreinterpret_s16_u16(vget_high_u16(pu2_pred_1_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_1 = vsubq_s32(transpose_val_32q_1.val[1], o_2_val);
                e_sub_o_val_1 = vrshlq_s32(e_sub_o_val_1, shift_val_neg);
                shift_res_2 = vqmovn_s32(e_sub_o_val_1);
                shift_res_2 = vrev64_s16(shift_res_2);

                pred_add_val_t = vqadd_s16(shift_res_2, vreinterpret_s16_u16(vget_high_u16(pu2_pred_2_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;

                e_sub_o_val_0 = vsubq_s32(transpose_val_32q_2.val[1], o_3_val);
                e_sub_o_val_0 = vrshlq_s32(e_sub_o_val_0, shift_val_neg);
                shift_res_3 = vqmovn_s32(e_sub_o_val_0);
                shift_res_3 = vrev64_s16(shift_res_3);

                pred_add_val_t = vqadd_s16(shift_res_3, vreinterpret_s16_u16(vget_high_u16(pu2_pred_3_pred_strd_val)));
                pred_add_val_t = vmax_s16(pred_add_val_t, const_0);
                pred_add_val_t = vmin_s16(pred_add_val_t, dup_const_clip_limit);
                vst1_u16(pu2_dst_tmp_0_4_dst_strd, vreinterpret_u16_s16(pred_add_val_t));
                pu2_dst_tmp_0_4_dst_strd += dst_strd;
            }
            pi2_tmp += 4;
        }
    }
}
