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
*  ihevcd_hbd_itrans_recon_dc_neon_intr.c
*
* @brief
*  Contains ARM NEON intrinsics functions for High Bit Depth DC inverse transform
*  and reconstruction (Luma and Chroma)
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevcd_hbd_itrans_recon_dc_luma_neonintr()
*  - ihevcd_hbd_itrans_recon_dc_chroma_neonintr()
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
#include "ihevc_trans_macros.h"
#include "ihevcd_itrans_recon_dc.h"
#include <arm_neon.h>

void ihevcd_hbd_itrans_recon_dc_luma_neonintr(UWORD16 *pu2_pred,
                                              UWORD16 *pu2_dst,
                                              WORD32 pred_strd,
                                              WORD32 dst_strd,
                                              WORD32 log2_trans_size,
                                              WORD16 i2_coeff_value,
                                              WORD32 bit_depth)
{
    WORD32 row;
    WORD32 add, shift;
    WORD32 dc_value, quant_out;
    WORD32 trans_size;
    UWORD16 max_val;

    trans_size = (1 << log2_trans_size);
    quant_out = (WORD32)i2_coeff_value;

    shift = IT_SHIFT_STAGE_1;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((quant_out * 64 + add) >> shift);
    shift = 20 - bit_depth;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((dc_value * 64 + add) >> shift);

    max_val = (1 << bit_depth) - 1;
    uint16x8_t v_max = vdupq_n_u16(max_val);
    uint16x4_t v_max_4 = vdup_n_u16(max_val);

    if(trans_size == 4)
    {
        if(dc_value >= 0)
        {
            uint16x4_t v_dc = vdup_n_u16((uint16_t)dc_value);
            for(row = 0; row < 4; row++)
            {
                uint16x4_t v_p = vld1_u16(pu2_pred);
                vst1_u16(pu2_dst, vmin_u16(vqadd_u16(v_p, v_dc), v_max_4));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x4_t v_dc = vdup_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 4; row++)
            {
                uint16x4_t v_p = vld1_u16(pu2_pred);
                vst1_u16(pu2_dst, vmin_u16(vqsub_u16(v_p, v_dc), v_max_4));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 8)
    {
        if(dc_value >= 0)
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)dc_value);
            for(row = 0; row < 8; row++)
            {
                uint16x8_t v_p = vld1q_u16(pu2_pred);
                vst1q_u16(pu2_dst, vminq_u16(vqaddq_u16(v_p, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 8; row++)
            {
                uint16x8_t v_p = vld1q_u16(pu2_pred);
                vst1q_u16(pu2_dst, vminq_u16(vqsubq_u16(v_p, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 16)
    {
        if(dc_value >= 0)
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)dc_value);
            for(row = 0; row < 16; row++)
            {
                uint16x8_t v_p0 = vld1q_u16(pu2_pred);
                uint16x8_t v_p1 = vld1q_u16(pu2_pred + 8);
                vst1q_u16(pu2_dst, vminq_u16(vqaddq_u16(v_p0, v_dc), v_max));
                vst1q_u16(pu2_dst + 8, vminq_u16(vqaddq_u16(v_p1, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 16; row++)
            {
                uint16x8_t v_p0 = vld1q_u16(pu2_pred);
                uint16x8_t v_p1 = vld1q_u16(pu2_pred + 8);
                vst1q_u16(pu2_dst, vminq_u16(vqsubq_u16(v_p0, v_dc), v_max));
                vst1q_u16(pu2_dst + 8, vminq_u16(vqsubq_u16(v_p1, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 32)
    {
        if(dc_value >= 0)
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)dc_value);
            for(row = 0; row < 32; row++)
            {
                uint16x8_t v_p0 = vld1q_u16(pu2_pred);
                uint16x8_t v_p1 = vld1q_u16(pu2_pred + 8);
                uint16x8_t v_p2 = vld1q_u16(pu2_pred + 16);
                uint16x8_t v_p3 = vld1q_u16(pu2_pred + 24);
                vst1q_u16(pu2_dst, vminq_u16(vqaddq_u16(v_p0, v_dc), v_max));
                vst1q_u16(pu2_dst + 8, vminq_u16(vqaddq_u16(v_p1, v_dc), v_max));
                vst1q_u16(pu2_dst + 16, vminq_u16(vqaddq_u16(v_p2, v_dc), v_max));
                vst1q_u16(pu2_dst + 24, vminq_u16(vqaddq_u16(v_p3, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x8_t v_dc = vdupq_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 32; row++)
            {
                uint16x8_t v_p0 = vld1q_u16(pu2_pred);
                uint16x8_t v_p1 = vld1q_u16(pu2_pred + 8);
                uint16x8_t v_p2 = vld1q_u16(pu2_pred + 16);
                uint16x8_t v_p3 = vld1q_u16(pu2_pred + 24);
                vst1q_u16(pu2_dst, vminq_u16(vqsubq_u16(v_p0, v_dc), v_max));
                vst1q_u16(pu2_dst + 8, vminq_u16(vqsubq_u16(v_p1, v_dc), v_max));
                vst1q_u16(pu2_dst + 16, vminq_u16(vqsubq_u16(v_p2, v_dc), v_max));
                vst1q_u16(pu2_dst + 24, vminq_u16(vqsubq_u16(v_p3, v_dc), v_max));
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }
}

void ihevcd_hbd_itrans_recon_dc_chroma_neonintr(UWORD16 *pu2_pred,
                                                UWORD16 *pu2_dst,
                                                WORD32 pred_strd,
                                                WORD32 dst_strd,
                                                WORD32 log2_trans_size,
                                                WORD16 i2_coeff_value,
                                                WORD32 bit_depth)
{
    WORD32 row;
    WORD32 add, shift;
    WORD32 dc_value, quant_out;
    WORD32 trans_size;
    UWORD16 max_val;

    trans_size = (1 << log2_trans_size);
    quant_out = (WORD32)i2_coeff_value;

    shift = IT_SHIFT_STAGE_1;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((quant_out * 64 + add) >> shift);
    shift = 20 - bit_depth;
    add = 1 << (shift - 1);
    dc_value = CLIP_S16((dc_value * 64 + add) >> shift);

    max_val = (1 << bit_depth) - 1;
    uint16x4_t v_max_4 = vdup_n_u16(max_val);

    if(trans_size == 4)
    {
        if(dc_value >= 0)
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)dc_value);
            for(row = 0; row < 4; row++)
            {
                uint16x4_t v_p = vdup_n_u16(0);
                v_p = vld1_lane_u16(&pu2_pred[0], v_p, 0);
                v_p = vld1_lane_u16(&pu2_pred[2], v_p, 1);
                v_p = vld1_lane_u16(&pu2_pred[4], v_p, 2);
                v_p = vld1_lane_u16(&pu2_pred[6], v_p, 3);

                uint16x4_t v_res = vmin_u16(vqadd_u16(v_p, v_dc_4), v_max_4);

                vst1_lane_u16(&pu2_dst[0], v_res, 0);
                vst1_lane_u16(&pu2_dst[2], v_res, 1);
                vst1_lane_u16(&pu2_dst[4], v_res, 2);
                vst1_lane_u16(&pu2_dst[6], v_res, 3);

                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 4; row++)
            {
                uint16x4_t v_p = vdup_n_u16(0);
                v_p = vld1_lane_u16(&pu2_pred[0], v_p, 0);
                v_p = vld1_lane_u16(&pu2_pred[2], v_p, 1);
                v_p = vld1_lane_u16(&pu2_pred[4], v_p, 2);
                v_p = vld1_lane_u16(&pu2_pred[6], v_p, 3);

                uint16x4_t v_res = vmin_u16(vqsub_u16(v_p, v_dc_4), v_max_4);

                vst1_lane_u16(&pu2_dst[0], v_res, 0);
                vst1_lane_u16(&pu2_dst[2], v_res, 1);
                vst1_lane_u16(&pu2_dst[4], v_res, 2);
                vst1_lane_u16(&pu2_dst[6], v_res, 3);

                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 8)
    {
        if(dc_value >= 0)
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)dc_value);
            for(row = 0; row < 8; row++)
            {
                uint16x4_t v_p0 = vdup_n_u16(0);
                v_p0 = vld1_lane_u16(&pu2_pred[0], v_p0, 0);
                v_p0 = vld1_lane_u16(&pu2_pred[2], v_p0, 1);
                v_p0 = vld1_lane_u16(&pu2_pred[4], v_p0, 2);
                v_p0 = vld1_lane_u16(&pu2_pred[6], v_p0, 3);

                uint16x4_t v_p1 = vdup_n_u16(0);
                v_p1 = vld1_lane_u16(&pu2_pred[8], v_p1, 0);
                v_p1 = vld1_lane_u16(&pu2_pred[10], v_p1, 1);
                v_p1 = vld1_lane_u16(&pu2_pred[12], v_p1, 2);
                v_p1 = vld1_lane_u16(&pu2_pred[14], v_p1, 3);

                uint16x4_t v_r0 = vmin_u16(vqadd_u16(v_p0, v_dc_4), v_max_4);
                uint16x4_t v_r1 = vmin_u16(vqadd_u16(v_p1, v_dc_4), v_max_4);

                vst1_lane_u16(&pu2_dst[0], v_r0, 0);
                vst1_lane_u16(&pu2_dst[2], v_r0, 1);
                vst1_lane_u16(&pu2_dst[4], v_r0, 2);
                vst1_lane_u16(&pu2_dst[6], v_r0, 3);

                vst1_lane_u16(&pu2_dst[8], v_r1, 0);
                vst1_lane_u16(&pu2_dst[10], v_r1, 1);
                vst1_lane_u16(&pu2_dst[12], v_r1, 2);
                vst1_lane_u16(&pu2_dst[14], v_r1, 3);

                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 8; row++)
            {
                uint16x4_t v_p0 = vdup_n_u16(0);
                v_p0 = vld1_lane_u16(&pu2_pred[0], v_p0, 0);
                v_p0 = vld1_lane_u16(&pu2_pred[2], v_p0, 1);
                v_p0 = vld1_lane_u16(&pu2_pred[4], v_p0, 2);
                v_p0 = vld1_lane_u16(&pu2_pred[6], v_p0, 3);

                uint16x4_t v_p1 = vdup_n_u16(0);
                v_p1 = vld1_lane_u16(&pu2_pred[8], v_p1, 0);
                v_p1 = vld1_lane_u16(&pu2_pred[10], v_p1, 1);
                v_p1 = vld1_lane_u16(&pu2_pred[12], v_p1, 2);
                v_p1 = vld1_lane_u16(&pu2_pred[14], v_p1, 3);

                uint16x4_t v_r0 = vmin_u16(vqsub_u16(v_p0, v_dc_4), v_max_4);
                uint16x4_t v_r1 = vmin_u16(vqsub_u16(v_p1, v_dc_4), v_max_4);

                vst1_lane_u16(&pu2_dst[0], v_r0, 0);
                vst1_lane_u16(&pu2_dst[2], v_r0, 1);
                vst1_lane_u16(&pu2_dst[4], v_r0, 2);
                vst1_lane_u16(&pu2_dst[6], v_r0, 3);

                vst1_lane_u16(&pu2_dst[8], v_r1, 0);
                vst1_lane_u16(&pu2_dst[10], v_r1, 1);
                vst1_lane_u16(&pu2_dst[12], v_r1, 2);
                vst1_lane_u16(&pu2_dst[14], v_r1, 3);

                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 16)
    {
        WORD32 col;
        if(dc_value >= 0)
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)dc_value);
            for(row = 0; row < 16; row++)
            {
                for(col = 0; col < 16; col += 8)
                {
                    uint16x4_t v_p0 = vdup_n_u16(0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 0) * 2], v_p0, 0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 1) * 2], v_p0, 1);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 2) * 2], v_p0, 2);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 3) * 2], v_p0, 3);

                    uint16x4_t v_p1 = vdup_n_u16(0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 4) * 2], v_p1, 0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 5) * 2], v_p1, 1);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 6) * 2], v_p1, 2);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 7) * 2], v_p1, 3);

                    uint16x4_t v_r0 = vmin_u16(vqadd_u16(v_p0, v_dc_4), v_max_4);
                    uint16x4_t v_r1 = vmin_u16(vqadd_u16(v_p1, v_dc_4), v_max_4);

                    vst1_lane_u16(&pu2_dst[(col + 0) * 2], v_r0, 0);
                    vst1_lane_u16(&pu2_dst[(col + 1) * 2], v_r0, 1);
                    vst1_lane_u16(&pu2_dst[(col + 2) * 2], v_r0, 2);
                    vst1_lane_u16(&pu2_dst[(col + 3) * 2], v_r0, 3);

                    vst1_lane_u16(&pu2_dst[(col + 4) * 2], v_r1, 0);
                    vst1_lane_u16(&pu2_dst[(col + 5) * 2], v_r1, 1);
                    vst1_lane_u16(&pu2_dst[(col + 6) * 2], v_r1, 2);
                    vst1_lane_u16(&pu2_dst[(col + 7) * 2], v_r1, 3);
                }
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 16; row++)
            {
                for(col = 0; col < 16; col += 8)
                {
                    uint16x4_t v_p0 = vdup_n_u16(0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 0) * 2], v_p0, 0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 1) * 2], v_p0, 1);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 2) * 2], v_p0, 2);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 3) * 2], v_p0, 3);

                    uint16x4_t v_p1 = vdup_n_u16(0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 4) * 2], v_p1, 0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 5) * 2], v_p1, 1);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 6) * 2], v_p1, 2);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 7) * 2], v_p1, 3);

                    uint16x4_t v_r0 = vmin_u16(vqsub_u16(v_p0, v_dc_4), v_max_4);
                    uint16x4_t v_r1 = vmin_u16(vqsub_u16(v_p1, v_dc_4), v_max_4);

                    vst1_lane_u16(&pu2_dst[(col + 0) * 2], v_r0, 0);
                    vst1_lane_u16(&pu2_dst[(col + 1) * 2], v_r0, 1);
                    vst1_lane_u16(&pu2_dst[(col + 2) * 2], v_r0, 2);
                    vst1_lane_u16(&pu2_dst[(col + 3) * 2], v_r0, 3);

                    vst1_lane_u16(&pu2_dst[(col + 4) * 2], v_r1, 0);
                    vst1_lane_u16(&pu2_dst[(col + 5) * 2], v_r1, 1);
                    vst1_lane_u16(&pu2_dst[(col + 6) * 2], v_r1, 2);
                    vst1_lane_u16(&pu2_dst[(col + 7) * 2], v_r1, 3);
                }
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }

    if(trans_size == 32)
    {
        WORD32 col;
        if(dc_value >= 0)
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)dc_value);
            for(row = 0; row < 32; row++)
            {
                for(col = 0; col < 32; col += 8)
                {
                    uint16x4_t v_p0 = vdup_n_u16(0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 0) * 2], v_p0, 0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 1) * 2], v_p0, 1);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 2) * 2], v_p0, 2);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 3) * 2], v_p0, 3);

                    uint16x4_t v_p1 = vdup_n_u16(0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 4) * 2], v_p1, 0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 5) * 2], v_p1, 1);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 6) * 2], v_p1, 2);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 7) * 2], v_p1, 3);

                    uint16x4_t v_r0 = vmin_u16(vqadd_u16(v_p0, v_dc_4), v_max_4);
                    uint16x4_t v_r1 = vmin_u16(vqadd_u16(v_p1, v_dc_4), v_max_4);

                    vst1_lane_u16(&pu2_dst[(col + 0) * 2], v_r0, 0);
                    vst1_lane_u16(&pu2_dst[(col + 1) * 2], v_r0, 1);
                    vst1_lane_u16(&pu2_dst[(col + 2) * 2], v_r0, 2);
                    vst1_lane_u16(&pu2_dst[(col + 3) * 2], v_r0, 3);

                    vst1_lane_u16(&pu2_dst[(col + 4) * 2], v_r1, 0);
                    vst1_lane_u16(&pu2_dst[(col + 5) * 2], v_r1, 1);
                    vst1_lane_u16(&pu2_dst[(col + 6) * 2], v_r1, 2);
                    vst1_lane_u16(&pu2_dst[(col + 7) * 2], v_r1, 3);
                }
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        else
        {
            uint16x4_t v_dc_4 = vdup_n_u16((uint16_t)(-(WORD32)dc_value));
            for(row = 0; row < 32; row++)
            {
                for(col = 0; col < 32; col += 8)
                {
                    uint16x4_t v_p0 = vdup_n_u16(0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 0) * 2], v_p0, 0);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 1) * 2], v_p0, 1);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 2) * 2], v_p0, 2);
                    v_p0 = vld1_lane_u16(&pu2_pred[(col + 3) * 2], v_p0, 3);

                    uint16x4_t v_p1 = vdup_n_u16(0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 4) * 2], v_p1, 0);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 5) * 2], v_p1, 1);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 6) * 2], v_p1, 2);
                    v_p1 = vld1_lane_u16(&pu2_pred[(col + 7) * 2], v_p1, 3);

                    uint16x4_t v_r0 = vmin_u16(vqsub_u16(v_p0, v_dc_4), v_max_4);
                    uint16x4_t v_r1 = vmin_u16(vqsub_u16(v_p1, v_dc_4), v_max_4);

                    vst1_lane_u16(&pu2_dst[(col + 0) * 2], v_r0, 0);
                    vst1_lane_u16(&pu2_dst[(col + 1) * 2], v_r0, 1);
                    vst1_lane_u16(&pu2_dst[(col + 2) * 2], v_r0, 2);
                    vst1_lane_u16(&pu2_dst[(col + 3) * 2], v_r0, 3);

                    vst1_lane_u16(&pu2_dst[(col + 4) * 2], v_r1, 0);
                    vst1_lane_u16(&pu2_dst[(col + 5) * 2], v_r1, 1);
                    vst1_lane_u16(&pu2_dst[(col + 6) * 2], v_r1, 2);
                    vst1_lane_u16(&pu2_dst[(col + 7) * 2], v_r1, 3);
                }
                pu2_pred += pred_strd;
                pu2_dst += dst_strd;
            }
        }
        return;
    }
}
