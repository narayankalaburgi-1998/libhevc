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
*  ihevc_hbd_padding_neon_intr.c
*
* @brief
*  Contains ARM NEON intrinsic function definitions for High Bit Depth (HBD)
*  padding operations in HEVC.
*
* @author
*  Ittiam
*
* @par List of Functions:
*  - ihevc_hbd_pad_left_luma_neonintr()
*  - ihevc_hbd_pad_right_luma_neonintr()
*  - ihevc_hbd_pad_top_neonintr()
*  - ihevc_hbd_pad_bottom_neonintr()
*  - ihevc_hbd_pad_left_chroma_neonintr()
*  - ihevc_hbd_pad_right_chroma_neonintr()
*  - ihevc_hbd_pad_horz_chroma_neonintr()
*
* @remarks
*  Optimized with ARM NEON intrinsics (ARMv8 / AArch64 / AArch32 NEON).
*  Exact bit-exact replacement for C implementations in ihevc_hbd_padding.c.
*
*******************************************************************************
*/

#include <arm_neon.h>
#include "ihevc_typedefs.h"
#include "ihevc_macros.h"
#include "ihevc_platform_macros.h"
#include "ihevc_padding.h"

/**
*******************************************************************************
*
* @brief
*   Padding (luma block) at the left of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The left column of a 2D array (val = *pu2_src) is replicated pad_size times
*   to the left for ht rows.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the source (col 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad to the left
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_left_luma_neonintr(UWORD16 *pu2_src,
                                      WORD32 src_strd,
                                      WORD32 ht,
                                      WORD32 pad_size)
{
    WORD32 row;
    WORD32 col;

    /* Process 4 rows per iteration (640 bytes stored per 80-sample chunk) */
    for(row = 0; row <= ht - 4; row += 4)
    {
        uint16x8_t v0 = vdupq_n_u16(pu2_src[0]);
        uint16x8_t v1 = vdupq_n_u16(pu2_src[src_strd]);
        uint16x8_t v2 = vdupq_n_u16(pu2_src[2 * src_strd]);
        uint16x8_t v3 = vdupq_n_u16(pu2_src[3 * src_strd]);

        for(col = 0; col < pad_size; col += 80)
        {
            UWORD16 *dst0 = pu2_src - pad_size + col;
            UWORD16 *dst1 = dst0 + src_strd;
            UWORD16 *dst2 = dst1 + src_strd;
            UWORD16 *dst3 = dst2 + src_strd;

            /* Row 0 */
            vst1q_u16(dst0 + 0,  v0);
            vst1q_u16(dst0 + 8,  v0);
            vst1q_u16(dst0 + 16, v0);
            vst1q_u16(dst0 + 24, v0);
            vst1q_u16(dst0 + 32, v0);
            vst1q_u16(dst0 + 40, v0);
            vst1q_u16(dst0 + 48, v0);
            vst1q_u16(dst0 + 56, v0);
            vst1q_u16(dst0 + 64, v0);
            vst1q_u16(dst0 + 72, v0);

            /* Row 1 */
            vst1q_u16(dst1 + 0,  v1);
            vst1q_u16(dst1 + 8,  v1);
            vst1q_u16(dst1 + 16, v1);
            vst1q_u16(dst1 + 24, v1);
            vst1q_u16(dst1 + 32, v1);
            vst1q_u16(dst1 + 40, v1);
            vst1q_u16(dst1 + 48, v1);
            vst1q_u16(dst1 + 56, v1);
            vst1q_u16(dst1 + 64, v1);
            vst1q_u16(dst1 + 72, v1);

            /* Row 2 */
            vst1q_u16(dst2 + 0,  v2);
            vst1q_u16(dst2 + 8,  v2);
            vst1q_u16(dst2 + 16, v2);
            vst1q_u16(dst2 + 24, v2);
            vst1q_u16(dst2 + 32, v2);
            vst1q_u16(dst2 + 40, v2);
            vst1q_u16(dst2 + 48, v2);
            vst1q_u16(dst2 + 56, v2);
            vst1q_u16(dst2 + 64, v2);
            vst1q_u16(dst2 + 72, v2);

            /* Row 3 */
            vst1q_u16(dst3 + 0,  v3);
            vst1q_u16(dst3 + 8,  v3);
            vst1q_u16(dst3 + 16, v3);
            vst1q_u16(dst3 + 24, v3);
            vst1q_u16(dst3 + 32, v3);
            vst1q_u16(dst3 + 40, v3);
            vst1q_u16(dst3 + 48, v3);
            vst1q_u16(dst3 + 56, v3);
            vst1q_u16(dst3 + 64, v3);
            vst1q_u16(dst3 + 72, v3);
        }

        pu2_src += 4 * src_strd;
    }

    /* Remainder rows if ht is not a multiple of 4 */
    for(; row < ht; row++)
    {
        uint16x8_t v = vdupq_n_u16(*pu2_src);

        for(col = 0; col < pad_size; col += 80)
        {
            UWORD16 *dst = pu2_src - pad_size + col;

            vst1q_u16(dst + 0,  v);
            vst1q_u16(dst + 8,  v);
            vst1q_u16(dst + 16, v);
            vst1q_u16(dst + 24, v);
            vst1q_u16(dst + 32, v);
            vst1q_u16(dst + 40, v);
            vst1q_u16(dst + 48, v);
            vst1q_u16(dst + 56, v);
            vst1q_u16(dst + 64, v);
            vst1q_u16(dst + 72, v);
        }

        pu2_src += src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (luma block) at the right of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The rightmost column of a 2D array (val = *(pu2_src - 1)) is replicated
*   pad_size times to the right for ht rows.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the right padding area (col wd)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad to the right
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_right_luma_neonintr(UWORD16 *pu2_src,
                                       WORD32 src_strd,
                                       WORD32 ht,
                                       WORD32 pad_size)
{
    WORD32 row;
    WORD32 col;

    /* Process 4 rows per iteration (640 bytes stored per 80-sample chunk) */
    for(row = 0; row <= ht - 4; row += 4)
    {
        uint16x8_t v0 = vdupq_n_u16(*(pu2_src - 1));
        uint16x8_t v1 = vdupq_n_u16(*(pu2_src + src_strd - 1));
        uint16x8_t v2 = vdupq_n_u16(*(pu2_src + 2 * src_strd - 1));
        uint16x8_t v3 = vdupq_n_u16(*(pu2_src + 3 * src_strd - 1));

        for(col = 0; col < pad_size; col += 80)
        {
            UWORD16 *dst0 = pu2_src + col;
            UWORD16 *dst1 = dst0 + src_strd;
            UWORD16 *dst2 = dst1 + src_strd;
            UWORD16 *dst3 = dst2 + src_strd;

            /* Row 0 */
            vst1q_u16(dst0 + 0,  v0);
            vst1q_u16(dst0 + 8,  v0);
            vst1q_u16(dst0 + 16, v0);
            vst1q_u16(dst0 + 24, v0);
            vst1q_u16(dst0 + 32, v0);
            vst1q_u16(dst0 + 40, v0);
            vst1q_u16(dst0 + 48, v0);
            vst1q_u16(dst0 + 56, v0);
            vst1q_u16(dst0 + 64, v0);
            vst1q_u16(dst0 + 72, v0);

            /* Row 1 */
            vst1q_u16(dst1 + 0,  v1);
            vst1q_u16(dst1 + 8,  v1);
            vst1q_u16(dst1 + 16, v1);
            vst1q_u16(dst1 + 24, v1);
            vst1q_u16(dst1 + 32, v1);
            vst1q_u16(dst1 + 40, v1);
            vst1q_u16(dst1 + 48, v1);
            vst1q_u16(dst1 + 56, v1);
            vst1q_u16(dst1 + 64, v1);
            vst1q_u16(dst1 + 72, v1);

            /* Row 2 */
            vst1q_u16(dst2 + 0,  v2);
            vst1q_u16(dst2 + 8,  v2);
            vst1q_u16(dst2 + 16, v2);
            vst1q_u16(dst2 + 24, v2);
            vst1q_u16(dst2 + 32, v2);
            vst1q_u16(dst2 + 40, v2);
            vst1q_u16(dst2 + 48, v2);
            vst1q_u16(dst2 + 56, v2);
            vst1q_u16(dst2 + 64, v2);
            vst1q_u16(dst2 + 72, v2);

            /* Row 3 */
            vst1q_u16(dst3 + 0,  v3);
            vst1q_u16(dst3 + 8,  v3);
            vst1q_u16(dst3 + 16, v3);
            vst1q_u16(dst3 + 24, v3);
            vst1q_u16(dst3 + 32, v3);
            vst1q_u16(dst3 + 40, v3);
            vst1q_u16(dst3 + 48, v3);
            vst1q_u16(dst3 + 56, v3);
            vst1q_u16(dst3 + 64, v3);
            vst1q_u16(dst3 + 72, v3);
        }

        pu2_src += 4 * src_strd;
    }

    /* Remainder rows if ht is not a multiple of 4 */
    for(; row < ht; row++)
    {
        uint16x8_t v = vdupq_n_u16(*(pu2_src - 1));

        for(col = 0; col < pad_size; col += 80)
        {
            UWORD16 *dst = pu2_src + col;

            vst1q_u16(dst + 0,  v);
            vst1q_u16(dst + 8,  v);
            vst1q_u16(dst + 16, v);
            vst1q_u16(dst + 24, v);
            vst1q_u16(dst + 32, v);
            vst1q_u16(dst + 40, v);
            vst1q_u16(dst + 48, v);
            vst1q_u16(dst + 56, v);
            vst1q_u16(dst + 64, v);
            vst1q_u16(dst + 72, v);
        }

        pu2_src += src_strd;
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding at the top of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The top row (row 0) is replicated pad_size times upwards.
*   For common HEVC widths (64, 32, 16, 8, 4), the source row is preloaded
*   ONCE into NEON registers prior to the loop, completely eliminating all
*   memory loads and cache-miss overhead inside the vertical replication loop.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the top valid row (row 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD16 units
*
* @param[in] pad_size
*   Number of rows to pad above
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_top_neonintr(UWORD16 *pu2_src,
                                WORD32 src_strd,
                                WORD32 wd,
                                WORD32 pad_size)
{
    WORD32 row;

    if((wd <= 0) || (pad_size <= 0))
    {
        return;
    }

    /* Fast path: wd == 64 (Standard CTB width) - Preload once into registers, ascending stores */
    if(wd == 64)
    {
        uint16x8_t v0 = vld1q_u16(pu2_src + 0);
        uint16x8_t v1 = vld1q_u16(pu2_src + 8);
        uint16x8_t v2 = vld1q_u16(pu2_src + 16);
        uint16x8_t v3 = vld1q_u16(pu2_src + 24);
        uint16x8_t v4 = vld1q_u16(pu2_src + 32);
        uint16x8_t v5 = vld1q_u16(pu2_src + 40);
        uint16x8_t v6 = vld1q_u16(pu2_src + 48);
        uint16x8_t v7 = vld1q_u16(pu2_src + 56);
        UWORD16 *dst = pu2_src - pad_size * src_strd;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);
            vst1q_u16(dst + 16, v2);
            vst1q_u16(dst + 24, v3);
            vst1q_u16(dst + 32, v4);
            vst1q_u16(dst + 40, v5);
            vst1q_u16(dst + 48, v6);
            vst1q_u16(dst + 56, v7);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 32 */
    if(wd == 32)
    {
        uint16x8_t v0 = vld1q_u16(pu2_src + 0);
        uint16x8_t v1 = vld1q_u16(pu2_src + 8);
        uint16x8_t v2 = vld1q_u16(pu2_src + 16);
        uint16x8_t v3 = vld1q_u16(pu2_src + 24);
        UWORD16 *dst = pu2_src - pad_size * src_strd;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);
            vst1q_u16(dst + 16, v2);
            vst1q_u16(dst + 24, v3);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 16 */
    if(wd == 16)
    {
        uint16x8_t v0 = vld1q_u16(pu2_src + 0);
        uint16x8_t v1 = vld1q_u16(pu2_src + 8);
        UWORD16 *dst = pu2_src - pad_size * src_strd;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 8 */
    if(wd == 8)
    {
        uint16x8_t v0 = vld1q_u16(pu2_src + 0);
        UWORD16 *dst = pu2_src - pad_size * src_strd;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 4 */
    if(wd == 4)
    {
        uint16x4_t v0 = vld1_u16(pu2_src + 0);
        UWORD16 *dst = pu2_src - pad_size * src_strd;

        for(row = 0; row < pad_size; row++)
        {
            vst1_u16(dst + 0, v0);
            dst += src_strd;
        }
        return;
    }

    /* Generic streaming path for arbitrary wd */
    for(row = 1; row <= pad_size; row++)
    {
        UWORD16 *dst = pu2_src - row * src_strd;
        const UWORD16 *src = pu2_src;
        WORD32 col;

        for(col = 0; col + 32 <= wd; col += 32)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
        }

        if(col + 16 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            col += 16;
        }

        if(col + 8 <= wd)
        {
            vst1q_u16(dst + col, vld1q_u16(src + col));
            col += 8;
        }

        if(col + 4 <= wd)
        {
            vst1_u16(dst + col, vld1_u16(src + col));
            col += 4;
        }

        for(; col < wd; col++)
        {
            dst[col] = src[col];
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding at the bottom of a 2D array using ARM NEON intrinsics.
*
* @par Description:
*   The bottom row (pu2_src - 1 * src_strd) is replicated pad_size times downwards
*   into pu2_src + (row - 1) * src_strd for row = 1 .. pad_size.
*   For common HEVC widths, the bottom valid row is preloaded ONCE into NEON
*   registers prior to the loop.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the bottom padding area (row ht)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] wd
*   Width of the row to replicate in UWORD16 units
*
* @param[in] pad_size
*   Number of rows to pad below
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_bottom_neonintr(UWORD16 *pu2_src,
                                   WORD32 src_strd,
                                   WORD32 wd,
                                   WORD32 pad_size)
{
    WORD32 row;
    const UWORD16 *src;

    if((wd <= 0) || (pad_size <= 0))
    {
        return;
    }

    src = pu2_src - src_strd;

    /* Fast path: wd == 64 (Standard CTB width) - Preload once into registers */
    if(wd == 64)
    {
        uint16x8_t v0 = vld1q_u16(src + 0);
        uint16x8_t v1 = vld1q_u16(src + 8);
        uint16x8_t v2 = vld1q_u16(src + 16);
        uint16x8_t v3 = vld1q_u16(src + 24);
        uint16x8_t v4 = vld1q_u16(src + 32);
        uint16x8_t v5 = vld1q_u16(src + 40);
        uint16x8_t v6 = vld1q_u16(src + 48);
        uint16x8_t v7 = vld1q_u16(src + 56);
        UWORD16 *dst = pu2_src;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);
            vst1q_u16(dst + 16, v2);
            vst1q_u16(dst + 24, v3);
            vst1q_u16(dst + 32, v4);
            vst1q_u16(dst + 40, v5);
            vst1q_u16(dst + 48, v6);
            vst1q_u16(dst + 56, v7);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 32 */
    if(wd == 32)
    {
        uint16x8_t v0 = vld1q_u16(src + 0);
        uint16x8_t v1 = vld1q_u16(src + 8);
        uint16x8_t v2 = vld1q_u16(src + 16);
        uint16x8_t v3 = vld1q_u16(src + 24);
        UWORD16 *dst = pu2_src;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);
            vst1q_u16(dst + 16, v2);
            vst1q_u16(dst + 24, v3);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 16 */
    if(wd == 16)
    {
        uint16x8_t v0 = vld1q_u16(src + 0);
        uint16x8_t v1 = vld1q_u16(src + 8);
        UWORD16 *dst = pu2_src;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            vst1q_u16(dst + 8, v1);

            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 8 */
    if(wd == 8)
    {
        uint16x8_t v0 = vld1q_u16(src + 0);
        UWORD16 *dst = pu2_src;

        for(row = 0; row < pad_size; row++)
        {
            vst1q_u16(dst + 0, v0);
            dst += src_strd;
        }
        return;
    }

    /* Fast path: wd == 4 */
    if(wd == 4)
    {
        uint16x4_t v0 = vld1_u16(src + 0);
        UWORD16 *dst = pu2_src;

        for(row = 0; row < pad_size; row++)
        {
            vst1_u16(dst + 0, v0);
            dst += src_strd;
        }
        return;
    }

    /* Generic streaming path for arbitrary wd */
    for(row = 0; row < pad_size; row++)
    {
        UWORD16 *dst = pu2_src + row * src_strd;
        WORD32 col;

        for(col = 0; col + 32 <= wd; col += 32)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);
            uint16x8_t v2 = vld1q_u16(src + col + 16);
            uint16x8_t v3 = vld1q_u16(src + col + 24);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            vst1q_u16(dst + col + 16, v2);
            vst1q_u16(dst + col + 24, v3);
        }

        if(col + 16 <= wd)
        {
            uint16x8_t v0 = vld1q_u16(src + col + 0);
            uint16x8_t v1 = vld1q_u16(src + col + 8);

            vst1q_u16(dst + col + 0, v0);
            vst1q_u16(dst + col + 8, v1);
            col += 16;
        }

        if(col + 8 <= wd)
        {
            vst1q_u16(dst + col, vld1q_u16(src + col));
            col += 8;
        }

        if(col + 4 <= wd)
        {
            vst1_u16(dst + col, vld1_u16(src + col));
            col += 4;
        }

        for(; col < wd; col++)
        {
            dst[col] = src[col];
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (chroma interleaved block) at the left of a 2D array.
*
* @par Description:
*   The leftmost UV pair (pu4_src[0]) is replicated pad_size/2 times to the left
*   for ht rows using 32-bit vector stores. Supports standard 4:2:0/4:2:2 (pad_size 80)
*   and 4:4:4 (pad_size 160) via dedicated unrolled paths.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the source (col 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad (halved for 32-bit UV pairs)
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_left_chroma_neonintr(UWORD16 *pu2_src,
                                        WORD32 src_strd,
                                        WORD32 ht,
                                        WORD32 pad_size)
{
    WORD32 row;
    UWORD32 *pu4_src = (UWORD32 *)pu2_src;
    WORD32 strd32 = src_strd >> 1;

    if(ht <= 0)
    {
        return;
    }

    if(pad_size == 160)
    {
        /* 4:4:4 format - 160 UWORD16 samples (80 UV pairs, 320 bytes per row) */
        /* Process 4 rows per iteration (1280 bytes stored per loop) */
        for(row = 0; row <= ht - 4; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32]);

            UWORD32 *dst0 = pu4_src - 80;
            UWORD32 *dst1 = dst0 + strd32;
            UWORD32 *dst2 = dst1 + strd32;
            UWORD32 *dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0,  v0); vst1q_u32(dst0 + 4,  v0);
            vst1q_u32(dst0 + 8,  v0); vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0); vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0); vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0); vst1q_u32(dst0 + 36, v0);
            vst1q_u32(dst0 + 40, v0); vst1q_u32(dst0 + 44, v0);
            vst1q_u32(dst0 + 48, v0); vst1q_u32(dst0 + 52, v0);
            vst1q_u32(dst0 + 56, v0); vst1q_u32(dst0 + 60, v0);
            vst1q_u32(dst0 + 64, v0); vst1q_u32(dst0 + 68, v0);
            vst1q_u32(dst0 + 72, v0); vst1q_u32(dst0 + 76, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0,  v1); vst1q_u32(dst1 + 4,  v1);
            vst1q_u32(dst1 + 8,  v1); vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1); vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1); vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1); vst1q_u32(dst1 + 36, v1);
            vst1q_u32(dst1 + 40, v1); vst1q_u32(dst1 + 44, v1);
            vst1q_u32(dst1 + 48, v1); vst1q_u32(dst1 + 52, v1);
            vst1q_u32(dst1 + 56, v1); vst1q_u32(dst1 + 60, v1);
            vst1q_u32(dst1 + 64, v1); vst1q_u32(dst1 + 68, v1);
            vst1q_u32(dst1 + 72, v1); vst1q_u32(dst1 + 76, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0,  v2); vst1q_u32(dst2 + 4,  v2);
            vst1q_u32(dst2 + 8,  v2); vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2); vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2); vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2); vst1q_u32(dst2 + 36, v2);
            vst1q_u32(dst2 + 40, v2); vst1q_u32(dst2 + 44, v2);
            vst1q_u32(dst2 + 48, v2); vst1q_u32(dst2 + 52, v2);
            vst1q_u32(dst2 + 56, v2); vst1q_u32(dst2 + 60, v2);
            vst1q_u32(dst2 + 64, v2); vst1q_u32(dst2 + 68, v2);
            vst1q_u32(dst2 + 72, v2); vst1q_u32(dst2 + 76, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0,  v3); vst1q_u32(dst3 + 4,  v3);
            vst1q_u32(dst3 + 8,  v3); vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3); vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3); vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3); vst1q_u32(dst3 + 36, v3);
            vst1q_u32(dst3 + 40, v3); vst1q_u32(dst3 + 44, v3);
            vst1q_u32(dst3 + 48, v3); vst1q_u32(dst3 + 52, v3);
            vst1q_u32(dst3 + 56, v3); vst1q_u32(dst3 + 60, v3);
            vst1q_u32(dst3 + 64, v3); vst1q_u32(dst3 + 68, v3);
            vst1q_u32(dst3 + 72, v3); vst1q_u32(dst3 + 76, v3);

            pu4_src += 4 * strd32;
        }

        /* Remainder rows if ht is not a multiple of 4 */
        for(; row < ht; row++)
        {
            uint32x4_t v = vdupq_n_u32(pu4_src[0]);
            UWORD32 *dst = pu4_src - 80;

            vst1q_u32(dst + 0,  v); vst1q_u32(dst + 4,  v);
            vst1q_u32(dst + 8,  v); vst1q_u32(dst + 12, v);
            vst1q_u32(dst + 16, v); vst1q_u32(dst + 20, v);
            vst1q_u32(dst + 24, v); vst1q_u32(dst + 28, v);
            vst1q_u32(dst + 32, v); vst1q_u32(dst + 36, v);
            vst1q_u32(dst + 40, v); vst1q_u32(dst + 44, v);
            vst1q_u32(dst + 48, v); vst1q_u32(dst + 52, v);
            vst1q_u32(dst + 56, v); vst1q_u32(dst + 60, v);
            vst1q_u32(dst + 64, v); vst1q_u32(dst + 68, v);
            vst1q_u32(dst + 72, v); vst1q_u32(dst + 76, v);

            pu4_src += strd32;
        }
    }
    else
    {
        /* Standard 4:2:0 and 4:2:2 format - 80 UWORD16 samples (40 UV pairs, 160 bytes per row) */
        /* Process 4 rows per iteration (640 bytes stored per loop) */
        for(row = 0; row <= ht - 4; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32]);

            UWORD32 *dst0 = pu4_src - 40;
            UWORD32 *dst1 = dst0 + strd32;
            UWORD32 *dst2 = dst1 + strd32;
            UWORD32 *dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0,  v0); vst1q_u32(dst0 + 4,  v0);
            vst1q_u32(dst0 + 8,  v0); vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0); vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0); vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0); vst1q_u32(dst0 + 36, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0,  v1); vst1q_u32(dst1 + 4,  v1);
            vst1q_u32(dst1 + 8,  v1); vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1); vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1); vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1); vst1q_u32(dst1 + 36, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0,  v2); vst1q_u32(dst2 + 4,  v2);
            vst1q_u32(dst2 + 8,  v2); vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2); vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2); vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2); vst1q_u32(dst2 + 36, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0,  v3); vst1q_u32(dst3 + 4,  v3);
            vst1q_u32(dst3 + 8,  v3); vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3); vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3); vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3); vst1q_u32(dst3 + 36, v3);

            pu4_src += 4 * strd32;
        }

        /* Remainder rows if ht is not a multiple of 4 */
        for(; row < ht; row++)
        {
            uint32x4_t v = vdupq_n_u32(pu4_src[0]);
            UWORD32 *dst = pu4_src - 40;

            vst1q_u32(dst + 0,  v); vst1q_u32(dst + 4,  v);
            vst1q_u32(dst + 8,  v); vst1q_u32(dst + 12, v);
            vst1q_u32(dst + 16, v); vst1q_u32(dst + 20, v);
            vst1q_u32(dst + 24, v); vst1q_u32(dst + 28, v);
            vst1q_u32(dst + 32, v); vst1q_u32(dst + 36, v);

            pu4_src += strd32;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (chroma interleaved block) at the right of a 2D array.
*
* @par Description:
*   The rightmost UV pair (pu4_src[-1]) is replicated pad_size/2 times to the
*   right for ht rows using 32-bit vector stores. Supports standard 4:2:0/4:2:2 (pad_size 80)
*   and 4:4:4 (pad_size 160) via dedicated unrolled paths.
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the start of the right padding area (col wd)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad (halved for 32-bit UV pairs)
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_right_chroma_neonintr(UWORD16 *pu2_src,
                                         WORD32 src_strd,
                                         WORD32 ht,
                                         WORD32 pad_size)
{
    WORD32 row;
    UWORD32 *pu4_src = (UWORD32 *)pu2_src;
    WORD32 strd32 = src_strd >> 1;

    if(ht <= 0)
    {
        return;
    }

    if(pad_size == 160)
    {
        /* 4:4:4 format - 160 UWORD16 samples (80 UV pairs, 320 bytes per row) */
        /* Process 4 rows per iteration (1280 bytes stored per loop) */
        for(row = 0; row <= ht - 4; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[-1]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32 - 1]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32 - 1]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32 - 1]);

            UWORD32 *dst0 = pu4_src;
            UWORD32 *dst1 = dst0 + strd32;
            UWORD32 *dst2 = dst1 + strd32;
            UWORD32 *dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0,  v0); vst1q_u32(dst0 + 4,  v0);
            vst1q_u32(dst0 + 8,  v0); vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0); vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0); vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0); vst1q_u32(dst0 + 36, v0);
            vst1q_u32(dst0 + 40, v0); vst1q_u32(dst0 + 44, v0);
            vst1q_u32(dst0 + 48, v0); vst1q_u32(dst0 + 52, v0);
            vst1q_u32(dst0 + 56, v0); vst1q_u32(dst0 + 60, v0);
            vst1q_u32(dst0 + 64, v0); vst1q_u32(dst0 + 68, v0);
            vst1q_u32(dst0 + 72, v0); vst1q_u32(dst0 + 76, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0,  v1); vst1q_u32(dst1 + 4,  v1);
            vst1q_u32(dst1 + 8,  v1); vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1); vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1); vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1); vst1q_u32(dst1 + 36, v1);
            vst1q_u32(dst1 + 40, v1); vst1q_u32(dst1 + 44, v1);
            vst1q_u32(dst1 + 48, v1); vst1q_u32(dst1 + 52, v1);
            vst1q_u32(dst1 + 56, v1); vst1q_u32(dst1 + 60, v1);
            vst1q_u32(dst1 + 64, v1); vst1q_u32(dst1 + 68, v1);
            vst1q_u32(dst1 + 72, v1); vst1q_u32(dst1 + 76, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0,  v2); vst1q_u32(dst2 + 4,  v2);
            vst1q_u32(dst2 + 8,  v2); vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2); vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2); vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2); vst1q_u32(dst2 + 36, v2);
            vst1q_u32(dst2 + 40, v2); vst1q_u32(dst2 + 44, v2);
            vst1q_u32(dst2 + 48, v2); vst1q_u32(dst2 + 52, v2);
            vst1q_u32(dst2 + 56, v2); vst1q_u32(dst2 + 60, v2);
            vst1q_u32(dst2 + 64, v2); vst1q_u32(dst2 + 68, v2);
            vst1q_u32(dst2 + 72, v2); vst1q_u32(dst2 + 76, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0,  v3); vst1q_u32(dst3 + 4,  v3);
            vst1q_u32(dst3 + 8,  v3); vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3); vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3); vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3); vst1q_u32(dst3 + 36, v3);
            vst1q_u32(dst3 + 40, v3); vst1q_u32(dst3 + 44, v3);
            vst1q_u32(dst3 + 48, v3); vst1q_u32(dst3 + 52, v3);
            vst1q_u32(dst3 + 56, v3); vst1q_u32(dst3 + 60, v3);
            vst1q_u32(dst3 + 64, v3); vst1q_u32(dst3 + 68, v3);
            vst1q_u32(dst3 + 72, v3); vst1q_u32(dst3 + 76, v3);

            pu4_src += 4 * strd32;
        }

        /* Remainder rows if ht is not a multiple of 4 */
        for(; row < ht; row++)
        {
            uint32x4_t v = vdupq_n_u32(pu4_src[-1]);
            UWORD32 *dst = pu4_src;

            vst1q_u32(dst + 0,  v); vst1q_u32(dst + 4,  v);
            vst1q_u32(dst + 8,  v); vst1q_u32(dst + 12, v);
            vst1q_u32(dst + 16, v); vst1q_u32(dst + 20, v);
            vst1q_u32(dst + 24, v); vst1q_u32(dst + 28, v);
            vst1q_u32(dst + 32, v); vst1q_u32(dst + 36, v);
            vst1q_u32(dst + 40, v); vst1q_u32(dst + 44, v);
            vst1q_u32(dst + 48, v); vst1q_u32(dst + 52, v);
            vst1q_u32(dst + 56, v); vst1q_u32(dst + 60, v);
            vst1q_u32(dst + 64, v); vst1q_u32(dst + 68, v);
            vst1q_u32(dst + 72, v); vst1q_u32(dst + 76, v);

            pu4_src += strd32;
        }
    }
    else
    {
        /* Standard 4:2:0 and 4:2:2 format - 80 UWORD16 samples (40 UV pairs, 160 bytes per row) */
        /* Process 4 rows per iteration (640 bytes stored per loop) */
        for(row = 0; row <= ht - 4; row += 4)
        {
            uint32x4_t v0 = vdupq_n_u32(pu4_src[-1]);
            uint32x4_t v1 = vdupq_n_u32(pu4_src[strd32 - 1]);
            uint32x4_t v2 = vdupq_n_u32(pu4_src[2 * strd32 - 1]);
            uint32x4_t v3 = vdupq_n_u32(pu4_src[3 * strd32 - 1]);

            UWORD32 *dst0 = pu4_src;
            UWORD32 *dst1 = dst0 + strd32;
            UWORD32 *dst2 = dst1 + strd32;
            UWORD32 *dst3 = dst2 + strd32;

            /* Row 0 */
            vst1q_u32(dst0 + 0,  v0); vst1q_u32(dst0 + 4,  v0);
            vst1q_u32(dst0 + 8,  v0); vst1q_u32(dst0 + 12, v0);
            vst1q_u32(dst0 + 16, v0); vst1q_u32(dst0 + 20, v0);
            vst1q_u32(dst0 + 24, v0); vst1q_u32(dst0 + 28, v0);
            vst1q_u32(dst0 + 32, v0); vst1q_u32(dst0 + 36, v0);

            /* Row 1 */
            vst1q_u32(dst1 + 0,  v1); vst1q_u32(dst1 + 4,  v1);
            vst1q_u32(dst1 + 8,  v1); vst1q_u32(dst1 + 12, v1);
            vst1q_u32(dst1 + 16, v1); vst1q_u32(dst1 + 20, v1);
            vst1q_u32(dst1 + 24, v1); vst1q_u32(dst1 + 28, v1);
            vst1q_u32(dst1 + 32, v1); vst1q_u32(dst1 + 36, v1);

            /* Row 2 */
            vst1q_u32(dst2 + 0,  v2); vst1q_u32(dst2 + 4,  v2);
            vst1q_u32(dst2 + 8,  v2); vst1q_u32(dst2 + 12, v2);
            vst1q_u32(dst2 + 16, v2); vst1q_u32(dst2 + 20, v2);
            vst1q_u32(dst2 + 24, v2); vst1q_u32(dst2 + 28, v2);
            vst1q_u32(dst2 + 32, v2); vst1q_u32(dst2 + 36, v2);

            /* Row 3 */
            vst1q_u32(dst3 + 0,  v3); vst1q_u32(dst3 + 4,  v3);
            vst1q_u32(dst3 + 8,  v3); vst1q_u32(dst3 + 12, v3);
            vst1q_u32(dst3 + 16, v3); vst1q_u32(dst3 + 20, v3);
            vst1q_u32(dst3 + 24, v3); vst1q_u32(dst3 + 28, v3);
            vst1q_u32(dst3 + 32, v3); vst1q_u32(dst3 + 36, v3);

            pu4_src += 4 * strd32;
        }

        /* Remainder rows if ht is not a multiple of 4 */
        for(; row < ht; row++)
        {
            uint32x4_t v = vdupq_n_u32(pu4_src[-1]);
            UWORD32 *dst = pu4_src;

            vst1q_u32(dst + 0,  v); vst1q_u32(dst + 4,  v);
            vst1q_u32(dst + 8,  v); vst1q_u32(dst + 12, v);
            vst1q_u32(dst + 16, v); vst1q_u32(dst + 20, v);
            vst1q_u32(dst + 24, v); vst1q_u32(dst + 28, v);
            vst1q_u32(dst + 32, v); vst1q_u32(dst + 36, v);

            pu4_src += strd32;
        }
    }
}

/**
*******************************************************************************
*
* @brief
*   Padding (chroma interleaved block) horizontally on both left and right edges.
*
* @par Description:
*   Replicates leftmost UV pair (pu4_src[0]) to the left (pad_size/2 times) and
*   rightmost UV pair (pu4_src[wd/2 - 1]) to the right (pad_size/2 times) for ht rows.
*   Supports standard 4:2:0/4:2:2 (pad_size 80) and 4:4:4 (pad_size 160).
*
* @param[in,out] pu2_src
*   UWORD16 pointer to the valid chroma block start (col 0)
*
* @param[in] src_strd
*   Source buffer stride in UWORD16 units
*
* @param[in] ht
*   Height of the array to pad
*
* @param[in] wd
*   Width of the valid chroma array in UWORD16 units (2x chroma samples)
*
* @param[in] pad_size
*   Number of UWORD16 samples to pad to each side (halved for 32-bit UV pairs)
*
* @returns
*   Void
*
*******************************************************************************
*/
void ihevc_hbd_pad_horz_chroma_neonintr(UWORD16 *pu2_src,
                                        WORD32 src_strd,
                                        WORD32 ht,
                                        WORD32 wd,
                                        WORD32 pad_size)
{
    WORD32 row;
    UWORD32 *pu4_src = (UWORD32 *)pu2_src;
    WORD32 strd32 = src_strd >> 1;
    WORD32 wd32 = wd >> 1;

    if((ht <= 0) || (wd32 <= 0))
    {
        return;
    }

    if(pad_size == 160)
    {
        /* 4:4:4 format (pad_size 160 -> 80 UV pairs) */
        for(row = 0; row <= ht - 2; row += 2)
        {
            uint32x4_t vl0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t vr0 = vdupq_n_u32(pu4_src[wd32 - 1]);
            uint32x4_t vl1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t vr1 = vdupq_n_u32(pu4_src[strd32 + wd32 - 1]);

            UWORD32 *dst_l0 = pu4_src - 80;
            UWORD32 *dst_r0 = pu4_src + wd32;
            UWORD32 *dst_l1 = dst_l0 + strd32;
            UWORD32 *dst_r1 = dst_r0 + strd32;

            /* Left row 0 */
            vst1q_u32(dst_l0 + 0,  vl0); vst1q_u32(dst_l0 + 4,  vl0);
            vst1q_u32(dst_l0 + 8,  vl0); vst1q_u32(dst_l0 + 12, vl0);
            vst1q_u32(dst_l0 + 16, vl0); vst1q_u32(dst_l0 + 20, vl0);
            vst1q_u32(dst_l0 + 24, vl0); vst1q_u32(dst_l0 + 28, vl0);
            vst1q_u32(dst_l0 + 32, vl0); vst1q_u32(dst_l0 + 36, vl0);
            vst1q_u32(dst_l0 + 40, vl0); vst1q_u32(dst_l0 + 44, vl0);
            vst1q_u32(dst_l0 + 48, vl0); vst1q_u32(dst_l0 + 52, vl0);
            vst1q_u32(dst_l0 + 56, vl0); vst1q_u32(dst_l0 + 60, vl0);
            vst1q_u32(dst_l0 + 64, vl0); vst1q_u32(dst_l0 + 68, vl0);
            vst1q_u32(dst_l0 + 72, vl0); vst1q_u32(dst_l0 + 76, vl0);

            /* Right row 0 */
            vst1q_u32(dst_r0 + 0,  vr0); vst1q_u32(dst_r0 + 4,  vr0);
            vst1q_u32(dst_r0 + 8,  vr0); vst1q_u32(dst_r0 + 12, vr0);
            vst1q_u32(dst_r0 + 16, vr0); vst1q_u32(dst_r0 + 20, vr0);
            vst1q_u32(dst_r0 + 24, vr0); vst1q_u32(dst_r0 + 28, vr0);
            vst1q_u32(dst_r0 + 32, vr0); vst1q_u32(dst_r0 + 36, vr0);
            vst1q_u32(dst_r0 + 40, vr0); vst1q_u32(dst_r0 + 44, vr0);
            vst1q_u32(dst_r0 + 48, vr0); vst1q_u32(dst_r0 + 52, vr0);
            vst1q_u32(dst_r0 + 56, vr0); vst1q_u32(dst_r0 + 60, vr0);
            vst1q_u32(dst_r0 + 64, vr0); vst1q_u32(dst_r0 + 68, vr0);
            vst1q_u32(dst_r0 + 72, vr0); vst1q_u32(dst_r0 + 76, vr0);

            /* Left row 1 */
            vst1q_u32(dst_l1 + 0,  vl1); vst1q_u32(dst_l1 + 4,  vl1);
            vst1q_u32(dst_l1 + 8,  vl1); vst1q_u32(dst_l1 + 12, vl1);
            vst1q_u32(dst_l1 + 16, vl1); vst1q_u32(dst_l1 + 20, vl1);
            vst1q_u32(dst_l1 + 24, vl1); vst1q_u32(dst_l1 + 28, vl1);
            vst1q_u32(dst_l1 + 32, vl1); vst1q_u32(dst_l1 + 36, vl1);
            vst1q_u32(dst_l1 + 40, vl1); vst1q_u32(dst_l1 + 44, vl1);
            vst1q_u32(dst_l1 + 48, vl1); vst1q_u32(dst_l1 + 52, vl1);
            vst1q_u32(dst_l1 + 56, vl1); vst1q_u32(dst_l1 + 60, vl1);
            vst1q_u32(dst_l1 + 64, vl1); vst1q_u32(dst_l1 + 68, vl1);
            vst1q_u32(dst_l1 + 72, vl1); vst1q_u32(dst_l1 + 76, vl1);

            /* Right row 1 */
            vst1q_u32(dst_r1 + 0,  vr1); vst1q_u32(dst_r1 + 4,  vr1);
            vst1q_u32(dst_r1 + 8,  vr1); vst1q_u32(dst_r1 + 12, vr1);
            vst1q_u32(dst_r1 + 16, vr1); vst1q_u32(dst_r1 + 20, vr1);
            vst1q_u32(dst_r1 + 24, vr1); vst1q_u32(dst_r1 + 28, vr1);
            vst1q_u32(dst_r1 + 32, vr1); vst1q_u32(dst_r1 + 36, vr1);
            vst1q_u32(dst_r1 + 40, vr1); vst1q_u32(dst_r1 + 44, vr1);
            vst1q_u32(dst_r1 + 48, vr1); vst1q_u32(dst_r1 + 52, vr1);
            vst1q_u32(dst_r1 + 56, vr1); vst1q_u32(dst_r1 + 60, vr1);
            vst1q_u32(dst_r1 + 64, vr1); vst1q_u32(dst_r1 + 68, vr1);
            vst1q_u32(dst_r1 + 72, vr1); vst1q_u32(dst_r1 + 76, vr1);

            pu4_src += 2 * strd32;
        }

        if(ht & 1)
        {
            uint32x4_t vl = vdupq_n_u32(pu4_src[0]);
            uint32x4_t vr = vdupq_n_u32(pu4_src[wd32 - 1]);
            UWORD32 *dst_l = pu4_src - 80;
            UWORD32 *dst_r = pu4_src + wd32;

            /* Left row */
            vst1q_u32(dst_l + 0,  vl); vst1q_u32(dst_l + 4,  vl);
            vst1q_u32(dst_l + 8,  vl); vst1q_u32(dst_l + 12, vl);
            vst1q_u32(dst_l + 16, vl); vst1q_u32(dst_l + 20, vl);
            vst1q_u32(dst_l + 24, vl); vst1q_u32(dst_l + 28, vl);
            vst1q_u32(dst_l + 32, vl); vst1q_u32(dst_l + 36, vl);
            vst1q_u32(dst_l + 40, vl); vst1q_u32(dst_l + 44, vl);
            vst1q_u32(dst_l + 48, vl); vst1q_u32(dst_l + 52, vl);
            vst1q_u32(dst_l + 56, vl); vst1q_u32(dst_l + 60, vl);
            vst1q_u32(dst_l + 64, vl); vst1q_u32(dst_l + 68, vl);
            vst1q_u32(dst_l + 72, vl); vst1q_u32(dst_l + 76, vl);

            /* Right row */
            vst1q_u32(dst_r + 0,  vr); vst1q_u32(dst_r + 4,  vr);
            vst1q_u32(dst_r + 8,  vr); vst1q_u32(dst_r + 12, vr);
            vst1q_u32(dst_r + 16, vr); vst1q_u32(dst_r + 20, vr);
            vst1q_u32(dst_r + 24, vr); vst1q_u32(dst_r + 28, vr);
            vst1q_u32(dst_r + 32, vr); vst1q_u32(dst_r + 36, vr);
            vst1q_u32(dst_r + 40, vr); vst1q_u32(dst_r + 44, vr);
            vst1q_u32(dst_r + 48, vr); vst1q_u32(dst_r + 52, vr);
            vst1q_u32(dst_r + 56, vr); vst1q_u32(dst_r + 60, vr);
            vst1q_u32(dst_r + 64, vr); vst1q_u32(dst_r + 68, vr);
            vst1q_u32(dst_r + 72, vr); vst1q_u32(dst_r + 76, vr);
        }
    }
    else
    {
        /* Standard 4:2:0 / 4:2:2 format (pad_size 80 -> 40 UV pairs) */
        for(row = 0; row <= ht - 2; row += 2)
        {
            uint32x4_t vl0 = vdupq_n_u32(pu4_src[0]);
            uint32x4_t vr0 = vdupq_n_u32(pu4_src[wd32 - 1]);
            uint32x4_t vl1 = vdupq_n_u32(pu4_src[strd32]);
            uint32x4_t vr1 = vdupq_n_u32(pu4_src[strd32 + wd32 - 1]);

            UWORD32 *dst_l0 = pu4_src - 40;
            UWORD32 *dst_r0 = pu4_src + wd32;
            UWORD32 *dst_l1 = dst_l0 + strd32;
            UWORD32 *dst_r1 = dst_r0 + strd32;

            /* Left row 0 */
            vst1q_u32(dst_l0 + 0,  vl0); vst1q_u32(dst_l0 + 4,  vl0);
            vst1q_u32(dst_l0 + 8,  vl0); vst1q_u32(dst_l0 + 12, vl0);
            vst1q_u32(dst_l0 + 16, vl0); vst1q_u32(dst_l0 + 20, vl0);
            vst1q_u32(dst_l0 + 24, vl0); vst1q_u32(dst_l0 + 28, vl0);
            vst1q_u32(dst_l0 + 32, vl0); vst1q_u32(dst_l0 + 36, vl0);

            /* Right row 0 */
            vst1q_u32(dst_r0 + 0,  vr0); vst1q_u32(dst_r0 + 4,  vr0);
            vst1q_u32(dst_r0 + 8,  vr0); vst1q_u32(dst_r0 + 12, vr0);
            vst1q_u32(dst_r0 + 16, vr0); vst1q_u32(dst_r0 + 20, vr0);
            vst1q_u32(dst_r0 + 24, vr0); vst1q_u32(dst_r0 + 28, vr0);
            vst1q_u32(dst_r0 + 32, vr0); vst1q_u32(dst_r0 + 36, vr0);

            /* Left row 1 */
            vst1q_u32(dst_l1 + 0,  vl1); vst1q_u32(dst_l1 + 4,  vl1);
            vst1q_u32(dst_l1 + 8,  vl1); vst1q_u32(dst_l1 + 12, vl1);
            vst1q_u32(dst_l1 + 16, vl1); vst1q_u32(dst_l1 + 20, vl1);
            vst1q_u32(dst_l1 + 24, vl1); vst1q_u32(dst_l1 + 28, vl1);
            vst1q_u32(dst_l1 + 32, vl1); vst1q_u32(dst_l1 + 36, vl1);

            /* Right row 1 */
            vst1q_u32(dst_r1 + 0,  vr1); vst1q_u32(dst_r1 + 4,  vr1);
            vst1q_u32(dst_r1 + 8,  vr1); vst1q_u32(dst_r1 + 12, vr1);
            vst1q_u32(dst_r1 + 16, vr1); vst1q_u32(dst_r1 + 20, vr1);
            vst1q_u32(dst_r1 + 24, vr1); vst1q_u32(dst_r1 + 28, vr1);
            vst1q_u32(dst_r1 + 32, vr1); vst1q_u32(dst_r1 + 36, vr1);

            pu4_src += 2 * strd32;
        }

        if(ht & 1)
        {
            uint32x4_t vl = vdupq_n_u32(pu4_src[0]);
            uint32x4_t vr = vdupq_n_u32(pu4_src[wd32 - 1]);
            UWORD32 *dst_l = pu4_src - 40;
            UWORD32 *dst_r = pu4_src + wd32;

            /* Left row */
            vst1q_u32(dst_l + 0,  vl); vst1q_u32(dst_l + 4,  vl);
            vst1q_u32(dst_l + 8,  vl); vst1q_u32(dst_l + 12, vl);
            vst1q_u32(dst_l + 16, vl); vst1q_u32(dst_l + 20, vl);
            vst1q_u32(dst_l + 24, vl); vst1q_u32(dst_l + 28, vl);
            vst1q_u32(dst_l + 32, vl); vst1q_u32(dst_l + 36, vl);

            /* Right row */
            vst1q_u32(dst_r + 0,  vr); vst1q_u32(dst_r + 4,  vr);
            vst1q_u32(dst_r + 8,  vr); vst1q_u32(dst_r + 12, vr);
            vst1q_u32(dst_r + 16, vr); vst1q_u32(dst_r + 20, vr);
            vst1q_u32(dst_r + 24, vr); vst1q_u32(dst_r + 28, vr);
            vst1q_u32(dst_r + 32, vr); vst1q_u32(dst_r + 36, vr);
        }
    }
}

