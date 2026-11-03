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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
extern "C" {
#include "ihevcd_fmt_conv.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

namespace {

enum class FmtConvType {
  k422spTo420p,
  k422spTo422p,
  k444spTo420p,
  k444spTo444p,
};

struct FmtConvBenchConfig {
  int wd;
  int ht;
  FmtConvType type;
  IV_ARCH_T arch;
};

using HbdFmtConv13Fn = ihevcd_hbd_fmt_conv_420sp_to_420p_ft;
using HbdFmtConv11Fn = ihevcd_fmt_conv_hbd_444sp_to_444p_ft;

static HbdFmtConv13Fn* GetHbdFmtConv13Fn(FmtConvType type, IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    switch (type) {
      case FmtConvType::k422spTo420p:
        return ihevcd_hbd_fmt_conv_422sp_to_420p;
      case FmtConvType::k422spTo422p:
        return ihevcd_hbd_fmt_conv_422sp_to_422p;
      case FmtConvType::k444spTo420p:
        return ihevcd_hbd_fmt_conv_444sp_to_420p;
      default:
        return nullptr;
    }
  }
#if defined(__aarch64__) || defined(__arm__)
  if (arch == ARCH_ARMV8_GENERIC || arch == ARCH_ARM_A9Q ||
      arch == ARCH_ARM_NEONINTR) {
    switch (type) {
      case FmtConvType::k422spTo420p:
        return ihevcd_hbd_fmt_conv_422sp_to_420p_neonintr;
      case FmtConvType::k422spTo422p:
        return ihevcd_hbd_fmt_conv_422sp_to_422p_neonintr;
      case FmtConvType::k444spTo420p:
        return ihevcd_hbd_fmt_conv_444sp_to_420p_neonintr;
      default:
        return nullptr;
    }
  }
#endif
  return nullptr;
}

static HbdFmtConv11Fn* GetHbdFmtConv11Fn(FmtConvType type, IV_ARCH_T arch) {
  if (type != FmtConvType::k444spTo444p) {
    return nullptr;
  }
  if (arch == ARCH_NA) {
    return ihevcd_fmt_conv_hbd_444sp_to_444p;
  }
#if defined(__aarch64__) || defined(__arm__)
  if (arch == ARCH_ARMV8_GENERIC || arch == ARCH_ARM_A9Q ||
      arch == ARCH_ARM_NEONINTR) {
    return ihevcd_fmt_conv_hbd_444sp_to_444p_neonintr;
  }
#endif
  return nullptr;
}

bool HasHbdFmtConvFn(IV_ARCH_T arch, FmtConvType type) {
  if (type == FmtConvType::k444spTo444p) {
    return GetHbdFmtConv11Fn(type, arch) != nullptr;
  }
  return GetHbdFmtConv13Fn(type, arch) != nullptr;
}

void BM_HbdFmtConv(benchmark::State& state, FmtConvBenchConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const FmtConvType type = config.type;
  const IV_ARCH_T arch = config.arch;
  const int disable_luma_copy = 0;
  const int is_u_first = 1;

  const bool is_444_src = (type == FmtConvType::k444spTo420p ||
                           type == FmtConvType::k444spTo444p);
  const bool is_444_dst = (type == FmtConvType::k444spTo444p);
  const bool is_422_dst = (type == FmtConvType::k422spTo422p);

  const int src_y_strd = wd;
  const int dst_y_strd = wd;
  const int src_uv_strd = is_444_src ? (2 * wd) : wd;
  const int dst_uv_wd = is_444_dst ? wd : (wd / 2);
  const int dst_uv_strd = dst_uv_wd;
  const int dst_uv_ht = (is_444_dst || is_422_dst) ? ht : (ht / 2);

  std::vector<UWORD16> y_src(src_y_strd * ht);
  std::vector<UWORD16> uv_src(src_uv_strd * ht);
  FillRandom(y_src, static_cast<UWORD16>(0), static_cast<UWORD16>(1023), 101);
  FillRandom(uv_src, static_cast<UWORD16>(0), static_cast<UWORD16>(1023), 202);

  std::vector<UWORD16> y_dst(dst_y_strd * ht, 0xCDCD);
  std::vector<UWORD16> u_dst(dst_uv_strd * dst_uv_ht, 0xCDCD);
  std::vector<UWORD16> v_dst(dst_uv_strd * dst_uv_ht, 0xCDCD);

  std::vector<UWORD16> y_ref(dst_y_strd * ht, 0xCDCD);
  std::vector<UWORD16> u_ref(dst_uv_strd * dst_uv_ht, 0xCDCD);
  std::vector<UWORD16> v_ref(dst_uv_strd * dst_uv_ht, 0xCDCD);

  if (type == FmtConvType::k444spTo444p) {
    HbdFmtConv11Fn* fn = GetHbdFmtConv11Fn(type, arch);
    if (!fn) {
      state.SkipWithError("Target function pointer is null");
      return;
    }

    if (arch != ARCH_NA) {
      HbdFmtConv11Fn* ref_fn = GetHbdFmtConv11Fn(type, ARCH_NA);
      if (ref_fn) {
        ref_fn(y_src.data(), uv_src.data(), y_ref.data(), u_ref.data(),
               v_ref.data(), wd, ht, src_y_strd, src_uv_strd, dst_y_strd,
               dst_uv_strd);
        fn(y_src.data(), uv_src.data(), y_dst.data(), u_dst.data(),
           v_dst.data(), wd, ht, src_y_strd, src_uv_strd, dst_y_strd,
           dst_uv_strd);

        if (!VerifyOutput2D(y_ref.data(), y_dst.data(), wd, ht, dst_y_strd)) {
          state.SkipWithError("Luma output mismatch between SIMD and C reference");
          return;
        }
        if (!VerifyOutput2D(u_ref.data(), u_dst.data(), dst_uv_wd, dst_uv_ht,
                            dst_uv_strd)) {
          state.SkipWithError("Chroma U output mismatch between SIMD and C reference");
          return;
        }
        if (!VerifyOutput2D(v_ref.data(), v_dst.data(), dst_uv_wd, dst_uv_ht,
                            dst_uv_strd)) {
          state.SkipWithError("Chroma V output mismatch between SIMD and C reference");
          return;
        }
      }
    }

    for (auto _ : state) {
      fn(y_src.data(), uv_src.data(), y_dst.data(), u_dst.data(), v_dst.data(),
         wd, ht, src_y_strd, src_uv_strd, dst_y_strd, dst_uv_strd);
      benchmark::DoNotOptimize(y_dst.data());
      benchmark::DoNotOptimize(u_dst.data());
      benchmark::DoNotOptimize(v_dst.data());
      benchmark::ClobberMemory();
    }
  } else {
    HbdFmtConv13Fn* fn = GetHbdFmtConv13Fn(type, arch);
    if (!fn) {
      state.SkipWithError("Target function pointer is null");
      return;
    }

    if (arch != ARCH_NA) {
      HbdFmtConv13Fn* ref_fn = GetHbdFmtConv13Fn(type, ARCH_NA);
      if (ref_fn) {
        ref_fn(y_src.data(), uv_src.data(), y_ref.data(), u_ref.data(),
               v_ref.data(), wd, ht, src_y_strd, src_uv_strd, dst_y_strd,
               dst_uv_strd, is_u_first, disable_luma_copy);
        fn(y_src.data(), uv_src.data(), y_dst.data(), u_dst.data(),
           v_dst.data(), wd, ht, src_y_strd, src_uv_strd, dst_y_strd,
           dst_uv_strd, is_u_first, disable_luma_copy);

        if (!disable_luma_copy &&
            !VerifyOutput2D(y_ref.data(), y_dst.data(), wd, ht, dst_y_strd)) {
          state.SkipWithError("Luma output mismatch between SIMD and C reference");
          return;
        }
        if (!VerifyOutput2D(u_ref.data(), u_dst.data(), dst_uv_wd, dst_uv_ht,
                            dst_uv_strd)) {
          state.SkipWithError("Chroma U output mismatch between SIMD and C reference");
          return;
        }
        if (!VerifyOutput2D(v_ref.data(), v_dst.data(), dst_uv_wd, dst_uv_ht,
                            dst_uv_strd)) {
          state.SkipWithError("Chroma V output mismatch between SIMD and C reference");
          return;
        }
      }
    }

    for (auto _ : state) {
      fn(y_src.data(), uv_src.data(), y_dst.data(), u_dst.data(), v_dst.data(),
         wd, ht, src_y_strd, src_uv_strd, dst_y_strd, dst_uv_strd,
         is_u_first, disable_luma_copy);
      benchmark::DoNotOptimize(y_dst.data());
      benchmark::DoNotOptimize(u_dst.data());
      benchmark::DoNotOptimize(v_dst.data());
      benchmark::ClobberMemory();
    }
  }

  state.SetItemsProcessed(state.iterations() * wd * ht);
  const size_t bytes_per_iter =
      (y_src.size() + y_dst.size() + uv_src.size() + u_dst.size() +
       v_dst.size()) *
      sizeof(UWORD16);
  state.SetBytesProcessed(state.iterations() * bytes_per_iter);
}

}  // namespace

void RegisterAllBenchmarks() {
  struct FmtConvOpInfo {
    FmtConvType type;
    const char* name;
  };

  const FmtConvOpInfo ops[] = {
      {FmtConvType::k422spTo420p, "422sp_to_420p"},
      {FmtConvType::k422spTo422p, "422sp_to_422p"},
      {FmtConvType::k444spTo420p, "444sp_to_420p"},
      {FmtConvType::k444spTo444p, "444sp_to_444p"},
  };

  const std::pair<int, int> resolutions[] = {
      {640, 360},
      {854, 480},
      {1280, 720},
      {1920, 1080},
      {3840, 2160},
  };

  for (const auto& op : ops) {
    for (const auto& res : resolutions) {
      const int wd = res.first;
      const int ht = res.second;
      for (auto arch : GetBenchmarkArchitectures()) {
        if (arch != ARCH_NA && !HasHbdFmtConvFn(arch, op.type)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);

        std::string name = "BM_HbdFmtConv/" + std::string(op.name) + "/" +
                           std::to_string(wd) + "x" + std::to_string(ht) + "/" +
                           arch_name;

        FmtConvBenchConfig cfg{wd, ht, op.type, arch};
        benchmark::RegisterBenchmark(name.c_str(),
                                     [cfg](benchmark::State& st) {
                                       BM_HbdFmtConv(st, cfg);
                                     })
            ->Unit(benchmark::kMicrosecond);
      }
    }
  }
}
