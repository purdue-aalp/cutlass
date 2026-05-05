/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example demonstrate a simple way to instantiate and run a TF32 GEMM using the new CUTLASS 3.0
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:

    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.

    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. Another advantage is that TMA can load in FP32 data and
    convert them implicitly to TF32.

    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).

    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the 
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can 
    improve performance.

    Examples:

      $ ./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm --m=2048 --n=2048 --k=2048 --rasterization=N --swizzle=2
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Simulator compatibility shims
// -----------------------------------------------------------------------------
// This translation unit preserves the original CUTLASS Hopper warp-specialized
// kernel selection, but replaces two PTX features that some simulators do not
// implement:
//   1. cute::elect_one_sync() / elect_one_leader_sync() no longer lower to
//      elect.sync. Instead, one active lane per warp is chosen via
//      __activemask() and __ffs(), which keeps the "exactly one active lane"
//      behavior even in diverged regions.
//   2. cutlass::arch::warpgroup_reg_alloc/dealloc() are compiled as no-ops by
//      forcing CUDA_CTA_RECONFIG_ACTIVATED=0 before CUTLASS pulls in
//      cutlass/arch/reg_reconfig.h. This removes setmaxnreg.{inc,dec} PTX while
//      leaving the surrounding control flow unchanged.
// -----------------------------------------------------------------------------

#ifndef CUTLASS_SIM_DISABLE_SETMAXNREG
#define CUTLASS_SIM_DISABLE_SETMAXNREG 1
#endif

#if CUTLASS_SIM_DISABLE_SETMAXNREG
#ifndef CUDA_CTA_RECONFIG_ACTIVATED
#define CUDA_CTA_RECONFIG_ACTIVATED 0
#endif
#endif

#ifndef CUTLASS_SIM_DISABLE_ELECT_SYNC
#define CUTLASS_SIM_DISABLE_ELECT_SYNC 1
#endif

#if CUTLASS_SIM_DISABLE_ELECT_SYNC
#define elect_one_sync cutlass48_sim_disabled_elect_one_sync_ptx
#define elect_one_leader_sync cutlass48_sim_disabled_elect_one_leader_sync_ptx
#include "cute/arch/cluster_sm90.hpp"
#undef elect_one_leader_sync
#undef elect_one_sync

namespace cute {

CUTE_HOST_DEVICE uint32_t elect_one_sync() {
#if defined(__CUDA_ARCH__)
  unsigned int active_mask = __activemask();
  __syncwarp(active_mask);
  unsigned int leader_lane = active_mask ? static_cast<unsigned int>(__ffs(active_mask) - 1) : 0u;
  unsigned int lane = static_cast<unsigned int>(threadIdx.x) & 0x1fu;
  return lane == leader_lane;
#else
  return true;
#endif
}

CUTE_HOST_DEVICE ElectOneLaneIdReturnType elect_one_leader_sync() {
#if defined(__CUDA_ARCH__)
  unsigned int active_mask = __activemask();
  __syncwarp(active_mask);
  unsigned int leader_lane = active_mask ? static_cast<unsigned int>(__ffs(active_mask) - 1) : 0u;
  unsigned int lane = static_cast<unsigned int>(threadIdx.x) & 0x1fu;
  return {lane == leader_lane, leader_lane};
#else
  return {true, 0};
#endif
}

} // namespace cute
#endif
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = float;                                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = float;                                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;                                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_32>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // GeForce Blackwell does not support TMA multicast
using StageCountType = cutlass::gemm::collective::StageCount<2>;            // Constrain pipeline stages to fit GeForce Blackwell shared memory
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    StageCountType,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  RasterOrderOptions raster;
  int swizzle;

  Options():
    help(false),
    // GPGPU-Sim default: a single TileShape (128x128) with K=128 (=4 K-tiles).
    // Functional simulation runs every thread sequentially, so the stock 5120x
    // 4096x4096 default would take days. The CUTLASS GEMM still dispatches a
    // real warp-specialized Hopper kernel here, just on a single CTA.
    m(128), n(128), k(128),
    alpha(1.f), beta(0.f),
    // Skip the profiling loop by default. The correctness path runs gemm.run()
    // exactly once and compares against the reference GEMM; replaying it 1000
    // times only stresses the simulator without adding signal.
    iterations(0),
    raster(RasterOrderOptions::Heuristic),
    swizzle(1)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Deterministic, well-conditioned initializer for the GEMM operands.
///
/// The stock CUTLASS example uses `BlockFillRandomUniform`, which fans out
/// to ~33k threads running the xorwow skip-ahead matrix; that is prohibitive
/// under functional simulation. Earlier sim_compat revisions worked around
/// the cost by zero-filling, but that turns the GEMM result into a trivial
/// zero buffer and makes verify() vacuous.
///
/// This initializer instead launches a tiny custom kernel that writes a
/// purely index-derived pattern. The pattern is:
///
///     val(i) = float((i + bias) % MOD - HALF) / float(HALF)
///
/// with MOD=17 and HALF=8, giving values in roughly [-1, +1] in 1/8 steps.
/// Properties:
///   * Deterministic and reproducible across runs/seeds, which makes it easy
///     to bisect simulator regressions.
///   * Bounded magnitude so FP32 accumulation across K stays well below
///     overflow even for moderate K extents.
///   * Different `bias` per buffer (A vs B vs C) so the result is not a
///     single constant tiled across the output, which makes the reference
///     comparison sensitive to per-(m,n) errors rather than just a global
///     magnitude check.
///   * One thread per element (capped block size 256), which keeps the
///     functional-sim launch cheap relative to the random PRNG kernel.
// NOTE: keep this kernel in a *named* namespace. CUTLASS's cute headers use
// their own anonymous namespace for some inline definitions, and adding a
// second anonymous namespace at file scope here causes nvcc/cudafe to emit a
// host stub that ambiguously references _GLOBAL__N__... between cute's
// anonymous namespace and ours. Naming the namespace sidesteps the collision.
namespace cutlass48_sim_compat {

template <class Element>
__global__ void pattern_fill(Element* dst, size_t n, uint32_t bias) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  // 32-bit arithmetic on purpose: a 64-bit `i % MOD` would lower to
  // mul.hi.u64 (now implemented in this simulator), but a 32-bit modulo is
  // strictly cheaper and avoids exercising additional simulator paths
  // during initialization. The truncation of `idx` to u32 is fine because
  // problem sizes here are <<< 2^32 elements.
  uint32_t const MOD  = 17u;
  int      const HALF = 8;
  uint32_t i32 = static_cast<uint32_t>(idx) + bias;
  int v = static_cast<int>(i32 % MOD) - HALF;  // [-8, +8]
  dst[idx] = static_cast<Element>(static_cast<float>(v) / static_cast<float>(HALF));
}

}  // namespace cutlass48_sim_compat

template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed = 2023) {

  size_t const count = block.size();
  if (count == 0) return true;

  int const block_size = 256;
  size_t const grid_size = (count + block_size - 1) / block_size;
  // Seed contributes a per-buffer bias so block_A / block_B / block_C all get
  // distinct patterns even though they share this single helper.
  uint32_t bias = static_cast<uint32_t>(seed & 0xffffffffu);

  cutlass48_sim_compat::pattern_fill<Element>
      <<<static_cast<unsigned>(grid_size), block_size>>>(
          block.get(), count, bias);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cerr << "cutlass48_sim_compat::pattern_fill launch failed: "
              << cudaGetErrorString(launch_err) << std::endl;
    return false;
  }
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    std::cerr << "cutlass48_sim_compat::pattern_fill sync failed: "
              << cudaGetErrorString(sync_err) << std::endl;
    return false;
  }
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    kernel_hw_info
  };

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8) 
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options &options) {
  // Step 1 of the post-bring-up plan: actually validate numerical
  // correctness against the reference device GEMM. The earlier sim_compat
  // revision short-circuited this to a tautology so the kernel could reach
  // graceful exit; now that the warp-specialized Hopper kernel runs
  // end-to-end we want a real correctness check on a small problem.
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  // Compute reference output using the basic CUTLASS device reference GEMM.
  // This is a plain CUDA kernel (no Hopper-specific PTX), so it can run
  // under functional simulation without exercising additional simulator
  // bring-up paths.
  DeviceGemmReference gemm_reference;
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Use a *relative* equality check rather than bit-exact.
  //
  // Even though both the CUTLASS Hopper kernel and the reference GEMM
  // operate on FP32 tensors with FP32 accumulation, the warp-specialized
  // path lowers to wgmma.mma_async tensor-core instructions which (on real
  // hardware) accumulate via the TF32 mode and may differ from the strict
  // FP32 reference at the last few mantissa bits. The simulator's
  // functional wgmma model may approximate this differently, so a relative
  // tolerance is the right comparison shape.
  //
  // For the 128x128x128 default this gives plenty of headroom while still
  // catching meaningful divergence in the wgmma / mbarrier / TMA paths.
  ElementC const epsilon(1e-3f);
  ElementC const nonzero_floor(1e-3f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      epsilon, nonzero_floor);

  if (!passed) {
    // Diagnostic dump: copy back the input buffers (A, B, C) plus the two
    // outputs so we can distinguish "pattern_fill did not actually run /
    // produced garbage" from "GEMM output is wrong" from "wgmma rounding
    // noise". The latter would manifest as small relative deltas; the
    // former manifests as structurally wrong values (zeros, NaNs, or a
    // single value tiled across the whole buffer).
    size_t const total   = block_D.size();
    size_t const a_total = block_A.size();
    size_t const b_total = block_B.size();
    size_t const c_total = block_C.size();
    size_t const dump    = total < 16 ? total : 16;

    std::vector<ElementA> a_host(a_total);
    std::vector<ElementB> b_host(b_total);
    std::vector<ElementC> c_host(c_total);
    std::vector<ElementC> got(total), ref(total);
    cudaMemcpy(a_host.data(), block_A.get(),    a_total * sizeof(ElementA), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host.data(), block_B.get(),    b_total * sizeof(ElementB), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_host.data(), block_C.get(),    c_total * sizeof(ElementC), cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(),    block_D.get(),    total   * sizeof(ElementC), cudaMemcpyDeviceToHost);
    cudaMemcpy(ref.data(),    block_ref_D.get(), total  * sizeof(ElementC), cudaMemcpyDeviceToHost);

    auto summarize = [](const char* name, auto const& v) {
      size_t n = v.size();
      size_t nans = 0, infs = 0, zeros = 0;
      double mn = std::numeric_limits<double>::infinity();
      double mx = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < n; ++i) {
        float x = static_cast<float>(v[i]);
        if (std::isnan(x)) { ++nans; continue; }
        if (std::isinf(x)) { ++infs; continue; }
        if (x == 0.0f) ++zeros;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
      }
      std::cerr << "  " << name << " n=" << n
                << " nans=" << nans << " infs=" << infs
                << " zeros=" << zeros
                << " min=" << mn << " max=" << mx << "\n";
    };
    summarize("A     ", a_host);
    summarize("B     ", b_host);
    summarize("C     ", c_host);
    summarize("D     ", got);
    summarize("ref_D ", ref);
    std::cerr << "  first " << dump << " of A: ";
    for (size_t i = 0; i < dump; ++i) std::cerr << static_cast<float>(a_host[i]) << ' ';
    std::cerr << "\n  first " << dump << " of B: ";
    for (size_t i = 0; i < dump; ++i) std::cerr << static_cast<float>(b_host[i]) << ' ';
    std::cerr << "\n  first " << dump << " of C: ";
    for (size_t i = 0; i < dump; ++i) std::cerr << static_cast<float>(c_host[i]) << ' ';
    std::cerr << "\n";

    size_t mismatches = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t worst_idx = 0;
    size_t nan_count = 0;
    size_t zero_got_nonzero_ref = 0;
    for (size_t i = 0; i < total; ++i) {
      float g = static_cast<float>(got[i]);
      float r = static_cast<float>(ref[i]);
      if (std::isnan(g) || std::isnan(r)) { ++nan_count; ++mismatches; continue; }
      double a = std::fabs(static_cast<double>(g) - static_cast<double>(r));
      double denom = std::max<double>(std::fabs(r), 1e-3);
      double rel = a / denom;
      bool eq = (a < 1e-3) || (rel < 1e-3);
      if (!eq) {
        ++mismatches;
        if (a > max_abs_diff) { max_abs_diff = a; max_rel_diff = rel; worst_idx = i; }
      }
      if (g == 0.0f && r != 0.0f) ++zero_got_nonzero_ref;
    }
    std::cerr << "  verify: " << mismatches << "/" << total
              << " mismatched elements (epsilon=1e-3)\n"
              << "  worst |got-ref| = " << max_abs_diff
              << " (rel=" << max_rel_diff << ") at flat idx=" << worst_idx
              << " got=" << static_cast<float>(got[worst_idx])
              << " ref=" << static_cast<float>(ref[worst_idx]) << "\n"
              << "  NaNs: " << nan_count
              << ", got==0 while ref!=0: " << zero_got_nonzero_ref << "\n";
    std::cerr << "  first " << dump << " (got | ref):\n";
    for (size_t i = 0; i < dump; ++i) {
      std::cerr << "    [" << i << "] " << static_cast<float>(got[i])
                << " | " << static_cast<float>(ref[i]) << "\n";
    }
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU with compute capability 9.0 or newer.\n";
    return 0;
  }

  
  

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
