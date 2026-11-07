// 3D 7-point stencil benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt, which sets the LANGUAGE
// property of this source to CUDA).  The computation is performed entirely on
// the GPU; the host only initializes/launches and copies the final grid back
// for result reporting and validation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Grid initialization kernel: grid[idx] = (idx % 19) * 1.0
// ---------------------------------------------------------------------------
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (Real)(idx % 19) * 1.0;
    }
}

// ---------------------------------------------------------------------------
// Correctly-rounded division by 7.0 without the hardware FP64 division sequence.
//
// On consumer GPUs the FP64 pipe is narrow (1/64 of FP32 on GA10x) and the IEEE
// double division expands to ~15 dependent FP64 instructions, which made the
// division - not memory bandwidth - the bottleneck of this stencil.
//
// Markstein's reciprocal-correction sequence: with y = RN(1/7),
//     q  = RN(x * y)
//     r  = x - 7*q        (exact, single FMA)
//     q' = RN(q + r * y)  == RN(x / 7)
// This equals the correctly rounded quotient for every normal x whose quotient
// neither overflows nor underflows (7's significand is not all ones, which is the
// only exclusion of Markstein's theorem).  Zero, subnormals, Inf, NaN and extreme
// magnitudes take the hardware division, so the result is bit-identical to
// x / 7.0 for every input.  Verified against x / 7.0 on 5e10 inputs covering
// arbitrary bit patterns, specials, and the value ranges of this benchmark.
// The guard uses only integer ops on the exponent field so the FP64 pipe sees
// exactly 3 instructions (DMUL + 2 DFMA) instead of ~15.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real div7(const Real x) {
    const Real y = 1.0 / 7.0;  // compile-time RN(1/7)
    const Real q = x * y;
    const Real r = fma(-q, 7.0, x);
    Real result = fma(r, y, q);
    // Biased exponent in [63, 1983]  <=>  |x| in [2^-960, 2^960]
    const unsigned e = ((unsigned)__double2hiint(x) >> 20) & 0x7ffu;
    if (e - 63u > 1920u) result = x / 7.0;
    return result;
}

// ---------------------------------------------------------------------------
// 7-point stencil kernel: one thread per interior cell.
//
// Thread blocks are BX x BY x BZ tiles of interior points; the x dimension maps to
// consecutive addresses so each warp reads/writes fully coalesced 8-byte elements,
// and the x/y/z neighbours are served from L1/L2 (every input element is touched
// by 7 threads, but only once from DRAM).  The stencil is FP64-throughput bound on
// GeForce parts, so the one-cell-per-thread form with maximal occupancy and no
// synchronisation won over shared-memory / register-marching variants.
//
// Floating point semantics match the CPU version exactly: the same left-to-right
// summation order and a correctly rounded division by 7.0 (see div7).  No
// fast-math flags are used and there are no multiplies that could be contracted.
// ---------------------------------------------------------------------------
constexpr int BX = 64;  // block extent in x (consecutive addresses)
constexpr int BY = 2;   // block extent in y
constexpr int BZ = 2;   // block extent in z

__global__ void __launch_bounds__(BX * BY * BZ)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz) {
    // Interior coordinates (offset by 1 for the boundary layer).
    const int x = blockIdx.x * BX + threadIdx.x + 1;
    const int y = blockIdx.y * BY + threadIdx.y + 1;
    const int z = blockIdx.z * BZ + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const size_t plane = (size_t)nx * ny;
    const size_t idx = (size_t)z * plane + (size_t)y * nx + x;

    const Real center = __ldg(input + idx);
    const Real left   = __ldg(input + idx - 1);
    const Real right  = __ldg(input + idx + 1);
    const Real front  = __ldg(input + idx - nx);
    const Real back   = __ldg(input + idx + nx);
    const Real bottom = __ldg(input + idx - plane);
    const Real top    = __ldg(input + idx + plane);

    // Simple averaging stencil (identical evaluation order to the CPU code)
    output[idx] = div7(center + left + right + front + back + bottom + top);
}

// Launch one stencil iteration on the device (asynchronous).
//
// Boundary handling: the CPU version copies input boundary values to the output
// every iteration.  Since the boundary of both buffers is initialised to the
// same values before the first iteration (see main), and only interior points
// are ever written, the boundary copy is a no-op and is skipped here.  The
// resulting grids are bit-for-bit identical to the CPU version.
void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return;  // no interior points

    const int inx = (int)(nx - 2);
    const int iny = (int)(ny - 2);
    const int inz = (int)(nz - 2);

    const dim3 block(BX, BY, BZ);
    const dim3 grid((inx + BX - 1) / BX, (iny + BY - 1) / BY, (inz + BZ - 1) / BZ);

    stencilKernel<<<grid, block>>>(input, output, (int)nx, (int)ny, (int)nz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids on the device (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    {
        const int threads = 256;
        size_t blocks = (gridSize + threads - 1) / threads;
        blocks = std::min<size_t>(std::max<size_t>(blocks, 1), 65535 * 16);
        initializeGridKernel<<<(unsigned)blocks, threads>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    // Both buffers carry identical boundary values; the stencil only writes interior
    // points, which makes the per-iteration boundary copy of the CPU version redundant.
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    // Use the full-resolution duration for the rate: GPU runs can finish in < 1 ms.
    const double seconds = std::chrono::duration<double>(end - start).count();
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy the final grid back to the host for result reporting / validation
    std::vector<Real> finalGrid(gridSize);
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
