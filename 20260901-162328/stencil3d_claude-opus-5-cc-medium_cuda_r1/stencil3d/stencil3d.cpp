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

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err__ = (call);                                                                    \
        if (err__ != cudaSuccess) {                                                                          \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__, __LINE__,              \
                   cudaGetErrorString(err__));                                                               \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread block shape of the stencil kernel. Each thread owns one (x, y) column
// and marches along z, keeping the z-neighbours in registers; the x/y
// neighbours are served by the L1/L2 caches (they are read by the same block).
static constexpr int BX = 32;
static constexpr int BY = 8;
// Blocks per SM targeted when partitioning the z range: more, shorter z chunks
// expose more parallelism, which matters because the kernel saturates the FP64
// pipelines of the device.
static constexpr int BLOCKS_PER_SM = 64;

// Correctly rounded x / 7.0 at the cost of three FP64 operations instead of the
// ~20 of a full IEEE double division. r = RN(1/7); q = RN(x*r) is within one
// ulp of x/7, e = RN(x - 7q) is exact, and the final FMA rounds once. Because 7
// is odd, x/7 is never close enough to a rounding midpoint for the residual
// correction to pick the wrong neighbour, so the result is bit-identical to
// x / 7.0 (checked against the hardware divide over billions of doubles).
__device__ __forceinline__ Real div7(const Real x) {
    constexpr Real r = 1.0 / 7.0;
    const Real q = x * r;
    const Real e = __fma_rn(-7.0, q, x);
    return __fma_rn(e, r, q);
}

__global__ void initializeGridKernel(Real* __restrict__ grid1, Real* __restrict__ grid2, const size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        const Real value = static_cast<Real>(idx % 19) * 1.0;
        // Both buffers are initialized: boundary values are copied unchanged by
        // every stencil iteration, so they stay at their initial value forever.
        grid1[idx] = value;
        grid2[idx] = value;
    }
}

// 7-point stencil computation over the interior points.
__global__ void __launch_bounds__(BX * BY, 6)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz, const int zChunk) {
    const int x = 1 + static_cast<int>(blockIdx.x) * BX + static_cast<int>(threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y) * BY + static_cast<int>(threadIdx.y);

    const int zBegin = 1 + static_cast<int>(blockIdx.z) * zChunk;
    const int zEnd = min(zBegin + zChunk, nz - 1); // exclusive
    if (zBegin >= zEnd || x >= nx - 1 || y >= ny - 1) return;

    const size_t nxny = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    size_t idx = idx3(x, y, zBegin, nx, ny);

    Real bottom = input[idx - nxny];
    Real center = input[idx];
    Real top = input[idx + nxny];

    for (int z = zBegin; z < zEnd; ++z) {
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil
        const Real value = div7(center + left + right + front + back + bottom + top);
        // The output buffer is not read by this kernel, so keep it out of the
        // caches and leave the whole L2 to the input stream.
        __stwt(&output[idx], value);

        bottom = center;
        center = top;
        idx += nxny;
        if (z + 2 < nz) {
            top = input[idx + nxny];
        }
    }
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
    if (gridSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    }

    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        const int initBlock = 256;
        const size_t initGrid = std::min<size_t>((gridSize + initBlock - 1) / initBlock, 65535);
        initializeGridKernel<<<static_cast<unsigned>(initGrid), initBlock>>>(d_grid1, d_grid2, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Launch configuration: enough blocks along z to fill the device
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    dim3 block(BX, BY);
    dim3 grid(1, 1, 1);
    int zChunk = 1;
    if (hasInterior) {
        const unsigned gx = static_cast<unsigned>((nx - 2 + BX - 1) / BX);
        const unsigned gy = static_cast<unsigned>((ny - 2 + BY - 1) / BY);
        const size_t interiorZ = nz - 2;

        int smCount = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
        // Split the z range into enough chunks to keep every SM oversubscribed.
        const size_t desiredBlocks = static_cast<size_t>(smCount) * BLOCKS_PER_SM;
        size_t gz = (desiredBlocks + static_cast<size_t>(gx) * gy - 1) / (static_cast<size_t>(gx) * gy);
        gz = std::max<size_t>(1, std::min(gz, interiorZ));
        zChunk = static_cast<int>((interiorZ + gz - 1) / gz);
        gz = (interiorZ + zChunk - 1) / zChunk;
        grid = dim3(gx, gy, static_cast<unsigned>(gz));
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const auto launchIteration = [&](const int iter) {
        const Real* in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* out = (iter % 2 == 0) ? d_grid2 : d_grid1;
        stencilKernel<<<grid, block, 0, stream>>>(in, out, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), zChunk);
    };

    // Small grids run the kernel in a few tens of microseconds, which is the
    // same order as the launch overhead. Replaying the iteration sequence from
    // a CUDA graph removes that overhead. The chunk size is even so that the
    // buffer roles are the same again after every replay.
    const int graphChunk = std::min(iterations - iterations % 2, 256);
    cudaGraphExec_t graphExec = nullptr;
    if (hasInterior && graphChunk >= 2) {
        cudaGraph_t graphDef = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int iter = 0; iter < graphChunk; ++iter) {
            launchIteration(iter);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graphDef));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graphDef, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graphDef));
    }

    // Warm up: bring the device out of its idle clock state and populate the
    // instruction/TLB caches before timing. This recomputes exactly what the
    // first timed iteration writes, so it cannot change the result.
    if (hasInterior && iterations > 0) {
        const auto warmupBegin = std::chrono::high_resolution_clock::now();
        const auto warmupBudget = std::chrono::milliseconds(300);
        do {
            launchIteration(0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } while (std::chrono::high_resolution_clock::now() - warmupBegin < warmupBudget);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (hasInterior) {
        int iter = 0;
        if (graphExec != nullptr) {
            for (; iter + graphChunk <= iterations; iter += graphChunk) {
                CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            }
        }
        for (; iter < iterations; ++iter) {
            launchIteration(iter);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    // On the GPU a single millisecond covers tens of millions of cell updates,
    // so the elapsed time is taken at microsecond resolution and only the
    // report is rounded to milliseconds.
    auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    long duration = (durationUs.count() + 500) / 1000;

    printf("Computation time: %ld ms\n", duration);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (durationUs.count() / 1000000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the final grid back to the host
    std::vector<Real> finalGrid(gridSize);
    if (gridSize > 0) {
        const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (graphExec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
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
