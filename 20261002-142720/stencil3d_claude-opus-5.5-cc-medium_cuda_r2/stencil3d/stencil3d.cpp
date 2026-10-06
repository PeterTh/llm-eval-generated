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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// 3D index calculation
__host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Correctly rounded x / 7.0 without the FP64 divide sequence (FP64
// instruction throughput is the bottleneck on many GPUs).
// q = RN(x * RN(1/7)) is a faithful quotient, and rem = x - 7q is computed
// exactly by one FMA. The correctly rounded result is therefore q or one of
// its neighbours: it moves to the neighbour on rem's side iff |rem| / 7 exceeds
// half the spacing to that neighbour, i.e. |rem| > 3.5 * spacing. Since both
// sides are positive doubles, that test is an exact integer compare of bit
// patterns. (Ties cannot occur for division by 7.) This yields exactly the
// same bits as x / 7.0. Zero, non-finite or extreme-magnitude inputs, where
// the no-overflow/underflow preconditions might not hold, fall back to true
// IEEE division.
__device__ __forceinline__ Real divideBy7(const Real x) {
    const unsigned biasedExp = ((unsigned)__double2hiint(x) >> 20) & 0x7FFu;
    if (biasedExp - (1023u - 960u) > 2u * 960u) {
        return x / 7.0;
    }
    const Real q = x * (1.0 / 7.0);
    const Real rem = fma(-q, 7.0, x);

    const unsigned long long qBits = (unsigned long long)__double_as_longlong(q);
    const unsigned long long rBits = (unsigned long long)__double_as_longlong(rem);
    constexpr unsigned long long absMask = 0x7FFFFFFFFFFFFFFFull;
    const unsigned long long qMag = qBits & absMask;
    const unsigned long long rMag = rBits & absMask;
    const bool awayFromZero = ((qBits ^ rBits) >> 63) == 0;
    // Spacing towards zero halves when q is an exact power of two.
    const bool halfSpacing = !awayFromZero && (qMag & 0xFFFFFFFFFFFFFull) == 0;
    // threshold = 3.5 * spacing = 1.75 * 2^(exp(q) - 51 [- 1])
    const unsigned long long threshold =
        (((qMag >> 52) - 51ull - (halfSpacing ? 1ull : 0ull)) << 52) | 0xC000000000000ull;
    unsigned long long resBits = qBits;
    if (rMag > threshold) {
        resBits = awayFromZero ? qBits + 1ull : qBits - 1ull;
    }
    return __longlong_as_double((long long)resBits);
}

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

// Initialise a slab of the global grid. Local element i corresponds to global
// element globalOffset + i (the slab may include halo planes).
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n, const size_t globalOffset) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t idx = globalOffset + i;
        grid[i] = (idx % 19) * 1.0;
    }
}

// 7-point stencil computation over global planes [zBegin, zEnd).
// Each thread owns one (x, y) column and marches along a chunk of z, keeping
// the z-neighbours in registers. Boundary points are copied unchanged from
// input to output. Buffers hold a z-slab whose local plane 0 is global plane
// zOffset (planes zBegin-1 .. zEnd must be present when interior).
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const size_t nx, const size_t ny, const size_t nz,
              const size_t zBegin, const size_t zEnd, const size_t zOffset,
              const size_t zChunk) {
    const size_t x = blockIdx.x * (size_t)BLOCK_X + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t zFirst = zBegin + blockIdx.z * zChunk;
    if (zFirst >= zEnd) return;
    const size_t zLast = min(zFirst + zChunk, zEnd);
    const size_t plane = nx * ny;

    size_t idx = idx3(x, y, zFirst - zOffset, nx, ny);

    // Lateral boundary column: plain copy
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        for (size_t z = zFirst; z < zLast; ++z, idx += plane) {
            output[idx] = __ldg(&input[idx]);
        }
        return;
    }

    size_t z = zFirst;
    if (z == 0) {
        output[idx] = __ldg(&input[idx]);
        ++z;
        idx += plane;
    }
    const size_t zInEnd = min(zLast, nz - 1);  // interior z: [1, nz-1)

    if (z < zInEnd) {
        Real bottom = __ldg(&input[idx - plane]);
        Real center = __ldg(&input[idx]);
        for (; z < zInEnd; ++z, idx += plane) {
            const Real top = __ldg(&input[idx + plane]);
            const Real left = __ldg(&input[idx - 1]);
            const Real right = __ldg(&input[idx + 1]);
            const Real front = __ldg(&input[idx - nx]);
            const Real back = __ldg(&input[idx + nx]);

            // Simple averaging stencil (same summation order as reference)
            output[idx] = divideBy7(center + left + right + front + back + bottom + top);

            bottom = center;
            center = top;
        }
    }

    if (z < zLast && z == nz - 1) {
        output[idx] = __ldg(&input[idx]);
    }
}

// Per-GPU state for the z-slab domain decomposition. Each device owns global
// planes [z0, z1) and stores planes [z0-1, z1+1) (halo planes on both sides,
// clamped to the global domain).
struct Slab {
    int device = 0;
    int numSMs = 1;
    size_t z0 = 0, z1 = 0;        // owned planes
    size_t zLo = 0, zHi = 0;      // stored planes (owned + halos)
    Real* buf[2] = {nullptr, nullptr};
    cudaStream_t compute = nullptr;  // interior planes
    cudaStream_t edge = nullptr;     // edge planes + halo exchange (high priority)
    cudaEvent_t computeDone = nullptr;   // interior planes computed
    cudaEvent_t edgeKernDone = nullptr;  // edge planes computed
    cudaEvent_t edgeDone = nullptr;      // edge planes computed, staged and halos received
    cudaEvent_t sent = nullptr;      // edge planes staged to host
    Real* hostLow = nullptr;         // pinned staging of plane z0 (for g-1)
    Real* hostHigh = nullptr;        // pinned staging of plane z1-1 (for g+1)
};

void launchStencil(const Slab& s, const Real* in, Real* out,
                   const size_t nx, const size_t ny, const size_t nz,
                   const size_t zBegin, const size_t zEnd, cudaStream_t stream) {
    if (zBegin >= zEnd) return;
    const size_t bx = (nx + BLOCK_X - 1) / BLOCK_X;
    const size_t by = (ny + BLOCK_Y - 1) / BLOCK_Y;
    const size_t blocksXY = std::max<size_t>(bx * by, 1);
    const size_t depth = zEnd - zBegin;

    // Split z into chunks so there are enough blocks to fill the GPU while
    // keeping chunks long enough for register reuse along z.
    const size_t targetBlocks = (size_t)s.numSMs * 16;
    size_t numChunks = std::max<size_t>((targetBlocks + blocksXY - 1) / blocksXY, 1);
    numChunks = std::min<size_t>(numChunks, std::max<size_t>(depth / 8, 1));
    numChunks = std::min<size_t>(numChunks, 65535);
    const size_t zChunk = (depth + numChunks - 1) / numChunks;
    numChunks = (depth + zChunk - 1) / zChunk;

    const dim3 grid((unsigned)bx, (unsigned)by, (unsigned)numChunks);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, zBegin, zEnd, s.zLo, zChunk);
}

// Choose how many GPUs to use: enough work per device to amortise the
// per-iteration halo exchange (which is staged through host memory when
// peer access is unavailable), and at least a few planes per device.
int chooseNumDevices(const size_t nx, const size_t ny, const size_t nz) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(EXIT_FAILURE);
    }
    constexpr size_t minCellsPerDevice = size_t(4) << 20;
    constexpr size_t minPlanesPerDevice = 4;
    const size_t cells = nx * ny * nz;
    size_t n = std::min<size_t>((size_t)deviceCount, std::max<size_t>(cells / minCellsPerDevice, 1));
    n = std::min<size_t>(n, std::max<size_t>(nz / minPlanesPerDevice, 1));
    return (int)std::max<size_t>(n, 1);
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
    
    const size_t plane = nx * ny;
    const bool nonEmpty = gridSize > 0;

    // Domain decomposition along z across the available GPUs
    const int numDevices = nonEmpty ? chooseNumDevices(nx, ny, nz) : 1;
    std::vector<Slab> slabs(numDevices);
    for (int g = 0; g < numDevices; ++g) {
        Slab& s = slabs[g];
        s.device = g;
        s.z0 = nz * g / numDevices;
        s.z1 = nz * (g + 1) / numDevices;
        s.zLo = s.z0 > 0 ? s.z0 - 1 : 0;
        s.zHi = std::min(s.z1 + 1, nz);
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceGetAttribute(&s.numSMs, cudaDevAttrMultiProcessorCount, g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
        int leastPriority = 0, greatestPriority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s.edge, cudaStreamNonBlocking, greatestPriority));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.computeDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.edgeKernDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.edgeDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.sent, cudaEventDisableTiming));
        if (numDevices > 1) {
            // Peer access is often unavailable (e.g. GeForce over PCIe), so
            // halo planes are exchanged through pinned host buffers.
            CUDA_CHECK(cudaHostAlloc(&s.hostLow, plane * sizeof(Real), cudaHostAllocPortable));
            CUDA_CHECK(cudaHostAlloc(&s.hostHigh, plane * sizeof(Real), cudaHostAllocPortable));
        }
    }

    // Allocate device grids (double buffering)
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        const size_t n = (s.zHi - s.zLo) * plane;
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&s.buf[b], std::max<size_t>(n, 1) * sizeof(Real)));
        }
        // grid2 starts zero-initialised, as with std::vector
        CUDA_CHECK(cudaMemsetAsync(s.buf[1], 0, n * sizeof(Real), s.compute));
    }

    // Initialize (each device initialises its slab including halo planes)
    printf("Initializing grid...\n");
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        const size_t n = (s.zHi - s.zLo) * plane;
        if (n == 0) continue;
        const int threads = 256;
        const size_t blocks = std::min<size_t>((n + threads - 1) / threads, (size_t)s.numSMs * 32);
        initializeGridKernel<<<(unsigned)blocks, threads, 0, s.compute>>>(s.buf[0], n, s.zLo * plane);
        CUDA_CHECK(cudaGetLastError());
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaEventRecord(s.computeDone, s.compute));
        CUDA_CHECK(cudaEventRecord(s.edgeKernDone, s.edge));
        CUDA_CHECK(cudaEventRecord(s.edgeDone, s.edge));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (nonEmpty) {
        for (int iter = 0; iter < iterations; ++iter) {
            const int inBuf = iter % 2;  // grid1 -> grid2 on even iterations
            const int outBuf = 1 - inBuf;

            if (numDevices == 1) {
                const Slab& s = slabs[0];
                launchStencil(s, s.buf[inBuf], s.buf[outBuf], nx, ny, nz, 0, nz, s.compute);
                continue;
            }

            // Dependencies on the previous iteration (all waits are issued
            // before any event of this iteration is recorded):
            //  - interior planes only touch owned planes, so they just need
            //    the previous edge planes (RAW on z0 / z1-1, WAR on z0+1 /
            //    z1-2);
            //  - edge planes need the previous interior (RAW / WAR) and the
            //    neighbours' halo receives (they read our staging buffers),
            //    the own halo receive is ordered by the stream itself.
            for (int g = 0; g < numDevices; ++g) {
                Slab& s = slabs[g];
                CUDA_CHECK(cudaSetDevice(s.device));
                CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.edgeKernDone, 0));
                CUDA_CHECK(cudaStreamWaitEvent(s.edge, s.computeDone, 0));
                if (g > 0) CUDA_CHECK(cudaStreamWaitEvent(s.edge, slabs[g - 1].edgeDone, 0));
                if (g + 1 < numDevices) CUDA_CHECK(cudaStreamWaitEvent(s.edge, slabs[g + 1].edgeDone, 0));
            }

            // Edge planes first (edge stream), staged to host for the
            // neighbours; interior planes overlap on the compute stream.
            for (int g = 0; g < numDevices; ++g) {
                Slab& s = slabs[g];
                CUDA_CHECK(cudaSetDevice(s.device));
                const Real* in = s.buf[inBuf];
                Real* out = s.buf[outBuf];
                const size_t lowEdgeEnd = std::min(s.z0 + 1, s.z1);
                const size_t highEdgeBegin = std::max(s.z1 - 1, lowEdgeEnd);

                launchStencil(s, in, out, nx, ny, nz, s.z0, lowEdgeEnd, s.edge);
                launchStencil(s, in, out, nx, ny, nz, highEdgeBegin, s.z1, s.edge);
                CUDA_CHECK(cudaEventRecord(s.edgeKernDone, s.edge));
                if (g > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(s.hostLow, out + (s.z0 - s.zLo) * plane,
                                               plane * sizeof(Real), cudaMemcpyDeviceToHost, s.edge));
                }
                if (g + 1 < numDevices) {
                    CUDA_CHECK(cudaMemcpyAsync(s.hostHigh, out + (s.z1 - 1 - s.zLo) * plane,
                                               plane * sizeof(Real), cudaMemcpyDeviceToHost, s.edge));
                }
                CUDA_CHECK(cudaEventRecord(s.sent, s.edge));

                launchStencil(s, in, out, nx, ny, nz, lowEdgeEnd, highEdgeBegin, s.compute);
                CUDA_CHECK(cudaEventRecord(s.computeDone, s.compute));
            }

            // Receive halo planes from the neighbours
            for (int g = 0; g < numDevices; ++g) {
                Slab& s = slabs[g];
                CUDA_CHECK(cudaSetDevice(s.device));
                Real* out = s.buf[outBuf];
                if (g > 0) {
                    const Slab& d = slabs[g - 1];
                    CUDA_CHECK(cudaStreamWaitEvent(s.edge, d.sent, 0));
                    CUDA_CHECK(cudaMemcpyAsync(out, d.hostHigh,
                                               plane * sizeof(Real), cudaMemcpyHostToDevice, s.edge));
                }
                if (g + 1 < numDevices) {
                    const Slab& u = slabs[g + 1];
                    CUDA_CHECK(cudaStreamWaitEvent(s.edge, u.sent, 0));
                    CUDA_CHECK(cudaMemcpyAsync(out + (s.zHi - 1 - s.zLo) * plane, u.hostLow,
                                               plane * sizeof(Real), cudaMemcpyHostToDevice, s.edge));
                }
                CUDA_CHECK(cudaEventRecord(s.edgeDone, s.edge));
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Gather the final grid (grid1 after an even number of iterations)
    const int finalBuf = iterations % 2;
    std::vector<Real> finalGrid(gridSize);
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMemcpy(finalGrid.data() + s.z0 * plane,
                              s.buf[finalBuf] + (s.z0 - s.zLo) * plane,
                              (s.z1 - s.z0) * plane * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(s.buf[0]));
        CUDA_CHECK(cudaFree(s.buf[1]));
        CUDA_CHECK(cudaEventDestroy(s.computeDone));
        CUDA_CHECK(cudaEventDestroy(s.edgeKernDone));
        CUDA_CHECK(cudaEventDestroy(s.edgeDone));
        CUDA_CHECK(cudaEventDestroy(s.sent));
        if (s.hostLow) CUDA_CHECK(cudaFreeHost(s.hostLow));
        if (s.hostHigh) CUDA_CHECK(cudaFreeHost(s.hostHigh));
        CUDA_CHECK(cudaStreamDestroy(s.compute));
        CUDA_CHECK(cudaStreamDestroy(s.edge));
    }
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
