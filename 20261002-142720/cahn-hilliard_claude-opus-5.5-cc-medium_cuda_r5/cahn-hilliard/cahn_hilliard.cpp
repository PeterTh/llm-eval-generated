#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

// Physical parameters (compile-time constants so the device code can fold them)
constexpr double kDx = 1.0;
constexpr double kDy = 1.0;
constexpr double kDz = 1.0;
constexpr double kDt = 0.01;
constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = (2.0 / 9.0);
constexpr double kGamma = 0.5;
constexpr double kD = 1.0;

// Tile of output cells handled by one thread block (marching along z)
constexpr int TX = 32;
constexpr int TY = 16;
constexpr int HX = TX + 2;  // mu tile incl. 1-cell halo
constexpr int HY = TY + 2;

__device__ __forceinline__ int clampi(const int v, const int lo, const int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Chemical potential at (x, y, z) with clamped boundary conditions
__device__ __forceinline__ double chemicalPotential(const double* __restrict__ c, const int nx, const int ny,
                                                   const int nz, const size_t plane, const int zBase,
                                                   const int x, const int y, const int z) {
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : 0;

    const double* row = c + (size_t)(z - zBase) * plane + (size_t)y * nx;
    const double cv = __ldg(row + x);
    const double cxx = (__ldg(row + xp) + __ldg(row + xn) - 2.0 * cv) / (kDx * kDx);
    const double cyy = (__ldg(row + (ptrdiff_t)(yp - y) * nx + x) + __ldg(row + (ptrdiff_t)(yn - y) * nx + x) -
                        2.0 * cv) / (kDy * kDy);
    const double czz = (__ldg(row + (ptrdiff_t)(zp - z) * (ptrdiff_t)plane + x) +
                        __ldg(row + (ptrdiff_t)(zn - z) * (ptrdiff_t)plane + x) - 2.0 * cv) / (kDz * kDz);
    const double lap = cxx + cyy + czz;

    return 4.5 * ((cv + 1.0) * kEAA + (cv - 1.0) * kEBB - 2.0 * cv * kEAB)
           + 3.0 * cv + cv * cv * cv
           - kGamma * lap;
}

// Fused time step: computes mu (with halo) into shared memory plane by plane
// and applies the Cahn-Hilliard update, marching along z over a chunk.
// Arrays hold the global planes [zBase, ...); this call updates planes [zStart, zEnd).
__global__ void __launch_bounds__(TX * TY)
cahnHilliardStep(const double* __restrict__ cold, double* __restrict__ cnew,
                 const int nx, const int ny, const int nz,
                 const int zBase, const int zStart, const int zEnd, const int zChunk) {
    __shared__ double smu[3][HY][HX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int x0 = blockIdx.x * TX;
    const int y0 = blockIdx.y * TY;
    const int z0 = zStart + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    const size_t plane = (size_t)nx * ny;

    const int x = x0 + tx;
    const int y = y0 + ty;
    const bool active = (x < nx) && (y < ny);

    auto computePlane = [&](const int p) {
        double (*dst)[HX] = smu[p % 3];
        for (int k = tid; k < HX * HY; k += TX * TY) {
            const int j = k / HX;
            const int i = k - j * HX;
            const int gx = clampi(x0 - 1 + i, 0, nx - 1);
            const int gy = clampi(y0 - 1 + j, 0, ny - 1);
            dst[j][i] = chemicalPotential(cold, nx, ny, nz, plane, zBase, gx, gy, p);
        }
    };

    if (z0 > 0) computePlane(z0 - 1);
    computePlane(z0);
    int last = z0;

    const int lx = tx + 1;
    const int ly = ty + 1;
    // Neighbor positions in the shared tile (clamped at the domain boundary)
    const int lxp = (x < nx - 1) ? lx + 1 : lx;
    const int lxn = (x > 0) ? lx - 1 : lx;
    const int lyp = (y < ny - 1) ? ly + 1 : ly;
    const int lyn = (y > 0) ? ly - 1 : ly;

    for (int z = z0; z < z1; ++z) {
        const int zp = (z < nz - 1) ? z + 1 : z;
        const int zn = (z > 0) ? z - 1 : 0;
        if (zp > last) {
            computePlane(zp);
            last = zp;
        }
        __syncthreads();

        if (active) {
            const double (*mc)[HX] = smu[z % 3];
            const double (*mp)[HX] = smu[zp % 3];
            const double (*mn)[HX] = smu[zn % 3];
            const double m = mc[ly][lx];
            const double mxx = (mc[ly][lxp] + mc[ly][lxn] - 2.0 * m) / (kDx * kDx);
            const double myy = (mc[lyp][lx] + mc[lyn][lx] - 2.0 * m) / (kDy * kDy);
            const double mzz = (mp[ly][lx] + mn[ly][lx] - 2.0 * m) / (kDz * kDz);
            const size_t idx = (size_t)(z - zBase) * plane + (size_t)y * nx + x;
            cnew[idx] = cold[idx] + kDt * kD * (mxx + myy + mzz);
        }
        __syncthreads();
    }
}

// Initialize concentration field
__global__ void initializeConcentration(double* __restrict__ c, const size_t vol, const size_t first,
                                        const size_t count) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const size_t linear_id = first + i;
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Per-GPU slab of the domain decomposed along z
struct Slab {
    int dev = 0;
    int zs = 0, ze = 0;  // owned global planes [zs, ze)
    int zb = 0, zt = 0;  // stored global planes [zb, zt) incl. halo of up to 2 planes
    double* buf[2] = {nullptr, nullptr};
    double* sendLo = nullptr;  // pinned host staging of planes [zs, zs + 2) for the lower neighbor
    double* sendHi = nullptr;  // pinned host staging of planes [ze - 2, ze) for the upper neighbor
    cudaStream_t stream = nullptr;      // compute stream
    cudaStream_t copyStream = nullptr;  // halo exchange stream
    cudaEvent_t evBoundary = nullptr;   // boundary planes of the current step computed
    cudaEvent_t evSent = nullptr;       // boundary planes staged to host
    cudaEvent_t evHaloIn = nullptr;     // halo planes received
    dim3 grid;
    int zChunk = 1;
};

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // Decompose the domain along z over the available GPUs. Each slab needs at
    // least kMinPlanes planes, and small problems stay on a single GPU.
    constexpr int kMinPlanes = 16;
    constexpr size_t kMinCellsPerGpu = size_t(1) << 18;
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    int numGpus = std::max(1, numDevices);
    numGpus = std::min<size_t>(numGpus, std::max<size_t>(1, nz / kMinPlanes));
    numGpus = std::min<size_t>(numGpus, std::max<size_t>(1, gridSize / kMinCellsPerGpu));

    std::vector<Slab> slabs(numGpus);
    const dim3 block(TX, TY, 1);
    const int gx = (int)((nx + TX - 1) / TX);
    const int gy = (int)((ny + TY - 1) / TY);
    const size_t haloBytes = 2 * planeSize * sizeof(double);
    for (int g = 0; g < numGpus; ++g) {
        Slab& s = slabs[g];
        s.dev = g;
        s.zs = (int)(nz * g / numGpus);
        s.ze = (int)(nz * (g + 1) / numGpus);
        s.zb = (g > 0) ? s.zs - 2 : s.zs;
        s.zt = (g + 1 < numGpus) ? s.ze + 2 : s.ze;
        CUDA_CHECK(cudaSetDevice(s.dev));
        const size_t stored = (size_t)(s.zt - s.zb) * planeSize;
        CUDA_CHECK(cudaMalloc(&s.buf[0], stored * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.buf[1], stored * sizeof(double)));
        if (g > 0) CUDA_CHECK(cudaHostAlloc(&s.sendLo, haloBytes, cudaHostAllocPortable));
        if (g + 1 < numGpus) CUDA_CHECK(cudaHostAlloc(&s.sendHi, haloBytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.copyStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.evBoundary, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.evSent, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.evHaloIn, cudaEventDisableTiming));

        // Launch configuration: split z into chunks so that there are enough blocks
        int numSMs = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, s.dev));
        const int owned = s.ze - s.zs;
        s.zChunk = 64;
        while (s.zChunk > 4 && (long long)gx * gy * ((owned + s.zChunk - 1) / s.zChunk) < 4LL * numSMs) s.zChunk /= 2;
    }

    auto launchStep = [&](const Slab& s, const double* cold, double* cnew, const int zStart, const int zEnd,
                          const int zChunk) {
        if (zEnd <= zStart) return;
        const dim3 grid(gx, gy, (zEnd - zStart + zChunk - 1) / zChunk);
        cahnHilliardStep<<<grid, block, 0, s.stream>>>(cold, cnew, (int)nx, (int)ny, (int)nz, s.zb, zStart, zEnd,
                                                       zChunk);
        CUDA_CHECK(cudaGetLastError());
    };

    // Initialize concentration field (including halo planes, so no initial exchange is needed)
    printf("Initializing concentration field...\n");
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.dev));
        initializeConcentration<<<1024, 256, 0, s.stream>>>(s.buf[0], gridSize, (size_t)s.zb * planeSize,
                                                            (size_t)(s.zt - s.zb) * planeSize);
        CUDA_CHECK(cudaGetLastError());
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.dev));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    for (int t = 0; t < iterations; ++t) {
        const int nxt = cur ^ 1;
        // Compute chemical potential and update concentration on every slab:
        // boundary planes first so their exchange overlaps with the interior.
        for (int g = 0; g < numGpus; ++g) {
            Slab& s = slabs[g];
            CUDA_CHECK(cudaSetDevice(s.dev));
            if (numGpus == 1) {
                launchStep(s, s.buf[cur], s.buf[nxt], s.zs, s.ze, s.zChunk);
                continue;
            }
            const bool hasLo = g > 0;
            const bool hasHi = g + 1 < numGpus;
            if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(s.stream, s.evHaloIn, 0));
            if (hasLo) launchStep(s, s.buf[cur], s.buf[nxt], s.zs, s.zs + 2, 2);
            if (hasHi) launchStep(s, s.buf[cur], s.buf[nxt], s.ze - 2, s.ze, 2);
            CUDA_CHECK(cudaEventRecord(s.evBoundary, s.stream));
            launchStep(s, s.buf[cur], s.buf[nxt], hasLo ? s.zs + 2 : s.zs, hasHi ? s.ze - 2 : s.ze, s.zChunk);

            // Stage boundary planes to host once neighbors consumed the previous ones
            if (t > 0) {
                if (hasLo) CUDA_CHECK(cudaStreamWaitEvent(s.copyStream, slabs[g - 1].evHaloIn, 0));
                if (hasHi) CUDA_CHECK(cudaStreamWaitEvent(s.copyStream, slabs[g + 1].evHaloIn, 0));
            }
            CUDA_CHECK(cudaStreamWaitEvent(s.copyStream, s.evBoundary, 0));
            if (hasLo)
                CUDA_CHECK(cudaMemcpyAsync(s.sendLo, s.buf[nxt] + (size_t)(s.zs - s.zb) * planeSize, haloBytes,
                                           cudaMemcpyDeviceToHost, s.copyStream));
            if (hasHi)
                CUDA_CHECK(cudaMemcpyAsync(s.sendHi, s.buf[nxt] + (size_t)(s.ze - 2 - s.zb) * planeSize, haloBytes,
                                           cudaMemcpyDeviceToHost, s.copyStream));
            CUDA_CHECK(cudaEventRecord(s.evSent, s.copyStream));
        }
        // Receive halo planes from the neighbors' staging buffers
        if (numGpus > 1) {
            for (int g = 0; g < numGpus; ++g) {
                Slab& s = slabs[g];
                CUDA_CHECK(cudaSetDevice(s.dev));
                if (g > 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(s.copyStream, slabs[g - 1].evSent, 0));
                    CUDA_CHECK(cudaMemcpyAsync(s.buf[nxt], slabs[g - 1].sendHi, haloBytes, cudaMemcpyHostToDevice,
                                               s.copyStream));
                }
                if (g + 1 < numGpus) {
                    CUDA_CHECK(cudaStreamWaitEvent(s.copyStream, slabs[g + 1].evSent, 0));
                    CUDA_CHECK(cudaMemcpyAsync(s.buf[nxt] + (size_t)(s.ze - s.zb) * planeSize, slabs[g + 1].sendLo,
                                               haloBytes, cudaMemcpyHostToDevice, s.copyStream));
                }
                CUDA_CHECK(cudaEventRecord(s.evHaloIn, s.copyStream));
            }
        }
        // Swap buffers
        cur = nxt;
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.dev));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.copyStream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Gather the owned planes of every slab
    std::vector<double> cold(gridSize);
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.dev));
        CUDA_CHECK(cudaMemcpy(cold.data() + (size_t)s.zs * planeSize, s.buf[cur] + (size_t)(s.zs - s.zb) * planeSize,
                              (size_t)(s.ze - s.zs) * planeSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(s.buf[0]));
        CUDA_CHECK(cudaFree(s.buf[1]));
        if (s.sendLo) CUDA_CHECK(cudaFreeHost(s.sendLo));
        if (s.sendHi) CUDA_CHECK(cudaFreeHost(s.sendHi));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
        CUDA_CHECK(cudaStreamDestroy(s.copyStream));
        CUDA_CHECK(cudaEventDestroy(s.evBoundary));
        CUDA_CHECK(cudaEventDestroy(s.evSent));
        CUDA_CHECK(cudaEventDestroy(s.evHaloIn));
    }

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
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
