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
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),        \
                    __FILE__, __LINE__);                                                  \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Tile configuration for the fused stencil kernel.
// Each block computes a TX x TY tile in x-y and marches through a chunk of z planes.
// Chemical potential (mu) is computed on the tile plus a 1-cell halo and kept in shared
// memory, so mu never round-trips through global memory.
constexpr int TX = 32;
constexpr int TY = 16;
constexpr int CX = TX + 4;  // c tile width  (halo 2)
constexpr int CY = TY + 4;  // c tile height (halo 2)
constexpr int MX = TX + 2;  // mu tile width  (halo 1)
constexpr int MY = TY + 2;  // mu tile height (halo 1)
constexpr int RING = 4;     // ring buffer depth for planes

struct Params {
    int nx, ny, nz;
    int zChunk;
    double idx2, idy2, idz2;  // 1/(dx*dx), ...
    double gamma, e_AA, e_BB, e_AB;
    double dtD;
};

// Fused Cahn-Hilliard step: cnew = cold + dt*D*Lap(mu(cold)), with clamped boundaries.
// UNIT: grid spacing is exactly 1 in all directions, so the (bit-exact) multiplications
// by 1/(d*d) == 1.0 can be skipped, saving FP64 throughput.
template <bool UNIT>
__global__ void __launch_bounds__(TX * TY)
cahnHilliardStepKernel(const double* __restrict__ cold, double* __restrict__ cnew, const Params p) {
    __shared__ double cs[RING][CY][CX];
    __shared__ double ms[RING][MY][MX];

    const int nx = p.nx, ny = p.ny, nz = p.nz;
    const double idx2 = UNIT ? 1.0 : p.idx2;
    const double idy2 = UNIT ? 1.0 : p.idy2;
    const double idz2 = UNIT ? 1.0 : p.idz2;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int x0 = blockIdx.x * TX;
    const int y0 = blockIdx.y * TY;
    const int z0 = blockIdx.z * p.zChunk;
    const int z1 = min(z0 + p.zChunk, nz);  // exclusive
    const size_t plane = (size_t)nx * ny;

    // Load c plane k (global coords clamped) into ring slot k % RING.
    auto loadC = [&](int k) {
        const double* src = cold + (size_t)k * plane;
        double (*dst)[CX] = cs[k & (RING - 1)];
        for (int i = tid; i < CX * CY; i += TX * TY) {
            const int ly = i / CX;
            const int lx = i - ly * CX;
            const int gx = min(max(x0 - 2 + lx, 0), nx - 1);
            const int gy = min(max(y0 - 2 + ly, 0), ny - 1);
            dst[ly][lx] = __ldg(src + (size_t)gy * nx + gx);
        }
    };

    // Compute mu on plane k for the tile + 1 halo (halo positions clamped to domain).
    auto computeMu = [&](int k) {
        const double (*cc)[CX] = cs[k & (RING - 1)];
        const double (*cn)[CX] = cs[max(k - 1, 0) & (RING - 1)];
        const double (*cp)[CX] = cs[min(k + 1, nz - 1) & (RING - 1)];
        double (*dst)[MX] = ms[k & (RING - 1)];
        for (int i = tid; i < MX * MY; i += TX * TY) {
            const int ly = i / MX;
            const int lx = i - ly * MX;
            const int gx = min(max(x0 - 1 + lx, 0), nx - 1);
            const int gy = min(max(y0 - 1 + ly, 0), ny - 1);
            // positions in the c tile (c tile origin is (x0-2, y0-2))
            const int cx = gx - (x0 - 2);
            const int cy = gy - (y0 - 2);
            const int cxp = min(gx + 1, nx - 1) - (x0 - 2);
            const int cxn = max(gx - 1, 0) - (x0 - 2);
            const int cyp = min(gy + 1, ny - 1) - (y0 - 2);
            const int cyn = max(gy - 1, 0) - (y0 - 2);

            const double cv = cc[cy][cx];
            const double cxx = (cc[cy][cxp] + cc[cy][cxn] - 2.0 * cv) * idx2;
            const double cyy = (cc[cyp][cx] + cc[cyn][cx] - 2.0 * cv) * idy2;
            const double czz = (cp[cy][cx] + cn[cy][cx] - 2.0 * cv) * idz2;
            const double lap = cxx + cyy + czz;

            dst[ly][lx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - p.gamma * lap;
        }
    };

    const int gx = x0 + tx;
    const int gy = y0 + ty;
    const bool active = (gx < nx) && (gy < ny);
    // positions in the mu tile (mu tile origin is (x0-1, y0-1))
    const int mx = tx + 1, my = ty + 1;
    const int mxp = min(gx + 1, nx - 1) - (x0 - 1);
    const int mxn = max(gx - 1, 0) - (x0 - 1);
    const int myp = min(gy + 1, ny - 1) - (y0 - 1);
    const int myn = max(gy - 1, 0) - (y0 - 1);

    auto update = [&](int z) {
        if (!active) return;
        const double (*mc)[MX] = ms[z & (RING - 1)];
        const double (*mn)[MX] = ms[max(z - 1, 0) & (RING - 1)];
        const double (*mp)[MX] = ms[min(z + 1, nz - 1) & (RING - 1)];
        const double mv = mc[my][mx];
        const double mxx = (mc[my][mxp] + mc[my][mxn] - 2.0 * mv) * idx2;
        const double myy = (mc[myp][mx] + mc[myn][mx] - 2.0 * mv) * idy2;
        const double mzz = (mp[my][mx] + mn[my][mx] - 2.0 * mv) * idz2;
        const double c0 = cs[z & (RING - 1)][ty + 2][tx + 2];
        cnew[(size_t)z * plane + (size_t)gy * nx + gx] = c0 + p.dtD * (mxx + myy + mzz);
    };

    // mu planes needed: [zs, ze]
    const int zs = max(z0 - 1, 0);
    const int ze = min(z1, nz - 1);

    loadC(max(zs - 1, 0));
    loadC(zs);
    for (int k = zs; k <= ze; ++k) {
        if (k + 1 <= nz - 1) loadC(k + 1);
        __syncthreads();
        computeMu(k);
        __syncthreads();
        if (k - 1 >= z0 && k - 1 < z1) update(k - 1);
        if (k == nz - 1 && k >= z0 && k < z1) update(k);
    }
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t linear_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x; linear_id < vol; linear_id += stride) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
    }
}

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
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        initializeConcentrationKernel<<<std::min<size_t>((gridSize + 255) / 256, 65535 * 4), 256>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Kernel launch configuration
    int device = 0, numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaFuncSetCacheConfig(cahnHilliardStepKernel<true>, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(cahnHilliardStepKernel<false>, cudaFuncCachePreferShared));
    
    Params p;
    p.nx = (int)nx; p.ny = (int)ny; p.nz = (int)nz;
    p.idx2 = 1.0 / (dx * dx);
    p.idy2 = 1.0 / (dy * dy);
    p.idz2 = 1.0 / (dz * dz);
    p.gamma = gamma; p.e_AA = e_AA; p.e_BB = e_BB; p.e_AB = e_AB;
    p.dtD = dt * D;
    
    const unsigned bx = (unsigned)((nx + TX - 1) / TX);
    const unsigned by = (unsigned)((ny + TY - 1) / TY);
    // Split z into chunks so there are enough blocks to fill the GPU, but keep chunks
    // long enough to amortize the 2 redundant mu planes per chunk.
    const size_t targetBlocks = (size_t)numSMs * 8;
    const size_t xyBlocks = (size_t)bx * by;
    size_t zChunks = (targetBlocks + xyBlocks - 1) / std::max<size_t>(xyBlocks, 1);
    zChunks = std::clamp<size_t>(zChunks, 1, std::max<size_t>(nz / 8, 1));
    zChunks = std::min<size_t>(zChunks, 65535);
    p.zChunk = (int)((nz + zChunks - 1) / std::max<size_t>(zChunks, 1));
    if (p.zChunk < 1) p.zChunk = 1;
    const unsigned bz = (unsigned)((nz + p.zChunk - 1) / p.zChunk);
    const dim3 block(TX, TY, 1);
    const dim3 grid(bx, by, bz);
    const bool runKernel = gridSize > 0;
    const bool unitSpacing = (p.idx2 == 1.0) && (p.idy2 == 1.0) && (p.idz2 == 1.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations && runKernel; ++t) {
        // Compute chemical potential and update concentration (fused)
        if (unitSpacing) {
            cahnHilliardStepKernel<true><<<grid, block>>>(d_cold, d_cnew, p);
        } else {
            cahnHilliardStepKernel<false><<<grid, block>>>(d_cold, d_cnew, p);
        }
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    
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
