#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                            \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

// Physical parameters (compile-time constants so the device code can fold them;
// e.g. division by dx*dx == 1.0 is exact and is eliminated without changing results)
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma_ = 0.5;
constexpr double D = 1.0;

// Thread block tile (x, y) and shared-memory halo extents
constexpr int TX = 32;
constexpr int TY = 16;
constexpr int NTHREADS = TX * TY;
constexpr int CW = TX + 4, CH = TY + 4;  // concentration tile with 2-cell halo
constexpr int MW = TX + 2, MH = TY + 2;  // chemical potential tile with 1-cell halo

__device__ __forceinline__ int clampi(const int v, const int lo, const int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Laplacian with clamped boundary conditions, same expression order as the original
// (2.0*center is exact, so fused/unfused evaluation of each term gives identical results)
__device__ __forceinline__ double lap7(const double center, const double xp, const double xn,
                                       const double yp, const double yn, const double zp, const double zn) {
    const double cxx = (xp + xn - 2.0 * center) / (dx * dx);
    const double cyy = (yp + yn - 2.0 * center) / (dy * dy);
    const double czz = (zp + zn - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

// Fused Cahn-Hilliard time step: computes the chemical potential mu for the tile plus a
// one-cell halo in shared memory and immediately applies the update, so mu never touches
// global memory. Each block owns a TX x TY column of the domain over a chunk of z-planes
// and marches through z, keeping rolling 3-plane windows of c and mu in shared memory.
// Plane slots are indexed by logical plane k (k may be -1 or nz; contents are clamped).
__global__ void __launch_bounds__(NTHREADS)
cahnHilliardStepKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                       const int nx, const int ny, const int nz, const int zchunk) {
    __shared__ double cs[3][CH][CW];
    __shared__ double ms[3][MH][MW];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int x0 = blockIdx.x * TX, y0 = blockIdx.y * TY;
    const size_t plane = static_cast<size_t>(nx) * ny;

    const int zb = blockIdx.z * zchunk;
    const int ze = min(nz, zb + zchunk);
    if (zb >= ze) return;
    const int jlo = max(zb - 1, 0);
    const int jhi = min(ze, nz - 1);

    auto loadC = [&](const int k) {
        const int kc = clampi(k, 0, nz - 1);
        const double* __restrict__ src = cold + static_cast<size_t>(kc) * plane;
        double (*dst)[CW] = cs[(k + 1) % 3];
        for (int i = tid; i < CW * CH; i += NTHREADS) {
            const int ly = i / CW, lx = i - ly * CW;
            const int gx = clampi(x0 + lx - 2, 0, nx - 1);
            const int gy = clampi(y0 + ly - 2, 0, ny - 1);
            dst[ly][lx] = __ldg(src + static_cast<size_t>(gy) * nx + gx);
        }
    };

    auto computeMu = [&](const int j) {
        const double (*cm)[CW] = cs[j % 3];        // logical plane j-1
        const double (*cc)[CW] = cs[(j + 1) % 3];  // logical plane j
        const double (*cp)[CW] = cs[(j + 2) % 3];  // logical plane j+1
        double (*dst)[MW] = ms[(j + 1) % 3];
        for (int i = tid; i < MW * MH; i += NTHREADS) {
            const int ly = i / MW, lx = i - ly * MW;
            const int gx = clampi(x0 + lx - 1, 0, nx - 1);
            const int gy = clampi(y0 + ly - 1, 0, ny - 1);
            const int gxp = (gx < nx - 1) ? gx + 1 : gx;
            const int gxn = (gx > 0) ? gx - 1 : 0;
            const int gyp = (gy < ny - 1) ? gy + 1 : gy;
            const int gyn = (gy > 0) ? gy - 1 : 0;
            const int cx = gx - x0 + 2, cy = gy - y0 + 2;
            const double cv = cc[cy][cx];
            const double l = lap7(cv, cc[cy][gxp - x0 + 2], cc[cy][gxn - x0 + 2],
                                  cc[gyp - y0 + 2][cx], cc[gyn - y0 + 2][cx], cp[cy][cx], cm[cy][cx]);
            // mu = 4.5*((cv+1)*e_AA + (cv-1)*e_BB - 2*cv*e_AB) + 3*cv + cv^3 - gamma*lap,
            // evaluated with the same fused multiply-add grouping as the optimized host build
            double m = fma(cv + 1.0, e_AA, (cv - 1.0) * e_BB);
            m = fma(-(2.0 * cv), e_AB, m);
            m = fma(m, 4.5, 3.0 * cv);
            m = fma(cv * cv, cv, m);
            dst[ly][lx] = fma(-l, gamma_, m);
        }
    };

    const int gx = x0 + tx, gy = y0 + ty;
    const bool active = gx < nx && gy < ny;
    const int mx = tx + 1, my = ty + 1;
    const int mxp = (gx < nx - 1) ? mx + 1 : mx;
    const int mxn = (gx > 0) ? mx - 1 : mx;
    const int myp = (gy < ny - 1) ? my + 1 : my;
    const int myn = (gy > 0) ? my - 1 : my;

    auto update = [&](const int z) {
        if (!active) return;
        const int zp = (z < nz - 1) ? z + 1 : z;
        const int zn = (z > 0) ? z - 1 : 0;
        const double (*mc)[MW] = ms[(z + 1) % 3];
        const double (*mp)[MW] = ms[(zp + 1) % 3];
        const double (*mn)[MW] = ms[(zn + 1) % 3];
        const double l = lap7(mc[my][mx], mc[my][mxp], mc[my][mxn], mc[myp][mx], mc[myn][mx],
                              mp[my][mx], mn[my][mx]);
        const size_t idx = static_cast<size_t>(z) * plane + static_cast<size_t>(gy) * nx + gx;
        cnew[idx] = fma(l, dt * D, cs[(z + 1) % 3][ty + 2][tx + 2]);  // cold + dt*D*lap
    };

    loadC(jlo - 1);
    loadC(jlo);
    for (int j = jlo; j <= jhi; ++j) {
        loadC(j + 1);
        __syncthreads();
        computeMu(j);
        __syncthreads();
        if (j - 1 >= zb) update(j - 1);
        __syncthreads();
    }
    if (ze == nz) update(nz - 1);
}

// Initialize concentration field on the device
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linear_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_id < vol;
         linear_id += stride) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = __dadd_rn(-1.0, __dmul_rn(2.0, pseudo));
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
    
    const size_t gridSize = nx * ny * nz;

    if (gridSize == 0 || iterations < 0) {
        // Nothing to compute on an empty grid
        iterations = std::max(iterations, 0);
    }

    std::vector<double> cold(gridSize);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Allocate device arrays
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    if (gridSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    }

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        const int initBlocks = static_cast<int>(std::min<size_t>((gridSize + 255) / 256, 65535));
        initializeConcentrationKernel<<<initBlocks, 256>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Launch configuration: TX x TY columns, z split into chunks to fill the GPU
    const dim3 block(TX, TY, 1);
    const unsigned gxBlocks = static_cast<unsigned>((nx + TX - 1) / TX);
    const unsigned gyBlocks = static_cast<unsigned>((ny + TY - 1) / TY);
    int blocksPerSM = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, cahnHilliardStepKernel, NTHREADS, 0));
    const size_t xyBlocks = std::max<size_t>(static_cast<size_t>(gxBlocks) * gyBlocks, 1);
    const size_t targetBlocks = static_cast<size_t>(prop.multiProcessorCount) * std::max(blocksPerSM, 1) * 2;
    const size_t minChunk = 8;
    const size_t maxChunks = std::max<size_t>((nz + minChunk - 1) / minChunk, 1);
    const size_t nChunks = std::clamp<size_t>((targetBlocks + xyBlocks - 1) / xyBlocks, 1, maxChunks);
    const int zchunk = static_cast<int>(std::max<size_t>((nz + nChunks - 1) / nChunks, 1));
    const dim3 grid(gxBlocks, gyBlocks, static_cast<unsigned>((nz + zchunk - 1) / zchunk));

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0) {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential and update concentration (fused)
            cahnHilliardStepKernel<<<grid, block>>>(d_cold, d_cnew, static_cast<int>(nx), static_cast<int>(ny),
                                                    static_cast<int>(nz), zchunk);
            // Swap buffers
            std::swap(d_cold, d_cnew);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
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
