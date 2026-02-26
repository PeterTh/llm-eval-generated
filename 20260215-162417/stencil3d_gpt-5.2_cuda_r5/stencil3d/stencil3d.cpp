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

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

template <int BX, int BY, int BZ>
__global__ void stencil7_shared_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                      int nx, int ny, int nz) {
    constexpr int SX = BX + 2;
    constexpr int SY = BY + 2;
    constexpr int SZ = BZ + 2;

    __shared__ Real sh[SX * SY * SZ];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);

    const int x = static_cast<int>(blockIdx.x) * BX + tx;
    const int y = static_cast<int>(blockIdx.y) * BY + ty;
    const int z = static_cast<int>(blockIdx.z) * BZ + tz;

    const int sx = tx + 1;
    const int sy = ty + 1;
    const int sz = tz + 1;

    auto shIdx = [](int lx, int ly, int lz) {
        return (lz * SY + ly) * SX + lx;
    };

    const bool inBounds = (x < nx) & (y < ny) & (z < nz);
    const size_t gIdx = (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
                            static_cast<size_t>(nx) +
                        static_cast<size_t>(x);

    sh[shIdx(sx, sy, sz)] = inBounds ? in[gIdx] : Real{0};

    if (tx == 0) {
        if (inBounds && x > 0) {
            sh[shIdx(sx - 1, sy, sz)] = in[gIdx - 1];
        } else {
            sh[shIdx(sx - 1, sy, sz)] = Real{0};
        }
    }
    if (tx == BX - 1) {
        if (inBounds && x + 1 < nx) {
            sh[shIdx(sx + 1, sy, sz)] = in[gIdx + 1];
        } else {
            sh[shIdx(sx + 1, sy, sz)] = Real{0};
        }
    }

    if (ty == 0) {
        if (inBounds && y > 0) {
            sh[shIdx(sx, sy - 1, sz)] = in[gIdx - static_cast<size_t>(nx)];
        } else {
            sh[shIdx(sx, sy - 1, sz)] = Real{0};
        }
    }
    if (ty == BY - 1) {
        if (inBounds && y + 1 < ny) {
            sh[shIdx(sx, sy + 1, sz)] = in[gIdx + static_cast<size_t>(nx)];
        } else {
            sh[shIdx(sx, sy + 1, sz)] = Real{0};
        }
    }

    const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (tz == 0) {
        if (inBounds && z > 0) {
            sh[shIdx(sx, sy, sz - 1)] = in[gIdx - slice];
        } else {
            sh[shIdx(sx, sy, sz - 1)] = Real{0};
        }
    }
    if (tz == BZ - 1) {
        if (inBounds && z + 1 < nz) {
            sh[shIdx(sx, sy, sz + 1)] = in[gIdx + slice];
        } else {
            sh[shIdx(sx, sy, sz + 1)] = Real{0};
        }
    }

    __syncthreads();

    if (!inBounds) return;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        out[gIdx] = sh[shIdx(sx, sy, sz)];
        return;
    }

    const Real c = sh[shIdx(sx, sy, sz)];
    const Real l = sh[shIdx(sx - 1, sy, sz)];
    const Real r = sh[shIdx(sx + 1, sy, sz)];
    const Real f = sh[shIdx(sx, sy - 1, sz)];
    const Real b = sh[shIdx(sx, sy + 1, sz)];
    const Real dn = sh[shIdx(sx, sy, sz - 1)];
    const Real up = sh[shIdx(sx, sy, sz + 1)];

    out[gIdx] = (c + l + r + f + b + dn + up) * (1.0 / 7.0);
}

static long runStencilCuda(std::vector<Real>& grid1, std::vector<Real>& grid2,
                           const size_t nx, const size_t ny, const size_t nz,
                           const int iterations, const bool needHostResult) {
    const size_t gridSize = nx * ny * nz;

    CUDA_CHECK(cudaSetDevice(0));

    Real* d1 = nullptr;
    Real* d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));

    constexpr int BX = 8;
    constexpr int BY = 8;
    constexpr int BZ = 4;
    dim3 block(BX, BY, BZ);
    dim3 grid((static_cast<unsigned int>(nx) + BX - 1) / BX,
              (static_cast<unsigned int>(ny) + BY - 1) / BY,
              (static_cast<unsigned int>(nz) + BZ - 1) / BZ);

    cudaEvent_t evStart{}, evStop{};
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencil7_shared_kernel<BX, BY, BZ><<<grid, block>>>(d1, d2, static_cast<int>(nx), static_cast<int>(ny),
                                                               static_cast<int>(nz));
        } else {
            stencil7_shared_kernel<BX, BY, BZ><<<grid, block>>>(d2, d1, static_cast<int>(nx), static_cast<int>(ny),
                                                               static_cast<int>(nz));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    if (needHostResult) {
        if ((iterations & 1) == 0) {
            CUDA_CHECK(cudaMemcpy(grid1.data(), d1, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(grid2.data(), d2, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
        }
    }

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d1));
    CUDA_CHECK(cudaFree(d2));

    long msLong = static_cast<long>(std::llround(static_cast<double>(ms)));
    if (iterations > 0 && msLong == 0) msLong = 1;
    return msLong;
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
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
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations (CUDA GPU)
    printf("Running stencil computation...\n");

    const bool needHostResult = printResults || validate;
    const long durationMs = runStencilCuda(grid1, grid2, nx, ny, nz, iterations, needHostResult);

    printf("Computation time: %ld ms\n", durationMs);

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
