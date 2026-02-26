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

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

namespace {
constexpr int BX = 8;
constexpr int BY = 8;
constexpr int BZ = 8;

__global__ void init_grid_kernel(Real* __restrict__ grid, const size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) grid[i] = static_cast<Real>(i % 19u);
}

__global__ void copy_xy_faces_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                    const int nx, const int ny, const int nz) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= nx || y >= ny) return;

    const size_t idx0 = (static_cast<size_t>(y) * static_cast<size_t>(nx)) + static_cast<size_t>(x);
    out[idx0] = in[idx0];
    if (nz > 1) {
        const size_t idx1 = (static_cast<size_t>(nz - 1) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
                            static_cast<size_t>(nx) + static_cast<size_t>(x);
        out[idx1] = in[idx1];
    }
}

__global__ void copy_xz_faces_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                    const int nx, const int ny, const int nz) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int z = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= nx || z >= nz) return;

    const size_t idx0 = (static_cast<size_t>(z) * static_cast<size_t>(ny)) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    out[idx0] = in[idx0];
    if (ny > 1) {
        const size_t idx1 = (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(ny - 1)) *
                            static_cast<size_t>(nx) + static_cast<size_t>(x);
        out[idx1] = in[idx1];
    }
}

__global__ void copy_yz_faces_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                    const int nx, const int ny, const int nz) {
    const int y = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int z = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (y >= ny || z >= nz) return;

    const size_t idx0 = (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
                        static_cast<size_t>(nx);
    out[idx0] = in[idx0];
    if (nx > 1) {
        const size_t idx1 = idx0 + static_cast<size_t>(nx - 1);
        out[idx1] = in[idx1];
    }
}

__global__ void stencil_interior_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                       const int nx, const int ny, const int nz) {
    constexpr int SX = BX + 2;
    constexpr int SY = BY + 2;
    constexpr int SZ = BZ + 2;
    constexpr int SXY = SX * SY;
    constexpr int STOTAL = SXY * SZ;

    extern __shared__ Real sh[];

    const int baseX = 1 + static_cast<int>(blockIdx.x) * BX;
    const int baseY = 1 + static_cast<int>(blockIdx.y) * BY;
    const int baseZ = 1 + static_cast<int>(blockIdx.z) * BZ;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);

    const int tid = (tz * BY + ty) * BX + tx;
    const int nthreads = BX * BY * BZ;

    for (int s = tid; s < STOTAL; s += nthreads) {
        const int sx = s % SX;
        const int sy = (s / SX) % SY;
        const int sz = s / SXY;
        const int gx = baseX + sx - 1;
        const int gy = baseY + sy - 1;
        const int gz = baseZ + sz - 1;

        Real v = 0.0;
        if (static_cast<unsigned>(gx) < static_cast<unsigned>(nx) &&
            static_cast<unsigned>(gy) < static_cast<unsigned>(ny) &&
            static_cast<unsigned>(gz) < static_cast<unsigned>(nz)) {
            const size_t g = (static_cast<size_t>(gz) * static_cast<size_t>(ny) + static_cast<size_t>(gy)) *
                             static_cast<size_t>(nx) + static_cast<size_t>(gx);
            v = in[g];
        }
        sh[s] = v;
    }
    __syncthreads();

    const int gx = baseX + tx;
    const int gy = baseY + ty;
    const int gz = baseZ + tz;
    if (gx > nx - 2 || gy > ny - 2 || gz > nz - 2) return;

    const int sx = tx + 1;
    const int sy = ty + 1;
    const int sz = tz + 1;
    const int sc = (sz * SXY) + (sy * SX) + sx;

    const Real center = sh[sc];
    const Real left   = sh[sc - 1];
    const Real right  = sh[sc + 1];
    const Real front  = sh[sc - SX];
    const Real back   = sh[sc + SX];
    const Real bottom = sh[sc - SXY];
    const Real top    = sh[sc + SXY];

    const size_t g = (static_cast<size_t>(gz) * static_cast<size_t>(ny) + static_cast<size_t>(gy)) *
                     static_cast<size_t>(nx) + static_cast<size_t>(gx);
    out[g] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
}
} // namespace

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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    const size_t bytes = gridSize * sizeof(Real);
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    // Initialize on GPU
    printf("Initializing grid...\n");
    {
        const int threads = 256;
        const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
        init_grid_kernel<<<blocks, threads>>>(d_a, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    const bool hasInterior = (nx > 2 && ny > 2 && nz > 2);
    dim3 block(BX, BY, BZ);
    dim3 grid(0, 0, 0);
    size_t shmemBytes = 0;
    if (hasInterior) {
        grid = dim3(static_cast<unsigned>((nx - 2 + BX - 1) / BX),
                    static_cast<unsigned>((ny - 2 + BY - 1) / BY),
                    static_cast<unsigned>((nz - 2 + BZ - 1) / BZ));
        shmemBytes = static_cast<size_t>(BX + 2) * static_cast<size_t>(BY + 2) * static_cast<size_t>(BZ + 2) * sizeof(Real);
    }

    const dim3 faceBlock(32, 8);
    const dim3 gridXY(static_cast<unsigned>((nx + faceBlock.x - 1) / faceBlock.x),
                      static_cast<unsigned>((ny + faceBlock.y - 1) / faceBlock.y));
    const dim3 gridXZ(static_cast<unsigned>((nx + faceBlock.x - 1) / faceBlock.x),
                      static_cast<unsigned>((nz + faceBlock.y - 1) / faceBlock.y));
    const dim3 gridYZ(static_cast<unsigned>((ny + faceBlock.x - 1) / faceBlock.x),
                      static_cast<unsigned>((nz + faceBlock.y - 1) / faceBlock.y));

    // Run stencil iterations on GPU
    printf("Running stencil computation...\n");
    cudaEvent_t evStart{}, evStop{};
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    Real* d_cur = d_a;
    Real* d_next = d_b;
    for (int iter = 0; iter < iterations; ++iter) {
        // Preserve boundary semantics by copying just the boundary faces.
        copy_xy_faces_kernel<<<gridXY, faceBlock>>>(d_cur, d_next, inx, iny, inz);
        copy_xz_faces_kernel<<<gridXZ, faceBlock>>>(d_cur, d_next, inx, iny, inz);
        copy_yz_faces_kernel<<<gridYZ, faceBlock>>>(d_cur, d_next, inx, iny, inz);

        if (hasInterior) {
            stencil_interior_kernel<<<grid, block, shmemBytes>>>(d_cur, d_next, inx, iny, inz);
        }
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_cur, d_next);
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, evStart, evStop));

    const long durationMs = static_cast<long>(std::llround(static_cast<double>(elapsedMs)));
    printf("Computation time: %ld ms\n", durationMs);

    // Calculate performance metrics
    double cellUpdates = 0.0;
    if (hasInterior && iterations > 0) {
        cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * static_cast<double>(iterations);
    }
    const double secs = std::max(1e-9, static_cast<double>(elapsedMs) / 1000.0);
    double mcups = cellUpdates / secs / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy back once for printing/validation
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_cur, bytes, cudaMemcpyDeviceToHost));

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        return valid ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
