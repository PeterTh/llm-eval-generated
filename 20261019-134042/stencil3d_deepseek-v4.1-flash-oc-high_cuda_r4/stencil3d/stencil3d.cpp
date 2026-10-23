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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

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

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 4;
constexpr int kBlockSize = kBlockX * kBlockY * kBlockZ;

constexpr int kTileX = kBlockX + 2;
constexpr int kTileY = kBlockY + 2;
constexpr int kTileZ = kBlockZ + 2;
constexpr int kTileSize = kTileX * kTileY * kTileZ;
constexpr int kTileSlice = kTileX * kTileY;

__global__ __launch_bounds__(kBlockSize)
void stencilInteriorKernel(const Real* __restrict__ input,
                           Real* __restrict__ output,
                           const int nx, const int ny, const int nz) {
    __shared__ Real tile[kTileSize];

    const int ox = static_cast<int>(blockIdx.x) * kBlockX;
    const int oy = static_cast<int>(blockIdx.y) * kBlockY;
    const int oz = static_cast<int>(blockIdx.z) * kBlockZ;

    const int tid = (static_cast<int>(threadIdx.z) * kBlockY +
                     static_cast<int>(threadIdx.y)) * kBlockX +
                    static_cast<int>(threadIdx.x);

    // Cooperative, coalesced tile load (interior + halo).
    for (int i = tid; i < kTileSize; i += kBlockSize) {
        const int sx = i % kTileX;
        const int rest = i / kTileX;
        const int sy = rest % kTileY;
        const int sz = rest / kTileY;

        int gx = ox + sx;
        int gy = oy + sy;
        int gz = oz + sz;
        if (gx >= nx) gx = nx - 1;
        if (gy >= ny) gy = ny - 1;
        if (gz >= nz) gz = nz - 1;

        tile[i] = input[static_cast<size_t>(gz) * ny * nx +
                        static_cast<size_t>(gy) * nx + gx];
    }
    __syncthreads();

    const int ix = ox + static_cast<int>(threadIdx.x);
    const int iy = oy + static_cast<int>(threadIdx.y);
    const int iz = oz + static_cast<int>(threadIdx.z);

    // Interior coordinates are ix+1, iy+1, iz+1; skip the partial edge block.
    if (ix >= nx - 2 || iy >= ny - 2 || iz >= nz - 2) {
        return;
    }

    const int c = (static_cast<int>(threadIdx.z) + 1) * kTileSlice +
                  (static_cast<int>(threadIdx.y) + 1) * kTileX +
                  (static_cast<int>(threadIdx.x) + 1);

    const Real center = tile[c];
    const Real left   = tile[c - 1];
    const Real right  = tile[c + 1];
    const Real front  = tile[c - kTileX];
    const Real back   = tile[c + kTileX];
    const Real bottom = tile[c - kTileSlice];
    const Real top    = tile[c + kTileSlice];

    const size_t idx = static_cast<size_t>(iz + 1) * ny * nx +
                       static_cast<size_t>(iy + 1) * nx + (ix + 1);

    const Real sum = center + left + right + front + back + bottom + top;

    // Correctly-rounded division by 7 via a multiply plus one Markstein
    // correction step, avoiding the much slower full IEEE divide sequence.
    constexpr Real kInv7 = 1.0 / 7.0;
    const Real q0 = sum * kInv7;
    const Real err = __fma_rn(-q0, 7.0, sum);
    output[idx] = __fma_rn(err, kInv7, q0);
}

inline unsigned int divUp(const size_t value, const size_t divisor) {
    return static_cast<unsigned int>((value + divisor - 1) / divisor);
}

void launchStencilIteration(const Real* input, Real* output,
                            const size_t nx, const size_t ny, const size_t nz) {
    // Nothing to compute if any dimension has no interior points.
    if (nx < 3 || ny < 3 || nz < 3) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(divUp(nx - 2, kBlockX),
                    divUp(ny - 2, kBlockY),
                    divUp(nz - 2, kBlockZ));

    stencilInteriorKernel<<<grid, block>>>(input, output,
                                           static_cast<int>(nx),
                                           static_cast<int>(ny),
                                           static_cast<int>(nz));
}

} // namespace

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
    
    // Device buffers.  Both are seeded with the initial grid so that the
    // (unchanging) boundary values are always correct, matching the CPU
    // reference which copies boundaries on every iteration.
    const size_t bytes = gridSize * sizeof(Real);
    Real* dGrid1 = nullptr;
    Real* dGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&dGrid2, bytes));
    CUDA_CHECK(cudaMemcpy(dGrid1, grid1.data(), bytes, cudaMemcpyHostToDevice));
    if (iterations > 0) {
        CUDA_CHECK(cudaMemcpy(dGrid2, grid1.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(dGrid2, 0, bytes));
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    Real* src = dGrid1;
    Real* dst = dGrid2;
    for (int iter = 0; iter < iterations; ++iter) {
        launchStencilIteration(src, dst, nx, ny, nz);
        std::swap(src, dst);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Copy the final device grid back for result printing / validation.
    const bool evenIterations = (iterations % 2 == 0);
    Real* dFinal = evenIterations ? dGrid1 : dGrid2;
    std::vector<Real>& gridFinal = evenIterations ? grid1 : grid2;
    CUDA_CHECK(cudaMemcpy(gridFinal.data(), dFinal, bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(dGrid1));
    CUDA_CHECK(cudaFree(dGrid2));
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = gridFinal;
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
