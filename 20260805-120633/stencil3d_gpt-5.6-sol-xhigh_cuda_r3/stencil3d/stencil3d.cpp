#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 4;
constexpr unsigned int kBlockZ = 2;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t cuda_check_error = (call);                                        \
        if (cuda_check_error != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(cuda_check_error));                             \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (false)

// X spans a complete warp for coalesced loads. The Y/Z tile gives each block
// neighbor reuse through the read-only cache while retaining full occupancy.
__global__ __launch_bounds__(kBlockX * kBlockY * kBlockZ)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   const size_t nx,
                   const size_t ny,
                   const size_t nz,
                   const size_t planeSize) {
    const unsigned int x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx - 1) return;

    const size_t yStart = 1 + static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t zStart = 1 + static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    const size_t yStride = static_cast<size_t>(blockDim.y) * gridDim.y;
    const size_t zStride = static_cast<size_t>(blockDim.z) * gridDim.z;

    // Grid-stride traversal keeps unusually long Y/Z dimensions valid even
    // beyond CUDA's 65,535-block limits; ordinary benchmark shapes run once.
    for (size_t z = zStart; z < nz - 1; z += zStride) {
        for (size_t y = yStart; y < ny - 1; y += yStride) {
            const size_t index = z * planeSize + y * nx + x;
            output[index] =
                (input[index] + input[index - 1] + input[index + 1] + input[index - nx] +
                 input[index + nx] + input[index - planeSize] + input[index + planeSize]) /
                7.0;
        }
    }
}

}  // namespace

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

    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be positive and iterations must be non-negative.\n");
        return 1;
    }
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions are too large.\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid dimensions are too large.\n");
        return 1;
    }
    const size_t gridBytes = gridSize * sizeof(Real);
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Allocate and initialize both device buffers.  Since boundary values never
    // change, initializing both buffers once avoids copying six faces every step.
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, gridBytes));
    CUDA_CHECK(cudaMemcpy(deviceGrid1, grid1.data(), gridBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGrid2, deviceGrid1, gridBytes, cudaMemcpyDeviceToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaFuncSetCacheConfig(stencilKernel, cudaFuncCachePreferL1));

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    if (hasInterior && iterations > 0) {
        const size_t interiorX = nx - 2;
        const size_t interiorY = ny - 2;
        const size_t interiorZ = nz - 2;
        const dim3 block(kBlockX, kBlockY, kBlockZ);
        const dim3 launchGrid(
            static_cast<unsigned int>((interiorX + kBlockX - 1) / kBlockX),
            static_cast<unsigned int>(std::min(
                (interiorY + block.y - 1) / block.y, static_cast<size_t>(65535))),
            static_cast<unsigned int>(std::min(
                (interiorZ + block.z - 1) / block.z, static_cast<size_t>(65535))));

        // Capturing all time steps into a graph preserves the Jacobi dependency
        // chain while eliminating repeated CPU launch overhead.
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        Real* input = deviceGrid1;
        Real* output = deviceGrid2;
        for (int iter = 0; iter < iterations; ++iter) {
            stencilKernel<<<launchGrid, block, 0, stream>>>(
                input, output, nx, ny, nz, nx * ny);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    // Run stencil iterations entirely on the GPU.
    printf("Running stencil computation...\n");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (graphExec != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));

    // Calculate performance metrics
    const double interiorCells = hasInterior
                                     ? static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                                           static_cast<double>(nz - 2)
                                     : 0.0;
    const double cellUpdates = interiorCells * static_cast<double>(iterations);
    const double mcups = elapsedMs > 0.0f ? cellUpdates / static_cast<double>(elapsedMs) / 1.0e3 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfer only the completed buffer back to the host after timing.
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    CUDA_CHECK(cudaMemcpyAsync(
        finalGrid.data(), finalDeviceGrid, gridBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (graphExec != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

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
