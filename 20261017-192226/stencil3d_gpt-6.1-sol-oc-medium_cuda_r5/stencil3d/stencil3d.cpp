#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each warp traverses contiguous X coordinates. Short Z strips reuse the
// center and both Z neighbors in registers, while retaining ample parallelism.
constexpr unsigned zStrip = 8;
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tileCount) {
    for (size_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const size_t tx = tile % tilesX;
        const size_t tyz = tile / tilesX;
        const size_t x = tx * blockDim.x + threadIdx.x + 1;
        const size_t y = (tyz % tilesY) * blockDim.y + threadIdx.y + 1;
        const size_t zBegin = (tyz / tilesY) * zStrip + 1;
        if (x >= nx - 1 || y >= ny - 1) continue;
        const size_t plane = nx * ny;
        size_t index = zBegin * plane + y * nx + x;
        Real bottom = input[index - plane];
        Real center = input[index];
        const size_t zEnd = (zBegin + zStrip < nz - 1) ? zBegin + zStrip : nz - 1;
        for (size_t z = zBegin; z < zEnd; ++z, index += plane) {
            const Real top = input[index + plane];
            output[index] = (center + input[index - 1] + input[index + 1]
                             + input[index - nx] + input[index + nx]
                             + bottom + top) / 7.0;
            bottom = center;
            center = top;
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Only the final result needs host storage. Both working grids stay resident
    // on the device, including their immutable boundary values.
    std::vector<Real> grid1(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&deviceGrid1, bytes), "allocate first grid");
    checkCuda(cudaMalloc(&deviceGrid2, bytes), "allocate second grid");
    checkCuda(cudaMemcpy(deviceGrid1, grid1.data(), bytes, cudaMemcpyHostToDevice), "upload grid");
    checkCuda(cudaMemcpy(deviceGrid2, deviceGrid1, bytes, cudaMemcpyDeviceToDevice), "initialize boundaries");
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = hasInterior ? (nx - 3) / 32 + 1 : 0;
    const size_t tilesY = hasInterior ? (ny - 3) / 4 + 1 : 0;
    const size_t tilesZ = hasInterior ? (nz - 3) / zStrip + 1 : 0;
    const size_t tileCount = tilesX * tilesY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), tileCount));
    auto launch = [&](int iteration) {
        const Real* input = (iteration % 2 == 0) ? deviceGrid1 : deviceGrid2;
        Real* output = (iteration % 2 == 0) ? deviceGrid2 : deviceGrid1;
        stencilIteration<<<blocks, dim3(32, 4), 0, stream>>>(
            input, output, nx, ny, nz, tilesX, tilesY, tileCount);
    };

    // An even batch can be replayed without changing the ping-pong ordering.
    // Capture outside the timed region; graphs reduce launch overhead for small grids.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    const int batch = std::min(iterations, 32);
    if (hasInterior && batch > 0) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
        for (int iter = 0; iter < batch; ++iter) launch(iter);
        checkCuda(cudaStreamEndCapture(stream, &graph), "end graph capture");
        checkCuda(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0), "instantiate graph");
        checkCuda(cudaGraphUpload(executableGraph, stream), "upload graph");
    }
    checkCuda(cudaStreamSynchronize(stream), "prepare computation");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (hasInterior && iterations > 0) {
        int iter = 0;
        for (; iter <= iterations - batch; iter += batch)
            checkCuda(cudaGraphLaunch(executableGraph, stream), "launch stencil graph");
        for (; iter < iterations; ++iter) launch(iter);
        checkCuda(cudaGetLastError(), "launch stencil");
    }
    checkCuda(cudaStreamSynchronize(stream), "finish stencil");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = hasInterior ? (double)(nx-2) * (ny-2) * (nz-2) * iterations : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    checkCuda(cudaMemcpy(grid1.data(), (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2,
                         bytes, cudaMemcpyDeviceToHost), "download result");
    if (executableGraph) checkCuda(cudaGraphExecDestroy(executableGraph), "destroy graph executable");
    if (graph) checkCuda(cudaGraphDestroy(graph), "destroy graph");
    checkCuda(cudaStreamDestroy(stream), "destroy stream");
    checkCuda(cudaFree(deviceGrid2), "free second grid");
    checkCuda(cudaFree(deviceGrid1), "free first grid");
    const std::vector<Real>& finalGrid = grid1;
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
