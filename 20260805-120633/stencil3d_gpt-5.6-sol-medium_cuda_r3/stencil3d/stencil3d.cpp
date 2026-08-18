#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kZChunk = 8;
constexpr int kTileX = kBlockX + 2;
constexpr int kTileY = kBlockY + 2;
constexpr int kTileSize = kTileX * kTileY;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A block owns an XY tile and eight consecutive Z planes. Three padded planes
// are kept in shared memory as a rolling window. This reduces the seven global
// reads of a direct stencil to about one (plus tile/chunk halos) per cell.
__global__ __launch_bounds__(kBlockX * kBlockY)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   size_t nx, size_t ny, size_t nz) {
    __shared__ Real planes[3][kTileSize];

    const int tid = static_cast<int>(threadIdx.y) * kBlockX +
                    static_cast<int>(threadIdx.x);
    const size_t planeStride = nx * ny;
    const size_t baseX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t zBegin = 1 + static_cast<size_t>(blockIdx.z) * kZChunk;
    const size_t zEnd = min(zBegin + kZChunk, nz - 1);

    // Cooperatively load a padded XY tile at the requested Z coordinate.
    auto loadPlane = [&](int slot, size_t z) {
        for (int tileIndex = tid; tileIndex < kTileSize;
             tileIndex += kBlockX * kBlockY) {
            const int tileY = tileIndex / kTileX;
            const int tileX = tileIndex - tileY * kTileX;
            const long long gx = static_cast<long long>(baseX) + tileX - 1;
            const long long gy = static_cast<long long>(baseY) + tileY - 1;
            Real value = 0.0;
            if (gx >= 0 && gy >= 0 && static_cast<size_t>(gx) < nx &&
                static_cast<size_t>(gy) < ny) {
                value = input[z * planeStride + static_cast<size_t>(gy) * nx +
                              static_cast<size_t>(gx)];
            }
            planes[slot][tileIndex] = value;
        }
    };

    loadPlane(0, zBegin - 1);
    loadPlane(1, zBegin);
    __syncthreads();

    const size_t x = baseX + threadIdx.x;
    const size_t y = baseY + threadIdx.y;
    const int centerIndex = (static_cast<int>(threadIdx.y) + 1) * kTileX +
                            static_cast<int>(threadIdx.x) + 1;
    int previous = 0;
    int current = 1;
    int next = 2;

    for (size_t z = zBegin; z < zEnd; ++z) {
        loadPlane(next, z + 1);
        __syncthreads();

        if (x > 0 && x + 1 < nx && y > 0 && y + 1 < ny) {
            const Real sum = planes[current][centerIndex] +
                             planes[current][centerIndex - 1] +
                             planes[current][centerIndex + 1] +
                             planes[current][centerIndex - kTileX] +
                             planes[current][centerIndex + kTileX] +
                             planes[previous][centerIndex] +
                             planes[next][centerIndex];
            output[z * planeStride + y * nx + x] = sum / 7.0;
        }

        // No thread may overwrite the previous plane until all consumers have
        // finished reading it.
        __syncthreads();
        const int oldPrevious = previous;
        previous = current;
        current = next;
        next = oldPrevious;
    }
}

void initializeGrid(std::vector<Real>& grid) {
    for (size_t i = 0; i < grid.size(); ++i) {
        grid[i] = static_cast<Real>(i % 19);
    }
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto bounds = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 1e6 || *bounds.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be at least 3 and iterations must be non-negative.\n");
        return 1;
    }
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions are too large.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid allocation size is too large.\n");
        return 1;
    }
    const size_t bytes = gridSize * sizeof(Real);
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    std::printf("Initializing grid...\n");
    initializeGrid(grid1);

    int device = 0;
    cudaCheck(cudaGetDevice(&device), "selecting a CUDA device");
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, device), "querying the CUDA device");
    std::printf("CUDA device: %s\n", properties.name);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    cudaCheck(cudaMalloc(&deviceGrid1, bytes), "allocating the first grid");
    cudaCheck(cudaMalloc(&deviceGrid2, bytes), "allocating the second grid");
    cudaCheck(cudaMemcpy(deviceGrid1, grid1.data(), bytes, cudaMemcpyHostToDevice),
              "uploading the initial grid");
    // Both buffers start with identical boundaries. Kernels only update the
    // interior, so the invariant boundary never needs to be copied again.
    cudaCheck(cudaMemcpy(deviceGrid2, grid1.data(), bytes, cudaMemcpyHostToDevice),
              "initializing the second grid");

    cudaStream_t stream{};
    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating stream");
    cudaCheck(cudaEventCreate(&startEvent), "creating start event");
    cudaCheck(cudaEventCreate(&stopEvent), "creating stop event");

    const dim3 block(kBlockX, kBlockY);
    const dim3 blocks(static_cast<unsigned>((nx + kBlockX - 1) / kBlockX),
                      static_cast<unsigned>((ny + kBlockY - 1) / kBlockY),
                      static_cast<unsigned>((nz - 2 + kZChunk - 1) / kZChunk));

    // Capture all iterations in a graph to pay the launch overhead only once.
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    if (iterations > 0) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                  "beginning graph capture");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            Real* input = (iteration & 1) ? deviceGrid2 : deviceGrid1;
            Real* output = (iteration & 1) ? deviceGrid1 : deviceGrid2;
            stencilKernel<<<blocks, block, 0, stream>>>(input, output, nx, ny, nz);
        }
        cudaCheck(cudaStreamEndCapture(stream, &graph), "ending graph capture");
        cudaCheck(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
                  "instantiating graph");
    }

    std::printf("Running stencil computation...\n");
    cudaCheck(cudaEventRecord(startEvent, stream), "recording start event");
    if (iterations > 0) {
        cudaCheck(cudaGraphLaunch(graphExec, stream), "launching stencil graph");
    }
    cudaCheck(cudaEventRecord(stopEvent, stream), "recording stop event");
    cudaCheck(cudaEventSynchronize(stopEvent), "waiting for stencil computation");
    cudaCheck(cudaGetLastError(), "executing stencil kernel");

    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "measuring execution time");
    std::printf("Computation time: %.3f ms\n", elapsedMs);

    const double cellUpdates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    Real* finalDeviceGrid = (iterations & 1) ? deviceGrid2 : deviceGrid1;
    std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;
    cudaCheck(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes, cudaMemcpyDeviceToHost),
              "downloading the final grid");

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(finalGrid);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (graphExec != nullptr) cudaCheck(cudaGraphExecDestroy(graphExec), "destroying graph executable");
    if (graph != nullptr) cudaCheck(cudaGraphDestroy(graph), "destroying graph");
    cudaCheck(cudaEventDestroy(stopEvent), "destroying stop event");
    cudaCheck(cudaEventDestroy(startEvent), "destroying start event");
    cudaCheck(cudaStreamDestroy(stream), "destroying stream");
    cudaCheck(cudaFree(deviceGrid2), "freeing the second grid");
    cudaCheck(cudaFree(deviceGrid1), "freeing the first grid");
    return valid ? 0 : 1;
}
