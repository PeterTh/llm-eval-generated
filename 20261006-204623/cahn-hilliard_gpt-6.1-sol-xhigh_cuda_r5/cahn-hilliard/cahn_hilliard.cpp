#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

struct DeviceBuffer {
    double* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct CudaStream {
    cudaStream_t stream = nullptr;

    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

struct CudaGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CudaGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    CudaGraph() = default;
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
};

// Keep the original expression: algebraic simplifications change rounding.
__device__ __forceinline__ double chemicalPotential(double c, double laplacian) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0;
    return 4.5 * ((c + 1.0) * e_AA + (c - 1.0) * e_BB - 2.0 * c * e_AB)
           + 3.0 * c + c * c * c - 0.5 * laplacian;
}

__global__ void initializeConcentration(double* c, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % count) / static_cast<double>(count);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Both stencils clamp neighbors at the domain edges. The original benchmark
// fixes all grid spacings to one and dt * D to 0.01.
// A compact grid also handles skinny domains without mostly inactive warps.
template<bool Chemical>
__global__ void flatStencil(const double* __restrict__ input,
                            double* __restrict__ output,
                            size_t nx, size_t ny, size_t nz, size_t count) {
    const size_t plane = nx * ny;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const size_t z = i / plane;
        const double c = input[i];
        const double cxx = input[i + (x + 1 < nx ? 1 : 0)] +
                           input[i - (x > 0 ? 1 : 0)] - 2.0 * c;
        const double cyy = input[i + (y + 1 < ny ? nx : 0)] +
                           input[i - (y > 0 ? nx : 0)] - 2.0 * c;
        const double czz = input[i + (z + 1 < nz ? plane : 0)] +
                           input[i - (z > 0 ? plane : 0)] - 2.0 * c;
        const double laplacian = cxx + cyy + czz;
        if constexpr (Chemical) {
            output[i] = chemicalPotential(c, laplacian);
        } else {
            output[i] = output[i] + 0.01 * laplacian;
        }
    }
}

constexpr unsigned tileX = 32;
constexpr unsigned tileY = 2;
constexpr unsigned tileZ = 2;

// Each warp covers adjacent X cells. Marching through Z reuses the three
// neighboring planes in registers, without shared-memory barriers.
template<bool Chemical>
__global__ void tiledStencil(const double* __restrict__ input,
                             double* __restrict__ output,
                             size_t nx, size_t ny, size_t nz,
                             unsigned tilesX, unsigned tilesXY) {
    const unsigned tile = blockIdx.x;
    const size_t x = static_cast<size_t>(tile % tilesX) * tileX + threadIdx.x;
    const size_t y = static_cast<size_t>((tile % tilesXY) / tilesX) * tileY + threadIdx.y;
    const size_t z0 = static_cast<size_t>(tile / tilesXY) * tileZ;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    size_t i = z0 * plane + y * nx + x;
    const size_t xp = x + 1 < nx ? 1 : 0;
    const size_t xn = x > 0 ? 1 : 0;
    const size_t yp = y + 1 < ny ? nx : 0;
    const size_t yn = y > 0 ? nx : 0;
    double below = input[i - (z0 > 0 ? plane : 0)];
    double center = input[i];
    double above = input[i + (z0 + 1 < nz ? plane : 0)];

    #pragma unroll
    for (unsigned k = 0; k < tileZ; ++k) {
        const size_t z = z0 + k;
        if (z >= nz) break;
        const double cxx = input[i + xp] + input[i - xn] - 2.0 * center;
        const double cyy = input[i + yp] + input[i - yn] - 2.0 * center;
        const double czz = above + below - 2.0 * center;
        const double laplacian = cxx + cyy + czz;
        if constexpr (Chemical) {
            output[i] = chemicalPotential(center, laplacian);
        } else {
            output[i] = output[i] + 0.01 * laplacian;
        }
        if (k + 1 < tileZ && z + 1 < nz) {
            i += plane;
            below = center;
            center = above;
            above = input[i + (z + 2 < nz ? plane : 0)];
        }
    }
}

struct StencilLaunch {
    size_t nx, ny, nz, count;
    unsigned blocks, tilesX, tilesXY;
    bool tiled;

    StencilLaunch(size_t x, size_t y, size_t z, size_t n)
        : nx(x), ny(y), nz(z), count(n) {
        tiled = nx >= tileX && ny >= tileY && nz >= tileZ;
        const size_t tx = (nx + tileX - 1) / tileX;
        const size_t txy = tx * ((ny + tileY - 1) / tileY);
        const size_t total = txy * ((nz + tileZ - 1) / tileZ);
        // CUDA's one-dimensional launch limit; the flat kernel can grid-stride.
        if (total > 2147483647u) tiled = false;
        tilesX = static_cast<unsigned>(tx);
        tilesXY = static_cast<unsigned>(txy);
        blocks = tiled ? static_cast<unsigned>(total)
                       : static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    }

    void step(double* current, double* mu, cudaStream_t stream) const {
        if (tiled) {
            const dim3 threads(tileX, tileY);
            tiledStencil<true><<<blocks, threads, 0, stream>>>(current, mu, nx, ny, nz, tilesX, tilesXY);
            tiledStencil<false><<<blocks, threads, 0, stream>>>(mu, current, nx, ny, nz, tilesX, tilesXY);
        } else {
            flatStencil<true><<<blocks, 256, 0, stream>>>(current, mu, nx, ny, nz, count);
            flatStencil<false><<<blocks, 256, 0, stream>>>(mu, current, nx, ny, nz, count);
        }
    }
};

} // namespace

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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / (2 * sizeof(double))) {
        std::fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold;

    try {
        CudaStream stream;
        DeviceBuffer fields(2 * gridSize);
        double* current = fields.data;
        double* mu = fields.data + gridSize;
        const StencilLaunch launch(nx, ny, nz, gridSize);

        printf("Initializing concentration field...\n");
        const unsigned initBlocks = static_cast<unsigned>(
            std::min<size_t>((gridSize + 255) / 256, 65535));
        initializeConcentration<<<initBlocks, 256, 0, stream.stream>>>(current, gridSize);
        CUDA_CHECK(cudaGetLastError());

        // The update reads concentration only at its own output cell, so
        // it can safely operate in place after the chemical-potential kernel.
        // Bounded graphs amortize launches without growing with iterations.
        const int graphSteps = std::min(std::max(iterations, 0), 32);
        const int tailSteps = graphSteps > 0 ? iterations % graphSteps : 0;
        CudaGraph graph;
        CudaGraph tailGraph;
        const auto captureSteps = [&](CudaGraph& captured, int steps) {
            if (steps == 0) return;
            CUDA_CHECK(cudaStreamBeginCapture(stream.stream, cudaStreamCaptureModeThreadLocal));
            for (int t = 0; t < steps; ++t) {
                launch.step(current, mu, stream.stream);
            }
            CUDA_CHECK(cudaStreamEndCapture(stream.stream, &captured.graph));
            CUDA_CHECK(cudaGraphInstantiate(&captured.executable, captured.graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(captured.executable, stream.stream));
        };
        captureSteps(graph, graphSteps);
        captureSteps(tailGraph, tailSteps);
        CUDA_CHECK(cudaStreamSynchronize(stream.stream));

        printf("Running Cahn-Hilliard simulation...\n");
        const auto start = std::chrono::steady_clock::now();
        if (graphSteps > 0) {
            for (int t = 0; iterations - t >= graphSteps; t += graphSteps) {
                CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.stream));
            }
        }
        if (tailSteps > 0) {
            CUDA_CHECK(cudaGraphLaunch(tailGraph.executable, stream.stream));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.stream));
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Computation time: %lld ms\n", static_cast<long long>(milliseconds));
        const double mcups = static_cast<double>(gridSize) * iterations / seconds / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Transfer the final field only when the existing host-side output
        // or validation needs it. All timestep data stays on the device.
        if (printResults || validate) {
            cold.resize(gridSize);
            CUDA_CHECK(cudaMemcpy(cold.data(), current, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA simulation failed: %s\n", error.what());
        return 1;
    }

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
