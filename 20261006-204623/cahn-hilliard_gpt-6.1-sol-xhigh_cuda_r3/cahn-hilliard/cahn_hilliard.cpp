#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA errors must not silently produce an incomplete result.
void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

class DeviceField {
public:
    explicit DeviceField(size_t bytes) { checkCuda(cudaMalloc(&data, bytes)); }
    ~DeviceField() { cudaFree(data); }
    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;
    double* data = nullptr;
};

class SimulationStream {
public:
    SimulationStream() { checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~SimulationStream() { cudaStreamDestroy(stream); }
    SimulationStream(const SimulationStream&) = delete;
    SimulationStream& operator=(const SimulationStream&) = delete;
    cudaStream_t stream = nullptr;
};

class TimeStepGraph {
public:
    TimeStepGraph() = default;
    TimeStepGraph(const TimeStepGraph&) = delete;
    TimeStepGraph& operator=(const TimeStepGraph&) = delete;
    ~TimeStepGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
};

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t{1299709}) % volume) /
                              static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Coalesced XY planes and a register window along Z avoid reloading the
// previous and current planes at every cell. Small grids use one plane per
// block to expose enough independent work to the GPU.
constexpr unsigned ZTile = 8;

__device__ __forceinline__ double chemicalPotential(double cv, double laplacian) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0;
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
}

template <bool Chemical, typename Index, unsigned Depth>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ concentration,
                        double* __restrict__ output,
                        unsigned nx, unsigned ny, unsigned nz) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const Index plane = static_cast<Index>(nx) * ny;
    for (unsigned zBase = blockIdx.z * Depth;
         zBase < nz; zBase += gridDim.z * Depth) {
        for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
             y < ny; y += gridDim.y * blockDim.y) {
            Index i = static_cast<Index>(zBase) * plane + static_cast<Index>(y) * nx + x;
            double previous = input[i - (zBase > 0 ? plane : 0)];
            double center = input[i];
            #pragma unroll
            for (unsigned offset = 0; offset < Depth; ++offset, i += plane) {
                if (zBase + offset < nz) {
                    const double following = input[i + (zBase + offset + 1 < nz ? plane : 0)];
                    const double cxx = input[i + (x + 1 < nx ? 1 : 0)] +
                                       input[i - (x > 0 ? 1 : 0)] - 2.0 * center;
                    const double cyy = input[i + (y + 1 < ny ? nx : 0)] +
                                       input[i - (y > 0 ? nx : 0)] - 2.0 * center;
                    const double czz = following + previous - 2.0 * center;
                    const double laplacian = cxx + cyy + czz;
                    if constexpr (Chemical) {
                        output[i] = chemicalPotential(center, laplacian);
                    } else {
                        output[i] = concentration[i] + 0.01 * laplacian;
                    }
                    previous = center;
                    center = following;
                }
            }
        }
    }
}

// A linear launch keeps very thin grids from wasting most lanes in XY tiles.
template <bool Chemical, typename Index>
__global__ void thinStencil(const double* __restrict__ input,
                            const double* __restrict__ concentration,
                            double* __restrict__ output,
                            unsigned nx, unsigned ny, unsigned nz) {
    const Index plane = static_cast<Index>(nx) * ny;
    const Index volume = plane * nz;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume;) {
        const unsigned x = static_cast<unsigned>(i % nx);
        const unsigned y = static_cast<unsigned>(i / nx % ny);
        const unsigned z = static_cast<unsigned>(i / plane);
        const double center = input[i];
        const double cxx = input[i + (x + 1 < nx ? 1 : 0)] +
                           input[i - (x > 0 ? 1 : 0)] - 2.0 * center;
        const double cyy = input[i + (y + 1 < ny ? nx : 0)] +
                           input[i - (y > 0 ? nx : 0)] - 2.0 * center;
        const double czz = input[i + (z + 1 < nz ? plane : 0)] +
                           input[i - (z > 0 ? plane : 0)] - 2.0 * center;
        const double laplacian = cxx + cyy + czz;
        if constexpr (Chemical) {
            output[i] = chemicalPotential(center, laplacian);
        } else {
            output[i] = concentration[i] + 0.01 * laplacian;
        }
        // Avoid overflow on the last grid-stride increment.
        if (volume - i <= stride) break;
        i += stride;
    }
}

template <typename Index>
void launchThinStencils(double* cold, double* cnew, double* mu,
                        unsigned nx, unsigned ny, unsigned nz,
                        cudaStream_t stream) {
    const size_t volume = static_cast<size_t>(nx) * ny * nz;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, (volume + 255) / 256));
    thinStencil<true, Index><<<blocks, 256, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
    thinStencil<false, Index><<<blocks, 256, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
}

template <typename Index, unsigned Depth>
void launchStencils(double* cold, double* cnew, double* mu,
                    unsigned nx, unsigned ny, unsigned nz,
                    dim3 grid, dim3 block, cudaStream_t stream) {
    stencil<true, Index, Depth><<<grid, block, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
    stencil<false, Index, Depth><<<grid, block, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
}

void launchTimeStep(double* cold, double* cnew, double* mu,
                    unsigned nx, unsigned ny, unsigned nz,
                    dim3 grid, dim3 block, cudaStream_t stream) {
    const size_t volume = static_cast<size_t>(nx) * ny * nz;
    if (static_cast<size_t>(nx) * ny < 256) {
        if (volume <= std::numeric_limits<unsigned>::max()) {
            launchThinStencils<unsigned>(cold, cnew, mu, nx, ny, nz, stream);
        } else {
            launchThinStencils<size_t>(cold, cnew, mu, nx, ny, nz, stream);
        }
    } else if (volume <= std::numeric_limits<unsigned>::max()) {
        if (volume < (1u << 20)) {
            launchStencils<unsigned, 1>(cold, cnew, mu, nx, ny, nz, grid, block, stream);
        } else {
            launchStencils<unsigned, ZTile>(cold, cnew, mu, nx, ny, nz, grid, block, stream);
        }
    } else {
        launchStencils<size_t, ZTile>(cold, cnew, mu, nx, ny, nz, grid, block, stream);
    }
}

void captureTimeSteps(TimeStepGraph& steps, int count,
                      double* current, double* next, double* mu,
                      unsigned nx, unsigned ny, unsigned nz,
                      dim3 grid, dim3 block, cudaStream_t stream) {
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int t = 0; t < count; t += 2) {
        launchTimeStep(current, next, mu, nx, ny, nz, grid, block, stream);
        launchTimeStep(next, current, mu, nx, ny, nz, grid, block, stream);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamEndCapture(stream, &steps.graph));
    checkCuda(cudaGraphInstantiate(&steps.executable, steps.graph, nullptr, nullptr, 0));
    checkCuda(cudaGraphUpload(steps.executable, stream));
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
    
    if (nx == 0 || nx > std::numeric_limits<unsigned>::max() ||
        ny > std::numeric_limits<unsigned>::max() ||
        nz > std::numeric_limits<unsigned>::max() ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or overflowing grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold;

    try {
        DeviceField field0(gridSize * sizeof(double));
        DeviceField field1(gridSize * sizeof(double));
        DeviceField potential(gridSize * sizeof(double));
        SimulationStream simulation;
        const cudaStream_t stream = simulation.stream;
        double* current = field0.data;
        double* next = field1.data;
        double* mu = potential.data;

        // Narrow grids still fill a block with useful rows. Limit Y/Z launch
        // dimensions and traverse any remaining slabs inside the kernel.
        unsigned minimumY = 1;
        while (minimumY < ny && minimumY < 8) minimumY *= 2;
        unsigned blockX = 1;
        while (blockX < nx && blockX < 256 / minimumY) blockX *= 2;
        unsigned blockY = 1;
        while (blockY < ny && blockY < 256 / blockX) blockY *= 2;
        const dim3 block(blockX, blockY);
        const unsigned depth = gridSize < (1u << 20) ? 1 : ZTile;
        const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                        static_cast<unsigned>(std::min<size_t>(65535, (ny + block.y - 1) / block.y)),
                        static_cast<unsigned>(std::min<size_t>(65535, (nz + depth - 1) / depth)));

        printf("Initializing concentration field...\n");
        const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>(65535, (gridSize + 255) / 256));
        initializeConcentration<<<initBlocks, 256, 0, stream>>>(current, gridSize);
        checkCuda(cudaGetLastError());

        // An even-sized graph restores buffer parity, so it can be replayed
        // for arbitrarily many steps with no per-step host synchronization.
        TimeStepGraph steps, tail;
        const int graphSteps = std::min(32, iterations > 1 ? iterations - iterations % 2 : 0);
        const int tailSteps = graphSteps > 0 ? (iterations % graphSteps) / 2 * 2 : 0;
        if (graphSteps > 0) {
            captureTimeSteps(steps, graphSteps, current, next, mu, nx, ny, nz, grid, block, stream);
        }
        if (tailSteps > 0) {
            captureTimeSteps(tail, tailSteps, current, next, mu, nx, ny, nz, grid, block, stream);
        }
        checkCuda(cudaStreamSynchronize(stream));

        printf("Running Cahn-Hilliard simulation...\n");
        const auto start = std::chrono::steady_clock::now();
        int t = 0;
        if (graphSteps > 0) {
            for (; t <= iterations - graphSteps; t += graphSteps) {
                checkCuda(cudaGraphLaunch(steps.executable, stream));
            }
        }
        if (tailSteps > 0) {
            checkCuda(cudaGraphLaunch(tail.executable, stream));
            t += tailSteps;
        }
        for (; t < iterations; ++t) {
            launchTimeStep(current, next, mu, nx, ny, nz, grid, block, stream);
            std::swap(current, next);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaStreamSynchronize(stream));
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        printf("Computation time: %.3f ms\n", milliseconds);
        const double mcups = static_cast<double>(gridSize) * iterations / (milliseconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Host transfers are needed only for the original reporting options.
        if (printResults || validate) {
            cold.resize(gridSize);
            checkCuda(cudaMemcpyAsync(cold.data(), current, gridSize * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaStreamSynchronize(stream));
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA simulation failed: %s\n", error.what());
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
