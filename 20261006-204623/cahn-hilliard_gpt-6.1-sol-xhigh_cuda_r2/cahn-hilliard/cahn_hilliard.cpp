#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

constexpr unsigned int blockX = 32;
constexpr unsigned int blockY = 4;
constexpr unsigned int planesPerBlock = 4;

// These are the physical parameters of the original benchmark. Unit grid
// spacing removes divisions without changing the order of the stencil sums.
__device__ __forceinline__ double laplacian(double center, double xp, double xn,
                                           double yp, double yn, double zp, double zn) {
    const double cxx = xp + xn - 2.0 * center;
    const double cyy = yp + yn - 2.0 * center;
    const double czz = zp + zn - 2.0 * center;
    return cxx + cyy + czz;
}

__device__ __forceinline__ double chemicalPotential(double cv, double lap) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0;
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv - 0.5 * lap;
}

__global__ void initializeConcentration(double* c, size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += stride) {
        const double pseudo = (((i + 1) * size_t{1299709}) % volume)
                              / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Each warp visits adjacent X coordinates. A thread streams several Z planes,
// keeping the preceding and following values in registers. Both passes use the
// same clamped stencil, with a kernel dependency between mu and concentration.
// Very long, thin grids use a flattened tile grid to avoid the smaller CUDA
// Y/Z grid limits; ordinary grids need no tile-coordinate divisions.
template <bool chemical, bool flat>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz,
                        size_t tilesX, size_t tilesY) {
    size_t tx = blockIdx.x;
    size_t ty = blockIdx.y;
    size_t tz = blockIdx.z;
    if constexpr (flat) {
        tx = blockIdx.x % tilesX;
        const size_t yz = blockIdx.x / tilesX;
        ty = yz % tilesY;
        tz = yz / tilesY;
    }
    const size_t x = tx * blockX + threadIdx.x;
    const size_t y = ty * blockY + threadIdx.y;
    const size_t z0 = tz * planesPerBlock;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    size_t i = z0 * plane + y * nx + x;
    const size_t xm = x > 0 ? 1 : 0;
    const size_t xp = x + 1 < nx ? 1 : 0;
    const size_t ym = y > 0 ? nx : 0;
    const size_t yp = y + 1 < ny ? nx : 0;
    double previous = input[i - (z0 > 0 ? plane : 0)];
    double center = input[i];

    #pragma unroll
    for (unsigned int k = 0; k < planesPerBlock; ++k) {
        const size_t z = z0 + k;
        if (z >= nz) break;
        const double next = input[i + (z + 1 < nz ? plane : 0)];
        const double lap = laplacian(center, input[i + xp], input[i - xm],
                                    input[i + yp], input[i - ym], next, previous);
        if constexpr (chemical) {
            output[i] = chemicalPotential(center, lap);
        } else {
            output[i] = cold[i] + 0.01 * lap;
        }
        previous = center;
        center = next;
        i += plane;
    }
}

// Device allocation and runtime resources have one owner. No CPU simulation
// path is provided: CUDA is required even when zero time steps are requested.
struct Simulation {
    double* first = nullptr;
    double* second = nullptr;
    double* mu = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    cudaGraphExec_t batch = nullptr;
    cudaGraphExec_t tail = nullptr;

    explicit Simulation(size_t bytes) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&first, bytes));
        CUDA_CHECK(cudaMalloc(&second, bytes));
        CUDA_CHECK(cudaMalloc(&mu, bytes));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
    }

    ~Simulation() {
        if (tail) cudaGraphExecDestroy(tail);
        if (batch) cudaGraphExecDestroy(batch);
        cudaEventDestroy(end);
        cudaEventDestroy(start);
        cudaFree(mu);
        cudaFree(second);
        cudaFree(first);
        cudaStreamDestroy(stream);
    }

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    cudaGraphExec_t makeGraph(int steps, size_t nx, size_t ny, size_t nz,
                             dim3 blocks, bool flat, size_t tilesX, size_t tilesY) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        double* old = first;
        double* next = second;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < steps; ++step) {
            if (flat) {
                stencil<true, true><<<blocks, dim3(blockX, blockY), 0, stream>>>(
                    old, nullptr, mu, nx, ny, nz, tilesX, tilesY);
                stencil<false, true><<<blocks, dim3(blockX, blockY), 0, stream>>>(
                    mu, old, next, nx, ny, nz, tilesX, tilesY);
            } else {
                stencil<true, false><<<blocks, dim3(blockX, blockY), 0, stream>>>(
                    old, nullptr, mu, nx, ny, nz, tilesX, tilesY);
                stencil<false, false><<<blocks, dim3(blockX, blockY), 0, stream>>>(
                    mu, old, next, nx, ny, nz, tilesX, tilesY);
            }
            std::swap(old, next);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        return executable;
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
    
    // Check sizes before computing products or making device allocations.
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxCells / ny || nx * ny > maxCells / nz) {
        fprintf(stderr, "Grid dimensions must be positive and fit in addressable memory\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t tilesX = (nx - 1) / blockX + 1;
    const size_t tilesY = (ny - 1) / blockY + 1;
    const size_t tilesZ = (nz - 1) / planesPerBlock + 1;
    const size_t tileCount = tilesX * tilesY * tilesZ;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (tileCount > static_cast<size_t>(properties.maxGridSize[0])) {
        fprintf(stderr, "Grid exceeds the CUDA launch size limit\n");
        return 1;
    }
    const bool flat = tilesY > static_cast<size_t>(properties.maxGridSize[1])
                      || tilesZ > static_cast<size_t>(properties.maxGridSize[2]);
    const dim3 blocks = flat ? dim3(static_cast<unsigned int>(tileCount))
                            : dim3(static_cast<unsigned int>(tilesX),
                                   static_cast<unsigned int>(tilesY),
                                   static_cast<unsigned int>(tilesZ));
    Simulation simulation(gridSize * sizeof(double));

    printf("Initializing concentration field...\n");
    const unsigned int initBlocks = static_cast<unsigned int>(std::min<size_t>(
        (gridSize - 1) / 256 + 1, static_cast<size_t>(properties.multiProcessorCount) * 32));
    initializeConcentration<<<initBlocks, 256, 0, simulation.stream>>>(simulation.first, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(simulation.stream));

    // Repeated batches contain 32 steps and return to the same buffer.
    // Capture the final partial batch separately, including an odd final step.
    const int steps = std::max(iterations, 0);
    const int batchSteps = std::min(steps, 32);
    const int batches = batchSteps ? steps / batchSteps : 0;
    const int tailSteps = batchSteps ? steps % batchSteps : 0;
    if (batchSteps) {
        simulation.batch = simulation.makeGraph(batchSteps, nx, ny, nz, blocks, flat, tilesX, tilesY);
    }
    if (tailSteps) {
        simulation.tail = simulation.makeGraph(tailSteps, nx, ny, nz, blocks, flat, tilesX, tilesY);
    }
    CUDA_CHECK(cudaStreamSynchronize(simulation.stream));

    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(simulation.start, simulation.stream));
    for (int batch = 0; batch < batches; ++batch) {
        CUDA_CHECK(cudaGraphLaunch(simulation.batch, simulation.stream));
    }
    if (tailSteps) CUDA_CHECK(cudaGraphLaunch(simulation.tail, simulation.stream));
    CUDA_CHECK(cudaEventRecord(simulation.end, simulation.stream));
    CUDA_CHECK(cudaEventSynchronize(simulation.end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, simulation.start, simulation.end));
    printf("Computation time: %.3f ms\n", static_cast<double>(milliseconds));
    const double cellUpdates = static_cast<double>(gridSize) * steps;
    const double mcups = milliseconds > 0.0f ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfers and host allocation are only needed for requested output.
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        const double* final = steps % 2 ? simulation.second : simulation.first;
        CUDA_CHECK(cudaMemcpy(cold.data(), final, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
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
