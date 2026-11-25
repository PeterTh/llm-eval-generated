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

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* concentration, size_t volume) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = first; i < volume; i += stride) {
        const double pseudo = ((i + 1) * 1299709 % volume) / static_cast<double>(volume);
        concentration[i] = fma(2.0, pseudo, -1.0);
    }
}

template <bool chemicalPotential, typename Index, unsigned planesPerThread>
__global__ void stencil(const double* __restrict__ field,
                        double* __restrict__ result, unsigned nx, unsigned ny, unsigned nz) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;

    const Index plane = static_cast<Index>(nx) * ny;
    const Index dxm = x > 0;
    const Index dxp = x + 1 < nx;
    const unsigned firstZ = (blockIdx.z * blockDim.z + threadIdx.z) * planesPerThread;
    const unsigned strideZ = gridDim.z * blockDim.z * planesPerThread;

    // Grid strides cover shapes that exceed CUDA's grid limits in Y or Z.
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
         y < ny; y += gridDim.y * blockDim.y) {
        const Index row = static_cast<Index>(y) * nx + x;
        const Index dym = y > 0 ? nx : 0;
        const Index dyp = y + 1 < ny ? nx : 0;
        for (unsigned z = firstZ; z < nz; z += strideZ) {
            Index i = z * plane + row;
            double center = field[i];
            double previous = z > 0 ? field[i - plane] : center;
#pragma unroll
            for (unsigned layer = 0; layer < planesPerThread; ++layer) {
                const bool hasNext = z + layer + 1 < nz;
                const double following = hasNext ? field[i + plane] : center;
                // Keep the original axis order and clamped neighbors at each face.
                const double cxx = fma(-2.0, center, field[i + dxp] + field[i - dxm]);
                const double cyy = fma(-2.0, center, field[i + dyp] + field[i - dym]);
                const double czz = fma(-2.0, center, following + previous);
                const double laplacian = (cxx + cyy) + czz;
                if constexpr (chemicalPotential) {
                    constexpr double energy = 2.0 / 9.0;
                    // Preserve the operation order and fused rounding of the serial build.
                    double interaction = fma(center + 1.0, -energy, (center - 1.0) * -energy);
                    interaction = fma(-2.0 * energy, center, interaction);
                    const double local = fma(center * center, center,
                                             fma(4.5, interaction, 3.0 * center));
                    result[i] = fma(-0.5, laplacian, local);
                } else {
                    result[i] = fma(0.01, laplacian, result[i]);
                }
                if (!hasNext) break;
                previous = center;
                center = following;
                i += plane;
            }
        }
    }
}

template <typename Index, unsigned planesPerThread>
void advance(double* concentration, double* potential, dim3 blocks, dim3 threads,
             unsigned nx, unsigned ny, unsigned nz, int steps, cudaStream_t stream) {
    for (int step = 0; step < steps; ++step) {
        stencil<true, Index, planesPerThread><<<blocks, threads, 0, stream>>>(concentration, potential, nx, ny, nz);
        // Potential is complete, and each update reads only its own concentration.
        stencil<false, Index, planesPerThread><<<blocks, threads, 0, stream>>>(potential, concentration, nx, ny, nz);
    }
    checkCuda(cudaGetLastError(), "Launching stencil kernels");
}

}  // namespace

bool validateResult(const std::vector<double>& c) {
    if (c.empty()) {
        printf("Validation failed: empty concentration field\n");
        return false;
    }

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

    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (std::max({nx, ny, nz}) > std::numeric_limits<int>::max() ||
        (nx && (ny > maxElements / nx || nz > maxElements / (nx * ny)))) {
        fprintf(stderr, "Grid dimensions are too large\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double* current = nullptr;
    double* potential = nullptr;
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Creating CUDA stream");

    printf("Initializing concentration field...\n");
    if (gridSize) {
        checkCuda(cudaMalloc(&current, bytes), "Allocating concentration field");
        constexpr unsigned initThreads = 256;
        const unsigned initBlocks = std::min<size_t>((gridSize + initThreads - 1) / initThreads, 65535);
        initializeConcentration<<<initBlocks, initThreads, 0, stream>>>(current, gridSize);
        checkCuda(cudaGetLastError(), "Launching concentration initialization");
        if (iterations > 0) {
            checkCuda(cudaMalloc(&potential, bytes), "Allocating chemical potential field");
        }
    }

    constexpr unsigned warpWidth = 32;
    constexpr unsigned blockSize = 128;
    constexpr unsigned gridLimitYZ = 65535;
    // Fill warps across X first, then use Y and Z for narrow grids.
    unsigned width = 1;
    while (width < warpWidth && width < nx) width *= 2;
    unsigned rows = 1;
    while (rows < blockSize / width && rows < ny) rows *= 2;
    // Large grids benefit from reusing adjacent Z values in registers.
    const bool reusePlanes = gridSize >= (size_t{1} << 24) && nx >= warpWidth && nz >= 2;
    const unsigned planesPerThread = reusePlanes ? 2 : 1;
    const unsigned maxLayers = std::min(64u, blockSize / (width * rows));
    unsigned layers = 1;
    while (layers < maxLayers && layers * planesPerThread < nz) layers *= 2;
    const dim3 threads(width, rows, layers);
    const unsigned planesPerBlock = layers * planesPerThread;
    const dim3 blocks((nx + threads.x - 1) / threads.x,
                      std::min<size_t>((ny + threads.y - 1) / threads.y, gridLimitYZ),
                      std::min<size_t>((nz + planesPerBlock - 1) / planesPerBlock, gridLimitYZ));
    const bool wideIndices = gridSize > std::numeric_limits<unsigned>::max();
    const auto singlePlane = wideIndices ? advance<size_t, 1> : advance<unsigned, 1>;
    const auto twoPlanes = wideIndices ? advance<size_t, 2> : advance<unsigned, 2>;
    const auto enqueue = reusePlanes ? twoPlanes : singlePlane;
    const auto runSteps = [&](int steps) {
        enqueue(current, potential, blocks, threads, nx, ny, nz, steps, stream);
    };

    cudaGraphExec_t graphExec = nullptr;
    int graphSteps = 0;
    if (gridSize && iterations >= 4) {
        // Reuse the same device buffers and batch launches to reduce dispatch overhead.
        graphSteps = std::min(iterations, 32);
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "Beginning CUDA graph capture");
        runSteps(graphSteps);
        cudaGraph_t graph;
        checkCuda(cudaStreamEndCapture(stream, &graph), "Ending CUDA graph capture");
        checkCuda(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "Instantiating CUDA graph");
        checkCuda(cudaGraphDestroy(graph), "Destroying CUDA graph");
        checkCuda(cudaGraphUpload(graphExec, stream), "Uploading CUDA graph");
    }
    checkCuda(cudaStreamSynchronize(stream), "Preparing CUDA simulation");

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (gridSize && iterations > 0) {
        int remaining = iterations;
        if (graphExec) {
            for (; remaining >= graphSteps; remaining -= graphSteps) {
                checkCuda(cudaGraphLaunch(graphExec, stream), "Launching CUDA graph");
            }
        }
        if (remaining) runSteps(remaining);
        checkCuda(cudaStreamSynchronize(stream), "Completing CUDA simulation");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %lld ms\n", static_cast<long long>(milliseconds.count()));
    const double cellUpdates = static_cast<double>(gridSize) * std::max(iterations, 0);
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / (elapsed.count() * 1e6));

    std::vector<double> concentration;
    if ((printResults || validate) && gridSize) {
        concentration.resize(gridSize);
        checkCuda(cudaMemcpy(concentration.data(), current, bytes, cudaMemcpyDeviceToHost), "Copying concentration results");
    }
    if (graphExec) checkCuda(cudaGraphExecDestroy(graphExec), "Releasing CUDA graph");
    if (potential) checkCuda(cudaFree(potential), "Releasing chemical potential field");
    if (current) checkCuda(cudaFree(current), "Releasing concentration field");
    checkCuda(cudaStreamDestroy(stream), "Destroying CUDA stream");

    // Print results for external validation
    if (printResults) {
        print_results(concentration, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(concentration);

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
