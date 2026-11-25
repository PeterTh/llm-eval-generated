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

constexpr unsigned blockSize = 128;
constexpr unsigned tileX = 32;
constexpr unsigned tileY = blockSize / tileX;
constexpr unsigned maxGridYZ = 65535;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* field, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const double pseudo = ((i + 1) * 1299709 % count) / static_cast<double>(count);
        field[i] = -1.0 + 2.0 * pseudo;
    }
}

template <bool chemicalPotential>
__device__ __forceinline__ double stencilValue(double center, double xm, double xp,
                                                double ym, double yp, double zm, double zp,
                                                const double* concentration, size_t i) {
    // Unit grid spacing in all three dimensions.
    const double twiceCenter = 2.0 * center;
    const double dxx = (xp + xm) - twiceCenter;
    const double dyy = (yp + ym) - twiceCenter;
    const double dzz = (zp + zm) - twiceCenter;
    const double laplacian = (dxx + dyy) + dzz;

    if constexpr (chemicalPotential) {
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        constexpr double gamma = 0.5;
        // Preserve the operation order of the optimized scalar expression.
        const double interaction = fma(center + 1.0, eAA, (center - 1.0) * eBB);
        const double energy = fma(-twiceCenter, eAB, interaction);
        const double local = fma(center * center, center, fma(4.5, energy, 3.0 * center));
        return fma(-gamma, laplacian, local);
    } else {
        constexpr double dt = 0.01;
        constexpr double diffusivity = 1.0;
        return fma(dt * diffusivity, laplacian, concentration[i]);
    }
}

// Each warp covers adjacent x coordinates. A thread walks a short z column,
// keeping the center and both z neighbors in registers as it advances.
template <bool chemicalPotential, unsigned tileZ>
__global__ void tiledStencil(const double* __restrict__ field,
                             const double* __restrict__ concentration,
                             double* __restrict__ result, unsigned nx, unsigned ny, unsigned nz) {
    const unsigned x = blockIdx.x * tileX + threadIdx.x;
    const unsigned y = blockIdx.y * tileY + threadIdx.y;
    const unsigned z = blockIdx.z * tileZ;
    if (x >= nx || y >= ny) return;

    const unsigned plane = nx * ny;
    const unsigned xm = x > 0;
    const unsigned xp = x + 1 < nx;
    const unsigned ym = y > 0 ? nx : 0;
    const unsigned yp = y + 1 < ny ? nx : 0;
    const unsigned depth = min(tileZ, nz - z);
    unsigned i = z * plane + y * nx + x;
    double previous = field[z > 0 ? i - plane : i];
    double center = field[i];

#pragma unroll
    for (unsigned layer = 0; layer < tileZ; ++layer) {
        if (layer >= depth) break;
        const double next = z + layer + 1 < nz ? field[i + plane] : center;
        result[i] = stencilValue<chemicalPotential>(center, field[i - xm], field[i + xp],
                    field[i - ym], field[i + yp], previous, next, concentration, i);
        previous = center;
        center = next;
        i += plane;
    }
}

// Flatten narrow and small domains so they still fill warps and expose all
// their cells in parallel, even when one or two dimensions have extent one.
template <bool chemicalPotential, typename Index>
__global__ void linearStencil(const double* __restrict__ field,
                              const double* __restrict__ concentration,
                              double* __restrict__ result, Index nx, Index ny, Index nz,
                              Index count) {
    const Index plane = nx * ny;
    const Index stride = gridDim.x * blockDim.x;
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const Index z = i / plane;
        const Index rowOffset = i - z * plane;
        const Index y = rowOffset / nx;
        const Index x = rowOffset - y * nx;
        const Index xm = x > 0;
        const Index xp = x + 1 < nx;
        const Index ym = y > 0 ? nx : 0;
        const Index yp = y + 1 < ny ? nx : 0;
        const Index zm = z > 0 ? plane : 0;
        const Index zp = z + 1 < nz ? plane : 0;
        result[i] = stencilValue<chemicalPotential>(field[i], field[i - xm], field[i + xp],
                    field[i - ym], field[i + yp], field[i - zm], field[i + zp], concentration, i);
    }
}

template <bool chemicalPotential>
void launchStencil(const double* field, const double* concentration, double* result,
                   size_t nx, size_t ny, size_t nz, size_t count, unsigned linearBlocks,
                   cudaStream_t stream) {
    const bool use32Bit = count <= std::numeric_limits<unsigned>::max() - linearBlocks * blockSize;
    const size_t xBlocks = (nx + tileX - 1) / tileX;
    const size_t yBlocks = (ny + tileY - 1) / tileY;
    const bool canTile = use32Bit && nx >= 16 && ny >= tileY && yBlocks <= maxGridYZ;

    // Shorter columns leave more blocks available to schedule on small grids.
    const size_t shortZBlocks = (nz + 1) / 2;
    const size_t longZBlocks = (nz + 3) / 4;
    const dim3 threads(tileX, tileY);
    if (canTile && count <= (size_t{1} << 20) && nz >= 2 && shortZBlocks <= maxGridYZ) {
        tiledStencil<chemicalPotential, 2><<<dim3(xBlocks, yBlocks, shortZBlocks), threads, 0, stream>>>(
            field, concentration, result, nx, ny, nz);
    } else if (canTile && nz >= 4 && longZBlocks <= maxGridYZ) {
        tiledStencil<chemicalPotential, 4><<<dim3(xBlocks, yBlocks, longZBlocks), threads, 0, stream>>>(
            field, concentration, result, nx, ny, nz);
    } else if (use32Bit) {
        linearStencil<chemicalPotential, unsigned><<<linearBlocks, blockSize, 0, stream>>>(
            field, concentration, result, nx, ny, nz, count);
    } else {
        linearStencil<chemicalPotential, size_t><<<linearBlocks, blockSize, 0, stream>>>(
            field, concentration, result, nx, ny, nz, count);
    }
}

} // namespace

bool validateResult(const std::vector<double>& c) {
    if (c.empty()) {
        printf("Validation failed: concentration field is empty\n");
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

    constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx != 0 && (ny > maxCells / nx || nz > maxCells / (nx * ny))) {
        fprintf(stderr, "Grid is too large\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double* current = nullptr;
    double* next = nullptr;
    double* potential = nullptr;

    printf("Initializing concentration field...\n");
    const unsigned linearBlocks = std::min<size_t>((gridSize + blockSize - 1) / blockSize, 65535);
    if (gridSize != 0) {
        checkCuda(cudaMalloc(&current, bytes), "Allocating concentration");
        if (iterations > 0) {
            checkCuda(cudaMalloc(&next, bytes), "Allocating next concentration");
            checkCuda(cudaMalloc(&potential, bytes), "Allocating chemical potential");
        }
        initializeConcentration<<<linearBlocks, blockSize>>>(current, gridSize);
        checkCuda(cudaGetLastError(), "Launching concentration initialization");
        checkCuda(cudaDeviceSynchronize(), "Initializing concentration");
    }

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (gridSize != 0 && iterations > 0) {
        const cudaStream_t stream = cudaStreamPerThread;
        const auto enqueueSteps = [&](int steps) {
            for (int t = 0; t < steps; ++t) {
                launchStencil<true>(current, nullptr, potential, nx, ny, nz, gridSize, linearBlocks, stream);
                launchStencil<false>(potential, current, next, nx, ny, nz, gridSize, linearBlocks, stream);
                std::swap(current, next);
            }
        };

        // Replay an even number of steps so the graph always starts with
        // the same buffer as input. This amortizes launch overhead on small grids.
        constexpr int graphSteps = 32;
        int remaining = iterations;
        cudaGraphExec_t executable = nullptr;
        if (gridSize <= 262144 && iterations >= 32 * graphSteps) {
            cudaGraph_t graph;
            checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                      "Capturing simulation steps");
            enqueueSteps(graphSteps);
            checkCuda(cudaStreamEndCapture(stream, &graph), "Finishing simulation capture");
            checkCuda(cudaGraphInstantiateWithFlags(&executable, graph, 0), "Instantiating simulation graph");
            checkCuda(cudaGraphDestroy(graph), "Destroying simulation graph");
            for (; remaining >= graphSteps; remaining -= graphSteps) {
                checkCuda(cudaGraphLaunch(executable, stream), "Launching simulation graph");
            }
        }
        enqueueSteps(remaining);
        checkCuda(cudaGetLastError(), "Launching simulation");
        checkCuda(cudaStreamSynchronize(stream), "Running simulation");
        if (executable != nullptr) {
            checkCuda(cudaGraphExecDestroy(executable), "Destroying executable simulation graph");
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", milliseconds.count());

    const double cellUpdates = static_cast<double>(gridSize) * std::max(0, iterations);
    const double seconds = std::chrono::duration<double>(elapsed).count();
    printf("Performance: %.3f MCellUpdates/s\n",
           seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0);

    std::vector<double> cold;
    if ((printResults || validate) && gridSize != 0) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), current, bytes, cudaMemcpyDeviceToHost),
                  "Copying final concentration");
    }
    if (current != nullptr) checkCuda(cudaFree(current), "Freeing concentration");
    if (next != nullptr) checkCuda(cudaFree(next), "Freeing next concentration");
    if (potential != nullptr) checkCuda(cudaFree(potential), "Freeing chemical potential");

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold);

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
