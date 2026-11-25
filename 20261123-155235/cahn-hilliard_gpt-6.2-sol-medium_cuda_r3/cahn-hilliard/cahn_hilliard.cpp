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

constexpr unsigned initThreads = 128;
constexpr unsigned maxGridYZ = 65535;
constexpr unsigned maxBlockZ = 64;
constexpr size_t tiledMinVolume = 8 * 1024 * 1024;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* concentration, size_t volume) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < volume) {
        const size_t seed = (index + 1) * 1299709;
        const size_t mask = volume - 1;
        const size_t residue = (volume & mask) == 0 ? (seed & mask) : (seed % volume);
        const double pseudo = __ddiv_rn(static_cast<double>(residue), static_cast<double>(volume));
        concentration[index] = -1.0 + 2.0 * pseudo;
    }
}

__device__ __forceinline__ double chemicalPotential(double c, double laplacian) {
    constexpr double eAA = -2.0 / 9.0;
    constexpr double eBB = -2.0 / 9.0;
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    // Explicit contractions preserve the rounding of the optimized CPU expression.
    const double interaction = fma(c + 1.0, eAA, (c - 1.0) * eBB);
    const double linear = fma(4.5, fma(-2.0 * c, eAB, interaction), 3.0 * c);
    return fma(-gamma, laplacian, fma(c * c, c, linear));
}

template <bool update, bool strided, unsigned planesPerThread, typename Index>
__global__ void advanceField(double* __restrict__ concentration, double* __restrict__ potential,
                             Index nx, Index ny, Index nz) {
    const Index x = blockIdx.x * blockDim.x + threadIdx.x;
    const Index firstY = blockIdx.y * blockDim.y + threadIdx.y;
    const Index firstZ = (blockIdx.z * blockDim.z + threadIdx.z) * planesPerThread;
    if (x >= nx || firstY >= ny || firstZ >= nz) {
        return;
    }

    const Index plane = nx * ny;
    const double* field = update ? potential : concentration;
    for (Index zStart = firstZ;; zStart += gridDim.z * blockDim.z * planesPerThread) {
        for (Index y = firstY;; y += gridDim.y * blockDim.y) {
            Index index = zStart * plane + y * nx + x;
            double center = field[index];
            double previous = zStart > 0 ? field[index - plane] : center;

            // Reuse center and z neighbors across adjacent planes in registers.
            #pragma unroll
            for (unsigned step = 0; step < planesPerThread; ++step) {
                const Index z = zStart + step;
                if (z >= nz) {
                    break;
                }
                // Each missing neighbor is clamped to the center. Grid spacing is one.
                const double next = z + 1 < nz ? field[index + plane] : center;
                const double twice = 2.0 * center;
                const double xx = (x + 1 < nx ? field[index + 1] : center)
                                + (x > 0 ? field[index - 1] : center) - twice;
                const double yy = (y + 1 < ny ? field[index + nx] : center)
                                + (y > 0 ? field[index - nx] : center) - twice;
                const double zz = (next + previous) - twice;
                const double laplacian = (xx + yy) + zz;
                if constexpr (update) {
                    // The potential sweep is complete. Only this thread accesses its
                    // concentration cell during the update, so it can write in place.
                    concentration[index] = fma(0.01, laplacian, concentration[index]);
                } else {
                    potential[index] = chemicalPotential(center, laplacian);
                }
                index += plane;
                previous = center;
                center = next;
            }
            if (!strided || y + gridDim.y * blockDim.y >= ny) {
                break;
            }
        }
        if (!strided || zStart + gridDim.z * blockDim.z * planesPerThread >= nz) {
            break;
        }
    }
}

template <bool strided, unsigned planesPerThread, typename Index>
void launchTimeSteps(double* concentration, double* potential, Index nx, Index ny, Index nz,
                     dim3 grid, dim3 block, int iterations) {
    for (int t = 0; t < iterations; ++t) {
        advanceField<false, strided, planesPerThread><<<grid, block>>>(concentration, potential, nx, ny, nz);
        advanceField<true, strided, planesPerThread><<<grid, block>>>(concentration, potential, nx, ny, nz);
    }
    checkCuda(cudaGetLastError(), "Launching simulation kernels");
}

template <unsigned planesPerThread, typename Index>
void runSimulation(double* concentration, double* potential, Index nx, Index ny, Index nz,
                   int iterations) {
    // Pack short rows and shallow slices into the same block. For long x rows
    // on thin grids, spread the warps across x instead.
    constexpr unsigned threadsPerBlock = planesPerThread == 1 ? 128 : 256;
    const unsigned maxWidth = ny <= 2 && nz <= planesPerThread
                                ? threadsPerBlock / ny : 32;
    unsigned width = 1;
    while (width < nx && width < maxWidth) {
        width *= 2;
    }
    unsigned height = 1;
    while (height < ny && width * height < threadsPerBlock) {
        height *= 2;
    }
    const dim3 block(width, height, std::min(threadsPerBlock / (width * height), maxBlockZ));
    const Index yBlocks = (ny - 1) / block.y + 1;
    const Index zBlocks = (nz - 1) / (block.z * planesPerThread) + 1;
    const dim3 grid((nx - 1) / block.x + 1, std::min<Index>(yBlocks, maxGridYZ),
                    std::min<Index>(zBlocks, maxGridYZ));
    if (yBlocks > maxGridYZ || zBlocks > maxGridYZ) {
        launchTimeSteps<true, planesPerThread>(concentration, potential, nx, ny, nz, grid, block, iterations);
    } else {
        launchTimeSteps<false, planesPerThread>(concentration, potential, nx, ny, nz, grid, block, iterations);
    }
}

template <typename Index>
void runSimulation(double* concentration, double* potential, Index nx, Index ny, Index nz,
                   int iterations) {
    // Tile along z once the grid has enough work to keep the GPU occupied.
    if (nz >= 2 && nx * ny * nz >= tiledMinVolume) {
        runSimulation<2>(concentration, potential, nx, ny, nz, iterations);
    } else {
        runSimulation<1>(concentration, potential, nx, ny, nz, iterations);
    }
}

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
    
    constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || nx > maxCells / ny || nx * ny > maxCells / nz) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    double* concentration = nullptr;
    double* potential = nullptr;
    checkCuda(cudaMalloc(&concentration, bytes), "Allocating concentration");
    if (iterations > 0) {
        checkCuda(cudaMalloc(&potential, bytes), "Allocating chemical potential");
    }

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");

    printf("Initializing concentration field...\n");
    const size_t initBlocks = (gridSize - 1) / initThreads + 1;
    initializeConcentration<<<initBlocks, initThreads>>>(concentration, gridSize);
    checkCuda(cudaGetLastError(), "Launching concentration initialization");

    printf("Running Cahn-Hilliard simulation...\n");
    checkCuda(cudaEventRecord(start), "Recording start event");
    if (gridSize <= std::numeric_limits<unsigned>::max()) {
        runSimulation(concentration, potential, static_cast<unsigned>(nx),
                      static_cast<unsigned>(ny), static_cast<unsigned>(nz), iterations);
    } else {
        runSimulation(concentration, potential, nx, ny, nz, iterations);
    }
    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Running simulation");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "Measuring simulation time");
    checkCuda(cudaEventDestroy(start), "Destroying start event");
    checkCuda(cudaEventDestroy(stop), "Destroying stop event");

    printf("Computation time: %.3f ms\n", elapsedMs);
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), concentration, bytes, cudaMemcpyDeviceToHost),
                  "Copying the final concentration");
    }
    checkCuda(cudaFree(concentration), "Freeing concentration");
    checkCuda(cudaFree(potential), "Freeing chemical potential");

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
