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

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(double)), "Allocating GPU fields");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double* data() const { return data_; }

private:
    double* data_ = nullptr;
};

template <typename Index>
struct Grid {
    unsigned nx, ny, nz;
    Index plane;
};

template <bool powerOfTwo>
__global__ void initializeConcentration(double* c, size_t count, double scale) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const size_t seed = (index + 1) * 1299709;
    // Division by a power of two is exact using its reciprocal.
    const double pseudo = powerOfTwo ? static_cast<double>(seed & (count - 1)) * scale
                                     : static_cast<double>(seed % count) / scale;
    c[index] = -1.0 + 2.0 * pseudo;
}

// Each axis uses unit spacing and a clamped neighbor at the boundary.
template <typename Index>
__device__ __forceinline__ double laplacian(const double* field, Index index, double center,
                                          Grid<Index> grid, unsigned x, unsigned y, unsigned z) {
    const double twiceCenter = 2.0 * center;
    const double xx = (field[index + (x + 1 < grid.nx ? 1 : 0)] +
                       field[index - (x > 0 ? 1 : 0)]) - twiceCenter;
    const double yy = (field[index + (y + 1 < grid.ny ? grid.nx : 0)] +
                       field[index - (y > 0 ? grid.nx : 0)]) - twiceCenter;
    const double zz = (field[index + (z + 1 < grid.nz ? grid.plane : 0)] +
                       field[index - (z > 0 ? grid.plane : 0)]) - twiceCenter;
    return (xx + yy) + zz;
}

template <bool chemicalPotential, typename Index>
__device__ __forceinline__ void stencilCell(const double* input, const double* cold, double* output,
                                            Grid<Index> grid, unsigned x, unsigned y, unsigned z) {
    const Index index = z * grid.plane + static_cast<Index>(y) * grid.nx + x;
    const double center = input[index];
    const double lap = laplacian(input, index, center, grid, x, y, z);

    if constexpr (chemicalPotential) {
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        constexpr double gamma = 0.5;
        // Keep the operation order of the original optimized double-precision expression.
        const double interaction = fma(-2.0 * center, eAB,
                                       fma(center + 1.0, eAA, (center - 1.0) * eBB));
        const double energy = fma(center * center, center, fma(4.5, interaction, 3.0 * center));
        output[index] = fma(-gamma, lap, energy);
    } else {
        constexpr double dt = 0.01;
        constexpr double diffusion = 1.0;
        output[index] = fma(dt * diffusion, lap, cold[index]);
    }
}

template <bool chemicalPotential, bool stride, typename Index>
__global__ void stencil(const double* __restrict__ input, const double* __restrict__ cold,
                        double* __restrict__ output, Grid<Index> grid) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned firstY = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned firstZ = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= grid.nx) return;

    if constexpr (stride) {
        for (unsigned z = firstZ; z < grid.nz; z += gridDim.z * blockDim.z) {
            for (unsigned y = firstY; y < grid.ny; y += gridDim.y * blockDim.y) {
                stencilCell<chemicalPotential>(input, cold, output, grid, x, y, z);
            }
        }
    } else if (firstY < grid.ny && firstZ < grid.nz) {
        stencilCell<chemicalPotential>(input, cold, output, grid, x, firstY, firstZ);
    }
}

template <typename Index>
double* evolve(Grid<Index> grid, double* cold, double* cnew, double* mu, int iterations) {
    constexpr unsigned maxGridYZ = 65535;
    constexpr unsigned targetThreads = 128;

    // Keep warps along contiguous X, packing extra rows or planes for narrow grids.
    unsigned bx = 32;
    while (bx > 1 && grid.nx <= bx / 2) bx /= 2;
    unsigned by = targetThreads / bx;
    while (by > 1 && grid.ny <= by / 2) by /= 2;
    const unsigned bz = std::min({targetThreads / (bx * by), grid.nz, 64u});
    const dim3 block(bx, by, bz);
    const unsigned tilesY = (grid.ny + by - 1) / by;
    const unsigned tilesZ = (grid.nz + bz - 1) / bz;
    const dim3 blocks((grid.nx + bx - 1) / bx,
                      std::min(tilesY, maxGridYZ), std::min(tilesZ, maxGridYZ));
    const bool needsStride = tilesY > maxGridYZ || tilesZ > maxGridYZ;

    // The default stream orders both stencil passes and successive time steps.
    for (int t = 0; t < iterations; ++t) {
        if (needsStride) {
            stencil<true, true><<<blocks, block>>>(cold, nullptr, mu, grid);
            stencil<false, true><<<blocks, block>>>(mu, cold, cnew, grid);
        } else {
            stencil<true, false><<<blocks, block>>>(cold, nullptr, mu, grid);
            stencil<false, false><<<blocks, block>>>(mu, cold, cnew, grid);
        }
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "Launching CUDA stencils");
    return cold;
}

} // namespace

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
    
    // Align the three fields independently within the allocation for coalesced access.
    constexpr size_t fieldAlignment = 32;
    const size_t fieldCount = iterations > 0 ? 3 : 1;
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double) / fieldCount
                            - (fieldAlignment - 1);
    if (nx > std::numeric_limits<int>::max() || ny > std::numeric_limits<int>::max() ||
        nz > std::numeric_limits<int>::max() ||
        (nx != 0 && (ny > maxCells / nx || nz > maxCells / (nx * ny)))) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t fieldStride = (gridSize + fieldAlignment - 1) / fieldAlignment * fieldAlignment;
    DeviceBuffer fields(fieldStride * fieldCount);
    double* cold = fields.data();

    printf("Initializing concentration field...\n");
    if (gridSize != 0) {
        constexpr unsigned threads = 256;
        const size_t blocks = (gridSize + threads - 1) / threads;
        if (blocks > std::numeric_limits<int>::max()) {
            fprintf(stderr, "Grid too large for CUDA initialization\n");
            return 1;
        }
        const double count = static_cast<double>(gridSize);
        if ((gridSize & (gridSize - 1)) == 0) {
            initializeConcentration<true><<<blocks, threads>>>(cold, gridSize, 1.0 / count);
        } else {
            initializeConcentration<false><<<blocks, threads>>>(cold, gridSize, count);
        }
        checkCuda(cudaGetLastError(), "Launching CUDA initialization");
        checkCuda(cudaDeviceSynchronize(), "Initializing concentration on GPU");
    }

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();

    if (gridSize != 0 && iterations > 0) {
        double* const cnew = cold + fieldStride;
        double* const mu = cnew + fieldStride;
        const unsigned x = static_cast<unsigned>(nx);
        const unsigned y = static_cast<unsigned>(ny);
        const unsigned z = static_cast<unsigned>(nz);
        if (gridSize <= std::numeric_limits<unsigned>::max()) {
            cold = evolve(Grid<unsigned>{x, y, z, static_cast<unsigned>(plane)}, cold, cnew, mu, iterations);
        } else {
            cold = evolve(Grid<size_t>{x, y, z, plane}, cold, cnew, mu, iterations);
        }
        checkCuda(cudaDeviceSynchronize(), "Running CUDA simulation");
    }

    const auto duration = std::chrono::steady_clock::now() - start;
    printf("Computation time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

    const double seconds = std::chrono::duration<double>(duration).count();
    const double cellUpdates = static_cast<double>(gridSize) * std::max(0, iterations);
    printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0);

    std::vector<double> result;
    if ((printResults || validate) && gridSize != 0) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost),
                  "Copying concentration from GPU");
    }

    if (printResults) {
        print_results(result, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result);
        
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
