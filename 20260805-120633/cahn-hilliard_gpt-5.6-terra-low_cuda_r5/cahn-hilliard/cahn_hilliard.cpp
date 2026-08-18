#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error));                                  \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

// Each thread owns one cell.  The linear layout preserves the original x-major
// indexing and makes x-neighbor loads fully coalesced for interior cells.
__device__ __forceinline__ double laplacian(const double* field, const size_t index,
                                             const size_t x, const size_t y, const size_t z,
                                             const size_t nx, const size_t ny, const size_t nz,
                                             const double invDx2, const double invDy2,
                                             const double invDz2) {
    const size_t plane = nx * ny;
    const double center = field[index];
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);
    return (field[z * plane + y * nx + xp] + field[z * plane + y * nx + xn] - 2.0 * center) * invDx2
         + (field[z * plane + yp * nx + x] + field[z * plane + yn * nx + x] - 2.0 * center) * invDy2
         + (field[zp * plane + y * nx + x] + field[zn * plane + y * nx + x] - 2.0 * center) * invDz2;
}

__global__ void initializeConcentration(double* c, const size_t volume) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < volume) {
        const double pseudo = (((index + 1) * static_cast<size_t>(1299709)) % volume) /
                              static_cast<double>(volume);
        c[index] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double invDx2, const double invDy2, const double invDz2,
                                         const double gamma, const double eAA, const double eBB, const double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = z * (nx * ny) + y * nx + x;
        const double cv = c[index];
        mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma * laplacian(c, index, x, y, z, nx, ny, nz, invDx2, invDy2, invDz2);
    }
}

__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                                   const double* __restrict__ mu,
                                   const size_t nx, const size_t ny, const size_t nz,
                                   const double dtD, const double invDx2,
                                   const double invDy2, const double invDz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = z * (nx * ny) + y * nx + x;
        cnew[index] = cold[index] + dtD * laplacian(mu, index, x, y, z, nx, ny, nz, invDx2, invDy2, invDz2);
    }
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
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Keep all three fields on the GPU for the complete simulation.  This avoids
    // the per-step PCIe transfers that would otherwise dominate small time steps.
    double* dCold = nullptr;
    double* dCnew = nullptr;
    double* dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dCold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMu, gridSize * sizeof(double)));
    constexpr dim3 blockShape(32, 4, 2);
    const dim3 gridShape(static_cast<unsigned int>((nx + blockShape.x - 1) / blockShape.x),
                         static_cast<unsigned int>((ny + blockShape.y - 1) / blockShape.y),
                         static_cast<unsigned int>((nz + blockShape.z - 1) / blockShape.z));
    constexpr unsigned int initializationThreads = 256;
    const unsigned int initializationBlocks = static_cast<unsigned int>((gridSize + initializationThreads - 1) / initializationThreads);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration<<<initializationBlocks, initializationThreads>>>(dCold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start{};
    cudaEvent_t end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential<<<gridShape, blockShape>>>(
            dCold, dMu, nx, ny, nz, 1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz),
            gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate<<<gridShape, blockShape>>>(
            dCnew, dCold, dMu, nx, ny, nz, dt * D,
            1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz));
        
        // Swap buffers
        std::swap(dCold, dCnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
    
    printf("Computation time: %ld ms\n", durationMilliseconds);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (elapsedMilliseconds / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Materialize the final field only for operations that consume it on the host.
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), dCold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
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
            CUDA_CHECK(cudaFree(dCold));
            CUDA_CHECK(cudaFree(dCnew));
            CUDA_CHECK(cudaFree(dMu));
            return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(dCold));
            CUDA_CHECK(cudaFree(dCnew));
            CUDA_CHECK(cudaFree(dMu));
            return 1;
        }
    }
    
    CUDA_CHECK(cudaFree(dCold));
    CUDA_CHECK(cudaFree(dCnew));
    CUDA_CHECK(cudaFree(dMu));
    return 0;
}
