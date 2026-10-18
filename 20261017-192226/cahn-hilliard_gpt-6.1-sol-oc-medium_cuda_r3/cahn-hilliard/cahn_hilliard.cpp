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

// The benchmark's fixed physical parameters are compile-time constants so the
// device compiler can eliminate divisions by unit grid spacing.
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gradientEnergy = 0.5;
constexpr double D = 1.0;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ __forceinline__ double computeLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
__global__ void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t yTiles = (ny + blockDim.y - 1) / blockDim.y;
    for (size_t tile = blockIdx.y; tile < yTiles * nz; tile += gridDim.y) {
        const size_t z = tile / yTiles;
        const size_t y = (tile % yTiles) * blockDim.y + threadIdx.y;
        if (x < nx && y < ny) {
            const size_t idx = idx3(x, y, z, nx, ny);
            const double cv = c[idx];

            mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - gradientEnergy * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
        }
    }
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t yTiles = (ny + blockDim.y - 1) / blockDim.y;
    for (size_t tile = blockIdx.y; tile < yTiles * nz; tile += gridDim.y) {
        const size_t z = tile / yTiles;
        const size_t y = (tile % yTiles) * blockDim.y + threadIdx.y;
        if (x < nx && y < ny) {
            const size_t idx = idx3(x, y, z, nx, ny);
            cnew[idx] = cold[idx] + dt * D *
                       computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
        }
    }
}

// Initialize concentration field
__global__ void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < vol; idx += size_t(blockDim.x) * gridDim.x) {
        // Generate pseudo-random value in [-1, 1]
        const size_t linear_id = idx;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

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
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // All time steps operate on device-resident arrays. Adjacent lanes access
    // adjacent X cells; flatten Y/Z tiles to avoid the CUDA grid-Z limit.
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const dim3 block(32, 4);
    const size_t xTiles = (nx + block.x - 1) / block.x;
    if (xTiles > 2147483647u) {
        fprintf(stderr, "X dimension exceeds CUDA grid limits\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned>(xTiles),
                    static_cast<unsigned>(std::min<size_t>(((ny + block.y - 1) / block.y) * nz, 65535)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration<<<static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535)),
                              256, 0, stream>>>(cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto step = [&]() {
        computeChemicalPotential<<<grid, block, 0, stream>>>(cold, mu, nx, ny, nz);
        cahnHilliardUpdate<<<grid, block, 0, stream>>>(cnew, cold, mu, nx, ny, nz);
        std::swap(cold, cnew);
    };

    // Capture a bounded, even number of steps so graph replay reuses the same
    // ping-pong pointers. This amortizes launch overhead for small grids without
    // growing the graph with the requested iteration count.
    const int batchSteps = std::min(iterations / 2, 16) * 2;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (batchSteps > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int t = 0; t < batchSteps; ++t) step();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executableGraph, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int t = 0;
    if (batchSteps > 0) {
        for (; t <= iterations - batchSteps; t += batchSteps) {
            CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
        }
    }
    for (; t < iterations; ++t) step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    
    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
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
