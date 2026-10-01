#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Fail explicitly if CUDA is unavailable; simulation always runs on the GPU.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

// Adjacent threads process adjacent X values. The grid-stride loop supports
// large volumes without depending on the device's grid dimension limits.
template <typename Index>
__device__ __forceinline__ double computeLaplacian(
    const double* __restrict__ c, Index i, Index nx, Index ny, Index nz) {
    const Index plane = nx * ny;
    const Index x = i % nx;
    const Index y = (i / nx) % ny;
    const Index z = i / plane;
    const double center = 2.0 * c[i];
    const double cxx = (c[x + 1 < nx ? i + 1 : i] + c[x ? i - 1 : i] - center);
    const double cyy = (c[y + 1 < ny ? i + nx : i] + c[y ? i - nx : i] - center);
    const double czz = (c[z + 1 < nz ? i + plane : i] + c[z ? i - plane : i] - center);
    // The benchmark's dx, dy and dz are all 1.0.
    return cxx + cyy + czz;
}

template <typename Index>
__global__ void computeChemicalPotential(
    const double* __restrict__ c, double* __restrict__ mu,
    Index nx, Index ny, Index nz, size_t volume) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        const double cv = c[i];
        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * computeLaplacian(c, static_cast<Index>(i), nx, ny, nz);
    }
}

template <typename Index>
__global__ void cahnHilliardUpdate(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu, Index nx, Index ny, Index nz, size_t volume) {
    constexpr double dt = 0.01;
    constexpr double D = 1.0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        cnew[i] = cold[i] + dt * D * computeLaplacian(mu, static_cast<Index>(i), nx, ny, nz);
    }
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(
        (gridSize + threads - 1) / threads, 65535));

    printf("Initializing concentration field...\n");
    initializeConcentration<<<blocks, threads, 0, stream>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Two steps restore buffer parity, allowing the same graph to be replayed
    // for any iteration count. Kernel dependencies provide the global barriers
    // required between potential computation and concentration updates.
    auto launchStep = [&](auto sx, auto sy, auto sz) {
        computeChemicalPotential<<<blocks, threads, 0, stream>>>(cold, mu, sx, sy, sz, gridSize);
        cahnHilliardUpdate<<<blocks, threads, 0, stream>>>(cnew, cold, mu, sx, sy, sz, gridSize);
        std::swap(cold, cnew);
    };
    auto step = [&]() {
        // Use faster 32-bit coordinate arithmetic whenever the volume permits.
        // The loop and memory offsets remain size_t for large allocations.
        if (gridSize <= std::numeric_limits<unsigned>::max()) {
            launchStep(static_cast<unsigned>(nx), static_cast<unsigned>(ny),
                       static_cast<unsigned>(nz));
        } else {
            launchStep(nx, ny, nz);
        }
    };
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (iterations >= 2) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        step();
        step();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations / 2; ++t) {
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
    if (iterations > 0 && iterations % 2 != 0) step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
