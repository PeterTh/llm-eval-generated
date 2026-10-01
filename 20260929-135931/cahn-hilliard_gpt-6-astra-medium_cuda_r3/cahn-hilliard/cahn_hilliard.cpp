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

// All CUDA failures are fatal: the benchmark always runs on the GPU.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// X is contiguous, so each warp reads and writes consecutive doubles.
// Both passes use exactly the same clamped, seven-point stencil. The
// chemical-potential pass completes globally before the update is launched.
template <bool ChemicalPotential>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    for (size_t z = size_t(blockIdx.z) * blockDim.z + threadIdx.z;
         z < nz; z += size_t(blockDim.z) * gridDim.z) {
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += size_t(blockDim.y) * gridDim.y) {
            for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < nx; x += size_t(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                const double cv = field[i];
                // dx = dy = dz = 1 in this benchmark.
                const double cxx = (field[i + (x + 1 < nx ? 1 : 0)] +
                                    field[i - (x > 0 ? 1 : 0)] - 2.0 * cv);
                const double cyy = (field[i + (y + 1 < ny ? nx : 0)] +
                                    field[i - (y > 0 ? nx : 0)] - 2.0 * cv);
                const double czz = (field[i + (z + 1 < nz ? plane : 0)] +
                                    field[i - (z > 0 ? plane : 0)] - 2.0 * cv);
                const double laplacian = cxx + cyy + czz;
                if constexpr (ChemicalPotential) {
                    constexpr double e_AA = -(2.0 / 9.0);
                    constexpr double e_BB = -(2.0 / 9.0);
                    constexpr double e_AB = 2.0 / 9.0;
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                                      - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
                } else {
                    output[i] = cold[i] + 0.01 * laplacian;
                }
            }
        }
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
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *deviceCold, *deviceNew, *deviceMu;
    cudaCheck(cudaMalloc(&deviceCold, bytes));
    cudaCheck(cudaMalloc(&deviceNew, bytes));
    cudaCheck(cudaMalloc(&deviceMu, bytes));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Initializing concentration field...\n");
    initializeConcentration<<<static_cast<unsigned>(std::min<size_t>(
        (gridSize + 255) / 256, 65535)), 256, 0, stream>>>(deviceCold, gridSize);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));

    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned>(std::min<size_t>((nx + 31) / 32, 65535)),
                    static_cast<unsigned>(std::min<size_t>((ny + 3) / 4, 65535)),
                    static_cast<unsigned>(std::min<size_t>((nz + 1) / 2, 65535)));
    auto step = [&]() {
        stencil<true><<<grid, block, 0, stream>>>(deviceCold, nullptr, deviceMu, nx, ny, nz);
        stencil<false><<<grid, block, 0, stream>>>(deviceMu, deviceCold, deviceNew, nx, ny, nz);
        std::swap(deviceCold, deviceNew);
    };

    // An even batch restores the ping-pong buffer addresses, allowing graph
    // replay without pointer updates or any host/device transfers per step.
    constexpr int batch = 16;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (iterations >= batch) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int t = 0; t < batch; ++t) step();
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(executable, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    int t = 0;
    for (; iterations - t >= batch; t += batch)
        cudaCheck(cudaGraphLaunch(executable, stream));
    for (; t < iterations; ++t) step();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Only requested output/validation needs a host copy of the final field.
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        cudaCheck(cudaMemcpy(cold.data(), deviceCold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executable) cudaCheck(cudaGraphExecDestroy(executable));
    if (graph) cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(deviceMu));
    cudaCheck(cudaFree(deviceNew));
    cudaCheck(cudaFree(deviceCold));

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
