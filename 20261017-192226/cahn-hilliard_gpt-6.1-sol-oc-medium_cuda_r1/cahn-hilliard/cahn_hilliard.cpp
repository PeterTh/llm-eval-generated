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

static void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// One warp spans consecutive X coordinates. Each thread advances through four
// Z planes, retaining the Z neighbors in registers between stencil evaluations.
// The two instantiations are separate launches: all chemical potentials must be
// available before the concentration update reads its neighboring values.
template <bool ChemicalPotential>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    // Fixed benchmark parameters are compile-time constants so unit-spacing
    // divisions disappear without using approximate division or fast math.
    constexpr double dx = 1.0, dy = 1.0, dz = 1.0;
    constexpr double dt = 0.01, D = 1.0, gamma = 0.5;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t ybase = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || ybase >= ny) return;
    const size_t plane = nx * ny;
    // Grid strides also support dimensions beyond CUDA's Y/Z grid limits.
    for (size_t y = ybase; y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        for (size_t zbase = static_cast<size_t>(blockIdx.z) * 4;
             zbase < nz; zbase += static_cast<size_t>(gridDim.z) * 4) {
            size_t i = zbase * plane + y * nx + x;
            double center = field[i];
            double previous = field[zbase ? i - plane : i];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const size_t z = zbase + k;
                if (z >= nz) break;
                const double next = field[z + 1 < nz ? i + plane : i];
                const double cxx = (field[x + 1 < nx ? i + 1 : i] +
                                    field[x ? i - 1 : i] - 2.0 * center) / (dx * dx);
                const double cyy = (field[y + 1 < ny ? i + nx : i] +
                                    field[y ? i - nx : i] - 2.0 * center) / (dy * dy);
                const double czz = (next + previous - 2.0 * center) / (dz * dz);
                const double laplacian = cxx + cyy + czz;
                if constexpr (ChemicalPotential) {
                    output[i] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB -
                                       2.0 * center * e_AB) + 3.0 * center +
                                center * center * center - gamma * laplacian;
                } else {
                    output[i] = cold[i] + dt * D * laplacian;
                }
                previous = center;
                center = next;
                i += plane;
            }
        }
    }
}

// Initialize concentration field
__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
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
        nx * ny * nz > std::numeric_limits<size_t>::max() / (3 * sizeof(double))) {
        fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double* storage = nullptr;
    checkCuda(cudaMalloc(&storage, 3 * bytes), "allocating fields");
    double* deviceCold = storage;
    double* deviceNew = storage + gridSize;
    double* mu = storage + 2 * gridSize;
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating stream");

    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>((nx + 31) / 32),
                    static_cast<unsigned>(std::min<size_t>((ny + 3) / 4, 65535)),
                    static_cast<unsigned>(std::min<size_t>((nz + 3) / 4, 65535)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration<<<static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535)),
                              256, 0, stream>>>(deviceCold, gridSize);
    checkCuda(cudaGetLastError(), "initializing concentration");

    auto step = [&]() {
        stencil<true><<<grid, block, 0, stream>>>(deviceCold, nullptr, mu,
            nx, ny, nz);
        stencil<false><<<grid, block, 0, stream>>>(mu, deviceCold, deviceNew,
            nx, ny, nz);
        std::swap(deviceCold, deviceNew);
    };

    // An even batch restores the ping-pong pointer parity, so it can be replayed
    // without graph updates or CPU synchronization between individual steps.
    const int batch = std::min(std::max(iterations, 0), 32) & ~1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (batch != 0) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "starting capture");
        for (int t = 0; t < batch; ++t) step();
        checkCuda(cudaStreamEndCapture(stream, &graph), "ending capture");
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiating graph");
        checkCuda(cudaGraphUpload(executable, stream), "uploading graph");
    }
    checkCuda(cudaStreamSynchronize(stream), "preparing simulation");
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int t = 0;
    if (batch != 0) {
        for (; t <= iterations - batch; t += batch)
            checkCuda(cudaGraphLaunch(executable, stream), "launching simulation graph");
    }
    for (; t < iterations; ++t) step();
    checkCuda(cudaGetLastError(), "launching stencils");
    checkCuda(cudaStreamSynchronize(stream), "running simulation");
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfer only the final field, and only when host-side output needs it.
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), deviceCold, bytes, cudaMemcpyDeviceToHost), "copying result");
    }
    if (executable) checkCuda(cudaGraphExecDestroy(executable), "destroying executable graph");
    if (graph) checkCuda(cudaGraphDestroy(graph), "destroying graph");
    checkCuda(cudaStreamDestroy(stream), "destroying stream");
    checkCuda(cudaFree(storage), "freeing fields");
    
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
