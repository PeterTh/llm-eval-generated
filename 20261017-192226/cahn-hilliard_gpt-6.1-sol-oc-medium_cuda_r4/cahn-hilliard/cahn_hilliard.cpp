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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % volume) /
                              static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Each warp follows X, so all stencil loads are coalesced. Process four Z
// planes per thread, retaining the preceding and current values in registers.
// Separate launches provide the global barrier between the two stencil passes.
template<bool Update>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t base = size_t(blockIdx.z) * 4; base < nz;
             base += size_t(gridDim.z) * 4) {
            size_t i = base * plane + y * nx + x;
            double center = field[i];
            double previous = field[base == 0 ? i : i - plane];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const size_t z = base + k;
                if (z >= nz) break;
                const double next = field[z + 1 < nz ? i + plane : i];
                // dx = dy = dz = 1, as in the original benchmark.
                const double cxx = field[x + 1 < nx ? i + 1 : i] +
                                   field[x > 0 ? i - 1 : i] - 2.0 * center;
                const double cyy = field[y + 1 < ny ? i + nx : i] +
                                   field[y > 0 ? i - nx : i] - 2.0 * center;
                const double czz = next + previous - 2.0 * center;
                const double laplacian = (cxx + cyy) + czz;
                if constexpr (Update) {
                    output[i] = cold[i] + 0.01 * laplacian;
                } else {
                    constexpr double e_AA = -(2.0 / 9.0);
                    constexpr double e_BB = -(2.0 / 9.0);
                    constexpr double e_AB = 2.0 / 9.0;
                    output[i] = 4.5 * ((center + 1.0) * e_AA +
                                (center - 1.0) * e_BB - 2.0 * center * e_AB) +
                                3.0 * center + center * center * center -
                                0.5 * laplacian;
                }
                previous = center;
                center = next;
                i += plane;
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
    double *cold, *cnew, *mu;
    checkCuda(cudaMalloc(&cold, bytes), "allocate concentration");
    checkCuda(cudaMalloc(&cnew, bytes), "allocate next concentration");
    checkCuda(cudaMalloc(&mu, bytes), "allocate chemical potential");
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const dim3 block(32, 4);
    const dim3 grid((nx + 31) / 32,
                    std::min<size_t>((ny + 3) / 4, 65535),
                    std::min<size_t>((nz + 3) / 4, 65535));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration<<<std::min<size_t>((gridSize + 255) / 256, 65535),
                              256, 0, stream>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "initialize concentration");
    checkCuda(cudaStreamSynchronize(stream), "finish initialization");

    auto step = [&]() {
        stencil<false><<<grid, block, 0, stream>>>(cold, nullptr, mu, nx, ny, nz);
        stencil<true><<<grid, block, 0, stream>>>(mu, cold, cnew, nx, ny, nz);
        std::swap(cold, cnew);
    };

    // An even batch returns the ping-pong pointers to their initial order, so
    // the same graph can be replayed without any parameter updates or copies.
    const int batch = std::min(iterations, 32) / 2 * 2;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (batch > 0) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph");
        for (int t = 0; t < batch; ++t) step();
        checkCuda(cudaStreamEndCapture(stream, &graph), "end graph");
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
        checkCuda(cudaGraphUpload(executable, stream), "upload graph");
        checkCuda(cudaStreamSynchronize(stream), "finish graph upload");
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int t = 0;
    if (batch > 0)
        for (; t <= iterations - batch; t += batch)
            checkCuda(cudaGraphLaunch(executable, stream), "launch time steps");
    for (; t < iterations; ++t) step();
    checkCuda(cudaGetLastError(), "launch stencil");
    checkCuda(cudaStreamSynchronize(stream), "finish simulation");
    
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
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copy result");
    }
    if (executable) checkCuda(cudaGraphExecDestroy(executable), "destroy executable graph");
    if (graph) checkCuda(cudaGraphDestroy(graph), "destroy graph");
    checkCuda(cudaFree(cold), "free concentration");
    checkCuda(cudaFree(cnew), "free next concentration");
    checkCuda(cudaFree(mu), "free chemical potential");
    checkCuda(cudaStreamDestroy(stream), "destroy stream");
    
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
