#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceMatrices {
public:
    double* data = nullptr;

    explicit DeviceMatrices(size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "Allocating GPU matrices");
    }
    ~DeviceMatrices() { cudaFree(data); }
    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
};

// A 16x16 thread block computes a 64x64 output tile. Each thread keeps
// a 4x4 microtile in registers, reusing each shared-memory load four times.
// Padding the transposed A tile avoids shared-memory bank conflicts.
template <bool FullTiles>
__global__ void multiplyTiled(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N) {
    constexpr int Tile = 64;
    constexpr int Depth = 16;
    __shared__ double tileA[Depth][Tile + 1];
    __shared__ double tileB[Depth][Tile];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * Tile;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += Depth) {
        // Both input tiles are read in contiguous, coalesced row segments.
        #pragma unroll
        for (int index = thread; index < Tile * Depth; index += 256) {
            const int ar = index / Depth;
            const int ak = index % Depth;
            const int bk = index / Tile;
            const int bc = index % Tile;
            tileA[ak][ar] = (FullTiles || (rowBase + ar < N && base + ak < N))
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (FullTiles || (base + bk < N && colBase + bc < N))
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < Depth; ++k) {
            // Skip padding rather than introducing extra arithmetic in the
            // final partial tile. Reduction order matches the serial loop.
            if (FullTiles || base + k < N) {
                double a[4], b[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) a[r] = tileA[k][ty + r * 16];
                #pragma unroll
                for (int c = 0; c < 4; ++c) b[c] = tileB[k][tx + c * 16];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        sums[r][c] += a[r] * b[c];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t row = rowBase + ty + r * 16;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t col = colBase + tx + c * 16;
            if (FullTiles || (row < N && col < N)) C[row * N + col] = sums[r][c];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    DeviceMatrices storage(3 * bytes);
    double* deviceA = storage.data;
    double* deviceB = deviceA + elements;
    double* deviceC = deviceB + elements;
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "Copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "Copying matrix B");

    const dim3 threads(16, 16);
    const dim3 blocks((N + 63) / 64, (N + 63) / 64);
    if (N % 64 == 0) {
        multiplyTiled<true><<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    } else {
        multiplyTiled<false><<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    }
    checkCuda(cudaGetLastError(), "Launching matrix multiplication");
    // The blocking copy also waits for the kernel, so benchmark timing includes
    // completed computation and both directions of host/device transfers.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "Reading matrix C");
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) try {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Matrix size must be a positive integer");
            }
            N = static_cast<size_t>(parsed);
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
    
    if (N > std::numeric_limits<size_t>::max() / N / (3 * sizeof(double))) {
        throw std::runtime_error("Matrix size is too large");
    }

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize the CUDA context before timing, just as host matrix setup is
    // outside the multiplication benchmark. There is no CPU fallback.
    checkCuda(cudaSetDevice(0), "Selecting CUDA device");
    checkCuda(cudaFree(nullptr), "Initializing CUDA context");
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / std::chrono::duration<double>(end - start).count() / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
}
