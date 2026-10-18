#include <chrono>
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

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceMatrices {
public:
    double* data = nullptr;

    explicit DeviceMatrices(size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "Allocate GPU matrices");
    }
    ~DeviceMatrices() { cudaFree(data); }
    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
};

// A block computes a 64x64 output tile. Each thread accumulates a 4x4
// register tile, amortizing shared-memory loads over sixteen products.
constexpr unsigned Tile = 64;
constexpr unsigned Depth = 16;

__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    // Padding prevents the two half-warps' A broadcasts from sharing banks.
    __shared__ double tileA[Tile][Depth + 1];
    __shared__ double tileB[Depth][Tile];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * 16 + tx;
    const size_t row = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t col = static_cast<size_t>(blockIdx.x) * Tile;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += Depth) {
        // Linear cooperative loads give coalesced global-memory accesses.
        #pragma unroll
        for (unsigned load = 0; load < Tile * Depth / 256; ++load) {
            const unsigned index = tid + load * 256;
            const unsigned ar = index / Depth;
            const unsigned ak = index % Depth;
            const unsigned bk = index / Tile;
            const unsigned bc = index % Tile;
            tileA[ar][ak] = (row + ar < N && base + ak < N)
                ? A[(row + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (base + bk < N && col + bc < N)
                ? B[(base + bk) * N + col + bc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < Depth; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                a[r] = tileA[ty + r * 16][k];
                b[r] = tileB[k][tx + r * 16];
            }
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                #pragma unroll
                for (unsigned c = 0; c < 4; ++c) {
                    sums[r][c] = fma(a[r], b[c], sums[r][c]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            const size_t i = row + ty + r * 16;
            const size_t j = col + tx + c * 16;
            if (i < N && j < N) C[i * N + j] = sums[r][c];
        }
    }
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    DeviceMatrices storage(3 * bytes);
    double* deviceA = storage.data;
    double* deviceB = deviceA + elements;
    double* deviceC = deviceB + elements;
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix A to GPU");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "Copy matrix B to GPU");

    const dim3 threads(16, 16);
    const dim3 blocks(static_cast<unsigned>((N + Tile - 1) / Tile),
                      static_cast<unsigned>((N + Tile - 1) / Tile));
    multiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "Launch matrix multiplication");
    // The synchronous copy waits for the kernel and reports execution errors;
    // the benchmark's timer therefore includes completed GPU work and transfers.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "Copy result from GPU");
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
            N = atoi(argv[++i]);
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
    
    if (N == 0 || N > std::numeric_limits<size_t>::max() / N / (3 * sizeof(double))) {
        fprintf(stderr, "Matrix size must be positive and fit in addressable memory.\n");
        return 1;
    }

    // Initialize CUDA before timing, and fail explicitly if no GPU is available.
    checkCuda(cudaSetDevice(0), "Select CUDA device");
    checkCuda(cudaFree(nullptr), "Initialize CUDA runtime");

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
