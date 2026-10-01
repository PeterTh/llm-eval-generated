#include <cuda_runtime.h>

#include <cerrno>
#include <limits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A block computes 64 x 64 outputs. Each thread keeps a 4 x 4
// microtile in registers, reusing both shared-memory operands.
constexpr unsigned Tile = 64;
constexpr unsigned Depth = 16;

__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N,
                               size_t tiles) {
    __shared__ double aTile[Tile][Depth];
    __shared__ double bTile[Depth][Tile];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * 16 + tx;

    // A flat, strided grid also supports matrices beyond the grid-y limit.
    for (size_t tile = blockIdx.x; tile < tiles * tiles; tile += gridDim.x) {
        const size_t row = (tile / tiles) * Tile;
        const size_t col = (tile % tiles) * Tile;
        double sums[4][4] = {};
        for (size_t base = 0; base < N; base += Depth) {
            #pragma unroll
            for (unsigned q = tid; q < Tile * Depth; q += 256) {
                const unsigned ar = q / Depth, ak = q % Depth;
                const unsigned bk = q / Tile, bc = q % Tile;
                aTile[ar][ak] = (row + ar < N && base + ak < N)
                    ? A[(row + ar) * N + base + ak] : 0.0;
                bTile[bk][bc] = (base + bk < N && col + bc < N)
                    ? B[(base + bk) * N + col + bc] : 0.0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < Depth; ++k) {
                double av[4], bv[4];
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) av[r] = aTile[ty + r * 16][k];
                #pragma unroll
                for (unsigned c = 0; c < 4; ++c) bv[c] = bTile[k][tx + c * 16];
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) {
                    #pragma unroll
                    for (unsigned c = 0; c < 4; ++c) {
                        // Match the original separate multiply/add rounding.
                        sums[r][c] += av[r] * bv[c];
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
                if (row + ty + r * 16 < N && col + tx + c * 16 < N)
                    C[(row + ty + r * 16) * N + col + tx + c * 16] = sums[r][c];
            }
        }
    }
}
} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocate A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocate B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocate C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "upload A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "upload B");
    const size_t tiles = (N + Tile - 1) / Tile;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(tiles * tiles, 65535));
    multiplyKernel<<<blocks, dim3(16, 16)>>>(deviceA, deviceB, deviceC, N, tiles);
    checkCuda(cudaGetLastError(), "launch multiplication");
    // This blocking copy includes completion of all GPU work in the timing.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "download C");
    checkCuda(cudaFree(deviceC), "free C");
    checkCuda(cudaFree(deviceB), "free B");
    checkCuda(cudaFree(deviceA), "free A");
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

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value == end || *end || *value == '-' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Matrix size must be a positive integer.\n");
                return 1;
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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }
    // Initialize the CUDA context before measuring the multiplication.
    checkCuda(cudaFree(nullptr), "initialize device");

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
}
