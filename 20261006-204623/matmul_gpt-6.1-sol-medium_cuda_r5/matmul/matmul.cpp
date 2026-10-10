#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block reuses two shared-memory tiles, and each thread accumulates
// a small output tile in registers. All dot products retain increasing k order.
template <int Tile, int Outputs, bool FullTiles>
__global__ void multiplyTiles(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N) {
    constexpr int Depth = 16;
    constexpr int Threads = Tile / Outputs;
    __shared__ double a[Tile][Depth + 1];
    __shared__ double b[Depth][Tile];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * Threads + tx;
    const size_t row = size_t(blockIdx.y) * Tile;
    const size_t col = size_t(blockIdx.x) * Tile;
    double sums[Outputs][Outputs] = {};

    for (size_t base = 0; base < N; base += Depth) {
        for (int index = tid; index < Tile * Depth; index += Threads * Threads) {
            const int ar = index / Depth;
            const int ak = index % Depth;
            const int bk = index / Tile;
            const int bc = index % Tile;
            a[ar][ak] = (FullTiles || (row + ar < N && base + ak < N))
                ? A[(row + ar) * N + base + ak] : 0.0;
            b[bk][bc] = (FullTiles || (base + bk < N && col + bc < N))
                ? B[(base + bk) * N + col + bc] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Depth; ++k) {
            if (FullTiles || base + k < N) {
                double av[Outputs], bv[Outputs];
                #pragma unroll
                for (int i = 0; i < Outputs; ++i) {
                    av[i] = a[ty + i * Threads][k];
                    bv[i] = b[k][tx + i * Threads];
                }
                #pragma unroll
                for (int i = 0; i < Outputs; ++i) {
                    #pragma unroll
                    for (int j = 0; j < Outputs; ++j) {
                        sums[i][j] += av[i] * bv[j];
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < Outputs; ++i) {
        #pragma unroll
        for (int j = 0; j < Outputs; ++j) {
            const size_t r = row + ty + i * Threads;
            const size_t c = col + tx + j * Threads;
            if (FullTiles || (r < N && c < N)) C[r * N + c] = sums[i][j];
        }
    }
}

class DeviceMatrices {
    double *a = nullptr, *b = nullptr, *c = nullptr;
    size_t bytes;
public:
    explicit DeviceMatrices(size_t N) : bytes(N * N * sizeof(double)) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&a), bytes));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&b), bytes));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&c), bytes));
    }
    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
    ~DeviceMatrices() {
        cudaFree(c);
        cudaFree(b);
        cudaFree(a);
    }
    void multiply(const std::vector<double>& A, const std::vector<double>& B,
                  std::vector<double>& C, size_t N) {
        checkCuda(cudaMemcpy(a, A.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(b, B.data(), bytes, cudaMemcpyHostToDevice));
        const dim3 threads(16, 16);
        if (N < 256) {
            const dim3 blocks((N + 31) / 32, (N + 31) / 32);
            if (N % 32 == 0)
                multiplyTiles<32, 2, true><<<blocks, threads>>>(a, b, c, N);
            else
                multiplyTiles<32, 2, false><<<blocks, threads>>>(a, b, c, N);
        } else {
            const dim3 blocks((N + 63) / 64, (N + 63) / 64);
            if (N % 64 == 0)
                multiplyTiles<64, 4, true><<<blocks, threads>>>(a, b, c, N);
            else
                multiplyTiles<64, 4, false><<<blocks, threads>>>(a, b, c, N);
        }
        checkCuda(cudaGetLastError());
        // The synchronous copy also waits for kernel completion and reports
        // execution errors, so the benchmark includes transfers and computation.
        checkCuda(cudaMemcpy(C.data(), c, bytes, cudaMemcpyDeviceToHost));
    }
};

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
            
            if (!std::isfinite(actual) || relError > 1e-6) {
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
    
    if (N == 0 || N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        fprintf(stderr, "Invalid matrix size\n");
        return 1;
    }
    // Set up CUDA outside the timed section. Multiplication always runs on GPU.
    DeviceMatrices device(N);

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
    
    device.multiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double seconds = std::chrono::duration<double>(end - start).count();
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
