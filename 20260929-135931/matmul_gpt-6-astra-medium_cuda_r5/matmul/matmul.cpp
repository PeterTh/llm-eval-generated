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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceMatrix {
    double* data = nullptr;
    explicit DeviceMatrix(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes), "Allocating GPU matrix");
    }
    ~DeviceMatrix() { cudaFree(data); }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
};

// A block computes a square output tile. Each warp reads contiguous columns,
// and each thread retains several independent dot products in registers.
template<int Tile, bool FullTiles>
__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    constexpr int Depth = 16;
    constexpr int Rows = Tile / 8;
    constexpr int Cols = Tile / 32;
    __shared__ double aTile[Tile][Depth + 1];
    __shared__ double bTile[Depth][Tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tid = y * 32 + x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * Tile;
    double sums[Rows][Cols] = {};

    for (size_t base = 0; base < N; base += Depth) {
        #pragma unroll
        for (int p = tid; p < Tile * Depth; p += 256) {
            const int ar = p / Depth;
            const int ak = p % Depth;
            const int bk = p / Tile;
            const int bc = p % Tile;
            aTile[ar][ak] = (FullTiles || (rowBase + ar < N && base + ak < N))
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            bTile[bk][bc] = (FullTiles || (base + bk < N && colBase + bc < N))
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Depth; ++k) {
            double a[Rows];
            double b[Cols];
            #pragma unroll
            for (int r = 0; r < Rows; ++r) a[r] = aTile[y + r * 8][k];
            #pragma unroll
            for (int c = 0; c < Cols; ++c) b[c] = bTile[k][x + c * 32];
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                #pragma unroll
                for (int c = 0; c < Cols; ++c) sums[r][c] += a[r] * b[c];
            }
        }
        // All consumers must finish before the next tile overwrites shared memory.
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int c = 0; c < Cols; ++c) {
            const size_t row = rowBase + y + r * 8;
            const size_t col = colBase + x + c * 32;
            if (FullTiles || (row < N && col < N)) C[row * N + col] = sums[r][c];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    DeviceMatrix a(bytes), b(bytes), c(bytes);
    checkCuda(cudaMemcpy(a.data, A.data(), bytes, cudaMemcpyHostToDevice), "Copying A to GPU");
    checkCuda(cudaMemcpy(b.data, B.data(), bytes, cudaMemcpyHostToDevice), "Copying B to GPU");
    const dim3 threads(32, 8);
    // Smaller tiles expose more parallel blocks for small matrices.
    if (N <= 256) {
        const dim3 blocks((N + 31) / 32, (N + 31) / 32);
        if (N % 32 == 0)
            multiplyKernel<32, true><<<blocks, threads>>>(a.data, b.data, c.data, N);
        else
            multiplyKernel<32, false><<<blocks, threads>>>(a.data, b.data, c.data, N);
    } else {
        const dim3 blocks((N + 63) / 64, (N + 63) / 64);
        if (N % 64 == 0)
            multiplyKernel<64, true><<<blocks, threads>>>(a.data, b.data, c.data, N);
        else
            multiplyKernel<64, false><<<blocks, threads>>>(a.data, b.data, c.data, N);
    }
    checkCuda(cudaGetLastError(), "Launching matrix multiplication");
    // The blocking copy also waits for the kernel, so timing includes GPU work
    // and both transfer directions and C is ready for all existing callers.
    checkCuda(cudaMemcpy(C.data(), c.data, bytes, cudaMemcpyDeviceToHost), "Copying C from GPU");
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
    
    if (N == 0 || N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        fprintf(stderr, "Matrix size must be positive and fit in memory.\n");
        return 1;
    }
    // Initialize the CUDA context before timing the actual operation.
    checkCuda(cudaFree(nullptr), "Initializing CUDA");
    cudaDeviceProp device{};
    int deviceId = 0;
    checkCuda(cudaGetDevice(&deviceId), "Getting CUDA device");
    checkCuda(cudaGetDeviceProperties(&device, deviceId), "Getting CUDA device limits");
    if ((N + 63) / 64 > static_cast<size_t>(device.maxGridSize[1])) {
        throw std::runtime_error("Matrix size exceeds CUDA grid limits");
    }

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
