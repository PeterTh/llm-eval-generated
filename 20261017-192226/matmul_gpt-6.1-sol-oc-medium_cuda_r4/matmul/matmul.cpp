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

// A block computes a 64x64 output tile. Each thread keeps 4x4 outputs in
// registers; adjacent threads load/store adjacent columns. Reuse each input
// tile across all 256 threads before fetching the next 16 inner products.
template <bool FullTiles>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    __shared__ double tileA[64][16];
    __shared__ double tileB[16][64];
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * 64;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * 64;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += 16) {
#pragma unroll
        for (unsigned int offset = 0; offset < 1024; offset += 256) {
            const unsigned int index = tid + offset;
            const unsigned int ar = index / 16;
            const unsigned int ak = index % 16;
            const unsigned int bk = index / 64;
            const unsigned int bc = index % 64;
            tileA[ar][ak] = (FullTiles || (rowBase + ar < N && base + ak < N))
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (FullTiles || (base + bk < N && colBase + bc < N))
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int k = 0; k < 16; ++k) {
            double a[4], b[4];
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
                a[i] = tileA[ty + i * 16][k];
                b[i] = tileB[k][tx + i * 16];
            }
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    sums[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const size_t row = rowBase + ty + i * 16;
            const size_t col = colBase + tx + j * 16;
            if (FullTiles || (row < N && col < N)) {
                C[row * N + col] = sums[i][j];
            }
        }
    }
}

// Allocate once, outside the measured region. The timed operation includes
// both input transfers, the kernel, and the completed result transfer.
class CudaMatmul {
public:
    explicit CudaMatmul(size_t N) : n(N), bytes(N * N * sizeof(double)) {
        int device = 0;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDevice(&device), "cudaGetDevice");
        checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        const size_t tiles = (N + 63) / 64;
        if (tiles > static_cast<size_t>(properties.maxGridSize[0]) ||
            tiles > static_cast<size_t>(properties.maxGridSize[1])) {
            throw std::runtime_error("Matrix exceeds CUDA grid dimensions");
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), 3 * bytes), "cudaMalloc");
    }

    ~CudaMatmul() { cudaFree(data); }
    CudaMatmul(const CudaMatmul&) = delete;
    CudaMatmul& operator=(const CudaMatmul&) = delete;

    void multiply(const std::vector<double>& A, const std::vector<double>& B,
                  std::vector<double>& C) const {
        double* deviceA = data;
        double* deviceB = data + n * n;
        double* deviceC = data + 2 * n * n;
        checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "Copy A to GPU");
        checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "Copy B to GPU");
        const dim3 block(16, 16);
        const unsigned int tiles = static_cast<unsigned int>((n + 63) / 64);
        const dim3 grid(tiles, tiles);
        if (n % 64 == 0) {
            matrixMultiplyKernel<true><<<grid, block>>>(deviceA, deviceB, deviceC, n);
        } else {
            matrixMultiplyKernel<false><<<grid, block>>>(deviceA, deviceB, deviceC, n);
        }
        checkCuda(cudaGetLastError(), "Launch matrix multiplication");
        // The synchronous copy also waits for the kernel and reports errors.
        checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "Copy C from GPU");
    }

private:
    size_t n;
    size_t bytes;
    double* data = nullptr;
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
            char* end = nullptr;
            const char* argument = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || end == argument || *end != '\0' ||
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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Check all size products, including the three-matrix device allocation.
    if (N > std::numeric_limits<size_t>::max() / N / (3 * sizeof(double))) {
        throw std::runtime_error("Matrix size is too large");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    CudaMatmul gpu(N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.multiply(A, B, C);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double seconds = std::chrono::duration<double>(end - start).count();
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
