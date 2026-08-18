#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE = 32;
constexpr int THREAD_TILE = 16;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each 16x16 thread block computes a 32x32 output tile. Every thread owns a
// 2x2 register tile, while A and B are loaded once into padded shared memory.
// Padding removes the 32-way bank conflict that otherwise occurs while B is
// consumed by columns.
__global__ __launch_bounds__(THREAD_TILE * THREAD_TILE)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C, int N) {
    __shared__ double tileA[TILE][TILE + 1];
    __shared__ double tileB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = static_cast<int>(blockIdx.y) * TILE + ty;
    const int row1 = row0 + THREAD_TILE;
    const int col0 = static_cast<int>(blockIdx.x) * TILE + tx;
    const int col1 = col0 + THREAD_TILE;

    double c00 = 0.0;
    double c01 = 0.0;
    double c10 = 0.0;
    double c11 = 0.0;

    for (int base = 0; base < N; base += TILE) {
        const int k0 = base + tx;
        const int k1 = k0 + THREAD_TILE;
        const int bk0 = base + ty;
        const int bk1 = bk0 + THREAD_TILE;

        tileA[ty][tx] = (row0 < N && k0 < N) ? A[row0 * N + k0] : 0.0;
        tileA[ty][tx + THREAD_TILE] =
            (row0 < N && k1 < N) ? A[row0 * N + k1] : 0.0;
        tileA[ty + THREAD_TILE][tx] =
            (row1 < N && k0 < N) ? A[row1 * N + k0] : 0.0;
        tileA[ty + THREAD_TILE][tx + THREAD_TILE] =
            (row1 < N && k1 < N) ? A[row1 * N + k1] : 0.0;

        tileB[ty][tx] = (bk0 < N && col0 < N) ? B[bk0 * N + col0] : 0.0;
        tileB[ty][tx + THREAD_TILE] =
            (bk0 < N && col1 < N) ? B[bk0 * N + col1] : 0.0;
        tileB[ty + THREAD_TILE][tx] =
            (bk1 < N && col0 < N) ? B[bk1 * N + col0] : 0.0;
        tileB[ty + THREAD_TILE][tx + THREAD_TILE] =
            (bk1 < N && col1 < N) ? B[bk1 * N + col1] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double a0 = tileA[ty][k];
            const double a1 = tileA[ty + THREAD_TILE][k];
            const double b0 = tileB[k][tx];
            const double b1 = tileB[k][tx + THREAD_TILE];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
        }
        __syncthreads();
    }

    if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
}

class DeviceMatrices {
public:
    explicit DeviceMatrices(size_t elements) : bytes_(elements * sizeof(double)) {
        checkCuda(cudaMalloc(&a_, bytes_), "allocating matrix A");
        checkCuda(cudaMalloc(&b_, bytes_), "allocating matrix B");
        checkCuda(cudaMalloc(&c_, bytes_), "allocating matrix C");
    }

    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;

    ~DeviceMatrices() {
        cudaFree(c_);
        cudaFree(b_);
        cudaFree(a_);
    }

    void upload(const std::vector<double>& A, const std::vector<double>& B) {
        checkCuda(cudaMemcpy(a_, A.data(), bytes_, cudaMemcpyHostToDevice),
                  "copying matrix A to the GPU");
        checkCuda(cudaMemcpy(b_, B.data(), bytes_, cudaMemcpyHostToDevice),
                  "copying matrix B to the GPU");
    }

    float multiply(int N) {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        checkCuda(cudaEventCreate(&start), "creating start event");
        checkCuda(cudaEventCreate(&stop), "creating stop event");

        const dim3 threads(THREAD_TILE, THREAD_TILE);
        const dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        checkCuda(cudaEventRecord(start), "recording start event");
        matrixMultiplyKernel<<<blocks, threads>>>(a_, b_, c_, N);
        checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
        checkCuda(cudaEventRecord(stop), "recording stop event");
        checkCuda(cudaEventSynchronize(stop), "waiting for matrix multiplication");

        float elapsedMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "timing kernel");
        checkCuda(cudaEventDestroy(stop), "destroying stop event");
        checkCuda(cudaEventDestroy(start), "destroying start event");
        return elapsedMs;
    }

    void download(std::vector<double>& C) const {
        checkCuda(cudaMemcpy(C.data(), c_, bytes_, cudaMemcpyDeviceToHost),
                  "copying matrix C from the GPU");
    }

private:
    double* a_ = nullptr;
    double* b_ = nullptr;
    double* c_ = nullptr;
    size_t bytes_;
};

}  // namespace

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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    if (N == 0 || N > static_cast<size_t>(INT_MAX) ||
        N > std::numeric_limits<size_t>::max() / N) {
        std::fprintf(stderr, "Matrix size must be between 1 and %d.\n", INT_MAX);
        return 1;
    }
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Device allocation and PCIe transfers are setup, not multiplication, so
    // keep them outside the benchmark interval just as host allocation and
    // initialization were outside it in the original CPU benchmark.
    DeviceMatrices device(N * N);
    device.upload(A, B);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    const float elapsedMs = device.multiply(static_cast<int>(N));
    device.download(C);

    printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    
    // Calculate GFLOPS
    double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                     static_cast<double>(N)) / (elapsedMs * 1.0e6);
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
