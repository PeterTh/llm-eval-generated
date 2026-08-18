#include <cmath>
#include <cuda_runtime.h>
#include <limits>
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

constexpr int TILE_SIZE = 32;
constexpr int THREAD_TILE = 4;
constexpr int THREADS_PER_DIM = TILE_SIZE / THREAD_TILE;

void checkCuda(const cudaError_t status, const char* operation, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while %s: %s\n", file, line, operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

class DeviceMatrix {
public:
    explicit DeviceMatrix(const size_t bytes) {
        CUDA_CHECK(cudaMalloc(&data_, bytes));
    }

    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    ~DeviceMatrix() {
        if (data_ != nullptr) {
            // Destructors must not terminate during stack unwinding. All CUDA operations that
            // can report an error are checked at their call sites.
            cudaFree(data_);
        }
    }

    [[nodiscard]] double* get() const noexcept { return data_; }

private:
    double* data_ = nullptr;
};

// Each 8x8 thread block produces a 32x32 output tile.  A thread owns a 4x4
// register tile, which raises arithmetic intensity while the padded shared
// tiles avoid shared-memory bank conflicts for both operand access patterns.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadLinear = ty * THREADS_PER_DIM + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE_SIZE;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    const size_t row = rowBase + static_cast<size_t>(ty * THREAD_TILE);
    const size_t col = colBase + static_cast<size_t>(tx * THREAD_TILE);

    double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
    double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
    double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
    double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

    for (size_t kBase = 0; kBase < N; kBase += TILE_SIZE) {
        for (int index = threadLinear; index < TILE_SIZE * TILE_SIZE;
             index += THREADS_PER_DIM * THREADS_PER_DIM) {
            const int tileRow = index / TILE_SIZE;
            const int tileCol = index % TILE_SIZE;
            const size_t aRow = rowBase + static_cast<size_t>(tileRow);
            const size_t aCol = kBase + static_cast<size_t>(tileCol);
            const size_t bRow = kBase + static_cast<size_t>(tileRow);
            const size_t bCol = colBase + static_cast<size_t>(tileCol);

            tileA[tileRow][tileCol] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0;
            tileB[tileRow][tileCol] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            const double a0 = tileA[ty * THREAD_TILE][k];
            const double a1 = tileA[ty * THREAD_TILE + 1][k];
            const double a2 = tileA[ty * THREAD_TILE + 2][k];
            const double a3 = tileA[ty * THREAD_TILE + 3][k];
            const double b0 = tileB[k][tx * THREAD_TILE];
            const double b1 = tileB[k][tx * THREAD_TILE + 1];
            const double b2 = tileB[k][tx * THREAD_TILE + 2];
            const double b3 = tileB[k][tx * THREAD_TILE + 3];

            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }
        __syncthreads();
    }

    if (row + 3 < N && col + 3 < N) {
        C[row * N + col] = c00;           C[row * N + col + 1] = c01;
        C[row * N + col + 2] = c02;       C[row * N + col + 3] = c03;
        C[(row + 1) * N + col] = c10;     C[(row + 1) * N + col + 1] = c11;
        C[(row + 1) * N + col + 2] = c12; C[(row + 1) * N + col + 3] = c13;
        C[(row + 2) * N + col] = c20;     C[(row + 2) * N + col + 1] = c21;
        C[(row + 2) * N + col + 2] = c22; C[(row + 2) * N + col + 3] = c23;
        C[(row + 3) * N + col] = c30;     C[(row + 3) * N + col + 1] = c31;
        C[(row + 3) * N + col + 2] = c32; C[(row + 3) * N + col + 3] = c33;
    } else {
        if (row < N && col < N) C[row * N + col] = c00;
        if (row < N && col + 1 < N) C[row * N + col + 1] = c01;
        if (row < N && col + 2 < N) C[row * N + col + 2] = c02;
        if (row < N && col + 3 < N) C[row * N + col + 3] = c03;
        if (row + 1 < N && col < N) C[(row + 1) * N + col] = c10;
        if (row + 1 < N && col + 1 < N) C[(row + 1) * N + col + 1] = c11;
        if (row + 1 < N && col + 2 < N) C[(row + 1) * N + col + 2] = c12;
        if (row + 1 < N && col + 3 < N) C[(row + 1) * N + col + 3] = c13;
        if (row + 2 < N && col < N) C[(row + 2) * N + col] = c20;
        if (row + 2 < N && col + 1 < N) C[(row + 2) * N + col + 1] = c21;
        if (row + 2 < N && col + 2 < N) C[(row + 2) * N + col + 2] = c22;
        if (row + 2 < N && col + 3 < N) C[(row + 2) * N + col + 3] = c23;
        if (row + 3 < N && col < N) C[(row + 3) * N + col] = c30;
        if (row + 3 < N && col + 1 < N) C[(row + 3) * N + col + 1] = c31;
        if (row + 3 < N && col + 2 < N) C[(row + 3) * N + col + 2] = c32;
        if (row + 3 < N && col + 3 < N) C[(row + 3) * N + col + 3] = c33;
    }
}

float matrixMultiplyCuda(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    const size_t gridSize = (N + TILE_SIZE - 1) / TILE_SIZE;
    if (gridSize > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Matrix is too large for CUDA grid dimensions\n");
        std::exit(EXIT_FAILURE);
    }

    DeviceMatrix deviceA(bytes);
    DeviceMatrix deviceB(bytes);
    DeviceMatrix deviceC(bytes);
    CUDA_CHECK(cudaMemcpy(deviceA.get(), A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB.get(), B.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const dim3 block(THREADS_PER_DIM, THREADS_PER_DIM);
    const dim3 grid(static_cast<unsigned int>(gridSize), static_cast<unsigned int>(gridSize));
    matrixMultiplyKernel<<<grid, block>>>(deviceA.get(), deviceB.get(), deviceC.get(), N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(C.data(), deviceC.get(), bytes, cudaMemcpyDeviceToHost));
    return elapsedMilliseconds;
}

}  // namespace

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
    const float elapsedMilliseconds = matrixMultiplyCuda(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    const double gflops = (2.0 * N * N * N) / (elapsedMilliseconds * 1.0e-3) / 1e9;
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
