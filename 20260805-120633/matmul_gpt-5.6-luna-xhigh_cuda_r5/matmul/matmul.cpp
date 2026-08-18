#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

constexpr unsigned int kTileSize = 32;
constexpr unsigned int kThreadsPerDimension = 16;
constexpr unsigned int kThreadsPerBlock =
    kThreadsPerDimension * kThreadsPerDimension;

void checkCuda(const cudaError_t status, const char* call, const char* file,
               const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", call, file,
                     line, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

// Each block computes one 32x32 tile of C. A 16x16 block gives each thread a
// 2x2 register tile, reducing the instruction and synchronization overhead
// while retaining coalesced global loads and stores.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double tileA[kTileSize][kTileSize];
    __shared__ double tileB[kTileSize][kTileSize];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int linearThread = ty * kThreadsPerDimension + tx;

    const size_t blockRow = static_cast<size_t>(blockIdx.y) * kTileSize;
    const size_t blockColumn = static_cast<size_t>(blockIdx.x) * kTileSize;
    const size_t row0 = blockRow + 2 * ty;
    const size_t row1 = row0 + 1;
    const size_t column0 = blockColumn + 2 * tx;
    const size_t column1 = column0 + 1;

    double c00 = 0.0;
    double c01 = 0.0;
    double c10 = 0.0;
    double c11 = 0.0;

    for (size_t tileColumn = 0; tileColumn < N; tileColumn += kTileSize) {
        // Load four elements of each input tile per thread. The linearized
        // accesses make every warp read contiguous doubles from global memory.
        #pragma unroll
        for (unsigned int load = 0; load < 4; ++load) {
            const unsigned int tileIndex = linearThread + load * kThreadsPerBlock;
            const unsigned int tileRow = tileIndex / kTileSize;
            const unsigned int tileColumnOffset = tileIndex % kTileSize;

            const size_t aRow = blockRow + tileRow;
            const size_t aColumn = tileColumn + tileColumnOffset;
            tileA[tileRow][tileColumnOffset] =
                (aRow < N && aColumn < N) ? A[aRow * N + aColumn] : 0.0;

            const size_t bRow = tileColumn + tileRow;
            const size_t bColumn = blockColumn + tileColumnOffset;
            tileB[tileRow][tileColumnOffset] =
                (bRow < N && bColumn < N) ? B[bRow * N + bColumn] : 0.0;
        }
        __syncthreads();

        // The fixed trip count lets the compiler fully unroll the hot loop.
        // Accumulation remains in increasing k order for each output element.
        #pragma unroll
        for (unsigned int k = 0; k < kTileSize; ++k) {
            const double a0 = tileA[2 * ty][k];
            const double a1 = tileA[2 * ty + 1][k];
            const double b0 = tileB[k][2 * tx];
            const double b1 = tileB[k][2 * tx + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < N && column0 < N) C[row0 * N + column0] = c00;
    if (row0 < N && column1 < N) C[row0 * N + column1] = c01;
    if (row1 < N && column0 < N) C[row1 * N + column0] = c10;
    if (row1 < N && column1 < N) C[row1 * N + column1] = c11;
}

class CudaMatrixMultiplier {
public:
    explicit CudaMatrixMultiplier(const size_t N)
        : N_(N), elements_(N * N), bytes_(elements_ * sizeof(double)) {
        if (N_ != 0 &&
            (N_ > std::numeric_limits<size_t>::max() / N_ ||
             elements_ > std::numeric_limits<size_t>::max() / sizeof(double))) {
            std::fprintf(stderr, "Matrix size is too large\n");
            std::exit(EXIT_FAILURE);
        }

        if (bytes_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA_), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB_), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC_), bytes_));
        }
    }

    CudaMatrixMultiplier(const CudaMatrixMultiplier&) = delete;
    CudaMatrixMultiplier& operator=(const CudaMatrixMultiplier&) = delete;

    ~CudaMatrixMultiplier() {
        if (deviceA_ != nullptr) cudaFree(deviceA_);
        if (deviceB_ != nullptr) cudaFree(deviceB_);
        if (deviceC_ != nullptr) cudaFree(deviceC_);
    }

    void uploadInputs(const std::vector<double>& A,
                      const std::vector<double>& B) const {
        if (bytes_ == 0) return;
        CUDA_CHECK(cudaMemcpy(deviceA_, A.data(), bytes_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceB_, B.data(), bytes_, cudaMemcpyHostToDevice));
    }

    void multiply(std::vector<double>& C) const {
        if (N_ == 0) return;

        const unsigned int gridX = static_cast<unsigned int>(
            N_ / kTileSize + (N_ % kTileSize != 0));
        const unsigned int gridY = gridX;
        const dim3 block(kThreadsPerDimension, kThreadsPerDimension);
        const dim3 grid(gridX, gridY);

        matrixMultiplyKernel<<<grid, block>>>(deviceA_, deviceB_, deviceC_, N_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C.data(), deviceC_, bytes_, cudaMemcpyDeviceToHost));
    }

private:
    size_t N_;
    size_t elements_;
    size_t bytes_;
    double* deviceA_ = nullptr;
    double* deviceB_ = nullptr;
    double* deviceC_ = nullptr;
};

}  // namespace

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return C.empty();

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
    
    // Allocate and populate device buffers before timing the computation.
    CudaMatrixMultiplier multiplier(N);
    multiplier.uploadInputs(A, B);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    multiplier.multiply(C);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
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
