#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Output tile per thread block is BM x BN, computed by a BLOCK_DIM x BLOCK_DIM
// thread block where each thread owns a TM x TN register sub-tile.
constexpr unsigned BLOCK_DIM = 16;
constexpr unsigned TM = 4;
constexpr unsigned TN = 4;
constexpr unsigned BM = BLOCK_DIM * TM;
constexpr unsigned BN = BLOCK_DIM * TN;
constexpr unsigned BK = 16;

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

// Computes C = A * B for a block of `M` rows of A (row-major, M x N) against
// the full B (N x N), producing C (M x N). Each output element is accumulated
// over k = 0..N-1 in ascending order, matching the sequential reference.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t M, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * BLOCK_DIM + tx;
    constexpr unsigned NUM_THREADS = BLOCK_DIM * BLOCK_DIM;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        const size_t kLen = min(static_cast<size_t>(BK), N - k0);

        // Stage A tile (transposed into shared memory) with bounds checks.
        for (unsigned idx = tid; idx < BM * BK; idx += NUM_THREADS) {
            const unsigned r = idx / BK;
            const unsigned c = idx % BK;
            const size_t gr = rowBase + r;
            const size_t gc = k0 + c;
            As[c][r] = (gr < M && c < kLen) ? A[gr * N + gc] : 0.0;
        }
        // Stage B tile with bounds checks.
        for (unsigned idx = tid; idx < BK * BN; idx += NUM_THREADS) {
            const unsigned r = idx / BN;
            const unsigned c = idx % BN;
            const size_t gr = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (r < kLen && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

        for (unsigned kk = 0; kk < kLen; ++kk) {
            double a[TM];
            double b[TN];
            for (unsigned m = 0; m < TM; ++m) a[m] = As[kk][ty * TM + m];
            for (unsigned n = 0; n < TN; ++n) b[n] = Bs[kk][tx * TN + n];
            for (unsigned m = 0; m < TM; ++m)
                for (unsigned n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }
        __syncthreads();
    }

    for (unsigned m = 0; m < TM; ++m) {
        const size_t r = rowBase + ty * TM + m;
        if (r >= M) continue;
        for (unsigned n = 0; n < TN; ++n) {
            const size_t c = colBase + tx * TN + n;
            if (c < N) C[r * N + c] = acc[m][n];
        }
    }
}

int getDeviceCount() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        exit(1);
    }
    return deviceCount;
}

// One-time CUDA runtime and per-GPU context creation. Called before timing so
// the benchmark measures computation rather than driver startup.
void initCuda() {
    const int deviceCount = getDeviceCount();
    for (int d = 0; d < deviceCount; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
    }
}

// Splits the rows of C across all visible GPUs; each GPU receives its row
// block of A and a full copy of B, and the device work runs concurrently.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const size_t numParts = std::min(static_cast<size_t>(getDeviceCount()), N);
    const size_t rowsPerPart = N / numParts;
    const size_t extraRows = N % numParts;

    std::vector<std::thread> workers;
    workers.reserve(numParts);
    size_t row0 = 0;
    for (size_t d = 0; d < numParts; ++d) {
        const size_t rows = rowsPerPart + (d < extraRows ? 1 : 0);
        workers.emplace_back([&A, &B, &C, N, d, row0, rows]() {
            CUDA_CHECK(cudaSetDevice(static_cast<int>(d)));

            double *dA = nullptr, *dB = nullptr, *dC = nullptr;
            CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            CUDA_CHECK(cudaMemcpyAsync(dA, A.data() + row0 * N, rows * N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), N * N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

            const dim3 block(BLOCK_DIM, BLOCK_DIM);
            const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                            static_cast<unsigned>((rows + BM - 1) / BM));
            matmulKernel<<<grid, block, 0, stream>>>(dA, dB, dC, rows, N);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(C.data() + row0 * N, dC, rows * N * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaStreamDestroy(stream));
            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        });
        row0 += rows;
    }

    for (auto& w : workers) w.join();
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
    initCuda();
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
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
