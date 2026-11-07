#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// ---------------------------------------------------------------------------
// Matrix initialization on the GPU (one thread per element)
// ---------------------------------------------------------------------------
__global__ void initMatrixKernel(double* __restrict__ mat, const size_t N) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        mat[i * N + j] = getPseudoRndValue(N, i, j);
    }
}

void initMatrix(double* d_mat, const size_t N) {
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                    static_cast<unsigned>((N + block.y - 1) / block.y));
    initMatrixKernel<<<grid, block>>>(d_mat, N);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Tiled, register-blocked double-precision GEMM: C = A * B (row-major, NxN)
//
// Each thread block computes a BM x BN tile of C, iterating over K in steps of
// BK. A and B tiles are staged in shared memory (A is stored transposed so that
// the inner product loop reads contiguous shared-memory columns). Each thread
// computes a TM x TN micro-tile held in registers.
// ---------------------------------------------------------------------------
namespace gemm {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN;  // 16
constexpr int THREADS_Y = BM / TM;  // 16
constexpr int THREADS = THREADS_X * THREADS_Y;  // 256
static_assert(BM * BK % THREADS == 0, "A tile load must be evenly divisible");
static_assert(BK * BN % THREADS == 0, "B tile load must be evenly divisible");
}  // namespace gemm

template <bool FULL_TILES>
__global__ void __launch_bounds__(gemm::THREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int N) {
    using namespace gemm;

    // As is stored transposed: As[k][m] for conflict-free, broadcast-friendly reads
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % THREADS_X;
    const int ty = tid / THREADS_X;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int rowBase = blockRow + ty * TM;
    const int colBase = blockCol + tx * TN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    // Load mapping for A tile (BM x BK): each thread loads BM*BK/THREADS = 4 elements
    constexpr int A_LOADS = BM * BK / THREADS;
    constexpr int B_LOADS = BK * BN / THREADS;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // ---- Load A tile: rows [blockRow, blockRow+BM), cols [k0, k0+BK)
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int idx = tid + l * THREADS;
            const int m = idx / BK;
            const int k = idx % BK;
            const int gRow = blockRow + m;
            const int gCol = k0 + k;
            double v = 0.0;
            if (FULL_TILES || (gRow < N && gCol < N)) {
                v = A[static_cast<size_t>(gRow) * N + gCol];
            }
            As[k][m] = v;
        }
        // ---- Load B tile: rows [k0, k0+BK), cols [blockCol, blockCol+BN)
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int idx = tid + l * THREADS;
            const int k = idx / BN;
            const int n = idx % BN;
            const int gRow = k0 + k;
            const int gCol = blockCol + n;
            double v = 0.0;
            if (FULL_TILES || (gRow < N && gCol < N)) {
                v = B[static_cast<size_t>(gRow) * N + gCol];
            }
            Bs[k][n] = v;
        }
        __syncthreads();

        // ---- Compute micro-tile
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- Store results
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = rowBase + i;
        if (!FULL_TILES && gRow >= N) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gCol = colBase + j;
            if (FULL_TILES || gCol < N) {
                C[static_cast<size_t>(gRow) * N + gCol] = acc[i][j];
            }
        }
    }
}

void matrixMultiply(const double* d_A, const double* d_B, double* d_C, const size_t N) {
    using namespace gemm;
    if (N == 0) return;
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((N + BM - 1) / BM));
    const int n = static_cast<int>(N);
    if (N % BM == 0 && N % BN == 0 && N % BK == 0) {
        matmulKernel<true><<<grid, block>>>(d_A, d_B, d_C, n);
    } else {
        matmulKernel<false><<<grid, block>>>(d_A, d_B, d_C, n);
    }
    CUDA_CHECK(cudaGetLastError());
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

    // Initialize CUDA context up front so it is not attributed to the timed region
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    
    // Allocate matrices (host copies are only needed for output / validation)
    const size_t bytes = N * N * sizeof(double);
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes > 0 ? bytes : 1));
    CUDA_CHECK(cudaMalloc(&d_B, bytes > 0 ? bytes : 1));
    CUDA_CHECK(cudaMalloc(&d_C, bytes > 0 ? bytes : 1));
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(d_A, N);
    initMatrix(d_B, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy results (and inputs, for validation) back to the host
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        if (validate) {
            CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
        }
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
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
