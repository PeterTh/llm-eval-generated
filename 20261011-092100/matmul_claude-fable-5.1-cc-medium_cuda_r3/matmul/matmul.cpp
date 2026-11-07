// CUDA-parallelized matrix multiplication benchmark.
// This file is compiled as CUDA (see CMakeLists.txt: LANGUAGE CUDA).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__global__ void initMatrixKernel(double* __restrict__ mat, const size_t N) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        mat[i * N + j] = getPseudoRndValue(N, i, j);
    }
}

// Tiled, register-blocked double-precision GEMM: C = A * B (row-major, NxN).
//   Block tile:  BM x BN of C, iterating over K in steps of BK.
//   Thread tile: TM x TN outputs per thread => (BM/TM)*(BN/TN) threads per block.
// Global->shared loads are double-buffered through registers so the loads of
// tile t+1 overlap with the FMAs of tile t. Arbitrary N is handled by bounds
// checks on loads (zero fill) and on the final store.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
static_assert(THREADS == 256, "thread tiling must give 256 threads");
static_assert((BM * BK) % THREADS == 0 && (BN * BK) % THREADS == 0, "load split");
constexpr int A_LOADS = (BM * BK) / THREADS;  // 4 per thread
constexpr int B_LOADS = (BN * BK) / THREADS;  // 4 per thread

__global__ void __launch_bounds__(THREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int N) {
    // As is stored transposed ([k][m]) so that the inner product loop reads
    // TM consecutive doubles per thread (conflict-free, vectorizable).
    __shared__ double As[2][BK][BM];
    __shared__ double Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);  // 0..15  -> column group
    const int ty = tid / (BN / TN);  // 0..15  -> row group

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // Load mapping for A tile (BM x BK): element e = tid + l*THREADS,
    // row = e / BK, k = e % BK  -> coalesced along k (contiguous in memory).
    // Load mapping for B tile (BK x BN): row k = e / BN, col = e % BN -> coalesced along n.
    double regA[A_LOADS];
    double regB[B_LOADS];

    auto loadTileToRegs = [&](int kBase) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int e = tid + l * THREADS;
            const int m = e / BK;
            const int k = e % BK;
            const int gi = rowBase + m;
            const int gk = kBase + k;
            regA[l] = (gi < N && gk < N) ? A[static_cast<size_t>(gi) * N + gk] : 0.0;
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int e = tid + l * THREADS;
            const int k = e / BN;
            const int n = e % BN;
            const int gk = kBase + k;
            const int gj = colBase + n;
            regB[l] = (gk < N && gj < N) ? B[static_cast<size_t>(gk) * N + gj] : 0.0;
        }
    };

    auto storeRegsToSmem = [&](int buf) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int e = tid + l * THREADS;
            As[buf][e % BK][e / BK] = regA[l];
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int e = tid + l * THREADS;
            Bs[buf][e / BN][e % BN] = regB[l];
        }
    };

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    const int numTiles = (N + BK - 1) / BK;

    loadTileToRegs(0);
    storeRegsToSmem(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;
        if (t + 1 < numTiles) {
            loadTileToRegs((t + 1) * BK);  // prefetch next tile into registers
        }

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (t + 1 < numTiles) {
            storeRegsToSmem(nxt);
        }
        __syncthreads();
    }

    // Store results with bounds checks.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gi = rowBase + ty * TM + i;
        if (gi >= N) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = colBase + tx * TN + j;
            if (gj < N) C[static_cast<size_t>(gi) * N + gj] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

// Initialize a matrix on the device and copy it to the host vector.
void initMatrix(std::vector<double>& mat, double* d_mat, const size_t N) {
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                    static_cast<unsigned>((N + block.y - 1) / block.y));
    initMatrixKernel<<<grid, block>>>(d_mat, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(mat.data(), d_mat, N * N * sizeof(double), cudaMemcpyDeviceToHost));
}

// C = A * B on the GPU. A and B must already reside on the device; the result
// is copied back into the host vector C. The call is synchronous.
void matrixMultiply(const double* d_A, const double* d_B, double* d_C,
                    std::vector<double>& C, const size_t N) {
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((N + BM - 1) / BM));
    matmulKernel<<<grid, block>>>(d_A, d_B, d_C, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));
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

    // Initialize the CUDA context up front so that it is not part of the timing.
    CUDA_CHECK(cudaFree(0));
    {
        int dev = 0;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("CUDA device: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);
    }
    
    // Allocate device buffers
    const size_t bytes = N * N * sizeof(double);
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices (on the GPU; host copies kept for validation)
    printf("Initializing matrices...\n");
    initMatrix(A, d_A, N);
    initMatrix(B, d_B, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Perform matrix multiplication (timed: kernel + device->host copy of C)
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(d_A, d_B, d_C, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }

    int ret = 0;
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return ret;
}
