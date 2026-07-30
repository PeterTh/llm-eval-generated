#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// High-performance kernel with vectorized loads
// Thread block: 16x16 threads (256 threads)
// Each thread computes 4x4 output elements
// Block output: 64x64
// Tile K: 16
#define THREAD_M 16
#define THREAD_N 16
#define TM 4
#define TN 4
#define BM (THREAD_M * TM)  // 64
#define BN (THREAD_N * TN)  // 64
#define TK 16

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const int N) {
    __shared__ double tileA[BM][TK + 1];  // +1 to avoid bank conflicts
    __shared__ double tileB[TK][BN + 1];

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int rowBase = by * BM + ty * TM;
    const int colBase = bx * BN + tx * TN;

    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0;

    const int numTiles = (N + TK - 1) / TK;
    const int tid = ty * THREAD_M + tx;

    for (int t = 0; t < numTiles; ++t) {
        const int tileK = t * TK;

        // Load tileA[BM x TK] = 64x16 = 1024 elements, 256 threads, 4 loads each
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int r = idx % BM;
            int c = idx / BM;
            int gr = by * BM + r;
            int gc = tileK + c;
            tileA[r][c] = (gr < N && gc < N) ? A[gr * N + gc] : 0.0;
        }

        // Load tileB[TK x BN] = 16x64 = 1024 elements, 256 threads, 4 loads each
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int r = idx % TK;
            int c = idx / TK;
            int gr = tileK + r;
            int gc = bx * BN + c;
            tileB[r][c] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
        }

        __syncthreads();

        // Compute with register blocking
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = tileA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b[j] = tileB[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

    // Write results with vectorized stores
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = rowBase + i;
        if (gr < N) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int gc = colBase + j;
                if (gc < N)
                    C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void matrixMultiplyGPU(const double* d_A, const double* d_B, double* d_C, const size_t N) {
    dim3 block(THREAD_M, THREAD_N);
    dim3 grid((N + BM - 1) / BM, (N + BN - 1) / BN);

    matmulKernel<<<grid, block>>>(d_A, d_B, d_C, static_cast<int>(N));
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
    
    // Print GPU info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Allocate device memory
    const size_t bytes = N * N * sizeof(double);
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    // Perform matrix multiplication on GPU
    printf("Computing matrix multiplication...\n");

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));
    matrixMultiplyGPU(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
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
