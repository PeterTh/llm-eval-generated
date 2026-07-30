#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Tile size for shared-memory tiled matmul
#define TILE_SIZE 32

// Register-tiled matmul kernel: each thread computes TM x TN output elements
// Block dimensions: (TILE_M/TM) x (TILE_N/TN) threads
// Output tile per block: TILE_M x TILE_N
#define TILE_M  128
#define TILE_N  128
#define TILE_K  16
#define TM      8
#define TN      8
#define BLOCK_THREADS_X (TILE_N / TN)  // 16
#define BLOCK_THREADS_Y (TILE_M / TM)  // 16

__global__ void matmulKernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              const size_t N) {
    // Shared memory for A and B tiles (padded to avoid bank conflicts)
    __shared__ double sA[TILE_M][TILE_K + 1];
    __shared__ double sB[TILE_K][TILE_N + 1];

    const int blockRow = blockIdx.y * TILE_M;
    const int blockCol = blockIdx.x * TILE_N;

    const int tidY = threadIdx.y; // 0..15
    const int tidX = threadIdx.x; // 0..15

    // Each thread accumulates a TM x TN sub-tile
    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    // Thread's position in global output
    const int threadRow0 = blockRow + tidY * TM;
    const int threadCol0 = blockCol + tidX * TN;

    // Linear thread id for cooperative loading
    const int linearTid = tidY * BLOCK_THREADS_X + tidX;
    const int numThreads = BLOCK_THREADS_X * BLOCK_THREADS_Y; // 256

    // Iterate over K dimension in chunks of TILE_K
    for (size_t kk = 0; kk < N; kk += TILE_K) {

        // Cooperative load of A tile: TILE_M x TILE_K elements, 256 threads
        const int elemsA = TILE_M * TILE_K; // 128*16 = 2048
        for (int idx = linearTid; idx < elemsA; idx += numThreads) {
            const int r = idx / TILE_K;
            const int c = idx % TILE_K;
            const int gr = blockRow + r;
            const int gc = (int)kk + c;
            sA[r][c] = (gr < (int)N && gc < (int)N) ? __ldg(&A[gr * N + gc]) : 0.0;
        }

        // Cooperative load of B tile: TILE_K x TILE_N elements, 256 threads
        const int elemsB = TILE_K * TILE_N; // 16*128 = 2048
        for (int idx = linearTid; idx < elemsB; idx += numThreads) {
            const int r = idx / TILE_N;
            const int c = idx % TILE_N;
            const int gr = (int)kk + r;
            const int gc = blockCol + c;
            sB[r][c] = (gr < (int)N && gc < (int)N) ? __ldg(&B[gr * N + gc]) : 0.0;
        }

        __syncthreads();

        // Compute partial products using register tiling
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Load A fragment into registers
            double aFrag[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                aFrag[i] = sA[tidY * TM + i][k];

            // Load B fragment into registers
            double bFrag[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bFrag[j] = sB[k][tidX * TN + j];

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aFrag[i] * bFrag[j];
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = threadRow0 + i;
        if (gr < (int)N) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gc = threadCol0 + j;
                if (gc < (int)N)
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

struct GPUMatrices {
    double *d_A, *d_B, *d_C;
    size_t bytes;
};

GPUMatrices allocateGPU(const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    GPUMatrices g;
    g.bytes = bytes;
    CUDA_CHECK(cudaMalloc(&g.d_A, bytes));
    CUDA_CHECK(cudaMalloc(&g.d_B, bytes));
    CUDA_CHECK(cudaMalloc(&g.d_C, bytes));
    return g;
}

void freeGPU(GPUMatrices& g) {
    CUDA_CHECK(cudaFree(g.d_A));
    CUDA_CHECK(cudaFree(g.d_B));
    CUDA_CHECK(cudaFree(g.d_C));
}

void matrixMultiplyGPU(const GPUMatrices& g, const std::vector<double>& A,
                       const std::vector<double>& B, std::vector<double>& C,
                       const size_t N, float& kernelTimeMs) {
    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(g.d_A, A.data(), g.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_B, B.data(), g.bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y); // 16x16 = 256 threads
    dim3 grid((N + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    // Create CUDA events for accurate kernel timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Record start, launch kernel, record stop
    CUDA_CHECK(cudaEventRecord(startEvent));
    matmulKernel<<<grid, block>>>(g.d_A, g.d_B, g.d_C, N);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    // Get kernel execution time
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, startEvent, stopEvent));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), g.d_C, g.bytes, cudaMemcpyDeviceToHost));

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
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
    
    // Print GPU info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Matrix Multiplication Benchmark (CUDA)\n");
    printf("GPU: %s\n", prop.name);
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
    
    // Allocate GPU memory once
    printf("Allocating GPU memory...\n");
    GPUMatrices gpu = allocateGPU(N);
    
    // Perform matrix multiplication on GPU
    printf("Computing matrix multiplication...\n");

    // Warm up GPU
    CUDA_CHECK(cudaFree(0));

    float kernelTimeMs = 0.0f;
    matrixMultiplyGPU(gpu, A, B, C, N, kernelTimeMs);
    
    printf("Computation time: %.3f ms\n", kernelTimeMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (kernelTimeMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Free GPU memory
    freeGPU(gpu);
    
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
