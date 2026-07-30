#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Tile size for shared memory tiling (32x32 = 1024 threads, 16KB shared mem per tile)
constexpr int TILE_SIZE = 32;

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

// CUDA kernel: tiled matrix multiplication with shared memory.
// Each block computes a TILE_SIZE x TILE_SIZE output tile.
__global__ void __launch_bounds__(TILE_SIZE * TILE_SIZE)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N) {
    // Use dynamic shared memory allocated at launch time
    extern __shared__ double shared[];
    double (*tileA)[TILE_SIZE] = (double (*)[TILE_SIZE])shared;
    double (*tileB)[TILE_SIZE] = (double (*)[TILE_SIZE])(shared + TILE_SIZE * TILE_SIZE);

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Coalesced load from A: row-by-row within warp
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Coalesced load from B: row-by-row within warp
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Persistent device memory pointers to avoid repeated allocation
static double *d_A_global = nullptr;
static double *d_B_global = nullptr;
static double *d_C_global = nullptr;
static size_t d_N_global = 0;

// Initialize/reinitialize device memory for a given matrix size
static void ensureDeviceMemory(const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    if (d_A_global != nullptr && d_N_global == N) return;
    // Free any previously allocated memory
    if (d_A_global) { cudaFree(d_A_global); cudaFree(d_B_global); cudaFree(d_C_global); }
    CUDA_CHECK(cudaMalloc(&d_A_global, bytes));
    CUDA_CHECK(cudaMalloc(&d_B_global, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_global, bytes));
    d_N_global = N;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // Ensure device memory is allocated (may be reused across calls)
    ensureDeviceMemory(N);

    const size_t bytes = N * N * sizeof(double);

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpyAsync(d_A_global, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_B_global, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel with 2D grid and 2D blocks
    const dim3 blockDim(TILE_SIZE, TILE_SIZE);
    const dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                       (N + TILE_SIZE - 1) / TILE_SIZE);

    // 2 tiles * TILE_SIZE^2 doubles * 8 bytes
    const size_t sharedMemBytes = 2 * TILE_SIZE * TILE_SIZE * sizeof(double);

    matmulKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_A_global, d_B_global, d_C_global, N);

    // Check for kernel launch errors and copy result back in one sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C.data(), d_C_global, bytes, cudaMemcpyDeviceToHost));
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
    
    // Warmup CUDA driver to avoid context init overhead in timed section
    cudaFree(0);

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Parallelization: CUDA (GPU)\n");
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication with CUDA event timing
    printf("Computing matrix multiplication...\n");

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));
    matrixMultiply(A, B, C, N);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    printf("Computation time: %.3f ms\n", elapsedMs);

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
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
