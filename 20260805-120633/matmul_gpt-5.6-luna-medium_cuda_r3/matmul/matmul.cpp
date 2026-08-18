#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr unsigned TILE_SIZE = 32;
constexpr unsigned BLOCK_ROWS = 8;

// Each block computes an 8x32 output tile. The 256 threads cooperatively load
// both input tiles, while each thread accumulates one output element.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[BLOCK_ROWS][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * BLOCK_ROWS + ty;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE_SIZE + tx;
    double sum = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE_SIZE) {
        const unsigned loadRow = threadIdx.y;
        const unsigned loadCol = threadIdx.x;
        const size_t aCol = tile + loadCol;

        tileA[loadRow][loadCol] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0;

        // There are four times as many B values as threads in this block.
        // Striding by the block size keeps all loads coalesced.
        const unsigned threadIndex = loadRow * TILE_SIZE + loadCol;
        for (unsigned bIndex = threadIndex; bIndex < TILE_SIZE * TILE_SIZE;
             bIndex += TILE_SIZE * BLOCK_ROWS) {
            const unsigned bTileRow = bIndex / TILE_SIZE;
            const unsigned bTileCol = bIndex % TILE_SIZE;
            const size_t bRow = tile + bTileRow;
            const size_t bCol = static_cast<size_t>(blockIdx.x) * TILE_SIZE + bTileCol;
            tileB[bTileRow][bTileCol] =
                (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;
        }
        __syncthreads();

        if (row < N && col < N) {
            #pragma unroll
            for (unsigned k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[ty][k] * tileB[k][tx];
            }
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes),
              "allocating matrix A");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes),
              "allocating matrix B");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes),
              "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix B");

    const dim3 block(TILE_SIZE, BLOCK_ROWS);
    const dim3 grid(static_cast<unsigned>((N + TILE_SIZE - 1) / TILE_SIZE),
                    static_cast<unsigned>((N + BLOCK_ROWS - 1) / BLOCK_ROWS));
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaDeviceSynchronize(), "executing matrix multiplication kernel");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
              "copying matrix C");

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
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
    // Exclude one-time CUDA context creation from the benchmark measurement.
    checkCuda(cudaFree(nullptr), "initializing CUDA context");
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
