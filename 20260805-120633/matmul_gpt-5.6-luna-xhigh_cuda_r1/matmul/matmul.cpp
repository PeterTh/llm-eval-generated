#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
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

constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int THREADS_Y = 8;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block computes a 32x32 output tile.  There are only 8 warps in a
// block; each thread computes four rows, which keeps the block size at 256
// threads while retaining the data reuse of a full 32x32 tile.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const size_t blockRow = static_cast<size_t>(blockIdx.y) * TILE_SIZE;
    const size_t blockCol = static_cast<size_t>(blockIdx.x) * TILE_SIZE;

    double sums[4] = {0.0, 0.0, 0.0, 0.0};
    const size_t tileCount = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t tile = 0; tile < tileCount; ++tile) {
        // The 256 threads cooperatively load both 32x32 input tiles.  The
        // loads are coalesced across each warp and out-of-bounds elements
        // are zero-padded for arbitrary matrix sizes.
        #pragma unroll
        for (unsigned int row = ty; row < TILE_SIZE; row += THREADS_Y) {
            const size_t globalRow = blockRow + row;
            const size_t globalK = tile * TILE_SIZE + row;
            const size_t globalACol = tile * TILE_SIZE + tx;
            const size_t globalCol = blockCol + tx;
            tileA[row][tx] = (globalRow < N && globalACol < N)
                ? A[globalRow * N + globalACol]
                : 0.0;
            tileB[row][tx] = (globalK < N && globalCol < N)
                ? B[globalK * N + globalCol]
                : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < TILE_SIZE; ++k) {
            const double b = tileB[k][tx];
            #pragma unroll
            for (unsigned int row = 0; row < 4; ++row) {
                sums[row] = fma(tileA[ty + row * THREADS_Y][k], b, sums[row]);
            }
        }
        __syncthreads();
    }

    const size_t globalCol = blockCol + tx;
    if (globalCol < N) {
        #pragma unroll
        for (unsigned int row = 0; row < 4; ++row) {
            const size_t globalRow = blockRow + ty + row * THREADS_Y;
            if (globalRow < N) {
                C[globalRow * N + globalCol] = sums[row];
            }
        }
    }
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t matrixBytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), matrixBytes), "allocating A");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixBytes), "allocating B");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), matrixBytes), "allocating C");

    checkCuda(cudaMemcpy(deviceA, A.data(), matrixBytes, cudaMemcpyHostToDevice),
              "copying A to device");
    checkCuda(cudaMemcpy(deviceB, B.data(), matrixBytes, cudaMemcpyHostToDevice),
              "copying B to device");

    const size_t gridSize = (N + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 block(TILE_SIZE, THREADS_Y);
    const dim3 grid(static_cast<unsigned int>(gridSize),
                    static_cast<unsigned int>(gridSize));
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaMemcpy(C.data(), deviceC, matrixBytes, cudaMemcpyDeviceToHost),
              "copying C to host");

    checkCuda(cudaFree(deviceC), "freeing C");
    checkCuda(cudaFree(deviceB), "freeing B");
    checkCuda(cudaFree(deviceA), "freeing A");
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
    // Pay CUDA context initialization outside the measured multiplication,
    // just as host runtime/library startup is outside the original loop.
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
