#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        const cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err_));                                      \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

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

// Tile edge length of the shared-memory tiles; each thread block computes a
// TILE x TILE tile of C, with each thread accumulating THREAD_WORK rows of it.
constexpr unsigned TILE = 32;
constexpr unsigned THREAD_WORK = 4;  // C rows per thread (TILE x TILE/THREAD_WORK block)

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const unsigned N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE + 1];  // +1 padding avoids bank conflicts

    const unsigned tx = threadIdx.x;               // 0..TILE-1 (column within tile)
    const unsigned ty = threadIdx.y;               // 0..TILE/THREAD_WORK-1
    const unsigned col = blockIdx.x * TILE + tx;
    const unsigned rowBase = blockIdx.y * TILE;    // first row of this C tile

    double acc[THREAD_WORK] = {0.0};

    const unsigned numTiles = (N + TILE - 1) / TILE;
    for (unsigned t = 0; t < numTiles; ++t) {
        // Cooperatively load one TILE x TILE tile of A and B; each thread
        // loads THREAD_WORK elements of each.
        for (unsigned w = 0; w < THREAD_WORK; ++w) {
            const unsigned r = ty + w * (TILE / THREAD_WORK);
            const unsigned aRow = rowBase + r;
            const unsigned aCol = t * TILE + tx;
            As[r][tx] = (aRow < N && aCol < N) ? A[static_cast<size_t>(aRow) * N + aCol] : 0.0;

            const unsigned bRow = t * TILE + r;
            Bs[r][tx] = (bRow < N && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
#pragma unroll
            for (unsigned w = 0; w < THREAD_WORK; ++w) {
                acc[w] += As[ty + w * (TILE / THREAD_WORK)][k] * b;
            }
        }
        __syncthreads();
    }

    if (col < N) {
        for (unsigned w = 0; w < THREAD_WORK; ++w) {
            const unsigned row = rowBase + ty + w * (TILE / THREAD_WORK);
            if (row < N) {
                C[static_cast<size_t>(row) * N + col] = acc[w];
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE / THREAD_WORK);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulKernel<<<grid, block>>>(dA, dB, dC, static_cast<unsigned>(N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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

    // Initialize the CUDA context up front so setup cost is not timed
    CUDA_CHECK(cudaFree(nullptr));

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
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
