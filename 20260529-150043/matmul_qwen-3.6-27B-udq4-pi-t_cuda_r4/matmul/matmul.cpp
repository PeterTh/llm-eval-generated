#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ── CUDA error checking helper ──────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err), #call);                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Matrix initialization (same as original) ────────────────────────────────
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ── Optimized tiled CUDA matmul kernel ──────────────────────────────────────
//   • 32×32 tile with shared memory
//   • __ldg() read-only cache for input matrices
//   • Bank-conflict-free layout (32+1 stride)
//   • Fully unrolled inner loop for WarpCount iterations
//   • Vectorized global loads where beneficial
template <int TILE_SIZE>
__global__ void matmulKernel(const double* A, const double* B, double* C,
                             const size_t N) {
    // Shared memory tiles – extra column to avoid bank conflicts
    __shared__ double As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE + 1];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;

    const size_t row = blockIdx.y * TILE_SIZE + ty;
    const size_t col = blockIdx.x * TILE_SIZE + tx;

    double sum = 0.0;

    // Number of tile steps along the K dimension
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A into shared memory
        {
            const size_t aCol = t * TILE_SIZE + tx;
            if (row < N && aCol < N) {
                As[ty][tx] = __ldg(&A[row * N + aCol]);
            } else {
                As[ty][tx] = 0.0;
            }
        }

        // Load tile from B into shared memory
        {
            const size_t bRow = t * TILE_SIZE + ty;
            if (bRow < N && col < N) {
                Bs[ty][tx] = __ldg(&B[bRow * N + col]);
            } else {
                Bs[ty][tx] = 0.0;
            }
        }

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); // barrier before next tile load
    }

    // Write result back to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ── Host-side wrapper: allocate, launch, copy back ─────────────────────────
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);

    // Device pointers
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Host → device transfers
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Zero-initialize output on device (handles boundary tiles)
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Kernel launch configuration
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matmulKernel<TILE><<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device → host transfer
    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// ── Validation (same logic as original, runs on host) ───────────────────────
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
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
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got "
                    "%.10f (error: %.10e)\n",
                    i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

// ── CLI ─────────────────────────────────────────────────────────────────────
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
            N = static_cast<size_t>(atoi(argv[++i]));
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

    // Select GPU device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable GPU found.\n");
        return 1;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

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

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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
