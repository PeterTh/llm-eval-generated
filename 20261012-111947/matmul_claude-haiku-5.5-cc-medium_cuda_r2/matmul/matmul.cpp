#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Each thread block computes a TILE x TILE block of C using 16x16 threads,
// each thread owning a REG x REG sub-block of outputs in registers.
constexpr size_t TILE = 64;
constexpr size_t BK = 16;
constexpr size_t BLOCK_DIM = 16;
constexpr size_t REG = TILE / BLOCK_DIM;

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

// Tiled GEMM: A and B tiles are staged in shared memory, and the k dimension is
// walked in ascending order, so each C element accumulates in the same order as
// the serial reference loop.
__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double As[TILE][BK + 1];  // +1 padding avoids shared memory bank conflicts
    __shared__ double Bs[BK][TILE];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tid = ty * BLOCK_DIM + tx;
    const size_t rowBase = blockIdx.y * TILE;
    const size_t colBase = blockIdx.x * TILE;

    double acc[REG][REG] = {};

    for (size_t kt = 0; kt < N; kt += BK) {
        const size_t kLen = (N - kt < BK) ? N - kt : BK;

        for (size_t idx = tid; idx < TILE * BK; idx += BLOCK_DIM * BLOCK_DIM) {
            const size_t r = idx / BK;
            const size_t c = idx % BK;
            const size_t row = rowBase + r;
            As[r][c] = (row < N && c < kLen) ? A[row * N + kt + c] : 0.0;
        }
        for (size_t idx = tid; idx < BK * TILE; idx += BLOCK_DIM * BLOCK_DIM) {
            const size_t r = idx / TILE;
            const size_t c = idx % TILE;
            const size_t col = colBase + c;
            Bs[r][c] = (r < kLen && col < N) ? B[(kt + r) * N + col] : 0.0;
        }
        __syncthreads();

        for (size_t kk = 0; kk < kLen; ++kk) {
            double a[REG];
            double b[REG];
            for (size_t i = 0; i < REG; ++i) {
                a[i] = As[ty + i * BLOCK_DIM][kk];
            }
            for (size_t j = 0; j < REG; ++j) {
                b[j] = Bs[kk][tx + j * BLOCK_DIM];
            }
            for (size_t i = 0; i < REG; ++i) {
                for (size_t j = 0; j < REG; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < REG; ++i) {
        const size_t row = rowBase + ty + i * BLOCK_DIM;
        for (size_t j = 0; j < REG; ++j) {
            const size_t col = colBase + tx + j * BLOCK_DIM;
            if (row < N && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// Launches the kernel on device-resident matrices (asynchronous with respect to the host)
void matrixMultiply(const double* dA, const double* dB, double* dC, const size_t N) {
    if (N == 0) {
        return;
    }
    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N);
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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);

    // Allocate host matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Allocate device matrices and upload inputs (not included in the timing)
    const size_t bytes = N * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Perform matrix multiplication on the GPU
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    printf("Computation time: %.3f ms\n", duration.count());

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
