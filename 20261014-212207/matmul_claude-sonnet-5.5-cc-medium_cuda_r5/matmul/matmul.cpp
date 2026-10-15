#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int BM = 64, BN = 64, BK = 16, TX = 16, TY = 16;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// C[rows x N] = A[rows x N] * B[N x N]; each thread computes a 4x4 register tile.
// The k-summation order per element is sequential, as in the serial code.
__global__ void __launch_bounds__(TX * TY)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[4][4] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // A tile: 64 rows x 16 k (coalesced along k)
#pragma unroll
        for (int l = 0; l < (BM * BK) / (TX * TY); ++l) {
            const int idx = tid + l * TX * TY;
            const int m = idx / BK, k = idx % BK;
            const size_t gr = rowBase + m, gk = k0 + k;
            As[k][m] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
        }
        // B tile: 16 k x 64 cols (coalesced along columns)
#pragma unroll
        for (int l = 0; l < (BK * BN) / (TX * TY); ++l) {
            const int idx = tid + l * TX * TY;
            const int k = idx / BN, n = idx % BN;
            const size_t gk = k0 + k, gc = colBase + n;
            Bs[k][n] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[k][ty + TY * r];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Bs[k][tx + TX * c];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t gr = rowBase + ty + TY * r;
        if (gr >= rows) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t gc = colBase + tx + TX * c;
            if (gc < N) C[gr * N + gc] = acc[r][c];
        }
    }
}

// Rows of A/C are partitioned across all available GPUs; B is replicated.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    int numDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDev));
    if (numDev < 1) { fprintf(stderr, "No CUDA device found\n"); exit(1); }

    size_t chunk = (N + numDev - 1) / numDev;
    chunk = (chunk + BM - 1) / BM * BM;
    int used = (int)((N + chunk - 1) / chunk);
    if (used < 1) return;

    std::vector<double*> dA(used), dB(used), dC(used);
    std::vector<size_t> r0(used), nr(used);
    std::vector<cudaStream_t> st(used);

    for (int d = 0; d < used; ++d) {
        r0[d] = (size_t)d * chunk;
        nr[d] = std::min(chunk, N - r0[d]);
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&st[d]));
        CUDA_CHECK(cudaMalloc(&dA[d], nr[d] * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB[d], N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC[d], nr[d] * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dA[d], A.data() + r0[d] * N, nr[d] * N * sizeof(double),
                                   cudaMemcpyHostToDevice, st[d]));
        CUDA_CHECK(cudaMemcpyAsync(dB[d], B.data(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, st[d]));
        dim3 block(TX, TY);
        dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((nr[d] + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, st[d]>>>(dA[d], dB[d], dC[d], nr[d], N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C.data() + r0[d] * N, dC[d], nr[d] * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, st[d]));
    }
    for (int d = 0; d < used; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(st[d]));
        CUDA_CHECK(cudaFree(dA[d]));
        CUDA_CHECK(cudaFree(dB[d]));
        CUDA_CHECK(cudaFree(dC[d]));
        CUDA_CHECK(cudaStreamDestroy(st[d]));
    }
    CUDA_CHECK(cudaSetDevice(0));
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
    
    // Initialize CUDA contexts on all devices (outside the timed region)
    {
        int nd = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nd));
        for (int d = nd - 1; d >= 0; --d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaFree(0));
        }
    }

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
