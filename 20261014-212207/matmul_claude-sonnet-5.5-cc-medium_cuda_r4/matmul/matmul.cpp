#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(2);                                                              \
        }                                                                         \
    } while (0)

// Generate rows [row0, row0 + rows) of an N x N pseudo-random matrix on the device
__global__ void initKernel(double* mat, const size_t N, const size_t row0, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = row0 + idx / N;
        const size_t j = idx % N;
        mat[idx] = getPseudoRndValue(N, i, j);
    }
}

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

// C[rows x N] = A[rows x N] * B[N x N]; each block computes a 64x64 tile with 4x4 per thread.
// Each element accumulates over k in ascending order, as in the serial code.
__global__ void __launch_bounds__(256)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N, const size_t rows) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = tid + 256 * l;
            const int ar = idx / BK, ac = idx % BK;
            const size_t gr = rowBase + ar, gc = k0 + ac;
            As[ac][ar] = (gr < rows && gc < N) ? A[gr * N + gc] : 0.0;
            const int br = idx / BN, bc = idx % BN;
            const size_t gbr = k0 + br, gbc = colBase + bc;
            Bs[br][bc] = (gbr < N && gbc < N) ? B[gbr * N + gbc] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t gr = rowBase + ty + 16 * i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gc = colBase + tx + 16 * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Multiply on all available GPUs (rows of C split across devices).
// Returns computation time in ms; A and B are generated on the devices.
double matrixMultiplyGPU(std::vector<double>& C, const size_t N) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(2);
    }
    // Don't use more devices than there are row tiles
    const size_t rowTiles = (N + BM - 1) / BM;
    if ((size_t)ndev > rowTiles) ndev = (int)rowTiles;

    std::vector<size_t> r0(ndev), nr(ndev);
    const size_t tilesPer = rowTiles / ndev, extra = rowTiles % ndev;
    size_t curTile = 0;
    for (int d = 0; d < ndev; ++d) {
        const size_t t = tilesPer + ((size_t)d < extra ? 1 : 0);
        r0[d] = std::min(curTile * BM, N);
        nr[d] = std::min((curTile + t) * BM, N) - r0[d];
        curTile += t;
    }

    std::vector<double*> dA(ndev), dB(ndev), dC(ndev);
    std::vector<cudaStream_t> st(ndev);
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&st[d]));
        CUDA_CHECK(cudaMalloc(&dA[d], nr[d] * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB[d], N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC[d], nr[d] * N * sizeof(double)));
        initKernel<<<1024, 256, 0, st[d]>>>(dA[d], N, r0[d], nr[d]);
        initKernel<<<1024, 256, 0, st[d]>>>(dB[d], N, 0, N);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(st[d]));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((nr[d] + BM - 1) / BM));
        matmulKernel<<<grid, 256, 0, st[d]>>>(dA[d], dB[d], dC[d], N, nr[d]);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(st[d]));
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpyAsync(C.data() + r0[d] * N, dC[d], nr[d] * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, st[d]));
    }
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(st[d]));
        cudaFree(dA[d]);
        cudaFree(dB[d]);
        cudaFree(dC[d]);
        cudaStreamDestroy(st[d]);
    }
    return std::chrono::duration<double, std::milli>(end - start).count();
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
    // (A and B are generated directly on the GPUs; host copies only needed for validation)
    if (validate) {
        initMatrix(A, N);
        initMatrix(B, N);
    }
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    const double ms = matrixMultiplyGPU(C, N);
    
    printf("Computation time: %ld ms\n", (long)ms);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
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
