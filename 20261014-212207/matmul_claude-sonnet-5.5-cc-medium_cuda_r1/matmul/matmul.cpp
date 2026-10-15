#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int BM = 64, BN = 64, BK = 16, TX = 16, TY = 16, RM = BM / TY, RN = BN / TX;

// C[rows x N] = A[rows x N] * B[N x N]; register-blocked, shared-memory tiled.
// Each element accumulates over k in ascending order, like the sequential code.
__global__ void __launch_bounds__(TX * TY)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[RM][RN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // A tile: BM x BK, consecutive threads read consecutive k
        for (int l = tid; l < BM * BK; l += TX * TY) {
            const int m = l / BK, k = l % BK;
            const size_t r = rowBase + m, c = k0 + k;
            As[k][m] = (r < rows && c < N) ? A[r * N + c] : 0.0;
        }
        // B tile: BK x BN, consecutive threads read consecutive columns
        for (int l = tid; l < BK * BN; l += TX * TY) {
            const int k = l / BN, n = l % BN;
            const size_t r = k0 + k, c = colBase + n;
            Bs[k][n] = (r < N && c < N) ? B[r * N + c] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[RM], b[RN];
#pragma unroll
            for (int i = 0; i < RM; ++i) a[i] = As[k][ty + TY * i];
#pragma unroll
            for (int j = 0; j < RN; ++j) b[j] = Bs[k][tx + TX * j];
#pragma unroll
            for (int i = 0; i < RM; ++i)
#pragma unroll
                for (int j = 0; j < RN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < RM; ++i) {
        const size_t r = rowBase + ty + TY * i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < RN; ++j) {
            const size_t c = colBase + tx + TX * j;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}

// Splits the rows of C across all available GPUs; B is replicated.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    int numDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDev));
    if (numDev < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    // Rows per GPU, rounded to the tile height; skip GPUs that would get no work
    size_t chunk = (N + numDev - 1) / numDev;
    chunk = (chunk + BM - 1) / BM * BM;
    const int used = (int)((N + chunk - 1) / chunk);

    struct Part {
        size_t r0 = 0, rows = 0;
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaStream_t st = nullptr;
    };
    std::vector<Part> parts(used);

    for (int d = 0; d < used; ++d) {
        Part& p = parts[d];
        p.r0 = (size_t)d * chunk;
        p.rows = std::min(chunk, N - p.r0);
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&p.st));
        CUDA_CHECK(cudaMalloc(&p.dA, p.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p.dC, p.rows * N * sizeof(double)));
    }
    for (int d = 0; d < used; ++d) {
        Part& p = parts[d];
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpyAsync(p.dA, A.data() + p.r0 * N, p.rows * N * sizeof(double),
                                   cudaMemcpyHostToDevice, p.st));
        CUDA_CHECK(cudaMemcpyAsync(p.dB, B.data(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, p.st));
        const dim3 block(TX, TY);
        const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((p.rows + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, p.st>>>(p.dA, p.dB, p.dC, p.rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C.data() + p.r0 * N, p.dC, p.rows * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, p.st));
    }
    for (int d = 0; d < used; ++d) {
        Part& p = parts[d];
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(p.st));
        CUDA_CHECK(cudaFree(p.dA));
        CUDA_CHECK(cudaFree(p.dB));
        CUDA_CHECK(cudaFree(p.dC));
        CUDA_CHECK(cudaStreamDestroy(p.st));
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
    
    // Initialize CUDA contexts on all devices outside the timed region
    {
        int nd = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nd));
        for (int d = 0; d < nd; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaFree(0));
        }
        CUDA_CHECK(cudaSetDevice(0));
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
