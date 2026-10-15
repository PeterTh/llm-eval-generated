#include <algorithm>
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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int BM = 64;  // C tile rows per block
constexpr int BN = 64;  // C tile cols per block
constexpr int BK = 16;  // k-slab depth
constexpr int TS = 4;   // per-thread outputs: TS x TS
constexpr int TPB = 16; // threads per block dim (BM / TS)

// C[rows x N] = A[rows x N] * B[N x N]; each thread computes a 4x4 register tile,
// accumulating over k in ascending order (same summation order as the serial code).
__global__ void __launch_bounds__(TPB * TPB)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TPB + tx;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[TS][TS] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // A tile: BM x BK, consecutive threads read consecutive k (coalesced)
#pragma unroll
        for (int l = 0; l < (BM * BK) / (TPB * TPB); ++l) {
            const int idx = tid + l * TPB * TPB;
            const int r = idx / BK, c = idx % BK;
            const size_t gr = rowBase + r, gc = k0 + c;
            As[c][r] = (gr < rows && gc < N) ? A[gr * N + gc] : 0.0;
        }
        // B tile: BK x BN, consecutive threads read consecutive columns (coalesced)
#pragma unroll
        for (int l = 0; l < (BK * BN) / (TPB * TPB); ++l) {
            const int idx = tid + l * TPB * TPB;
            const int r = idx / BN, c = idx % BN;
            const size_t gr = k0 + r, gc = colBase + c;
            Bs[r][c] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TS], b[TS];
#pragma unroll
            for (int i = 0; i < TS; ++i) a[i] = As[k][ty + TPB * i];
#pragma unroll
            for (int j = 0; j < TS; ++j) b[j] = Bs[k][tx + TPB * j];
#pragma unroll
            for (int i = 0; i < TS; ++i)
#pragma unroll
                for (int j = 0; j < TS; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TS; ++i) {
        const size_t gr = rowBase + ty + TPB * i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TS; ++j) {
            const size_t gc = colBase + tx + TPB * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Create CUDA contexts on all devices up front (keeps one-time init out of the timed region)
void initGPUs() {
    int numDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDev));
    for (int d = numDev - 1; d >= 0; --d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
    }
}

// Multi-GPU matmul: rows of C are split across all available GPUs; B is replicated.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    int numDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDev));
    if (numDev < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(1);
    }
    // Don't use more devices than there are row-blocks
    const size_t rowBlocks = (N + BM - 1) / BM;
    if ((size_t)numDev > rowBlocks) numDev = (int)rowBlocks;

    // Pin host memory for fast asynchronous transfers
    const size_t bytes = N * N * sizeof(double);
    const double* hA = A.data();
    const double* hB = B.data();
    double* hC = C.data();
    const bool pinA = cudaHostRegister(const_cast<double*>(hA), bytes, cudaHostRegisterDefault) == cudaSuccess;
    const bool pinB = cudaHostRegister(const_cast<double*>(hB), bytes, cudaHostRegisterDefault) == cudaSuccess;
    const bool pinC = cudaHostRegister(hC, bytes, cudaHostRegisterDefault) == cudaSuccess;
    cudaGetLastError();

    struct Part {
        size_t r0, rows;
        double *dA, *dB, *dC;
        cudaStream_t stream;
    };
    std::vector<Part> parts(numDev);

    const size_t blocksPerDev = (rowBlocks + numDev - 1) / numDev;
    for (int d = 0; d < numDev; ++d) {
        Part& p = parts[d];
        const size_t b0 = std::min(rowBlocks, d * blocksPerDev);
        const size_t b1 = std::min(rowBlocks, (d + 1) * blocksPerDev);
        p.r0 = std::min(N, b0 * BM);
        p.rows = std::min(N, b1 * BM) - p.r0;
        if (p.rows == 0) { p.dA = p.dB = p.dC = nullptr; p.stream = nullptr; continue; }
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&p.stream));
        CUDA_CHECK(cudaMalloc(&p.dA, p.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p.dB, bytes));
        CUDA_CHECK(cudaMalloc(&p.dC, p.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(p.dA, hA + p.r0 * N, p.rows * N * sizeof(double),
                                   cudaMemcpyHostToDevice, p.stream));
        CUDA_CHECK(cudaMemcpyAsync(p.dB, hB, bytes, cudaMemcpyHostToDevice, p.stream));
        const dim3 block(TPB, TPB);
        const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((p.rows + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, p.stream>>>(p.dA, p.dB, p.dC, p.rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hC + p.r0 * N, p.dC, p.rows * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, p.stream));
    }
    for (int d = 0; d < numDev; ++d) {
        Part& p = parts[d];
        if (p.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(p.stream));
        cudaFree(p.dA);
        cudaFree(p.dB);
        cudaFree(p.dC);
        cudaStreamDestroy(p.stream);
    }
    CUDA_CHECK(cudaSetDevice(0));
    if (pinA) cudaHostUnregister(const_cast<double*>(hA));
    if (pinB) cudaHostUnregister(const_cast<double*>(hB));
    if (pinC) cudaHostUnregister(hC);
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
    
    initGPUs();

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
