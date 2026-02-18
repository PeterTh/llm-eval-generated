#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

template <int TILE>
__global__ void matmulTiledKernel(const double* __restrict__ A,
                                 const double* __restrict__ B,
                                 double* __restrict__ C,
                                 int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    double sum = 0.0;

    for (int m = 0; m < N; m += TILE) {
        const int aCol = m + tx;
        const int bRow = m + ty;

        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
}  // namespace

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    if (N == 0) return 0.0;
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Matrix size too large for CUDA launch.\n");
        std::exit(1);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    const int Ni = static_cast<int>(N);
    const size_t bytes = N * N * sizeof(double);

    // Pin host memory to speed up transfers (best-effort; may fail due to OS limits).
    bool pinnedA = false, pinnedB = false, pinnedC = false;
    if (cudaHostRegister(const_cast<double*>(A.data()), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        pinnedA = true;
    } else {
        cudaGetLastError();
    }
    if (cudaHostRegister(const_cast<double*>(B.data()), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        pinnedB = true;
    } else {
        cudaGetLastError();
    }
    if (cudaHostRegister(C.data(), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        pinnedC = true;
    } else {
        cudaGetLastError();
    }

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), bytes, cudaMemcpyHostToDevice, stream));

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((Ni + TILE - 1) / TILE, (Ni + TILE - 1) / TILE);

    CUDA_CHECK(cudaFuncSetCacheConfig(matmulTiledKernel<TILE>, cudaFuncCachePreferShared));

    // Warm-up to reduce first-launch overhead.
    matmulTiledKernel<TILE><<<grid, block, 0, stream>>>(dA, dB, dC, Ni);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    matmulTiledKernel<TILE><<<grid, block, 0, stream>>>(dA, dB, dC, Ni);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (pinnedA) CUDA_CHECK(cudaHostUnregister(const_cast<double*>(A.data())));
    if (pinnedB) CUDA_CHECK(cudaHostUnregister(const_cast<double*>(B.data())));
    if (pinnedC) CUDA_CHECK(cudaHostUnregister(C.data()));

    return static_cast<double>(ms);
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

    const double ms = matrixMultiply(A, B, C, N);

    printf("Computation time: %.3f ms\n", ms);

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
