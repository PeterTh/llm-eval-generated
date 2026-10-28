#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Tile edge (shared-memory tile is TILE x TILE) and thread block shape.
// Block is TILE columns x BLOCK_ROWS threads; each thread computes
// ROWS_PER_THREAD output rows, so TILE == BLOCK_ROWS * ROWS_PER_THREAD.
constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE / BLOCK_ROWS;

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

// Computes C[rows x N] = A[rows x N] * B[N x N] for the rows of this device's slice.
// Each element is accumulated over k in ascending order, as in the sequential version.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const int M, const int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE + tx;
    const int rowBase = blockIdx.y * TILE + ty;

    double sum[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) sum[r] = 0.0;

    for (int t = 0; t < N; t += TILE) {
        const int kLen = min(TILE, N - t);

#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = rowBase + r * BLOCK_ROWS;
            As[ty + r * BLOCK_ROWS][tx] =
                (row < M && t + tx < N) ? A[static_cast<size_t>(row) * N + t + tx] : 0.0;
            const int k = t + ty + r * BLOCK_ROWS;
            Bs[ty + r * BLOCK_ROWS][tx] =
                (k < N && col < N) ? B[static_cast<size_t>(k) * N + col] : 0.0;
        }
        __syncthreads();

        for (int k = 0; k < kLen; ++k) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                sum[r] += As[ty + r * BLOCK_ROWS][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = rowBase + r * BLOCK_ROWS;
        if (row < M && col < N) {
            C[static_cast<size_t>(row) * N + col] = sum[r];
        }
    }
}

// Ensure the CUDA runtime/context is initialized on all devices before timing.
void initCudaDevices(int& deviceCount) {
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: no CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    for (int d = 0; d < deviceCount; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(0));
    }
    CUDA_CHECK(cudaSetDevice(0));
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const int deviceCount) {
    if (N == 0) return;

    // Page-lock the host buffers so that per-device transfers can run asynchronously
    // (registration only pins existing memory; no copies are made).
    const size_t bytes = N * N * sizeof(double);
    CUDA_CHECK(cudaHostRegister(const_cast<double*>(A.data()), bytes, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(const_cast<double*>(B.data()), bytes, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(C.data(), bytes, cudaHostRegisterDefault));

    // Partition rows of C across devices (one slice per device, at most N slices).
    const int numDev = static_cast<int>(std::min<size_t>(static_cast<size_t>(deviceCount), N));

    struct Slice {
        size_t rowBegin = 0;
        size_t rows = 0;
        cudaStream_t stream = nullptr;
        double* dA = nullptr;
        double* dB = nullptr;
        double* dC = nullptr;
    };
    std::vector<Slice> slices(numDev);

    const size_t baseRows = N / numDev;
    const size_t extraRows = N % numDev;
    size_t row = 0;
    for (int d = 0; d < numDev; ++d) {
        slices[d].rowBegin = row;
        slices[d].rows = baseRows + (static_cast<size_t>(d) < extraRows ? 1 : 0);
        row += slices[d].rows;
    }

    // Launch all work asynchronously, one stream per device.
    for (int d = 0; d < numDev; ++d) {
        Slice& s = slices[d];
        const size_t M = s.rows;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&s.stream));
        CUDA_CHECK(cudaMalloc(&s.dA, M * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.dC, M * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpyAsync(s.dA, A.data() + s.rowBegin * N, M * N * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.dB, B.data(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));

        const dim3 block(TILE, BLOCK_ROWS);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((M + TILE - 1) / TILE));
        matmulKernel<<<grid, block, 0, s.stream>>>(s.dA, s.dB, s.dC,
                                                   static_cast<int>(M), static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(C.data() + s.rowBegin * N, s.dC, M * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, s.stream));
    }

    // Wait for all devices, then release resources.
    for (int d = 0; d < numDev; ++d) {
        Slice& s = slices[d];
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }
    for (int d = 0; d < numDev; ++d) {
        Slice& s = slices[d];
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(s.dA));
        CUDA_CHECK(cudaFree(s.dB));
        CUDA_CHECK(cudaFree(s.dC));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    CUDA_CHECK(cudaHostUnregister(const_cast<double*>(A.data())));
    CUDA_CHECK(cudaHostUnregister(const_cast<double*>(B.data())));
    CUDA_CHECK(cudaHostUnregister(C.data()));
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

    // Initialize CUDA devices (context creation is excluded from the timed region)
    int deviceCount = 0;
    initCudaDevices(deviceCount);
    printf("CUDA devices used: %d\n", static_cast<int>(std::min<size_t>(deviceCount, N)));
    
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
    
    matrixMultiply(A, B, C, N, deviceCount);
    
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
