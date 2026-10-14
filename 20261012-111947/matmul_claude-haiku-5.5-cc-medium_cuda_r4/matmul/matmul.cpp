#include <algorithm>
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

// Tiling parameters: each thread block computes a BM x BN tile of C with
// THREADS_X x THREADS_Y threads, each thread owning a TM x TN register block.
constexpr size_t BM = 64;
constexpr size_t BN = 64;
constexpr size_t BK = 16;
constexpr size_t TM = 4;
constexpr size_t TN = 4;
constexpr size_t THREADS_X = BN / TN;
constexpr size_t THREADS_Y = BM / TM;
constexpr size_t THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

// Computes C = A * B for a horizontal slice of rows. A holds `rows` rows of N
// elements, B is the full N x N matrix. Every C element accumulates its k-terms
// in ascending order, so the result matches the sequential reference exactly.
__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tid = ty * THREADS_X + tx;
    const size_t rowBase = blockIdx.y * BM;
    const size_t colBase = blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (size_t kt = 0; kt < N; kt += BK) {
        const size_t kLim = (N - kt < BK) ? (N - kt) : BK;

        // Stage the A tile transposed (As[k][m]) so compute reads are contiguous
        for (size_t idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
            const size_t m = idx / BK;
            const size_t k = idx % BK;
            As[k][m] = (rowBase + m < rows && k < kLim) ? A[(rowBase + m) * N + kt + k] : 0.0;
        }
        // Stage the B tile
        for (size_t idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
            const size_t k = idx / BN;
            const size_t n = idx % BN;
            Bs[k][n] = (k < kLim && colBase + n < N) ? B[(kt + k) * N + colBase + n] : 0.0;
        }
        __syncthreads();

        for (size_t k = 0; k < kLim; ++k) {
            double a[TM];
            double b[TN];
            for (size_t i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
            for (size_t j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
            for (size_t i = 0; i < TM; ++i) {
                for (size_t j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < TM; ++i) {
        for (size_t j = 0; j < TN; ++j) {
            const size_t row = rowBase + ty * TM + i;
            const size_t col = colBase + tx * TN + j;
            if (row < rows && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
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

int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        exit(EXIT_FAILURE);
    }
    return count;
}

// Create a context on every GPU up front so one-time driver setup is not timed
void initDevices(int count) {
    for (int d = 0; d < count; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    // Split the rows of C into contiguous slices, one per GPU
    const size_t numDevices = std::min(static_cast<size_t>(getDeviceCount()), N);
    const size_t rowsPerDevice = (N + numDevices - 1) / numDevices;

    struct DeviceWork {
        int device;
        size_t rowStart;
        size_t rows;
        cudaStream_t stream;
        double* dA;
        double* dB;
        double* dC;
    };

    std::vector<DeviceWork> work;
    for (size_t d = 0; d < numDevices; ++d) {
        const size_t rowStart = d * rowsPerDevice;
        if (rowStart >= N) {
            break;
        }
        work.push_back({static_cast<int>(d), rowStart, std::min(rowsPerDevice, N - rowStart),
                        nullptr, nullptr, nullptr, nullptr});
    }

    // Pin the host matrices so the copies below are truly asynchronous and
    // do not stall the host thread that feeds the other GPUs
    CUDA_CHECK(cudaHostRegister(const_cast<double*>(A.data()), A.size() * sizeof(double),
                                cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(const_cast<double*>(B.data()), B.size() * sizeof(double),
                                cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(C.data(), C.size() * sizeof(double), cudaHostRegisterDefault));

    // Issue all GPU work asynchronously so the devices run concurrently
    for (auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaStreamCreate(&w.stream));
        CUDA_CHECK(cudaMalloc(&w.dA, w.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.dC, w.rows * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpyAsync(w.dA, A.data() + w.rowStart * N, w.rows * N * sizeof(double),
                                   cudaMemcpyHostToDevice, w.stream));
        CUDA_CHECK(cudaMemcpyAsync(w.dB, B.data(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, w.stream));

        const dim3 block(THREADS_X, THREADS_Y);
        const dim3 grid((N + BN - 1) / BN, (w.rows + BM - 1) / BM);
        matrixMultiplyKernel<<<grid, block, 0, w.stream>>>(w.dA, w.dB, w.dC, w.rows, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(C.data() + w.rowStart * N, w.dC, w.rows * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, w.stream));
    }

    // Wait for every GPU and release its resources
    for (auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
        CUDA_CHECK(cudaFree(w.dA));
        CUDA_CHECK(cudaFree(w.dB));
        CUDA_CHECK(cudaFree(w.dC));
        CUDA_CHECK(cudaStreamDestroy(w.stream));
    }

    CUDA_CHECK(cudaHostUnregister(const_cast<double*>(A.data())));
    CUDA_CHECK(cudaHostUnregister(const_cast<double*>(B.data())));
    CUDA_CHECK(cudaHostUnregister(C.data()));
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

    initDevices(getDeviceCount());
    
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
