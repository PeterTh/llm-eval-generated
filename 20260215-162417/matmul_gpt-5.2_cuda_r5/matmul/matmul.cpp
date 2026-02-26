#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

static inline void checkCuda(const cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static inline void checkCublas(const cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error (%s): status=%d\n", what, static_cast<int>(st));
        std::exit(1);
    }
}

// GPU matmul via cuBLAS DGEMM.
// Inputs/outputs are row-major; cuBLAS is column-major.
// We compute: (A*B)^T = B^T * A^T in column-major, which matches row-major storage of (A*B).
float matrixMultiplyCuda(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N) {
    if (N == 0) return 0.0f;

    const size_t bytes = N * N * sizeof(double);
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dA), bytes), "cudaMalloc(dA)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dB), bytes), "cudaMalloc(dB)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dC), bytes), "cudaMalloc(dC)");

    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "H2D B");

    cublasHandle_t handle{};
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetStream(handle, 0), "cublasSetStream");

    cudaEvent_t start{}, stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const double alpha = 1.0;
    const double beta = 0.0;

    // Time only the GEMM execution (excludes allocations and host/device copies).
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord(start)");

    // Column-major: Cc = Bc * Ac  ==> (A*B)^T
    // Memory of Cc (column-major of (A*B)^T) equals row-major of (A*B).
    checkCublas(
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(N), static_cast<int>(N),
                    &alpha,
                    dB, static_cast<int>(N),
                    dA, static_cast<int>(N),
                    &beta,
                    dC, static_cast<int>(N)),
        "cublasDgemm");

    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "cudaEventElapsedTime");

    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "D2H C");

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    checkCublas(cublasDestroy(handle), "cublasDestroy");

    checkCuda(cudaFree(dA), "cudaFree(dA)");
    checkCuda(cudaFree(dB), "cudaFree(dB)");
    checkCuda(cudaFree(dC), "cudaFree(dC)");

    return elapsedMs;
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    // Check a few deterministic positions
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

    // Perform matrix multiplication (GPU)
    printf("Computing matrix multiplication...\n");

    // Ensure a CUDA device is present/selected (unconditional GPU path).
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice(0)");

    const float durationMs = matrixMultiplyCuda(A, B, C, N);

    printf("Computation time: %.3f ms\n", durationMs);

    // Calculate GFLOPS
    const double seconds = static_cast<double>(durationMs) / 1000.0;
    const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) /
                          (seconds > 0.0 ? seconds : 1e-12) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(A, B, C, N);

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
