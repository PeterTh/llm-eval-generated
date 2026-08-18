#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFailure(const char* operation, const char* message) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, message);
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFailure(operation, cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        cudaFailure(operation, message);
    }
}

// Owns all GPU state so setup and input transfers can be kept outside the
// compute-only timing interval.
class CudaMatmul {
public:
    CudaMatmul(const std::vector<double>& A, const std::vector<double>& B, size_t N)
        : n_(static_cast<int>(N)), bytes_(N * N * sizeof(double)) {
        checkCublas(cublasCreate(&handle_), "creating cuBLAS handle");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_a_), bytes_), "allocating matrix A");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_b_), bytes_), "allocating matrix B");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_c_), bytes_), "allocating matrix C");
        checkCuda(cudaMemcpy(d_a_, A.data(), bytes_, cudaMemcpyHostToDevice),
                  "copying matrix A to the GPU");
        checkCuda(cudaMemcpy(d_b_, B.data(), bytes_, cudaMemcpyHostToDevice),
                  "copying matrix B to the GPU");
    }

    CudaMatmul(const CudaMatmul&) = delete;
    CudaMatmul& operator=(const CudaMatmul&) = delete;

    ~CudaMatmul() {
        if (d_c_ != nullptr) cudaFree(d_c_);
        if (d_b_ != nullptr) cudaFree(d_b_);
        if (d_a_ != nullptr) cudaFree(d_a_);
        if (handle_ != nullptr) cublasDestroy(handle_);
    }

    void multiply() {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;

        // cuBLAS is column-major. A row-major buffer is its own transpose when
        // viewed as column-major, so B*A here produces (A*B)^T in that view,
        // which has exactly the desired row-major byte layout.
        checkCublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                n_, n_, n_, &alpha,
                                d_b_, n_, d_a_, n_, &beta, d_c_, n_),
                    "multiplying matrices");
        checkCuda(cudaDeviceSynchronize(), "synchronizing matrix multiplication");
    }

    void copyResult(std::vector<double>& C) const {
        checkCuda(cudaMemcpy(C.data(), d_c_, bytes_, cudaMemcpyDeviceToHost),
                  "copying result from the GPU");
    }

private:
    cublasHandle_t handle_ = nullptr;
    double* d_a_ = nullptr;
    double* d_b_ = nullptr;
    double* d_c_ = nullptr;
    int n_;
    size_t bytes_;
};

}  // namespace

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
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    std::printf("Matrix Multiplication Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", N, N);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    std::printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Device setup and transfers are intentionally excluded, as the original
    // benchmark timed only the matrix multiplication itself.
    CudaMatmul gpu(A, B, N);
    // Trigger CUDA's lazy module loading and let cuBLAS select/cache its GEMM
    // implementation before measuring steady-state multiplication throughput.
    gpu.multiply();
    std::printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    gpu.multiply();
    const auto end = std::chrono::high_resolution_clock::now();
    gpu.copyResult(C);

    const std::chrono::duration<double> duration = end - start;
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    std::printf("Computation time: %lld ms\n", static_cast<long long>(milliseconds));

    const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                           static_cast<double>(N)) /
                          duration.count() / 1e9;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(C, "MatrixC");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(A, B, C, N)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
