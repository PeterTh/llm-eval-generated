#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
}
static void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(static_cast<int>(status)));
}

__global__ void addDiagonal(double* matrix, int n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) matrix[i * n + i] += n;
}

// cuSOLVER treats the row-major matrix as column-major, so its upper
// triangle is the lower triangle of the row-major result.
__global__ void clearUpper(double* matrix, int n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * n;
    if (idx < total && idx % n > idx / n) matrix[idx] = 0.0;
}

class GpuMatrix {
public:
    explicit GpuMatrix(int size) : n(size), count(static_cast<size_t>(size) * size) {
        checkBlas(cublasCreate(&blas));
        checkSolver(cusolverDnCreate(&solver));
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&matrix), count * sizeof(double)));
    }
    ~GpuMatrix() {
        if (matrix) cudaFree(matrix);
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
    }
    GpuMatrix(const GpuMatrix&) = delete;
    GpuMatrix& operator=(const GpuMatrix&) = delete;

    void generate() {
        if (!count) return;
        std::vector<double> b(count);
        unsigned int seed = 42;
        for (double& x : b) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        double* deviceB = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), count * sizeof(double)));
        checkCuda(cudaMemcpy(deviceB, b.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        // B is row-major; B^T * B in column-major is B * B^T in row-major.
        checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                              &one, deviceB, n, deviceB, n, &zero, matrix, n));
        addDiagonal<<<(n + 255LL) / 256, 256>>>(matrix, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaFree(deviceB));
    }

    bool decompose() {
        if (!count) return true;
        int workSize = 0;
        checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               n, matrix, n, &workSize));
        double* work = nullptr;
        int* deviceInfo = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&work), static_cast<size_t>(workSize) * sizeof(double)));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int)));
        checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, matrix, n,
                                    work, workSize, deviceInfo));
        int info = 0;
        checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaFree(deviceInfo));
        checkCuda(cudaFree(work));
        if (info > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
            return false;
        }
        if (info < 0) throw std::runtime_error("cuSOLVER potrf invalid argument " + std::to_string(-info));
        clearUpper<<<(count + 255) / 256, 256>>>(matrix, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        return true;
    }

    std::vector<double> copyToHost() const {
        std::vector<double> result(count);
        if (count) checkCuda(cudaMemcpy(result.data(), matrix, count * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    bool validate(const std::vector<double>& original) {
        std::vector<double> reconstructed(count);
        if (count) {
            double* deviceResult = nullptr;
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceResult), count * sizeof(double)));
            const double one = 1.0, zero = 0.0;
            checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                                  &one, matrix, n, matrix, n, &zero, deviceResult, n));
            checkCuda(cudaMemcpy(reconstructed.data(), deviceResult, count * sizeof(double), cudaMemcpyDeviceToHost));
            checkCuda(cudaFree(deviceResult));
        }
        double maxError = 0.0, relError = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double error = std::fabs(reconstructed[i] - original[i]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i]) + 1e-10));
        }
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
        return true;
    }
private:
    int n;
    size_t count;
    double* matrix = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > std::numeric_limits<int>::max()) {
                printf("Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
            if (n != 0 && static_cast<size_t>(n) >
                    std::numeric_limits<size_t>::max() / sizeof(double) / static_cast<size_t>(n)) {
                printf("Matrix size is too large: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    try {
        GpuMatrix A(n);
        printf("Generating positive definite matrix...\n");
        A.generate();
        std::vector<double> original;
        if (validate) original = A.copyToHost();

        printf("Computing Cholesky decomposition...\n");
        auto start = std::chrono::high_resolution_clock::now();
        bool success = A.decompose();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (!success) { printf("Cholesky decomposition failed\n"); return 1; }
        printf("Computation time: %ld ms\n", duration.count());
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A.copyToHost(), "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            bool valid = A.validate(original);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "CUDA error: %s\n", e.what());
        return 1;
    }
    return 0;
}
