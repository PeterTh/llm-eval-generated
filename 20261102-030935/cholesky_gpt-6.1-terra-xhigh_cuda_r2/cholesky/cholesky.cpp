#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
}
static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        exit(1);
    }
}
static void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER error: %d\n", static_cast<int>(status));
        exit(1);
    }
}

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_;
};

__global__ void addDiagonal(double* a, int n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) a[i * (static_cast<size_t>(n) + 1)] += n;
}

__global__ void clearUpper(double* a, int n) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(n) * n;
    if (index < count && index % n > index / n) a[index] = 0.0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    if (n > INT_MAX || (n && n > SIZE_MAX / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("Generating positive definite matrix...\n");

    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    checkBlas(cublasCreate(&blas));
    checkSolver(cusolverDnCreate(&solver));
    const int dim = static_cast<int>(n);
    const size_t count = n * n;
    const size_t bytes = count * sizeof(double);
    DeviceBuffer<double> a(count);
    DeviceBuffer<double> original(validate ? count : 0);
    const double one = 1.0, zero = 0.0;
    if (n) {
        // rand_r is kept on the host to preserve the original seeded input sequence.
        std::vector<double> b(count);
        unsigned int seed = 42;
        for (size_t i = 0; i < count; ++i)
            b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        DeviceBuffer<double> deviceB(count);
        checkCuda(cudaMemcpy(deviceB.get(), b.data(), bytes, cudaMemcpyHostToDevice));
        // Row-major B is seen by cuBLAS as B^T in column-major order.
        checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                              &one, deviceB.get(), dim, deviceB.get(), dim,
                              &zero, a.get(), dim));
        addDiagonal<<<(n + 255) / 256, 256>>>(a.get(), dim);
        checkCuda(cudaGetLastError());
        if (validate) checkCuda(cudaMemcpy(original.get(), a.get(), bytes, cudaMemcpyDeviceToDevice));
        checkCuda(cudaDeviceSynchronize());
    }

    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    int info = 0;
    if (n) {
        int workspaceSize = 0;
        // The upper triangle in column-major storage is the row-major lower triangle.
        checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                                 dim, a.get(), dim, &workspaceSize));
        DeviceBuffer<double> workspace(workspaceSize);
        DeviceBuffer<int> deviceInfo(1);
        checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, dim, a.get(), dim,
                                    workspace.get(), workspaceSize, deviceInfo.get()));
        checkCuda(cudaMemcpy(&info, deviceInfo.get(), sizeof(info), cudaMemcpyDeviceToHost));
        if (info == 0) {
            clearUpper<<<(count + 255) / 256, 256>>>(a.get(), dim);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (info != 0) {
        if (info > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        else fprintf(stderr, "cuSOLVER: invalid argument %d\n", -info);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    printf("Computation time: %ld ms\n", duration.count());
    double ops = static_cast<double>(n) * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        std::vector<double> hostA(count);
        if (n) checkCuda(cudaMemcpy(hostA.data(), a.get(), bytes, cudaMemcpyDeviceToHost));
        print_results(hostA, "CholeskyL");
    }
    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> reconstructed(count), hostOriginal(count);
        if (n) {
            DeviceBuffer<double> product(count);
            // The row-major lower factor is column-major L^T.
            checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                                  &one, a.get(), dim, a.get(), dim,
                                  &zero, product.get(), dim));
            checkCuda(cudaMemcpy(reconstructed.data(), product.get(), bytes, cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hostOriginal.data(), original.get(), bytes, cudaMemcpyDeviceToHost));
        }
        double maxError = 0.0, relError = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double error = fabs(reconstructed[i] - hostOriginal[i]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(hostOriginal[i]) + 1e-10));
        }
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            valid = false;
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    checkSolver(cusolverDnDestroy(solver));
    checkBlas(cublasDestroy(blas));
    return valid ? 0 : 1;
}
