#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status)); } while (0)
#define CUBLAS_CHECK(call) do { cublasStatus_t status = (call); if (status != CUBLAS_STATUS_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " failed with status " + std::to_string(status)); } while (0)
#define CUSOLVER_CHECK(call) do { cusolverStatus_t status = (call); if (status != CUSOLVER_STATUS_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " failed with status " + std::to_string(status)); } while (0)

// These wrappers keep device allocations and library handles exception-safe.
template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_;
};

class Handles {
public:
    Handles() : blas_(nullptr), solver_(nullptr) {
        CUBLAS_CHECK(cublasCreate(&blas_));
        try { CUSOLVER_CHECK(cusolverDnCreate(&solver_)); }
        catch (...) { cublasDestroy(blas_); throw; }
    }
    ~Handles() { cusolverDnDestroy(solver_); cublasDestroy(blas_); }
    cublasHandle_t blas() const { return blas_; }
    cusolverDnHandle_t solver() const { return solver_; }
private:
    cublasHandle_t blas_;
    cusolverDnHandle_t solver_;
};

__global__ void addDiagonal(double* matrix, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        matrix[i * n + i] += static_cast<double>(n);
    }
}

__global__ void clearUpper(double* matrix, size_t n) {
    const size_t count = n * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (i % n > i / n) matrix[i] = 0.0;
    }
}

static int blocksFor(size_t count) {
    return static_cast<int>(std::min<size_t>((count + 255) / 256, 65535));
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
            int value = atoi(argv[++i]);
            if (value < 0) { fprintf(stderr, "Invalid matrix size\n"); return 1; }
            n = static_cast<size_t>(value);
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

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("Generating positive definite matrix...\n");

    try {
        if (n > INT_MAX || (n && n > SIZE_MAX / n / sizeof(double)))
            throw std::runtime_error("Matrix size is too large");
        const size_t count = n * n;
        const size_t bytes = count * sizeof(double);
        Handles handles;
        DeviceBuffer<double> matrix(count);
        DeviceBuffer<double> original(validate ? count : 0);
        // Preserve the original rand_r sequence and B * B^T construction.
        if (n) {
            std::vector<double> B(count);
            unsigned int seed = 42;
            for (double& value : B) value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
            DeviceBuffer<double> deviceB(count);
            CUDA_CHECK(cudaMemcpy(deviceB.get(), B.data(), bytes, cudaMemcpyHostToDevice));
            const double one = 1.0, zero = 0.0;
            // Row-major B is viewed as B^T by column-major cuBLAS.
            CUBLAS_CHECK(cublasDgemm(handles.blas(), CUBLAS_OP_T, CUBLAS_OP_N,
                                     static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                                     &one, deviceB.get(), static_cast<int>(n), deviceB.get(),
                                     static_cast<int>(n), &zero, matrix.get(), static_cast<int>(n)));
            addDiagonal<<<blocksFor(n), 256>>>(matrix.get(), n);
            CUDA_CHECK(cudaGetLastError());
            if (validate) CUDA_CHECK(cudaMemcpy(original.get(), matrix.get(), bytes, cudaMemcpyDeviceToDevice));
        }
        int workspaceSize = 0;
        if (n) CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handles.solver(), CUBLAS_FILL_MODE_UPPER,
                            static_cast<int>(n), matrix.get(), static_cast<int>(n), &workspaceSize));
        DeviceBuffer<double> workspace(workspaceSize);
        DeviceBuffer<int> info(n ? 1 : 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Computing Cholesky decomposition...\n");
        auto start = std::chrono::high_resolution_clock::now();
        int hostInfo = 0;
        if (n) {
            // The column-major upper triangle corresponds to row-major lower L.
            CUSOLVER_CHECK(cusolverDnDpotrf(handles.solver(), CUBLAS_FILL_MODE_UPPER,
                          static_cast<int>(n), matrix.get(), static_cast<int>(n),
                          workspace.get(), workspaceSize, info.get()));
            CUDA_CHECK(cudaMemcpy(&hostInfo, info.get(), sizeof(hostInfo), cudaMemcpyDeviceToHost));
            if (hostInfo == 0) {
                clearUpper<<<blocksFor(count), 256>>>(matrix.get(), n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        if (hostInfo != 0) {
            if (hostInfo > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
            else fprintf(stderr, "cuSOLVER rejected argument %d\n", -hostInfo);
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        double ops = static_cast<double>(n) * n * n / 3.0;
        double seconds = std::chrono::duration<double>(end - start).count();
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            std::vector<double> result(count);
            if (n) CUDA_CHECK(cudaMemcpy(result.data(), matrix.get(), bytes, cudaMemcpyDeviceToHost));
            print_results(result, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> reconstructed(count), originalHost(count);
            if (n) {
                DeviceBuffer<double> product(count);
                const double one = 1.0, zero = 0.0;
                // In column-major storage, L is L^T, so calculate (L^T)^T * L^T.
                CUBLAS_CHECK(cublasDgemm(handles.blas(), CUBLAS_OP_T, CUBLAS_OP_N,
                             static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                             &one, matrix.get(), static_cast<int>(n), matrix.get(), static_cast<int>(n),
                             &zero, product.get(), static_cast<int>(n)));
                CUDA_CHECK(cudaMemcpy(reconstructed.data(), product.get(), bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(originalHost.data(), original.get(), bytes, cudaMemcpyDeviceToHost));
            }
            double maxError = 0.0, relError = 0.0;
            for (size_t i = 0; i < count; ++i) {
                double error = fabs(reconstructed[i] - originalHost[i]);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (fabs(originalHost[i]) + 1e-10));
            }
            printf("Max absolute error: %.10e\n", maxError);
            printf("Max relative error: %.10e\n", relError);
            if (relError > 1e-6) {
                printf("Validation failed: relative error too large\nValidation: FAILED\n");
                return 1;
            }
            printf("Validation: PASSED\n");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
