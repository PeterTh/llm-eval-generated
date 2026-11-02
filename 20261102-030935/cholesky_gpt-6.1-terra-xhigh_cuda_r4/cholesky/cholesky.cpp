#include <algorithm>
#include <chrono>
#include <climits>
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
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
}
static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
static void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error: " + std::to_string(static_cast<int>(status)));
}

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_;
};

class Handles {
public:
    Handles() : blas(nullptr), solver(nullptr) {
        checkBlas(cublasCreate(&blas));
        try { checkSolver(cusolverDnCreate(&solver)); }
        catch (...) { cublasDestroy(blas); throw; }
    }
    ~Handles() { cusolverDnDestroy(solver); cublasDestroy(blas); }
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
};

__global__ void addDiagonal(double* a, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) a[i * n + i] += n;
}

__global__ void zeroUpper(double* a, int n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < static_cast<size_t>(n) && col < static_cast<size_t>(n) && col > row)
        a[row * n + col] = 0.0;
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
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || *arg == '\0' || *end != '\0' || value > INT_MAX ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
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

    try {
        Handles handles;
        const int dim = static_cast<int>(n);
        const size_t elements = n * n;
        const size_t bytes = elements * sizeof(double);
        DeviceBuffer<double> matrix(elements);
        DeviceBuffer<double> original(validate ? elements : 0);
        const double one = 1.0, zero = 0.0;

        printf("Generating positive definite matrix...\n");
        if (n) {
            std::vector<double> randomMatrix(elements);
            unsigned int seed = 42;
            for (double& value : randomMatrix)
                value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            DeviceBuffer<double> randomDevice(elements);
            checkCuda(cudaMemcpy(randomDevice.get(), randomMatrix.data(), bytes, cudaMemcpyHostToDevice));
            // Row-major B appears as B^T to cuBLAS.  B^T * B in column-major
            // therefore produces the same symmetric matrix as row-major B * B^T.
            checkBlas(cublasDgemm(handles.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  dim, dim, dim, &one, randomDevice.get(), dim,
                                  randomDevice.get(), dim, &zero, matrix.get(), dim));
            addDiagonal<<<(n + 255) / 256, 256>>>(matrix.get(), dim);
            checkCuda(cudaGetLastError());
            if (validate)
                checkCuda(cudaMemcpy(original.get(), matrix.get(), bytes, cudaMemcpyDeviceToDevice));
        }

        int workspaceSize = 0;
        if (n) checkSolver(cusolverDnDpotrf_bufferSize(handles.solver, CUBLAS_FILL_MODE_UPPER,
                                                       dim, matrix.get(), dim, &workspaceSize));
        DeviceBuffer<double> workspace(workspaceSize);
        DeviceBuffer<int> info(n ? 1 : 0);
        checkCuda(cudaDeviceSynchronize());
        printf("Computing Cholesky decomposition...\n");
        auto start = std::chrono::high_resolution_clock::now();
        int hostInfo = 0;
        if (n) {
            // An upper column-major factor is the required lower row-major factor.
            checkSolver(cusolverDnDpotrf(handles.solver, CUBLAS_FILL_MODE_UPPER, dim,
                                        matrix.get(), dim, workspace.get(), workspaceSize, info.get()));
            const dim3 threads(32, 8);
            const dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
            zeroUpper<<<blocks, threads>>>(matrix.get(), dim);
            checkCuda(cudaGetLastError());
            checkCuda(cudaMemcpy(&hostInfo, info.get(), sizeof(hostInfo), cudaMemcpyDeviceToHost));
        }
        std::vector<double> result;
        if (printResults && n && hostInfo == 0) {
            result.resize(elements);
            checkCuda(cudaMemcpy(result.data(), matrix.get(), bytes, cudaMemcpyDeviceToHost));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (hostInfo != 0) {
            if (hostInfo > 0)
                printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
            else
                fprintf(stderr, "cuSOLVER rejected argument %d\n", -hostInfo);
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        printf("Computation time: %ld ms\n", duration.count());
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GFLOPS\n", ops / seconds / 1e9);

        if (printResults) print_results(result, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            double maxError = 0.0, relError = 0.0;
            if (n) {
                DeviceBuffer<double> reconstructed(elements);
                checkBlas(cublasDgemm(handles.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      dim, dim, dim, &one, matrix.get(), dim,
                                      matrix.get(), dim, &zero, reconstructed.get(), dim));
                std::vector<double> actual(elements), expected(elements);
                checkCuda(cudaMemcpy(actual.data(), reconstructed.get(), bytes, cudaMemcpyDeviceToHost));
                checkCuda(cudaMemcpy(expected.data(), original.get(), bytes, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < elements; ++i) {
                    const double error = std::fabs(actual[i] - expected[i]);
                    maxError = std::max(maxError, error);
                    relError = std::max(relError, error / (std::fabs(expected[i]) + 1e-10));
                }
            }
            printf("Max absolute error: %.10e\n", maxError);
            printf("Max relative error: %.10e\n", relError);
            if (relError > 1e-6) {
                printf("Validation failed: relative error too large\nValidation: FAILED\n");
                return 1;
            }
            printf("Validation: PASSED\n");
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
