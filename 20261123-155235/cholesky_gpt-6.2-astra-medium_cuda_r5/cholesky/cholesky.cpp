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

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename Status>
static void check(Status status, const char* operation) {
    if (status != 0) {
        fprintf(stderr, "%s failed (status %d)\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)), "Allocate GPU memory");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

// A row-major B is seen as B^T by cuBLAS. The column-major upper
// triangle of its Gram matrix is the row-major lower triangle we need.
static void gram(cublasHandle_t blas, int n, const double* b, double* a) {
    const double one = 1.0, zero = 0.0;
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n,
                      &one, b, n, &zero, a, n), "Compute symmetric product");
}

__global__ void addDiagonal(double* a, int n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) a[i * n + i] += n;
}

static void generateMatrix(cublasHandle_t blas, int n, double* a) {
    const size_t count = static_cast<size_t>(n) * n;
    DeviceBuffer<double> b(count);
    {
        std::vector<double> random(count);
        unsigned int seed = 42;
        for (double& value : random) value = rand_r(&seed) / double(RAND_MAX) - 0.5;
        check(cudaMemcpy(b.data, random.data(), count * sizeof(double),
                         cudaMemcpyHostToDevice), "Upload random matrix");
    }
    gram(blas, n, b.data, a);
    addDiagonal<<<(n + 255u) / 256, 256>>>(a, n);
    check(cudaGetLastError(), "Add diagonal");
}

__global__ void clearUpper(double* factor, int n) {
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < col && col < static_cast<unsigned>(n))
        factor[static_cast<size_t>(row) * n + col] = 0.0;
}

// Both matrices are symmetric, so comparing the stored triangle gives
// the same maxima as comparing the full reconstruction.
__global__ void measureErrors(const double* reconstructed, const double* original,
                              int n, size_t count, unsigned long long* maxima) {
    double absolute = 0.0, relative = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        if (i % n <= i / n) {
            const double error = fabs(reconstructed[i] - original[i]);
            absolute = fmax(absolute, error);
            relative = fmax(relative, error / (fabs(original[i]) + 1e-10));
        }
    }
    for (int offset = 16; offset; offset /= 2) {
        absolute = fmax(absolute, __shfl_down_sync(0xffffffff, absolute, offset));
        relative = fmax(relative, __shfl_down_sync(0xffffffff, relative, offset));
    }
    if (threadIdx.x % 32 == 0) {
        // Nonnegative doubles have monotonically ordered IEEE bit patterns.
        atomicMax(maxima, static_cast<unsigned long long>(__double_as_longlong(absolute)));
        atomicMax(maxima + 1, static_cast<unsigned long long>(__double_as_longlong(relative)));
    }
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
    int n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            const long size = std::strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || size < 0 || size > INT_MAX ||
                static_cast<size_t>(size) > SIZE_MAX / sizeof(double) /
                    static_cast<size_t>(std::max(1L, size))) {
                fprintf(stderr, "Invalid matrix size\n");
                return 1;
            }
            n = static_cast<int>(size);
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
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(double);
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    check(cublasCreate(&blas), "Initialize cuBLAS");
    check(cusolverDnCreate(&solver), "Initialize cuSOLVER");

    DeviceBuffer<double> factor(count);
    DeviceBuffer<double> original(validate ? count : 0);
    printf("Generating positive definite matrix...\n");
    int workspaceSize = 0;
    if (n) {
        generateMatrix(blas, n, factor.data);
        if (validate)
            check(cudaMemcpy(original.data, factor.data, bytes, cudaMemcpyDeviceToDevice),
                  "Save input matrix");
        check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n, factor.data,
                                          n, &workspaceSize), "Query Cholesky workspace");
    }
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> info(1);
    check(cudaDeviceSynchronize(), "Finish matrix generation");

    printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::steady_clock::now();
    if (n) {
        check(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, factor.data, n,
                               workspace.data, workspaceSize, info.data), "Factor matrix");
        const dim3 threads(32, 8);
        clearUpper<<<dim3((n + 31u) / 32, (n + 7u) / 8), threads>>>(factor.data, n);
        check(cudaGetLastError(), "Clear upper triangle");
        int status;
        check(cudaMemcpy(&status, info.data, sizeof(status), cudaMemcpyDeviceToHost),
              "Read factorization status");
        if (status != 0) {
            if (status > 0)
                printf("Error: Matrix is not positive definite at diagonal element %d\n", status - 1);
            printf("Cholesky decomposition failed\n");
            return 1;
        }
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", elapsed.count() * 1000);
    printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * elapsed.count()));

    if (printResults) {
        std::vector<double> host(count);
        if (n) check(cudaMemcpy(host.data(), factor.data, bytes, cudaMemcpyDeviceToHost),
                     "Download Cholesky factor");
        print_results(host, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        double errors[2] = {};
        if (n) {
            DeviceBuffer<double> reconstructed(count);
            gram(blas, n, factor.data, reconstructed.data);
            DeviceBuffer<unsigned long long> maxima(2);
            check(cudaMemset(maxima.data, 0, sizeof(errors)), "Initialize error maxima");
            const int blocks = static_cast<int>(std::min<size_t>(4096, (count + 255) / 256));
            measureErrors<<<blocks, 256>>>(reconstructed.data, original.data, n, count, maxima.data);
            check(cudaGetLastError(), "Compare reconstruction");
            check(cudaMemcpy(errors, maxima.data, sizeof(errors), cudaMemcpyDeviceToHost),
                  "Read validation errors");
        }
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        valid = errors[1] <= 1e-6;
        if (!valid) printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    check(cusolverDnDestroy(solver), "Destroy cuSOLVER handle");
    check(cublasDestroy(blas), "Destroy cuBLAS handle");
    return valid ? 0 : 1;
}
