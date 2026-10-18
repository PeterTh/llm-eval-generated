#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

static void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

static void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
}

// Resource owners also release allocations if a later CUDA operation fails.
template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count = 0) {
        if (count) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

struct Blas {
    cublasHandle_t value = nullptr;
    Blas() { check(cublasCreate(&value)); }
    ~Blas() { cublasDestroy(value); }
};

struct Solver {
    cusolverDnHandle_t value = nullptr;
    Solver() { check(cusolverDnCreate(&value)); }
    ~Solver() { cusolverDnDestroy(value); }
};

__global__ void addDiagonal(double* a, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        a[i * n + i] += double(n);
}

__global__ void zeroUpper(double* a, size_t n) {
    const size_t count = n * n;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        if (i % n > i / n) a[i] = 0.0;
    }
}

// Each block reduces its errors before updating the two global maxima.
__global__ void compareMatrices(const double* reconstructed, const double* original,
                                size_t count, double* maxima) {
    __shared__ double absolute[256];
    __shared__ double relative[256];
    double a = 0.0, r = 0.0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x) {
        double error = fabs(reconstructed[i] - original[i]);
        if (!isfinite(error)) error = CUDART_INF;
        a = fmax(a, error);
        r = fmax(r, error / (fabs(original[i]) + 1e-10));
    }
    absolute[threadIdx.x] = a;
    relative[threadIdx.x] = r;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + offset]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // IEEE positive doubles have the same ordering as unsigned integers.
        atomicMax(reinterpret_cast<unsigned long long*>(maxima),
                  static_cast<unsigned long long>(__double_as_longlong(absolute[0])));
        atomicMax(reinterpret_cast<unsigned long long*>(maxima + 1),
                  static_cast<unsigned long long>(__double_as_longlong(relative[0])));
    }
}

static unsigned int blocksFor(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
}

// All cubic work stays on the GPU, and the matrix remains device-resident
// between generation, factorization and optional validation.
class CholeskyBenchmark {
    size_t n;
    size_t count;
    Stream stream;
    Blas blas;
    Solver solver;
    DeviceBuffer<double> matrix;
    DeviceBuffer<double> original;
    DeviceBuffer<int> info;
    int workspaceSize = 0;
    DeviceBuffer<double> workspace;

    int getWorkspaceSize() {
        if (n) check(cusolverDnDpotrf_bufferSize(solver.value, CUBLAS_FILL_MODE_UPPER,
                                                int(n), matrix.data, int(n), &workspaceSize));
        return workspaceSize;
    }

public:
    CholeskyBenchmark(size_t size, bool validate)
        : n(size), count(n * n), matrix(count), original(validate ? count : 0),
          info(1), workspace(getWorkspaceSize()) {
        check(cublasSetStream(blas.value, stream.value));
        check(cusolverDnSetStream(solver.value, stream.value));
        check(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH));
    }

    void generatePositiveDefiniteMatrix() {
        if (!n) return;
        // Preserve the original rand_r sequence exactly; only the O(n^2)
        // random-number generation is on the host.
        std::vector<double> b(count);
        unsigned int seed = 42;
        for (size_t i = 0; i < count; ++i)
            b[i] = (rand_r(&seed) / double(RAND_MAX)) - 0.5;

        DeviceBuffer<double> deviceB(count);
        check(cudaMemcpyAsync(deviceB.data, b.data(), count * sizeof(double),
                              cudaMemcpyHostToDevice, stream.value));
        const double one = 1.0, zero = 0.0;
        // Row-major B is column-major B^T. The symmetric product has the
        // same storage in either layout: A = B * B^T + n I.
        check(cublasDgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                          &one, deviceB.data, int(n), deviceB.data, int(n),
                          &zero, matrix.data, int(n)));
        addDiagonal<<<blocksFor(n), 256, 0, stream.value>>>(matrix.data, n);
        check(cudaGetLastError());
        if (original.data)
            check(cudaMemcpyAsync(original.data, matrix.data, count * sizeof(double),
                                  cudaMemcpyDeviceToDevice, stream.value));
        check(cudaStreamSynchronize(stream.value));
    }

    bool choleskyDecomposition() {
        if (!n) return true;
        // Column-major upper Cholesky is row-major lower Cholesky. cuSOLVER
        // uses a blocked factorization with parallel BLAS trailing updates.
        check(cusolverDnDpotrf(solver.value, CUBLAS_FILL_MODE_UPPER, int(n),
                               matrix.data, int(n), workspace.data, workspaceSize, info.data));
        zeroUpper<<<blocksFor(count), 256, 0, stream.value>>>(matrix.data, n);
        check(cudaGetLastError());
        int hostInfo = 0;
        check(cudaMemcpyAsync(&hostInfo, info.data, sizeof(int), cudaMemcpyDeviceToHost, stream.value));
        check(cudaStreamSynchronize(stream.value));
        if (hostInfo < 0)
            throw std::runtime_error("cuSOLVER rejected argument " + std::to_string(-hostInfo));
        if (hostInfo > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
            return false;
        }
        return true;
    }

    std::vector<double> download() {
        std::vector<double> result(count);
        if (count) {
            check(cudaMemcpyAsync(result.data(), matrix.data, count * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream.value));
            check(cudaStreamSynchronize(stream.value));
        }
        return result;
    }

    bool validateCholesky() {
        double errors[2] = {0.0, 0.0};
        if (n) {
            DeviceBuffer<double> reconstructed(count);
            DeviceBuffer<double> maxima(2);
            check(cudaMemsetAsync(maxima.data, 0, sizeof(errors), stream.value));
            const double one = 1.0, zero = 0.0;
            // The stored column-major matrix is U = L^T: reconstruct U^T U.
            check(cublasDgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                              &one, matrix.data, int(n), matrix.data, int(n),
                              &zero, reconstructed.data, int(n)));
            compareMatrices<<<blocksFor(count), 256, 0, stream.value>>>(
                reconstructed.data, original.data, count, maxima.data);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(errors, maxima.data, sizeof(errors),
                                  cudaMemcpyDeviceToHost, stream.value));
            check(cudaStreamSynchronize(stream.value));
        }
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
        return true;
    }
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
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(text, &end, 10);
            if (errno || text == end || *end || *text == '-' || parsed > INT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
        CholeskyBenchmark benchmark(n, validate);
        printf("Generating positive definite matrix...\n");
        benchmark.generatePositiveDefiniteMatrix();

        printf("Computing Cholesky decomposition...\n");
        const auto start = std::chrono::steady_clock::now();
        const bool success = benchmark.choleskyDecomposition();
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        if (!success) {
            printf("Cholesky decomposition failed\n");
            return 1;
        }

        // Decomposition synchronizes the stream, so this measures completed
        // GPU work, including dispatch and status checking, not just launches.
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = double(n) * double(n) * double(n) / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);

        if (printResults) print_results(benchmark.download(), "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = benchmark.validateCholesky();
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
    return 0;
}
