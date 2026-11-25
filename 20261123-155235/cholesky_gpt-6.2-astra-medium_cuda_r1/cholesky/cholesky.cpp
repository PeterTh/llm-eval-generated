#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <climits>
#include <limits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace {

template <class Status>
void check(Status status, const char* operation) {
    if (status != 0) {
        fprintf(stderr, "%s failed (status %d)\n", operation, static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), #call)

template <class T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        if (count) CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

#ifdef __GLIBC__
// glibc rand_r advances its 32-bit LCG three times per sample.
// Compose affine maps by repeated squaring to start each chunk at the
// same state as the original sequential, row-major random stream.
__device__ unsigned int seedAt(size_t sample) {
    unsigned int seed = 42, scale = 1103515245u, offset = 12345u;
    for (size_t steps = 3 * sample; steps; steps >>= 1) {
        if (steps & 1) seed = scale * seed + offset;
        offset *= scale + 1;
        scale *= scale;
    }
    return seed;
}

__global__ void generateRandom(double* b, int n) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    constexpr int chunk = 64;
    const int first = blockIdx.y * chunk;
    unsigned int seed = seedAt(row * n + first);
    for (int col = first; col < min(n, first + chunk); ++col) {
        seed = seed * 1103515245u + 12345u;
        unsigned int value = (seed >> 16) & 2047u;
        for (int digit = 0; digit < 2; ++digit) {
            seed = seed * 1103515245u + 12345u;
            value = (value << 10) | ((seed >> 16) & 1023u);
        }
        b[size_t(col) * n + row] = double(value) / RAND_MAX - 0.5;
    }
}
#endif

void generateMatrix(cublasHandle_t blas, double* matrix, int n) {
    DeviceBuffer<double> random(size_t(n) * n);
#ifdef __GLIBC__
    generateRandom<<<dim3((n + 127) / 128, (n + 63) / 64), 128>>>(random.data, n);
    CHECK(cudaGetLastError());
    const auto op = CUBLAS_OP_N;
#else
    std::vector<double> host(size_t(n) * n);
    unsigned int seed = 42;
    for (double& value : host) value = double(rand_r(&seed)) / RAND_MAX - 0.5;
    CHECK(cudaMemcpy(random.data, host.data(), host.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    const auto op = CUBLAS_OP_T;
#endif
    const double one = 1, zero = 0;
    // Column-major upper storage is the benchmark's row-major lower storage.
    CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, op, n, n, &one,
                      random.data, n, &zero, matrix, n));
}

__global__ void shiftDiagonal(double* matrix, int n) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) matrix[row * n + row] += n;
}

__global__ void clearUpper(double* matrix, int n) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < col && col < n) matrix[row * n + col] = 0;
}

constexpr int threads = 256;
__global__ void measureErrors(const double* reconstructed, const double* original,
                              size_t n, double2* partial) {
    double2 error = {0, 0};
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n * n;
         i += size_t(gridDim.x) * blockDim.x) {
        if (i % n <= i / n) {
            double diff = fabs(reconstructed[i] - original[i]);
            error.x = fmax(error.x, diff);
            error.y = fmax(error.y, diff / (fabs(original[i]) + 1e-10));
        }
    }
    __shared__ double2 maxima[threads];
    maxima[threadIdx.x] = error;
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x].x = fmax(maxima[threadIdx.x].x, maxima[threadIdx.x + stride].x);
            maxima[threadIdx.x].y = fmax(maxima[threadIdx.x].y, maxima[threadIdx.x + stride].y);
        }
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x] = maxima[0];
}

bool validateCholesky(cublasHandle_t blas, const double* factor, const double* original, int n) {
    const size_t count = size_t(n) * n;
    DeviceBuffer<double> reconstructed(count);
    const double one = 1, zero = 0;
    CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n, &one,
                      factor, n, &zero, reconstructed.data, n));
    const int blocks = std::min<size_t>(1024, (count + threads - 1) / threads);
    DeviceBuffer<double2> partial(blocks);
    measureErrors<<<blocks, threads>>>(reconstructed.data, original, n, partial.data);
    CHECK(cudaGetLastError());
    std::vector<double2> host(blocks);
    CHECK(cudaMemcpy(host.data(), partial.data, blocks * sizeof(double2), cudaMemcpyDeviceToHost));
    double maxError = 0, relError = 0;
    for (auto error : host) {
        maxError = std::max(maxError, error.x);
        relError = std::max(relError, error.y);
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

} // namespace

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
    
    // Parse command line arguments
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (n > INT_MAX || (n && n > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is out of range\n");
        return 1;
    }
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));

    const size_t count = n * n;
    DeviceBuffer<double> matrix(count), original(validate ? count : 0);
    printf("Generating positive definite matrix...\n");
    if (n) {
        generateMatrix(blas, matrix.data, n);
        shiftDiagonal<<<(n + threads - 1) / threads, threads>>>(matrix.data, n);
        CHECK(cudaGetLastError());
        if (validate) CHECK(cudaMemcpy(original.data, matrix.data, count * sizeof(double),
                                       cudaMemcpyDeviceToDevice));
    }

    int workSize = 0;
    if (n) CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n,
                                             matrix.data, n, &workSize));
    DeviceBuffer<double> work(workSize);
    DeviceBuffer<int> info(1);
    CHECK(cudaDeviceSynchronize());

    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::steady_clock::now();
    int status = 0;
    if (n) {
        CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, matrix.data, n,
                               work.data, workSize, info.data));
        clearUpper<<<dim3((n + 31) / 32, (n + 7) / 8), dim3(32, 8)>>>(matrix.data, n);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&status, info.data, sizeof(status), cudaMemcpyDeviceToHost));
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    if (status != 0) {
        if (status > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n",
                               status - 1);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    printf("Computation time: %.3f ms\n", elapsed.count() * 1000);
    printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * elapsed.count()));

    if (printResults) {
        std::vector<double> host(count);
        if (n) CHECK(cudaMemcpy(host.data(), matrix.data, count * sizeof(double),
                                cudaMemcpyDeviceToHost));
        print_results(host, "CholeskyL");
    }
    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        if (n) valid = validateCholesky(blas, matrix.data, original.data, n);
        else printf("Max absolute error: %.10e\nMax relative error: %.10e\n", 0.0, 0.0);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return valid ? 0 : 1;
}
