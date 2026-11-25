#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "../common/results_output.hpp"

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class Status>
void check(Status status, const char* operation) {
    if (static_cast<int>(status) != 0)
        throw std::runtime_error(std::string(operation) + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
}

template<class T>
class DeviceBuffer {
public:
    T* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// A row-major lower triangle is an upper triangle to column-major CUDA libraries.
void gramMatrix(cublasHandle_t blas, const double* rows, double* gram, int n) {
    const double one = 1.0, zero = 0.0;
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n,
                      &one, rows, n, &zero, gram, n), "cublasDsyrk");
}

__global__ void shiftDiagonal(double* matrix, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) matrix[static_cast<size_t>(i) * n + i] += n;
}

#if defined(__GLIBC__)
// glibc rand_r consumes three steps of a 32-bit linear congruential generator.
// Compose its affine map by repeated squaring to start each lane at its index.
__device__ constexpr unsigned advanceSeed(unsigned seed, size_t steps) {
    unsigned multiplier = 1103515245u, increment = 12345u;
    while (steps) {
        if (steps & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
        steps >>= 1;
    }
    return seed;
}

constexpr int randomThreads = 256;
constexpr int randomRounds = 16;

__global__ void fillRandom(double* values, size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * randomThreads * randomRounds + threadIdx.x;
    unsigned seed = advanceSeed(42, 3 * index);
    // Jump from the end of one draw to the start of this lane's next draw.
    constexpr size_t skip = 3 * (randomThreads - 1);
    constexpr unsigned add = advanceSeed(0, skip);
    constexpr unsigned mul = advanceSeed(1, skip) - add;
    for (int round = 0; round < randomRounds && index < count; ++round, index += randomThreads) {
        seed = advanceSeed(seed, 1);
        unsigned value = (seed >> 16) & 2047;
        seed = advanceSeed(seed, 1);
        value = (value << 10) | ((seed >> 16) & 1023);
        seed = advanceSeed(seed, 1);
        value = (value << 10) | ((seed >> 16) & 1023);
        values[index] = __ddiv_rn(static_cast<double>(value), RAND_MAX) - 0.5;
        seed = mul * seed + add;
    }
}
#endif

void generatePositiveDefiniteMatrix(cublasHandle_t blas, double* matrix, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    DeviceBuffer<double> random(count);
#if defined(__GLIBC__)
    constexpr int perBlock = randomThreads * randomRounds;
    fillRandom<<<(count + perBlock - 1) / perBlock, randomThreads>>>(random.data, count);
    check(cudaGetLastError(), "fill random matrix");
#else
    // Other C libraries can implement a different rand_r sequence.
    std::vector<double> host(count);
    unsigned int seed = 42;
    for (double& value : host) value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    check(cudaMemcpy(random.data, host.data(), count * sizeof(double), cudaMemcpyHostToDevice),
          "upload random matrix");
#endif
    gramMatrix(blas, random.data, matrix, n);
    shiftDiagonal<<<(n + 255) / 256, 256>>>(matrix, n);
    check(cudaGetLastError(), "shift diagonal");
}

__global__ void clearUpperTriangle(double* matrix, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < col && col < n) matrix[static_cast<size_t>(row) * n + col] = 0.0;
}

struct Errors {
    double absolute, relative;
};

struct ElementError {
    const double* reconstructed;
    const double* original;
    int n;

    __device__ Errors operator()(size_t index) const {
        if (index % n > index / n) return {0.0, 0.0};
        const double error = fabs(reconstructed[index] - original[index]);
        return {error, error / (fabs(original[index]) + 1e-10)};
    }
};

struct MaxErrors {
    __host__ __device__ Errors operator()(Errors a, Errors b) const {
        return {fmax(a.absolute, b.absolute), fmax(a.relative, b.relative)};
    }
};

bool validateCholesky(cublasHandle_t blas, const double* factor, const double* original, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    DeviceBuffer<double> reconstructed(count);
    gramMatrix(blas, factor, reconstructed.data, n);
    // Both matrices are symmetric, so their lower triangles contain every error.
    const auto first = thrust::make_counting_iterator<size_t>(0);
    const Errors errors = thrust::transform_reduce(thrust::device, first, first + count,
        ElementError{reconstructed.data, original, n}, Errors{0.0, 0.0}, MaxErrors{});
    printf("Max absolute error: %.10e\n", errors.absolute);
    printf("Max relative error: %.10e\n", errors.relative);
    if (errors.relative > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

int benchmark(int n, bool validate, bool printResults) {
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    check(cublasCreate(&blas), "cublasCreate");
    check(cusolverDnCreate(&solver), "cusolverDnCreate");

    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(double);
    DeviceBuffer<double> matrix(count), original(validate ? count : 0);
    printf("Generating positive definite matrix...\n");
    if (n) generatePositiveDefiniteMatrix(blas, matrix.data, n);
    if (validate && n)
        check(cudaMemcpy(original.data, matrix.data, bytes, cudaMemcpyDeviceToDevice),
              "save original matrix");

    int workspaceSize = 0;
    if (n)
        check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n, matrix.data, n,
                                         &workspaceSize), "cusolverDnDpotrf_bufferSize");
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> info(1);
    check(cudaDeviceSynchronize(), "finish matrix generation");
    printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    if (n) {
        check(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, matrix.data, n,
                              workspace.data, workspaceSize, info.data), "cusolverDnDpotrf");
        const dim3 block(32, 8);
        const dim3 grid((n + 31) / 32, (n + 7) / 8);
        clearUpperTriangle<<<grid, block>>>(matrix.data, n);
        check(cudaGetLastError(), "clear upper triangle");
        check(cudaMemcpy(&status, info.data, sizeof(status), cudaMemcpyDeviceToHost),
              "read factorization status");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    if (status > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", status - 1);
    if (status) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    printf("Computation time: %.3f ms\n", elapsed.count() * 1e3);
    const double ops = static_cast<double>(n) * n * n / 3.0;
    printf("Performance: %.3f GFLOPS\n", ops / elapsed.count() / 1e9);

    if (printResults) {
        std::vector<double> host(count);
        if (n) check(cudaMemcpy(host.data(), matrix.data, bytes, cudaMemcpyDeviceToHost),
                     "download factor");
        print_results(host, "CholeskyL");
    }
    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        if (n) valid = validateCholesky(blas, matrix.data, original.data, n);
        else printf("Max absolute error: %.10e\nMax relative error: %.10e\n", 0.0, 0.0);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    check(cusolverDnDestroy(solver), "cusolverDnDestroy");
    check(cublasDestroy(blas), "cublasDestroy");
    return valid ? 0 : 1;
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
    
    if (n > INT_MAX || (n && n > SIZE_MAX / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    try {
        return benchmark(static_cast<int>(n), validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
