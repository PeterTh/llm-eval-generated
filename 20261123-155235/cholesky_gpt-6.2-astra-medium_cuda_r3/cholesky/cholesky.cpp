#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

template <typename Status>
static void check(Status status, const char* operation) {
    if (status != 0) {
        fprintf(stderr, "%s failed (status %d)\n", operation, int(status));
        exit(EXIT_FAILURE);
    }
}

#define CHECK(operation) check((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data;
};

constexpr int threads = 256;

static int blocksFor(size_t count) {
    return static_cast<int>(std::min<size_t>((count + threads - 1) / threads, 65535));
}

#if defined(__GLIBC__)
// glibc rand_r advances its 32-bit generator three times per result.
// Compose affine maps modulo 2^32 to start each lane at its original index.
struct Advance {
    unsigned int mul, add;
    __host__ __device__ unsigned int operator()(unsigned int x) const {
        return mul * x + add;
    }
};

__host__ __device__ static Advance skipSamples(size_t count) {
    constexpr unsigned int a = 1103515245u, c = 12345u;
    Advance step{a * a * a, (a * a + a + 1) * c};
    Advance jump{1, 0};
    while (count) {
        if (count & 1) jump = {step.mul * jump.mul, step(jump.add)};
        step = {step.mul * step.mul, step(step.add)};
        count >>= 1;
    }
    return jump;
}

__global__ void generateRandom(double* b, size_t count, Advance strideJump) {
    const size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = skipSamples(first)(42);
    for (size_t i = first; i < count; i += gridDim.x * blockDim.x) {
        constexpr Advance next{1103515245u, 12345u};
        unsigned int x = next(state);
        unsigned int value = (x >> 16) & 2047;
        x = next(x);
        value = (value << 10) | ((x >> 16) & 1023);
        x = next(x);
        value = (value << 10) | ((x >> 16) & 1023);
        b[i] = __ddiv_rn(double(value), double(RAND_MAX)) - 0.5;
        state = strideJump(state);
    }
}
#endif

__global__ void shiftDiagonal(double* a, int n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(n)) a[i * n + i] += n;
}

// In column-major BLAS notation the row-major input is B^T, and
// the upper triangle of the product is the row-major lower triangle.
static void gramMatrix(cublasHandle_t blas, const double* b, double* a, int n) {
    const double one = 1.0, zero = 0.0;
    CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n,
                      &one, b, n, &zero, a, n));
}

static void generateMatrix(cublasHandle_t blas, double* scratch, double* a, int n) {
    const size_t count = size_t(n) * n;
#if defined(__GLIBC__)
    const int blocks = blocksFor(count);
    generateRandom<<<blocks, threads>>>(scratch, count, skipSamples(blocks * threads));
    CHECK(cudaGetLastError());
#else
    std::vector<double> b(count);
    unsigned int seed = 42;
    for (double& x : b) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
    CHECK(cudaMemcpy(scratch, b.data(), count * sizeof(double), cudaMemcpyHostToDevice));
#endif
    gramMatrix(blas, scratch, a, n);
    shiftDiagonal<<<(size_t(n) + threads - 1) / threads, threads>>>(a, n);
    CHECK(cudaGetLastError());
}

__global__ void clearUpper(double* factor, int n) {
    const size_t row = blockIdx.x;
    for (size_t col = row + 1 + threadIdx.x; col < size_t(n); col += blockDim.x)
        factor[row * n + col] = 0.0;
}

struct Errors {
    double absolute, relative;
};

__global__ void compareLower(const double* reconstructed, const double* original,
                             int n, Errors* partial) {
    Errors local{};
    for (size_t row = blockIdx.x; row < size_t(n); row += gridDim.x) {
        for (size_t col = threadIdx.x; col <= row; col += blockDim.x) {
            const size_t i = row * n + col;
            double error = fabs(reconstructed[i] - original[i]);
            local.absolute = fmax(local.absolute, error);
            local.relative = fmax(local.relative, error / (fabs(original[i]) + 1e-10));
        }
    }
    __shared__ Errors maxima[threads];
    maxima[threadIdx.x] = local;
    __syncthreads();
    for (int offset = threads / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            Errors other = maxima[threadIdx.x + offset];
            maxima[threadIdx.x].absolute = fmax(maxima[threadIdx.x].absolute, other.absolute);
            maxima[threadIdx.x].relative = fmax(maxima[threadIdx.x].relative, other.relative);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = maxima[0];
}

static int run(int n, bool validate, bool printResults) {
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));
    const size_t count = size_t(n) * n;
    // Reuse the random input allocation for the saved original during validation.
    DeviceBuffer<double> factor(std::max<size_t>(1, count));
    DeviceBuffer<double> scratch(std::max<size_t>(1, count));
    printf("Generating positive definite matrix...\n");
    if (n) {
        generateMatrix(blas, scratch.data, factor.data, n);
        if (validate)
            CHECK(cudaMemcpy(scratch.data, factor.data, count * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    int workspaceSize = 0;
    if (n)
        CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n,
                                          factor.data, n, &workspaceSize));
    DeviceBuffer<double> work(std::max(1, workspaceSize));
    DeviceBuffer<int> info(1);
    CHECK(cudaDeviceSynchronize());
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::steady_clock::now();
    int status = 0;
    if (n) {
        CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, factor.data, n,
                               work.data, workspaceSize, info.data));
        clearUpper<<<n, threads>>>(factor.data, n);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&status, info.data, sizeof(status), cudaMemcpyDeviceToHost));
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    if (status) {
        if (status > 0)
            printf("Error: Matrix is not positive definite at diagonal element %d\n", status - 1);
        printf("Cholesky decomposition failed\n");
    } else {
        printf("Computation time: %ld ms\n", long(elapsed.count() * 1000));
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * elapsed.count()));
        if (printResults) {
            std::vector<double> result(count);
            if (n) CHECK(cudaMemcpy(result.data(), factor.data, count * sizeof(double), cudaMemcpyDeviceToHost));
            print_results(result, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            Errors maximum{};
            if (n) {
                DeviceBuffer<double> reconstructed(count);
                gramMatrix(blas, factor.data, reconstructed.data, n);
                const int blocks = std::min(1024, n);
                DeviceBuffer<Errors> partial(blocks);
                compareLower<<<blocks, threads>>>(reconstructed.data, scratch.data, n, partial.data);
                CHECK(cudaGetLastError());
                std::vector<Errors> errors(blocks);
                CHECK(cudaMemcpy(errors.data(), partial.data, blocks * sizeof(Errors), cudaMemcpyDeviceToHost));
                for (Errors e : errors) {
                    maximum.absolute = std::max(maximum.absolute, e.absolute);
                    maximum.relative = std::max(maximum.relative, e.relative);
                }
            }
            printf("Max absolute error: %.10e\n", maximum.absolute);
            printf("Max relative error: %.10e\n", maximum.relative);
            if (maximum.relative > 1e-6) {
                printf("Validation failed: relative error too large\n");
                status = 1;
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return status != 0;
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
        fprintf(stderr, "Matrix size is out of range\n");
        return 1;
    }
    return run(static_cast<int>(n), validate, printResults);
}
