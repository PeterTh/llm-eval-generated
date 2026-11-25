#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
    }
}

void check(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuSOLVER status " + std::to_string(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            check(cudaMalloc(&data_, count * sizeof(T)), "Allocating device memory");
        }
    }

    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct LibraryHandles {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;

    ~LibraryHandles() {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
    }
};

class Event {
public:
    Event() { check(cudaEventCreate(&event_), "Creating CUDA event"); }
    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

constexpr int threads = 256;
constexpr int tileRows = 32;
constexpr int tileCols = threads / tileRows;
constexpr int smallMatrixSize = 64;
constexpr double one = 1.0;
constexpr double zero = 0.0;

// glibc rand_r advances a 32-bit LCG three times per result. Compose its
// affine transitions to let each CUDA thread jump to its place in the stream.
#if defined(__GLIBC__)
constexpr unsigned int randomMultiplier = 1103515245u;
constexpr unsigned int randomIncrement = 12345u;

struct RandomJump {
    unsigned int multiplier = 1;
    unsigned int increment = 0;

    __host__ __device__ unsigned int apply(unsigned int state) const {
        return multiplier * state + increment;
    }
};

__host__ __device__ RandomJump jumpRandom(size_t steps) {
    unsigned int multiplier = randomMultiplier * randomMultiplier * randomMultiplier;
    unsigned int increment = randomIncrement * (randomMultiplier * randomMultiplier + randomMultiplier + 1u);
    RandomJump jump;
    while (steps != 0) {
        if (steps & 1) {
            jump.multiplier *= multiplier;
            jump.increment = multiplier * jump.increment + increment;
        }
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        steps >>= 1;
    }
    return jump;
}

__global__ void generateRandom(double* values, size_t count, RandomJump strideJump) {
    const size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    unsigned int seed = jumpRandom(first).apply(42u);

    for (size_t i = first; i < count; i += stride) {
        const unsigned int a = randomMultiplier * seed + randomIncrement;
        const unsigned int b = randomMultiplier * a + randomIncrement;
        const unsigned int c = randomMultiplier * b + randomIncrement;
        const unsigned int value = (((a >> 16) & 2047u) << 20) |
                                   (((b >> 16) & 1023u) << 10) |
                                   ((c >> 16) & 1023u);
        values[i] = __ddiv_rn(static_cast<double>(value), RAND_MAX) - 0.5;
        seed = strideJump.apply(seed);
    }
}
#endif

void fillRandom(double* values, size_t count) {
#if defined(__GLIBC__)
    const int blocks = static_cast<int>(std::min<size_t>((count + threads - 1) / threads, 1024));
    generateRandom<<<blocks, threads>>>(values, count, jumpRandom(blocks * threads));
    check(cudaGetLastError(), "Launching random matrix generation");
#else
    // Preserve the platform's rand_r sequence on other C libraries.
    std::vector<double> input(count);
    unsigned int seed = 42;
    for (double& value : input) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    check(cudaMemcpy(values, input.data(), count * sizeof(double), cudaMemcpyHostToDevice),
          "Uploading random matrix");
#endif
}

__global__ void prepareMatrix(double* matrix, double* original, int n) {
    const int firstRow = blockIdx.x * tileRows;
    const int firstCol = blockIdx.y * tileCols;
    if (!original && firstRow + tileRows <= firstCol) return;

    const int row = firstRow + threadIdx.x;
    const int col = firstCol + threadIdx.y;
    if (row >= n || col >= n) return;

    const size_t index = static_cast<size_t>(col) * n + row;
    if (row > col) {
        // cuSOLVER may load whole tiles even when it factors only the upper triangle.
        matrix[index] = 0.0;
    } else if (row == col) {
        matrix[index] += n;
        if (original) original[index] = matrix[index];
    } else if (original) {
        original[index] = matrix[index];
    }
}

__global__ void clearLowerTriangle(double* matrix, int n) {
    const int firstRow = blockIdx.x * tileRows;
    const int firstCol = blockIdx.y * tileCols;
    if (firstRow + tileRows <= firstCol) return;

    const int row = firstRow + threadIdx.x;
    const int col = firstCol + threadIdx.y;
    if (row < n && row > col) {
        matrix[static_cast<size_t>(col) * n + row] = 0.0;
    }
}

// Keep small matrices in shared memory and compute each factor row in one
// block. A single launch avoids the library dispatch overhead at these sizes.
__global__ void factorSmallMatrix(double* matrix, int n, int* info) {
    __shared__ double factor[smallMatrixSize][smallMatrixSize + 1];
    __shared__ int failedPivot;
    const int lane = threadIdx.x;

    if (lane == 0) failedPivot = 0;
    for (int col = 0; col < n; ++col) {
        if (lane <= col) factor[lane][col] = matrix[static_cast<size_t>(col) * n + lane];
    }
    __syncthreads();

    for (int pivot = 0; pivot < n; ++pivot) {
        double value = 0.0;
        if (pivot <= lane && lane < n) {
            double sum = 0.0;
            for (int k = 0; k < pivot; ++k) {
                sum = fma(factor[k][pivot], factor[k][lane], sum);
            }
            value = factor[pivot][lane] - sum;
        }
        if (lane == pivot) {
            if (value <= 0.0) failedPivot = pivot + 1;
            factor[pivot][pivot] = sqrt(value);
        }
        __syncthreads();
        if (failedPivot != 0) break;
        if (pivot < lane && lane < n) {
            factor[pivot][lane] = value / factor[pivot][pivot];
        }
        __syncthreads();
    }

    if (lane == 0) *info = failedPivot;
    if (failedPivot == 0) {
        for (int col = 0; col < n; ++col) {
            if (lane <= col) matrix[static_cast<size_t>(col) * n + lane] = factor[lane][col];
        }
    }
}

struct Errors {
    double absolute;
    double relative;
};

__global__ void measureErrors(const double* reconstructed, const double* original, int n,
                              Errors* partial) {
    __shared__ Errors warpErrors[tileCols];

    const int row = blockIdx.x * tileRows + threadIdx.x;
    Errors maximum{0.0, 0.0};
    if (row < n) {
        for (size_t col = blockIdx.y * tileCols + threadIdx.y; col < static_cast<size_t>(n);
             col += gridDim.y * tileCols) {
            if (static_cast<size_t>(row) <= col) {
                const size_t index = col * n + row;
                const double expected = original[index];
                const double error = fabs(reconstructed[index] - expected);
                maximum.absolute = fmax(maximum.absolute, error);
                maximum.relative = fmax(maximum.relative, error / (fabs(expected) + 1e-10));
            }
        }
    }

    for (int offset = tileRows / 2; offset > 0; offset /= 2) {
        maximum.absolute = fmax(maximum.absolute, __shfl_down_sync(0xffffffff, maximum.absolute, offset));
        maximum.relative = fmax(maximum.relative, __shfl_down_sync(0xffffffff, maximum.relative, offset));
    }
    if (threadIdx.x == 0) warpErrors[threadIdx.y] = maximum;
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int warp = 1; warp < tileCols; ++warp) {
            maximum.absolute = fmax(maximum.absolute, warpErrors[warp].absolute);
            maximum.relative = fmax(maximum.relative, warpErrors[warp].relative);
        }
        partial[blockIdx.y * gridDim.x + blockIdx.x] = maximum;
    }
}

bool validateCholesky(cublasHandle_t blas, const double* factor, const double* original, int n) {
    Errors maximum{0.0, 0.0};
    if (n != 0) {
        DeviceBuffer<double> reconstructed(static_cast<size_t>(n) * n);
        check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n, &one,
                          factor, n, &zero, reconstructed.get(), n), "Reconstructing the matrix");

        const dim3 grid(1 + (n - 1) / tileRows, std::min(64, 1 + (n - 1) / tileCols));
        std::vector<Errors> partial(static_cast<size_t>(grid.x) * grid.y);
        DeviceBuffer<Errors> devicePartial(partial.size());
        measureErrors<<<grid, dim3(tileRows, tileCols)>>>(reconstructed.get(), original, n,
                                                          devicePartial.get());
        check(cudaGetLastError(), "Launching validation reduction");
        check(cudaMemcpy(partial.data(), devicePartial.get(), partial.size() * sizeof(Errors),
                         cudaMemcpyDeviceToHost), "Reading validation errors");
        for (const auto& error : partial) {
            maximum.absolute = std::max(maximum.absolute, error.absolute);
            maximum.relative = std::max(maximum.relative, error.relative);
        }
    }

    printf("Max absolute error: %.10e\n", maximum.absolute);
    printf("Max relative error: %.10e\n", maximum.relative);
    if (maximum.relative > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

bool choleskyDecomposition(cusolverDnHandle_t solver, double* matrix, int n, float& milliseconds) {
    int workspaceSize = 0;
    if (n > smallMatrixSize) {
        check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n, matrix, n, &workspaceSize),
              "Querying Cholesky workspace");
    }
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> deviceInfo(1);
    Event start;
    Event stop;

    check(cudaEventRecord(start.get()), "Recording Cholesky start");
    if (n <= smallMatrixSize) {
        factorSmallMatrix<<<1, smallMatrixSize>>>(matrix, n, deviceInfo.get());
        check(cudaGetLastError(), "Launching Cholesky factorization");
    } else {
        check(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, matrix, n,
                             workspace.get(), workspaceSize, deviceInfo.get()), "Factoring the matrix");
    }
    check(cudaEventRecord(stop.get()), "Recording Cholesky completion");

    int info = 0;
    check(cudaMemcpy(&info, deviceInfo.get(), sizeof(info), cudaMemcpyDeviceToHost),
          "Reading Cholesky status");
    check(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "Measuring Cholesky time");
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
    } else if (info < 0) {
        fprintf(stderr, "Cholesky decomposition rejected argument %d\n", -info);
    }
    return info == 0;
}

int runBenchmark(int n, bool validate, bool printResults) {
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t count = static_cast<size_t>(n) * n;
    LibraryHandles handles;
    DeviceBuffer<double> matrix(count);
    DeviceBuffer<double> original(validate ? count : 0);

    printf("Generating positive definite matrix...\n");
    const dim3 tile(tileRows, tileCols);
    const dim3 grid(1 + (n - 1) / tileRows, 1 + (n - 1) / tileCols);
    if (n != 0) {
        check(cublasCreate(&handles.blas), "Creating cuBLAS handle");
        if (n > smallMatrixSize) {
            check(cusolverDnCreate(&handles.solver), "Creating cuSOLVER handle");
        }

        // Column-major upper triangles are row-major lower triangles in memory.
        // B is generated in row-major order, so A = B * B^T is a transposed SYRK.
        DeviceBuffer<double> random(validate ? 0 : count);
        double* input = validate ? original.get() : random.get();
        fillRandom(input, count);
        check(cublasDsyrk(handles.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n, &one,
                          input, n, &zero, matrix.get(), n), "Forming the Gram matrix");
        prepareMatrix<<<grid, tile>>>(matrix.get(), original.get(), n);
        check(cudaGetLastError(), "Launching matrix preparation");
    }

    printf("Computing Cholesky decomposition...\n");
    float milliseconds = 0.0f;
    if (n != 0 && !choleskyDecomposition(handles.solver, matrix.get(), n, milliseconds)) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double ops = static_cast<double>(n) * n * n / 3.0;
    const double gflops = milliseconds > 0.0f ? ops / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (n != 0 && (printResults || validate)) {
        clearLowerTriangle<<<grid, tile>>>(matrix.get(), n);
        check(cudaGetLastError(), "Launching triangular cleanup");
    }
    if (printResults) {
        std::vector<double> result(count);
        if (n != 0) {
            check(cudaMemcpy(result.data(), matrix.get(), count * sizeof(double),
                             cudaMemcpyDeviceToHost), "Reading the Cholesky factor");
        }
        print_results(result, "CholeskyL");
    }

    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateCholesky(handles.blas, matrix.get(), original.get(), n);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    int n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long size = std::strtol(argument, &end, 10);
            if (errno == ERANGE || end == argument || *end != '\0' || size < 0 || size > INT_MAX ||
                (size > 0 && static_cast<size_t>(size) >
                                 std::numeric_limits<size_t>::max() / sizeof(double) / size)) {
                fprintf(stderr, "Invalid matrix size: %s\n", argument);
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

    try {
        return runBenchmark(n, validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "Cholesky benchmark failed: %s\n", error.what());
        return 1;
    }
}
