#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include "../common/results_output.hpp"

namespace {

constexpr int blockSize = 256;
constexpr int maxRowBlocks = 4096;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
    }
}

void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct CudaHandles {
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~CudaHandles() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
    }
};

__global__ void shiftDiagonal(double* matrix, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[static_cast<size_t>(i) * (static_cast<size_t>(n) + 1)] += n;
    }
}

// cuBLAS and cuSOLVER see the transpose of the row-major matrix.
constexpr auto triangle = CUBLAS_FILL_MODE_UPPER;

void generatePositiveDefiniteMatrix(cublasHandle_t blas, double* matrix, double* input, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    std::unique_ptr<double[]> random(new double[count]);
    unsigned int seed = 42;
    for (size_t i = 0; i < count; ++i) {
        random[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }
    checkCuda(cudaMemcpy(input, random.get(), count * sizeof(double), cudaMemcpyHostToDevice));

    const double one = 1.0;
    const double zero = 0.0;
    checkBlas(cublasDsyrk(blas, triangle, CUBLAS_OP_T, n, n,
                         &one, input, n, &zero, matrix, n));
    shiftDiagonal<<<(static_cast<size_t>(n) + blockSize - 1) / blockSize, blockSize>>>(matrix, n);
    checkCuda(cudaGetLastError());
}

bool choleskyDecomposition(CudaHandles& cuda, double* matrix, int n, float& milliseconds) {
    checkSolver(cusolverDnCreate(&cuda.solver));
    checkCuda(cudaEventCreate(&cuda.start));
    checkCuda(cudaEventCreate(&cuda.stop));

    int workspaceSize = 0;
    checkSolver(cusolverDnDpotrf_bufferSize(cuda.solver, triangle, n, matrix, n, &workspaceSize));
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> deviceInfo(1);

    checkCuda(cudaEventRecord(cuda.start));
    checkSolver(cusolverDnDpotrf(cuda.solver, triangle, n, matrix, n,
                               workspace.get(), workspaceSize, deviceInfo.get()));
    checkCuda(cudaEventRecord(cuda.stop));

    int info = 0;
    checkCuda(cudaEventSynchronize(cuda.stop));
    checkCuda(cudaEventElapsedTime(&milliseconds, cuda.start, cuda.stop));
    checkCuda(cudaMemcpy(&info, deviceInfo.get(), sizeof(info), cudaMemcpyDeviceToHost));
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) {
        throw std::runtime_error("cuSOLVER POTRF rejected argument " + std::to_string(-info));
    }
    return true;
}

__global__ void zeroUpperTriangle(double* matrix, int n) {
    for (int row = blockIdx.x; row < n; row += gridDim.x) {
        double* values = matrix + static_cast<size_t>(row) * n;
        for (int col = row + 1 + threadIdx.x; col < n; col += blockDim.x) {
            values[col] = 0.0;
        }
    }
}

__global__ void copyLowerTriangle(double* destination, const double* source, int n) {
    for (int row = blockIdx.x; row < n; row += gridDim.x) {
        const size_t offset = static_cast<size_t>(row) * n;
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            destination[offset + col] = source[offset + col];
        }
    }
}

struct Errors {
    double absolute;
    double relative;
};

__global__ void reconstructionErrors(const double* reconstructed, const double* original,
                                     int n, Errors* blockErrors) {
    Errors worst{};
    for (int row = blockIdx.x; row < n; row += gridDim.x) {
        const size_t offset = static_cast<size_t>(row) * n;
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            const double expected = original[offset + col];
            const double error = fabs(reconstructed[offset + col] - expected);
            if (!isfinite(error)) {
                worst = {CUDART_INF, CUDART_INF};
                continue;
            }
            worst.absolute = fmax(worst.absolute, error);
            worst.relative = fmax(worst.relative, error / (fabs(expected) + 1e-10));
        }
    }

    __shared__ Errors shared[blockSize];
    const int lane = threadIdx.x;
    shared[lane] = worst;
    __syncthreads();
    for (int offset = blockSize / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            shared[lane].absolute = fmax(shared[lane].absolute, shared[lane + offset].absolute);
            shared[lane].relative = fmax(shared[lane].relative, shared[lane + offset].relative);
        }
        __syncthreads();
    }
    if (lane == 0) {
        blockErrors[blockIdx.x] = shared[0];
    }
}

Errors validateCholesky(cublasHandle_t blas, const double* factor, const double* original, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    DeviceBuffer<double> reconstructed(count);
    const int blocks = std::min(n, maxRowBlocks);
    DeviceBuffer<Errors> deviceErrors(blocks);
    const double one = 1.0;
    const double zero = 0.0;
    checkBlas(cublasDsyrk(blas, triangle, CUBLAS_OP_T, n, n,
                         &one, factor, n, &zero, reconstructed.get(), n));
    reconstructionErrors<<<blocks, blockSize>>>(reconstructed.get(), original, n,
                                                deviceErrors.get());
    checkCuda(cudaGetLastError());

    std::vector<Errors> blockErrors(blocks);
    checkCuda(cudaMemcpy(blockErrors.data(), deviceErrors.get(), blocks * sizeof(Errors),
                         cudaMemcpyDeviceToHost));
    Errors worst{};
    for (const auto& errors : blockErrors) {
        worst.absolute = std::max(worst.absolute, errors.absolute);
        worst.relative = std::max(worst.relative, errors.relative);
    }
    return worst;
}

int runBenchmark(int n, bool validate, bool printResults) {
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    CudaHandles cuda;
    checkBlas(cublasCreate(&cuda.blas));
    const size_t elements = static_cast<size_t>(n) * n;
    if (elements > SIZE_MAX / sizeof(double)) {
        throw std::runtime_error("Matrix size overflows the allocation size");
    }
    const size_t bytes = elements * sizeof(double);
    DeviceBuffer<double> matrix(elements);
    DeviceBuffer<double> original(validate ? elements : 0);

    printf("Generating positive definite matrix...\n");
    if (n != 0) {
        DeviceBuffer<double> input(validate ? 0 : elements);
        // Initialize the unused triangle for cuSOLVER's tiled loads.
        checkCuda(cudaMemsetAsync(matrix.get(), 0, bytes));
        // Reuse the random input allocation to save the original for validation.
        generatePositiveDefiniteMatrix(cuda.blas, matrix.get(),
                                       validate ? original.get() : input.get(), n);
        if (validate) {
            copyLowerTriangle<<<std::min(n, maxRowBlocks), blockSize>>>(
                original.get(), matrix.get(), n);
            checkCuda(cudaGetLastError());
        }
    }

    printf("Computing Cholesky decomposition...\n");
    float milliseconds = 0.0f;
    if (n != 0 && !choleskyDecomposition(cuda, matrix.get(), n, milliseconds)) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    printf("Computation time: %.3f ms\n", milliseconds);
    const double operations = static_cast<double>(n) * n * n / 3.0;
    const double gflops = milliseconds > 0 ? operations / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (n != 0 && (printResults || validate)) {
        zeroUpperTriangle<<<std::min(n, maxRowBlocks), blockSize>>>(matrix.get(), n);
        checkCuda(cudaGetLastError());
    }
    if (printResults) {
        std::vector<double> result(elements);
        if (n != 0) {
            checkCuda(cudaMemcpy(result.data(), matrix.get(), bytes, cudaMemcpyDeviceToHost));
        }
        print_results(result, "CholeskyL");
    }
    if (validate) {
        printf("Validating result...\n");
        const Errors errors = n != 0 ? validateCholesky(cuda.blas, matrix.get(), original.get(), n)
                                     : Errors{};
        printf("Max absolute error: %.10e\n", errors.absolute);
        printf("Max relative error: %.10e\n", errors.relative);
        if (errors.relative > 1e-6) {
            printf("Validation failed: relative error too large\n");
            printf("Validation: FAILED\n");
            return 1;
        }
        printf("Validation: PASSED\n");
    }
    return 0;
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
    int n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            const long size = std::strtol(argument, &end, 10);
            if (end == argument || *end != '\0' || size < 0 || size > INT_MAX) {
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
        fprintf(stderr, "CUDA benchmark failed: %s\n", error.what());
        return 1;
    }
}
