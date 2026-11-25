#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
    }
}

void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
    }
}

template <class Handle, auto create, auto destroy>
class CudaHandle {
public:
    CudaHandle() { check(create(&handle)); }
    ~CudaHandle() { destroy(handle); }

    CudaHandle(const CudaHandle&) = delete;
    CudaHandle& operator=(const CudaHandle&) = delete;

    operator Handle() const { return handle; }

private:
    Handle handle{};
};

using Blas = CudaHandle<cublasHandle_t, cublasCreate, cublasDestroy>;
using Solver = CudaHandle<cusolverDnHandle_t, cusolverDnCreate, cusolverDnDestroy>;
cudaError_t createEvent(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

using Event = CudaHandle<cudaEvent_t, createEvent, cudaEventDestroy>;

struct DeviceFree {
    void operator()(void* pointer) const { cudaFree(pointer); }
};

template <class T>
std::unique_ptr<T, DeviceFree> deviceBuffer(size_t count) {
    T* pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)));
    return std::unique_ptr<T, DeviceFree>(pointer);
}

// A row-major lower triangle is a column-major upper triangle to cuBLAS
// and cuSOLVER. Both libraries can work on that triangle in place.
void formGramMatrix(cublasHandle_t blas, const double* input, double* output, int n) {
    const double one = 1.0;
    const double zero = 0.0;
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n,
                      &one, input, n, &zero, output, n));
}

constexpr int blockSize = 256;

__global__ void addDiagonal(double* matrix, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[i * (n + 1)] += static_cast<double>(n);
    }
}

__global__ void clearUpperTriangle(double* matrix, size_t n) {
    const size_t count = n * n;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        if (i % n > i / n) {
            matrix[i] = 0.0;
        }
    }
}

struct ErrorPair {
    double absolute;
    double relative;
};

__device__ void reduceWarp(ErrorPair& error) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        error.absolute = fmax(error.absolute, __shfl_down_sync(0xffffffff, error.absolute, offset));
        error.relative = fmax(error.relative, __shfl_down_sync(0xffffffff, error.relative, offset));
    }
}

__global__ void validationErrors(const double* reconstructed, const double* original,
                                 size_t n, ErrorPair* blockErrors) {
    const size_t count = n * n;
    ErrorPair error{};
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        // Both Gram matrices are symmetric, so the stored triangle suffices.
        if (i % n <= i / n) {
            const double reference = original[i];
            const double absolute = fabs(reconstructed[i] - reference);
            error.absolute = fmax(error.absolute, absolute);
            error.relative = fmax(error.relative, absolute / (fabs(reference) + 1e-10));
        }
    }

    __shared__ ErrorPair warpErrors[blockSize / 32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    reduceWarp(error);
    if (lane == 0) {
        warpErrors[warp] = error;
    }
    __syncthreads();
    if (warp == 0) {
        error = lane < blockSize / warpSize ? warpErrors[lane] : ErrorPair{};
        reduceWarp(error);
        if (lane == 0) {
            blockErrors[blockIdx.x] = error;
        }
    }
}

bool validateFactor(cublasHandle_t blas, const double* factor, const double* original,
                    double* reconstructed, int n, int blocks) {
    auto deviceErrors = deviceBuffer<ErrorPair>(blocks);
    std::vector<ErrorPair> hostErrors(blocks);
    formGramMatrix(blas, factor, reconstructed, n);
    validationErrors<<<blocks, blockSize>>>(reconstructed, original, n, deviceErrors.get());
    check(cudaGetLastError());
    check(cudaMemcpy(hostErrors.data(), deviceErrors.get(), blocks * sizeof(ErrorPair),
                     cudaMemcpyDeviceToHost));

    ErrorPair error{};
    for (const auto& block : hostErrors) {
        error.absolute = std::max(error.absolute, block.absolute);
        error.relative = std::max(error.relative, block.relative);
    }
    printf("Max absolute error: %.10e\n", error.absolute);
    printf("Max relative error: %.10e\n", error.relative);
    if (error.relative > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

int runBenchmark(int n, bool validate, bool printResults) {
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("Generating positive definite matrix...\n");

    // Initialize CUDA even for the empty matrix, which has no work to launch.
    check(cudaFree(nullptr));
    if (n == 0) {
        printf("Computing Cholesky decomposition...\n");
        printf("Computation time: 0 ms\nPerformance: 0.000 GFLOPS\n");
        if (printResults) {
            print_results(std::vector<double>{}, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", 0.0, 0.0);
            printf("Validation: PASSED\n");
        }
        return 0;
    }

    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(double);
    const int blocks = static_cast<int>(std::min<size_t>((count + blockSize - 1) / blockSize, 4096));
    Blas blas;
    Solver solver;
    auto matrix = deviceBuffer<double>(count);
    auto scratch = deviceBuffer<double>(count);
    {
        std::unique_ptr<double[]> randomMatrix(new double[count]);
        unsigned int seed = 42;
        for (size_t i = 0; i < count; ++i) {
            randomMatrix[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        }
        check(cudaMemcpy(scratch.get(), randomMatrix.get(), bytes, cudaMemcpyHostToDevice));
    }
    formGramMatrix(blas, scratch.get(), matrix.get(), n);
    addDiagonal<<<(n - 1) / blockSize + 1, blockSize>>>(matrix.get(), n);
    check(cudaGetLastError());

    std::unique_ptr<double, DeviceFree> original;
    if (validate) {
        original = deviceBuffer<double>(count);
        check(cudaMemcpyAsync(original.get(), matrix.get(), bytes, cudaMemcpyDeviceToDevice));
    } else {
        scratch.reset();
    }

    int workspaceSize;
    check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n,
                                     matrix.get(), n, &workspaceSize));
    auto workspace = deviceBuffer<double>(workspaceSize);
    auto deviceInfo = deviceBuffer<int>(1);
    Event start;
    Event end;

    printf("Computing Cholesky decomposition...\n");
    check(cudaEventRecord(start));
    check(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, matrix.get(), n,
                          workspace.get(), workspaceSize, deviceInfo.get()));
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float milliseconds;
    check(cudaEventElapsedTime(&milliseconds, start, end));

    int info;
    check(cudaMemcpy(&info, deviceInfo.get(), sizeof(info), cudaMemcpyDeviceToHost));
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (info < 0) {
        throw std::runtime_error("Invalid cuSOLVER factorization argument " + std::to_string(-info));
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double ops = static_cast<double>(n) * n * n / 3.0;
    printf("Performance: %.3f GFLOPS\n", milliseconds > 0.0f ? ops / (milliseconds * 1e6) : 0.0);

    if (printResults || validate) {
        clearUpperTriangle<<<blocks, blockSize>>>(matrix.get(), n);
        check(cudaGetLastError());
    }
    if (printResults) {
        std::vector<double> factor(count);
        check(cudaMemcpy(factor.data(), matrix.get(), bytes, cudaMemcpyDeviceToHost));
        print_results(factor, "CholeskyL");
    }
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateFactor(blas, matrix.get(), original.get(), scratch.get(), n, blocks);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
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
    long long n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = strtoll(argv[++i], nullptr, 10);
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

    if (n < 0 || n > std::numeric_limits<int>::max() ||
        (n > 0 && static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
        fprintf(stderr, "Invalid matrix size: %lld\n", n);
        return 1;
    }

    try {
        return runBenchmark(static_cast<int>(n), validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "Cholesky benchmark failed: %s\n", error.what());
        return 1;
    }
}
