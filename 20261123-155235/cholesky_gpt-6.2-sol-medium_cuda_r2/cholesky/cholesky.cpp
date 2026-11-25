#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCuda(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(status));
    }
}

void checkCuda(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuSOLVER status " +
                                 std::to_string(status));
    }
}

template <auto destroy>
struct CudaDeleter {
    template <typename T>
    void operator()(T* resource) const {
        destroy(resource);
    }
};

template <auto destroy, typename Status, typename T>
auto createCudaResource(Status (*create)(T*), const char* operation) {
    T resource = nullptr;
    checkCuda(create(&resource), operation);
    return std::unique_ptr<std::remove_pointer_t<T>, CudaDeleter<destroy>>(resource);
}

template <typename T>
using DeviceBuffer = std::unique_ptr<T, CudaDeleter<cudaFree>>;

template <typename T>
DeviceBuffer<T> allocateDevice(size_t count) {
    T* buffer = nullptr;
    checkCuda(cudaMalloc(&buffer, count * sizeof(T)), "cudaMalloc");
    return DeviceBuffer<T>(buffer);
}

constexpr int threadsPerBlock = 256;

int matrixGridSize(size_t elements) {
    return static_cast<int>(
        std::min<size_t>((elements + threadsPerBlock - 1) / threadsPerBlock, 2048));
}

__global__ void addDiagonal(double* matrix, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[i * (static_cast<size_t>(n) + 1)] += n;
    }
}

__global__ void clearLowerTriangle(double* matrix, int n, size_t elements) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = start; index < elements; index += stride) {
        if (index % n > index / n) {
            matrix[index] = 0.0;
        }
    }
}

void generatePositiveDefiniteMatrix(cublasHandle_t blas, double* matrix, int n,
                                    size_t elements) {
    // Keep the serial RNG sequence, then form the Gram matrix in parallel.
    std::vector<double> randomMatrix(elements);
    unsigned int seed = 42;
    for (double& value : randomMatrix) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    auto randomDevice = allocateDevice<double>(elements);
    checkCuda(cudaMemcpy(randomDevice.get(), randomMatrix.data(), elements * sizeof(double),
                         cudaMemcpyHostToDevice), "upload random matrix");

    // The row-major random matrix appears transposed to column-major cuBLAS.
    // Use the column-major upper triangle throughout so the result can be read
    // directly as a row-major lower-triangular factor.
    const double one = 1.0;
    const double zero = 0.0;
    checkCuda(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n, &one,
                         randomDevice.get(), n, &zero, matrix, n), "form Gram matrix");
    addDiagonal<<<(static_cast<size_t>(n) + threadsPerBlock - 1) / threadsPerBlock,
                  threadsPerBlock>>>(matrix, n);
    checkCuda(cudaGetLastError(), "add matrix diagonal");
}

bool choleskyDecomposition(double* matrix, int n, float& milliseconds) {
    auto solver = createCudaResource<cusolverDnDestroy>(cusolverDnCreate, "cusolverDnCreate");
    int workspaceSize = 0;
    checkCuda(cusolverDnDpotrf_bufferSize(solver.get(), CUBLAS_FILL_MODE_UPPER, n, matrix,
                                         n, &workspaceSize), "query Cholesky workspace");
    auto workspace = allocateDevice<double>(std::max(workspaceSize, 1));
    auto infoDevice = allocateDevice<int>(1);
    auto start = createCudaResource<cudaEventDestroy>(cudaEventCreate, "cudaEventCreate");
    auto stop = createCudaResource<cudaEventDestroy>(cudaEventCreate, "cudaEventCreate");

    checkCuda(cudaEventRecord(start.get()), "start Cholesky timer");
    checkCuda(cusolverDnDpotrf(solver.get(), CUBLAS_FILL_MODE_UPPER, n, matrix, n,
                              workspace.get(), workspaceSize, infoDevice.get()),
              "factorize matrix");
    checkCuda(cudaEventRecord(stop.get()), "stop Cholesky timer");

    int info = 0;
    checkCuda(cudaMemcpy(&info, infoDevice.get(), sizeof(info), cudaMemcpyDeviceToHost),
              "read Cholesky status");
    checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "read Cholesky timer");

    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) {
        throw std::runtime_error("invalid Cholesky argument " + std::to_string(-info));
    }
    return true;
}

struct ValidationErrors {
    double absolute;
    double relative;
};

__global__ void reduceValidationErrors(const double* reconstructed, const double* packedOriginal,
                                       int n, size_t elements, unsigned long long* maxima) {
    __shared__ double absolute[threadsPerBlock];
    __shared__ double relative[threadsPerBlock];

    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    for (size_t index = start; index < elements; index += stride) {
        // Both matrices are symmetric, so one triangle yields the same maxima.
        const size_t column = index / n;
        const size_t row = index - column * n;
        if (row <= column) {
            const double reference = packedOriginal[column * (column + 1) / 2 + row];
            const double error = fabs(reconstructed[index] - reference);
            maxAbsolute = fmax(maxAbsolute, error);
            maxRelative = fmax(maxRelative, error / (fabs(reference) + 1e-10));
        }
    }

    absolute[threadIdx.x] = maxAbsolute;
    relative[threadIdx.x] = maxRelative;
    for (int offset = threadsPerBlock / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < offset) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + offset]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + offset]);
        }
    }
    if (threadIdx.x == 0) {
        // Nonnegative doubles have the same ordering as their unsigned bit patterns.
        atomicMax(maxima, static_cast<unsigned long long>(__double_as_longlong(absolute[0])));
        atomicMax(maxima + 1, static_cast<unsigned long long>(__double_as_longlong(relative[0])));
    }
}

ValidationErrors validateCholesky(cublasHandle_t blas, const double* factor,
                                 const double* original, int n, size_t elements) {
    auto reconstructed = allocateDevice<double>(elements);
    const double one = 1.0;
    const double zero = 0.0;
    checkCuda(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n, &one, factor,
                         n, &zero, reconstructed.get(), n), "reconstruct matrix");

    auto maxima = allocateDevice<unsigned long long>(2);
    checkCuda(cudaMemset(maxima.get(), 0, 2 * sizeof(unsigned long long)), "initialize errors");
    reduceValidationErrors<<<matrixGridSize(elements), threadsPerBlock>>>(
        reconstructed.get(), original, n, elements, maxima.get());
    checkCuda(cudaGetLastError(), "reduce validation errors");

    double errors[2];
    checkCuda(cudaMemcpy(errors, maxima.get(), sizeof(errors), cudaMemcpyDeviceToHost),
              "read validation errors");
    return {errors[0], errors[1]};
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int n, bool validate, bool printResults) {
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(double);
    std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, CudaDeleter<cublasDestroy>> blas;
    DeviceBuffer<double> matrix;
    DeviceBuffer<double> original;

    printf("Generating positive definite matrix...\n");
    if (n > 0) {
        blas = createCudaResource<cublasDestroy>(cublasCreate, "cublasCreate");
        matrix = allocateDevice<double>(elements);
        // cuSOLVER's tiled loads may touch the half that SYRK leaves unwritten.
        checkCuda(cudaMemset(matrix.get(), 0, bytes), "initialize matrix");
        generatePositiveDefiniteMatrix(blas.get(), matrix.get(), n, elements);
        if (validate) {
            original = allocateDevice<double>((elements + n) / 2);
            checkCuda(cublasDtrttp(blas.get(), CUBLAS_FILL_MODE_UPPER, n, matrix.get(), n,
                                  original.get()), "save original triangle");
        }
    }

    printf("Computing Cholesky decomposition...\n");
    float milliseconds = 0.0f;
    if (n > 0 && !choleskyDecomposition(matrix.get(), n, milliseconds)) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));
    const double ops = static_cast<double>(n) * n * n / 3.0;
    const double gflops = milliseconds > 0.0f ? ops / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (n > 0 && (validate || printResults)) {
        clearLowerTriangle<<<matrixGridSize(elements), threadsPerBlock>>>(matrix.get(), n,
                                                                        elements);
        checkCuda(cudaGetLastError(), "clear unused triangle");
    }

    if (printResults) {
        std::vector<double> result(elements);
        if (n > 0) {
            checkCuda(cudaMemcpy(result.data(), matrix.get(), bytes, cudaMemcpyDeviceToHost),
                      "download Cholesky factor");
        }
        print_results(result, "CholeskyL");
    }

    if (validate) {
        printf("Validating result...\n");
        const auto errors = n > 0
            ? validateCholesky(blas.get(), matrix.get(), original.get(), n, elements)
            : ValidationErrors{};
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

int main(int argc, char** argv) {
    int n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const long size = strtol(value, &end, 10);
            if (end == value || *end != '\0' || size < 0 || size > INT_MAX ||
                (size > 0 && static_cast<size_t>(size) >
                    std::numeric_limits<size_t>::max() / sizeof(double) / size)) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
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
