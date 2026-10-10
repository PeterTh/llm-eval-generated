#include <algorithm>
#include <cerrno>
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
#include <math_constants.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error " + std::to_string(status));
}

// All operations use the same stream, so dependent GPU work needs no host barriers.
struct GpuContext {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;

    GpuContext() {
        try {
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            checkBlas(cublasCreate(&blas));
            checkSolver(cusolverDnCreate(&solver));
            checkBlas(cublasSetStream(blas, stream));
            checkSolver(cusolverDnSetStream(solver, stream));
            // Preserve full double precision; do not enable reduced-precision math.
            checkBlas(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        } catch (...) {
            release();
            throw;
        }
    }
    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;
    ~GpuContext() { release(); }

    void release() noexcept {
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
};

template <class T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

constexpr unsigned int threads = 256;

unsigned int gridSize(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + threads - 1) / threads, 4096));
}

__global__ void addDiagonal(double* A, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        A[i * n + i] += static_cast<double>(n);
}

__global__ void zeroUpperTriangle(double* A, size_t n) {
    const size_t count = n * n;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        if (i % n > i / n) A[i] = 0.0;
}

// Retain the original rand_r sequence, then parallelize the cubic B * B^T work.
void generatePositiveDefiniteMatrix(GpuContext& gpu, double* A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    DeviceBuffer<double> deviceB(B.size());
    checkCuda(cudaMemcpyAsync(deviceB.data, B.data(), B.size() * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
    const double one = 1.0, zero = 0.0;
    // A row-major B is seen as B^T by column-major cuBLAS.
    checkBlas(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                          &one, deviceB.data, static_cast<int>(n),
                          deviceB.data, static_cast<int>(n), &zero, A, static_cast<int>(n)));
    addDiagonal<<<gridSize(n), threads, 0, gpu.stream>>>(A, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(gpu.stream));
}

bool choleskyDecomposition(GpuContext& gpu, double* A, size_t n,
                           double* workspace, int workspaceSize, int* deviceInfo) {
    // Column-major upper Cholesky produces exactly the row-major lower layout.
    // cuSOLVER uses a blocked factorization with parallel trailing-matrix updates.
    checkSolver(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_UPPER,
                                static_cast<int>(n), A, static_cast<int>(n),
                                workspace, workspaceSize, deviceInfo));
    zeroUpperTriangle<<<gridSize(n * n), threads, 0, gpu.stream>>>(A, n);
    checkCuda(cudaGetLastError());
    int info = 0;
    checkCuda(cudaMemcpyAsync(&info, deviceInfo, sizeof(info), cudaMemcpyDeviceToHost, gpu.stream));
    checkCuda(cudaStreamSynchronize(gpu.stream));
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) throw std::runtime_error("cuSOLVER rejected argument " + std::to_string(-info));
    return true;
}

__global__ void reduceErrors(const double* reconstructed, const double* original,
                             size_t count, double* blockErrors) {
    __shared__ double absolute[threads];
    __shared__ double relative[threads];
    double maxAbsolute = 0.0, maxRelative = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        double error = fabs(reconstructed[i] - original[i]);
        double rel = error / (fabs(original[i]) + 1e-10);
        if (!isfinite(error) || !isfinite(rel)) {
            error = CUDART_INF;
            rel = CUDART_INF;
        }
        maxAbsolute = fmax(maxAbsolute, error);
        maxRelative = fmax(maxRelative, rel);
    }
    absolute[threadIdx.x] = maxAbsolute;
    relative[threadIdx.x] = maxRelative;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + stride]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blockErrors[2 * blockIdx.x] = absolute[0];
        blockErrors[2 * blockIdx.x + 1] = relative[0];
    }
}

bool validateCholesky(GpuContext& gpu, const double* L, const double* A_orig, size_t n) {
    DeviceBuffer<double> reconstructed(n * n);
    const double one = 1.0, zero = 0.0;
    checkBlas(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                          &one, L, static_cast<int>(n), L, static_cast<int>(n),
                          &zero, reconstructed.data, static_cast<int>(n)));
    const unsigned int blocks = gridSize(n * n);
    DeviceBuffer<double> errors(2 * blocks);
    reduceErrors<<<blocks, threads, 0, gpu.stream>>>(reconstructed.data, A_orig, n * n, errors.data);
    checkCuda(cudaGetLastError());
    std::vector<double> hostErrors(2 * blocks);
    checkCuda(cudaMemcpyAsync(hostErrors.data(), errors.data, hostErrors.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, gpu.stream));
    checkCuda(cudaStreamSynchronize(gpu.stream));
    double maxError = 0.0, relError = 0.0;
    for (unsigned int i = 0; i < blocks; ++i) {
        maxError = std::max(maxError, hostErrors[2 * i]);
        relError = std::max(relError, hostErrors[2 * i + 1]);
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || end == value || *end || *value == '-' || parsed == 0 ||
                parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
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
        GpuContext gpu;
        DeviceBuffer<double> A(n * n);
        DeviceBuffer<double> A_orig(validate ? n * n : 0);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(gpu, A.data, n);
        if (validate)
            checkCuda(cudaMemcpyAsync(A_orig.data, A.data, n * n * sizeof(double),
                                      cudaMemcpyDeviceToDevice, gpu.stream));

        // Allocate solver scratch space before timing; keep the matrix on the GPU.
        int workspaceSize = 0;
        checkSolver(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_UPPER,
                                               static_cast<int>(n), A.data, static_cast<int>(n),
                                               &workspaceSize));
        DeviceBuffer<double> workspace(static_cast<size_t>(workspaceSize));
        DeviceBuffer<int> info(1);
        checkCuda(cudaStreamSynchronize(gpu.stream));
        printf("Computing Cholesky decomposition...\n");
        const auto start = std::chrono::steady_clock::now();
        const bool success = choleskyDecomposition(gpu, A.data, n, workspace.data, workspaceSize, info.data);
        const auto end = std::chrono::steady_clock::now();
        if (!success) {
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        const double seconds = std::chrono::duration<double>(end - start).count();
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / seconds / 1e9);

        if (printResults) {
            std::vector<double> hostA(n * n);
            checkCuda(cudaMemcpyAsync(hostA.data(), A.data, hostA.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, gpu.stream));
            checkCuda(cudaStreamSynchronize(gpu.stream));
            print_results(hostA, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(gpu, A.data, A_orig.data, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
    return 0;
}
