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
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
}
static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
static void checkSolver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER error: " + std::to_string(static_cast<int>(status)));
}

template <typename T> class DeviceBuffer {
    T* ptr_ = nullptr;
public:
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
};

class CudaHandles {
public:
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CudaHandles() {
        checkBlas(cublasCreate(&blas));
        try { checkSolver(cusolverDnCreate(&solver)); }
        catch (...) { cublasDestroy(blas); throw; }
    }
    ~CudaHandles() { cusolverDnDestroy(solver); cublasDestroy(blas); }
};

__global__ void addDiagonal(double* a, int n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) a[i * n + i] += n;
}

// cuSOLVER writes the column-major lower triangle. Move it to the
// row-major lower triangle, clearing each source after it has been read.
__global__ void makeRowMajorLower(double* a, int n) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * n;
    if (index < total) {
        size_t i = index / n, j = index % n;
        if (i > j) {
            size_t upper = j * n + i;
            a[index] = a[upper];
            a[upper] = 0.0;
        }
    }
}

__global__ void calculateErrors(const double* reconstructed, const double* original,
                                size_t count, unsigned long long* maxima) {
    __shared__ double absErrors[256], relErrors[256];
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double absError = 0.0, relError = 0.0;
    if (index < count) {
        absError = fabs(reconstructed[index] - original[index]);
        relError = absError / (fabs(original[index]) + 1e-10);
        // Match std::max's behavior for NaNs in the original validation loop.
        if (isnan(absError)) absError = 0.0;
        if (isnan(relError)) relError = 0.0;
    }
    absErrors[threadIdx.x] = absError;
    relErrors[threadIdx.x] = relError;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            absErrors[threadIdx.x] = fmax(absErrors[threadIdx.x], absErrors[threadIdx.x + stride]);
            relErrors[threadIdx.x] = fmax(relErrors[threadIdx.x], relErrors[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(maxima, __double_as_longlong(absErrors[0]));
        atomicMax(maxima + 1, __double_as_longlong(relErrors[0]));
    }
}

static unsigned int blocksFor(size_t count) {
    return static_cast<unsigned int>((count + 255) / 256);
}

// Preserve the original random sequence while evaluating B * B^T on the GPU.
static void generatePositiveDefiniteMatrix(double* a, int n, cublasHandle_t blas) {
    if (!n) return;
    size_t count = static_cast<size_t>(n) * n;
    std::vector<double> b(count);
    unsigned int seed = 42;
    for (size_t i = 0; i < count; ++i)
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    DeviceBuffer<double> deviceB(count);
    checkCuda(cudaMemcpy(deviceB.get(), b.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    // Row-major B appears as B^T to column-major cuBLAS.
    checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
                          deviceB.get(), n, deviceB.get(), n, &zero, a, n));
    addDiagonal<<<blocksFor(n), 256>>>(a, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

static bool choleskyDecomposition(double* a, int n, cusolverDnHandle_t solver) {
    if (!n) return true;
    int workspaceSize = 0;
    checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, n, a, n, &workspaceSize));
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> info(1);
    checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, n, a, n,
                                workspace.get(), workspaceSize, info.get()));
    int result = 0;
    checkCuda(cudaMemcpy(&result, info.get(), sizeof(result), cudaMemcpyDeviceToHost));
    if (result > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", result - 1);
        return false;
    }
    if (result < 0) throw std::runtime_error("cuSOLVER potrf received an invalid argument: " + std::to_string(-result));
    makeRowMajorLower<<<blocksFor(static_cast<size_t>(n) * n), 256>>>(a, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    return true;
}

static bool validateCholesky(const double* l, const double* original, int n, cublasHandle_t blas) {
    double errors[2] = {0.0, 0.0};
    if (n) {
        size_t count = static_cast<size_t>(n) * n;
        DeviceBuffer<double> reconstructed(count);
        DeviceBuffer<unsigned long long> maxima(2);
        checkCuda(cudaMemset(maxima.get(), 0, 2 * sizeof(unsigned long long)));
        const double one = 1.0, zero = 0.0;
        checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
                              l, n, l, n, &zero, reconstructed.get(), n));
        calculateErrors<<<blocksFor(count), 256>>>(reconstructed.get(), original, count, maxima.get());
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(errors, maxima.get(), sizeof(errors), cudaMemcpyDeviceToHost));
    }
    printf("Max absolute error: %.10e\n", errors[0]);
    printf("Max relative error: %.10e\n", errors[1]);
    if (errors[1] > 1e-6) {
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
    bool validate = false, printResults = false;
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
    if (n > INT_MAX || (n && n > std::numeric_limits<size_t>::max() / n / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    try {
        CudaHandles handles;
        size_t count = n * n;
        DeviceBuffer<double> a(count);
        DeviceBuffer<double> original(validate ? count : 0);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(a.get(), static_cast<int>(n), handles.blas);
        if (validate && count)
            checkCuda(cudaMemcpy(original.get(), a.get(), count * sizeof(double), cudaMemcpyDeviceToDevice));
        printf("Computing Cholesky decomposition...\n");
        auto start = std::chrono::high_resolution_clock::now();
        bool success = choleskyDecomposition(a.get(), static_cast<int>(n), handles.solver);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (!success) {
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        printf("Computation time: %ld ms\n", duration.count());
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            std::vector<double> result(count);
            if (count) checkCuda(cudaMemcpy(result.data(), a.get(), count * sizeof(double), cudaMemcpyDeviceToHost));
            print_results(result, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(a.get(), original.get(), static_cast<int>(n), handles.blas);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
