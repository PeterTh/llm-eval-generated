#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

template <typename Status>
void check(Status status, const char* operation) {
    if (status != 0) {
        fprintf(stderr, "%s failed (status %d)\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define GPU(call) check((call), #call)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) GPU(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

// The row-major input B is B^T to cuBLAS. The column-major upper
// triangle of the product occupies the row-major lower triangle.
void gramMatrix(cublasHandle_t blas, int n, const double* b, double* a,
                double beta) {
    const double one = 1.0;
    GPU(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, n,
                    &one, b, n, &beta, a, n));
}

void generatePositiveDefiniteMatrix(cublasHandle_t blas, int n, double* a) {
    if (!n) return;
    const size_t count = size_t(n) * n;
    std::vector<double> host(count);
    unsigned int seed = 42;
    for (double& value : host) {
        value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }
    DeviceBuffer<double> b(count);
    GPU(cudaMemcpy(b.data, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));

    // Start from n*I. SYRK adds B*B^T and leaves the unused triangle zero.
    GPU(cudaMemset(a, 0, count * sizeof(double)));
    std::fill_n(host.begin(), n, static_cast<double>(n));
    GPU(cudaMemcpy2D(a, (size_t(n) + 1) * sizeof(double), host.data(), sizeof(double),
                     sizeof(double), n, cudaMemcpyHostToDevice));
    gramMatrix(blas, n, b.data, a, 1.0);
}

bool validateCholesky(cublasHandle_t blas, const double* factor,
                      const std::vector<double>& original, int n) {
    double maxError = 0.0;
    double relError = 0.0;
    if (n) {
        DeviceBuffer<double> product(original.size());
        gramMatrix(blas, n, factor, product.data, 0.0);
        std::vector<double> reconstructed(original.size());
        GPU(cudaMemcpy(reconstructed.data(), product.data, original.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
        // Both matrices are symmetric, so one triangle contains every error.
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                const size_t index = size_t(i) * n + j;
                const double error = fabs(reconstructed[index] - original[index]);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (fabs(original[index]) + 1e-10));
            }
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
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
    
    if (n > std::numeric_limits<int>::max() ||
        (n && n > std::vector<double>().max_size() / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    GPU(cublasCreate(&blas));
    GPU(cusolverDnCreate(&solver));
    DeviceBuffer<double> a(n * n);
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(blas, n, a.data);

    std::vector<double> original;
    if (validate && n) {
        original.resize(n * n);
        GPU(cudaMemcpy(original.data(), a.data, original.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
    }

    int workspaceSize = 0;
    if (n) {
        GPU(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, n,
                                        a.data, n, &workspaceSize));
    }
    DeviceBuffer<double> workspace(workspaceSize);
    DeviceBuffer<int> deviceInfo(1);
    GPU(cudaDeviceSynchronize());
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::steady_clock::now();
    int info = 0;
    if (n) {
        GPU(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, a.data, n,
                             workspace.data, workspaceSize, deviceInfo.data));
        GPU(cudaMemcpy(&info, deviceInfo.data, sizeof(info), cudaMemcpyDeviceToHost));
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (info != 0) {
        if (info > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    double ops = static_cast<double>(n) * n * n / 3.0;
    printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);

    if (printResults) {
        std::vector<double> result(n * n);
        if (n) GPU(cudaMemcpy(result.data(), a.data, result.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        print_results(result, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        valid = validateCholesky(blas, a.data, original, n);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    GPU(cusolverDnDestroy(solver));
    GPU(cublasDestroy(blas));
    return valid ? 0 : 1;
}
