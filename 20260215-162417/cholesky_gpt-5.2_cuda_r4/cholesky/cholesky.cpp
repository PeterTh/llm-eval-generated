#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

__global__ void zeroUpperRowMajor(double* A, int n) {
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i < n && j < n && i < j) {
        A[(size_t)i * (size_t)n + (size_t)j] = 0.0;
    }
}

static inline bool cusolverOk(cusolverStatus_t st, const char* what) {
    if (st == CUSOLVER_STATUS_SUCCESS) return true;
    printf("cuSolver error in %s (status=%d)\n", what, (int)st);
    return false;
}

static inline bool cudaOk(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return true;
    printf("CUDA error in %s: %s\n", what, cudaGetErrorString(st));
    return false;
}

// CUDA Cholesky decomposition using cuSolver (POTRF).
// Input/Output: A stored in row-major order; output is L in the lower triangle with upper triangle zero.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, double& elapsedMs) {
    elapsedMs = 0.0;
    if (n == 0) return true;
    if (n > (size_t)std::numeric_limits<int>::max()) {
        printf("Error: matrix too large (n=%zu)\n", n);
        return false;
    }

    const int ni = (int)n;
    const size_t nn = n * n;
    if (n != 0 && nn / n != n) {
        printf("Error: matrix size overflow (n=%zu)\n", n);
        return false;
    }

    const size_t bytes = nn * sizeof(double);

    cusolverDnHandle_t handle = nullptr;
    double* d_A = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    cudaEvent_t evStart = nullptr;
    cudaEvent_t evStop = nullptr;

    int lwork = 0;
    float ms = 0.0f;
    int info_h = 0;
    dim3 block(16, 16);
    dim3 grid(1, 1);

    bool ok = false;

    if (!cudaOk(cudaMalloc((void**)&d_A, bytes), "cudaMalloc(d_A)")) goto cleanup;
    if (!cudaOk(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D")) goto cleanup;

    if (!cusolverOk(cusolverDnCreate(&handle), "cusolverDnCreate")) goto cleanup;

    if (!cusolverOk(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, ni, d_A, ni, &lwork),
                    "cusolverDnDpotrf_bufferSize")) {
        goto cleanup;
    }

    if (!cudaOk(cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double)), "cudaMalloc(d_work)")) goto cleanup;
    if (!cudaOk(cudaMalloc((void**)&d_info, sizeof(int)), "cudaMalloc(d_info)")) goto cleanup;

    if (!cudaOk(cudaEventCreate(&evStart), "cudaEventCreate(start)")) goto cleanup;
    if (!cudaOk(cudaEventCreate(&evStop), "cudaEventCreate(stop)")) goto cleanup;

    if (!cudaOk(cudaEventRecord(evStart, 0), "cudaEventRecord(start)")) goto cleanup;

    if (!cusolverOk(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, ni, d_A, ni, d_work, lwork, d_info),
                    "cusolverDnDpotrf")) {
        goto cleanup;
    }

    grid = dim3((unsigned)((ni + (int)block.x - 1) / (int)block.x), (unsigned)((ni + (int)block.y - 1) / (int)block.y));
    zeroUpperRowMajor<<<grid, block>>>(d_A, ni);
    if (!cudaOk(cudaGetLastError(), "zeroUpperRowMajor launch")) goto cleanup;

    if (!cudaOk(cudaEventRecord(evStop, 0), "cudaEventRecord(stop)")) goto cleanup;
    if (!cudaOk(cudaEventSynchronize(evStop), "cudaEventSynchronize(stop)")) goto cleanup;

    ms = 0.0f;
    if (!cudaOk(cudaEventElapsedTime(&ms, evStart, evStop), "cudaEventElapsedTime")) goto cleanup;
    elapsedMs = (double)ms;

    info_h = 0;
    if (!cudaOk(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy info")) goto cleanup;
    if (info_h != 0) {
        if (info_h > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", info_h - 1);
        } else {
            printf("Error: cuSolver potrf failed (info=%d)\n", info_h);
        }
        goto cleanup;
    }

    if (!cudaOk(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) goto cleanup;

    ok = true;

cleanup:
    if (evStop) cudaEventDestroy(evStop);
    if (evStart) cudaEventDestroy(evStart);
    if (d_info) cudaFree(d_info);
    if (d_work) cudaFree(d_work);
    if (d_A) cudaFree(d_A);
    if (handle) cusolverDnDestroy(handle);
    return ok;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    double elapsedMs = 0.0;

    bool success = choleskyDecomposition(A, n, elapsedMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    long timeMs = (long)llround(elapsedMs);
    if (timeMs < 0) timeMs = 0;
    printf("Computation time: %ld ms\n", timeMs);

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double seconds = std::max(1e-9, elapsedMs / 1000.0);
    double gflops = ops / seconds / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
