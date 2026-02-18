#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static const char* cudaGetErrorStringSafe(cudaError_t e) {
    const char* s = cudaGetErrorString(e);
    return s ? s : "<unknown>";
}

static bool checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        printf("CUDA error (%s): %s\n", what, cudaGetErrorStringSafe(e));
        return false;
    }
    return true;
}

static bool checkCusolver(cusolverStatus_t s, const char* what) {
    if (s != CUSOLVER_STATUS_SUCCESS) {
        printf("cuSOLVER error (%s): %d\n", what, (int)s);
        return false;
    }
    return true;
}

__global__ void zeroUpperTriangleRowMajor(double* A, int n) {
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i < n && j < n && j > i) {
        A[(size_t)i * (size_t)n + (size_t)j] = 0.0;
    }
}

// CUDA-accelerated Cholesky decomposition using cuSOLVER.
// Note: cuSOLVER expects column-major. Since the generated matrix is exactly symmetric,
// passing the row-major buffer is equivalent up to transpose; using UPLO=UPPER yields
// a lower-triangular factor in the row-major view (L = U^T), matching original semantics.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const size_t bytes = n * n * sizeof(double);

    // Best-effort pinning for faster H2D/D2H.
    bool pinned = (cudaHostRegister(A.data(), bytes, cudaHostRegisterDefault) == cudaSuccess);

    cudaStream_t stream{};
    if (!checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    cusolverDnHandle_t solver{};
    if (!checkCusolver(cusolverDnCreate(&solver), "cusolverDnCreate")) {
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }
    checkCusolver(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");

    double* d_A = nullptr;
    int* d_info = nullptr;
    if (!checkCuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A") ||
        !checkCuda(cudaMalloc((void**)&d_info, sizeof(int)), "cudaMalloc d_info")) {
        if (d_A) cudaFree(d_A);
        if (d_info) cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    if (!checkCuda(cudaMemcpyAsync(d_A, A.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D memcpy")) {
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    int lwork = 0;
    if (!checkCusolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, (int)n, d_A, (int)n, &lwork),
                       "cusolverDnDpotrf_bufferSize")) {
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    double* d_work = nullptr;
    if (!checkCuda(cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double)), "cudaMalloc d_work")) {
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    if (!checkCusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, (int)n, d_A, (int)n, d_work, lwork, d_info),
                       "cusolverDnDpotrf")) {
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    int info_h = 0;
    if (!checkCuda(cudaMemcpyAsync(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H info")) {
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    // Zero upper triangle in the row-major view.
    dim3 block(32, 8, 1);
    dim3 grid((unsigned)((n + block.x - 1) / block.x), (unsigned)((n + block.y - 1) / block.y), 1);
    zeroUpperTriangleRowMajor<<<grid, block, 0, stream>>>(d_A, (int)n);
    if (!checkCuda(cudaGetLastError(), "zeroUpperTriangleRowMajor launch")) {
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    if (!checkCuda(cudaMemcpyAsync(A.data(), d_A, bytes, cudaMemcpyDeviceToHost, stream), "D2H memcpy")) {
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_info);
        cusolverDnDestroy(solver);
        cudaStreamDestroy(stream);
        if (pinned) cudaHostUnregister(A.data());
        return false;
    }

    cudaFree(d_work);
    cudaFree(d_A);
    cudaFree(d_info);
    cusolverDnDestroy(solver);
    cudaStreamDestroy(stream);
    if (pinned) cudaHostUnregister(A.data());

    if (info_h > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info_h - 1);
        return false;
    } else if (info_h < 0) {
        printf("Error: Invalid argument %d in Cholesky factorization\n", -info_h);
        return false;
    }

    return true;
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

static void warmupCuda() {
    // Force CUDA context creation and cuSOLVER load/JIT outside the timed region.
    cudaFree(0);

    cudaStream_t stream{};
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return;

    cusolverDnHandle_t solver{};
    if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        return;
    }
    cusolverDnSetStream(solver, stream);

    double* d_A = nullptr;
    int* d_info = nullptr;
    double* d_work = nullptr;
    cudaMalloc((void**)&d_A, sizeof(double));
    cudaMalloc((void**)&d_info, sizeof(int));
    double one = 1.0;
    cudaMemcpyAsync(d_A, &one, sizeof(double), cudaMemcpyHostToDevice, stream);

    int lwork = 0;
    if (cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, 1, d_A, 1, &lwork) == CUSOLVER_STATUS_SUCCESS) {
        cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));
        cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, 1, d_A, 1, d_work, lwork, d_info);
    }

    cudaStreamSynchronize(stream);
    if (d_work) cudaFree(d_work);
    if (d_A) cudaFree(d_A);
    if (d_info) cudaFree(d_info);
    cusolverDnDestroy(solver);
    cudaStreamDestroy(stream);
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

    warmupCuda();

    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
