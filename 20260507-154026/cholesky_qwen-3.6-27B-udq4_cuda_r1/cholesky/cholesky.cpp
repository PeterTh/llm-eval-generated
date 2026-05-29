#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// 1.  Matrix generation (row-major, on device)
// ---------------------------------------------------------------------------

__global__ void generateRandomKernel(double* B, size_t n, unsigned int seed) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = n * n;
    for (size_t i = idx; i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        unsigned int state = seed + static_cast<unsigned int>(i);
        state = state * 1103515245u + 12345u;
        double val = (static_cast<double>((state >> 16) & 0x7FFF) /
                      static_cast<double>(RAND_MAX)) - 0.5;
        B[i] = val;
    }
}

__global__ void computePositiveDefiniteKernel(double* A, const double* B, size_t n) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k)
            sum += B[row * n + k] * B[col * n + k];
        A[row * n + col] = sum;
        if (row == col)
            A[row * n + col] += static_cast<double>(n);
    }
}

// ---------------------------------------------------------------------------
// 2.  Cholesky kernels (row-major, on device)
//
//  Two-phase approach: for each column j, first compute the diagonal
//  (single thread), then compute all off-diagonal elements (grid-stride).
//  CUDA stream ordering guarantees the diagonal is visible before reads.
// ---------------------------------------------------------------------------

__global__ void choleskyDiagonalKernel(double* A, size_t n, size_t j) {
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k)
        sum += A[j * n + k] * A[j * n + k];
    double val = A[j * n + j] - sum;
    A[j * n + j] = (val > 0.0) ? sqrt(val) : 0.0;
}

__global__ void choleskyOffDiagKernel(double* A, size_t n, size_t j) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row > j && row < n) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k)
            sum += A[row * n + k] * A[j * n + k];
        A[row * n + j] = (A[row * n + j] - sum) / A[j * n + j];
    }
}

// ---------------------------------------------------------------------------
// 3.  Upper-triangle zeroing  (row-major, grid-stride)
// ---------------------------------------------------------------------------

__global__ void zeroUpperTriangleKernel(double* A, size_t n) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row)
        A[row * n + col] = 0.0;
}

// ---------------------------------------------------------------------------
// 4.  Validation kernels  (row-major)
// ---------------------------------------------------------------------------

__global__ void computeReconstructionKernel(double* reconstructed,
                                             const double* L, size_t n) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k)
            sum += L[row * n + k] * L[col * n + k];
        reconstructed[row * n + col] = sum;
    }
}

__global__ void computeErrorKernel(const double* reconstructed,
                                    const double* A_orig, size_t n,
                                    double* d_maxAbs, double* d_maxRel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = n * n;
    double localAbs = 0.0, localRel = 0.0;
    for (size_t ii = idx; ii < total;
         ii += static_cast<size_t>(blockDim.x) * gridDim.x) {
        double e = fabs(reconstructed[ii] - A_orig[ii]);
        if (e > localAbs) localAbs = e;
        double r = e / (fabs(A_orig[ii]) + 1e-10);
        if (r > localRel) localRel = r;
    }
    __shared__ double sAbs[256], sRel[256];
    int tid = threadIdx.x;
    sAbs[tid] = localAbs;  sRel[tid] = localRel;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sAbs[tid] = fmax(sAbs[tid], sAbs[tid + s]);
            sRel[tid] = fmax(sRel[tid], sRel[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(d_maxAbs),
                  *reinterpret_cast<const unsigned long long*>(&sAbs[0]));
        atomicMax(reinterpret_cast<unsigned long long*>(d_maxRel),
                  *reinterpret_cast<const unsigned long long*>(&sRel[0]));
    }
}

// ---------------------------------------------------------------------------
// 5.  Public API functions
// ---------------------------------------------------------------------------

bool choleskyDecomposition(double* d_A, const size_t n) {
    const int THR = 256;
    int blocks = static_cast<int>((n + THR - 1) / THR);

    // Column-by-column Cholesky: two kernels per column
    for (size_t j = 0; j < n; ++j) {
        // Phase 1: compute diagonal element
        choleskyDiagonalKernel<<<1, 1>>>(d_A, n, j);
        // Phase 2: compute off-diagonal elements
        choleskyOffDiagKernel<<<blocks, THR>>>(d_A, n, j);
    }

    // Zero upper triangle
    {
        dim3 b(16, 16);
        dim3 g((n + 15) / 16, (n + 15) / 16);
        zeroUpperTriangleKernel<<<g, b>>>(d_A, n);
    }

    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess);
}

void generatePositiveDefiniteMatrix(double* d_A, const size_t n) {
    double* d_B = nullptr;
    cudaMalloc(&d_B, n * n * sizeof(double));
    size_t total = n * n;
    int thr = 256;
    int blk = static_cast<int>((total + thr - 1) / thr);
    generateRandomKernel<<<blk, thr>>>(d_B, n, 42);

    dim3 b(16, 16);
    dim3 g((n + 15) / 16, (n + 15) / 16);
    computePositiveDefiniteKernel<<<g, b>>>(d_A, d_B, n);
    cudaDeviceSynchronize();
    cudaFree(d_B);
}

bool validateCholesky(const double* d_L, const double* d_A_orig, const size_t n) {
    double* d_rec = nullptr;
    cudaMalloc(&d_rec, n * n * sizeof(double));

    dim3 b(16, 16);
    dim3 g((n + 15) / 16, (n + 15) / 16);
    computeReconstructionKernel<<<g, b>>>(d_rec, d_L, n);
    cudaDeviceSynchronize();

    double h_abs = 0.0, h_rel = 0.0;
    double* d_abs = nullptr, *d_rel = nullptr;
    cudaMalloc(&d_abs, sizeof(double));
    cudaMalloc(&d_rel, sizeof(double));
    cudaMemcpy(d_abs, &h_abs, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rel, &h_rel, sizeof(double), cudaMemcpyHostToDevice);

    int thr = 256;
    int blk = static_cast<int>((n * n + thr - 1) / thr);
    computeErrorKernel<<<blk, thr>>>(d_rec, d_A_orig, n, d_abs, d_rel);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_abs, d_abs, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rel, d_rel, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Max absolute error: %.10e\n", h_abs);
    printf("Max relative error: %.10e\n", h_rel);

    bool ok = (h_rel <= 1e-6);
    if (!ok) printf("Validation failed: relative error too large\n");

    cudaFree(d_rec); cudaFree(d_abs); cudaFree(d_rel);
    return ok;
}

// ---------------------------------------------------------------------------
// 6.  main
// ---------------------------------------------------------------------------

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
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]); return 1;
        }
    }

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    double* d_A = nullptr;
    cudaMalloc(&d_A, n * n * sizeof(double));
    double* d_A_orig = nullptr;
    if (validate)
        cudaMalloc(&d_A_orig, n * n * sizeof(double));

    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d_A, n);

    if (validate)
        cudaMemcpy(d_A_orig, d_A, n * n * sizeof(double),
                   cudaMemcpyDeviceToDevice);

    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(d_A, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        cudaFree(d_A);
        if (validate) cudaFree(d_A_orig);
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        std::vector<double> h_A(n * n);
        cudaMemcpy(h_A.data(), d_A, n * n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        print_results(h_A, "CholeskyL");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(d_A, d_A_orig, n);
        printf("%s\n", valid ? "Validation: PASSED" : "Validation: FAILED");
        cudaFree(d_A_orig);
        cudaFree(d_A);
        return valid ? 0 : 1;
    }

    cudaFree(d_A);
    return 0;
}
