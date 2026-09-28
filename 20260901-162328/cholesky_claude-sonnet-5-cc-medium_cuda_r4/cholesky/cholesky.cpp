#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition, parallelized on the GPU with CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Right-looking blocked algorithm: for each diagonal block k, factorize it,
// solve the panel below it (triangular solve), then update the trailing
// submatrix with a rank-B update. Each of these three phases is a CUDA kernel
// operating on many tiles in parallel; only the dependency between successive
// k iterations is sequential.

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                  \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

namespace {
constexpr int kBlockDim = 32; // Tile size (B). Matches warp width.
}

// Factorize the bs x bs diagonal block at (koff, koff) in place (unblocked
// Cholesky on a tile held in shared memory). Sets *failFlag/*failIndex if the
// matrix is not positive definite, mirroring the original algorithm's check.
__global__ void potrf_block_kernel(double* A, int n, int B, int k, int* failFlag, int* failIndex) {
    __shared__ double tile[kBlockDim][kBlockDim];

    const int koff = k * B;
    const int bs = min(B, n - koff);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (tx < bs && ty < bs) {
        tile[ty][tx] = A[(size_t)(koff + ty) * n + (koff + tx)];
    }
    __syncthreads();

    if (tx == 0 && ty == 0) {
        for (int j = 0; j < bs; ++j) {
            double sum = 0.0;
            for (int m = 0; m < j; ++m) {
                sum += tile[j][m] * tile[j][m];
            }
            const double val = tile[j][j] - sum;
            if (val <= 0.0) {
                *failFlag = 1;
                *failIndex = koff + j;
                break;
            }
            const double Ljj = sqrt(val);
            tile[j][j] = Ljj;
            for (int i = j + 1; i < bs; ++i) {
                double s = 0.0;
                for (int m = 0; m < j; ++m) {
                    s += tile[i][m] * tile[j][m];
                }
                tile[i][j] = (tile[i][j] - s) / Ljj;
            }
        }
    }
    __syncthreads();

    if (tx < bs && ty < bs) {
        A[(size_t)(koff + ty) * n + (koff + tx)] = (ty >= tx) ? tile[ty][tx] : 0.0;
    }
}

// Triangular solve of the panel blocks below the diagonal block k:
// for each row-block i > k, solves L_ik * L_kk^T = A_ik for L_ik (in place).
__global__ void trsm_panel_kernel(double* A, int n, int B, int k) {
    __shared__ double Lkk[kBlockDim][kBlockDim];
    __shared__ double Xrow[kBlockDim][kBlockDim];

    const int koff = k * B;
    const int bsk = min(B, n - koff);

    const int i = k + 1 + blockIdx.x;
    const int ioff = i * B;
    const int bsi = min(B, n - ioff);

    const int tx = threadIdx.x;

    if (tx < bsk) {
        for (int c = 0; c < bsk; ++c) {
            Lkk[c][tx] = A[(size_t)(koff + c) * n + (koff + tx)];
        }
    }
    __syncthreads();

    if (tx < bsi) {
        for (int c = 0; c < bsk; ++c) {
            double sum = 0.0;
            for (int m = 0; m < c; ++m) {
                sum += Xrow[tx][m] * Lkk[c][m];
            }
            const double aval = A[(size_t)(ioff + tx) * n + (koff + c)];
            Xrow[tx][c] = (aval - sum) / Lkk[c][c];
        }
    }
    __syncthreads();

    if (tx < bsi) {
        for (int c = 0; c < bsk; ++c) {
            A[(size_t)(ioff + tx) * n + (koff + c)] = Xrow[tx][c];
        }
    }
}

// Trailing submatrix rank-B update: for each lower-triangular block pair
// (i, j) with k < j <= i, computes A_ij -= L_ik * L_jk^T.
__global__ void gemm_update_kernel(double* A, int n, int B, int k) {
    const int j = k + 1 + blockIdx.y;
    const int i = k + 1 + blockIdx.x;
    if (j > i) {
        return;
    }

    __shared__ double Lik[kBlockDim][kBlockDim];
    __shared__ double Ljk[kBlockDim][kBlockDim];

    const int koff = k * B;
    const int bsk = min(B, n - koff);
    const int ioff = i * B;
    const int bsi = min(B, n - ioff);
    const int joff = j * B;
    const int bsj = min(B, n - joff);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (tx < bsk && ty < bsi) {
        Lik[ty][tx] = A[(size_t)(ioff + ty) * n + (koff + tx)];
    }
    if (tx < bsk && ty < bsj) {
        Ljk[ty][tx] = A[(size_t)(joff + ty) * n + (koff + tx)];
    }
    __syncthreads();

    if (tx < bsj && ty < bsi) {
        double sum = 0.0;
        for (int m = 0; m < bsk; ++m) {
            sum += Lik[ty][m] * Ljk[tx][m];
        }
        A[(size_t)(ioff + ty) * n + (joff + tx)] -= sum;
    }
}

// Zeros the strictly-upper-triangular part of A, matching the reference
// implementation's post-processing of each row.
__global__ void zero_upper_kernel(double* A, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) {
        A[(size_t)i * n + j] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const int B = kBlockDim;
    const int nInt = (int)n;
    const int nb = (nInt + B - 1) / B;

    double* dA = nullptr;
    int* dFailFlag = nullptr;
    int* dFailIndex = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFailFlag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dFailIndex, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    bool ok = true;

    for (int k = 0; k < nb; ++k) {
        const int zero = 0;
        CUDA_CHECK(cudaMemcpy(dFailFlag, &zero, sizeof(int), cudaMemcpyHostToDevice));

        potrf_block_kernel<<<1, dim3(B, B)>>>(dA, nInt, B, k, dFailFlag, dFailIndex);
        CUDA_CHECK(cudaGetLastError());

        int failFlag = 0;
        int failIndex = 0;
        CUDA_CHECK(cudaMemcpy(&failFlag, dFailFlag, sizeof(int), cudaMemcpyDeviceToHost));
        if (failFlag) {
            CUDA_CHECK(cudaMemcpy(&failIndex, dFailIndex, sizeof(int), cudaMemcpyDeviceToHost));
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failIndex);
            ok = false;
            break;
        }

        const int rem = nb - k - 1;
        if (rem > 0) {
            trsm_panel_kernel<<<rem, B>>>(dA, nInt, B, k);
            CUDA_CHECK(cudaGetLastError());

            const dim3 grid(rem, rem);
            gemm_update_kernel<<<grid, dim3(B, B)>>>(dA, nInt, B, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (ok) {
        const dim3 blockDim(32, 32);
        const dim3 gridDim((nInt + blockDim.x - 1) / blockDim.x, (nInt + blockDim.y - 1) / blockDim.y);
        zero_upper_kernel<<<gridDim, blockDim>>>(dA, nInt);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dFailFlag));
    CUDA_CHECK(cudaFree(dFailIndex));

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

    // Warm up the CUDA context/device so its one-time initialization cost
    // isn't counted as part of the timed decomposition below.
    CUDA_CHECK(cudaFree(0));

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
