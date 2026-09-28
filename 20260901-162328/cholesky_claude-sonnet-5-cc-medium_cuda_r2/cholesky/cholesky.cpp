#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition, parallelized with CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Mathematically equivalent to the classical unblocked algorithm: Cholesky
// factorization of an SPD matrix with positive diagonal is unique, so the
// blocked right-looking formulation produces the same L (up to floating
// point rounding from reordered operations).

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static constexpr int BLOCK_SIZE = 32;

// Factorizes a single diagonal block (m x m, m <= BLOCK_SIZE) in place using
// an unblocked Cholesky within shared memory. Runs as a single thread block.
__global__ void potf2_kernel(double* A, long n, long koff, int m, int* d_error, int* d_errorIdx) {
    __shared__ double sA[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (tx < m && ty < m) {
        sA[ty][tx] = A[(koff + ty) * n + (koff + tx)];
    }
    __syncthreads();

    if (ty == 0) {
        for (int j = 0; j < m; ++j) {
            if (tx == j) {
                double sum = 0.0;
                for (int p = 0; p < j; ++p) {
                    sum += sA[j][p] * sA[j][p];
                }
                const double val = sA[j][j] - sum;
                if (val <= 0.0) {
                    *d_error = 1;
                    *d_errorIdx = static_cast<int>(koff) + j;
                    sA[j][j] = 0.0;
                } else {
                    sA[j][j] = sqrt(val);
                }
            }
            __syncwarp();
            if (tx > j && tx < m) {
                double sum = 0.0;
                for (int p = 0; p < j; ++p) {
                    sum += sA[tx][p] * sA[j][p];
                }
                sA[tx][j] = (sA[tx][j] - sum) / sA[j][j];
            }
            __syncwarp();
        }
    }
    __syncthreads();

    if (tx < m && ty < m && ty >= tx) {
        A[(koff + ty) * n + (koff + tx)] = sA[ty][tx];
    }
}

// Panel update (triangular solve): for each row-block i > k, solves
// A_ik <- A_ik * inv(L_kk)^T via row-wise forward substitution.
// One CUDA block per panel row-block; blockIdx.x selects the row-block.
__global__ void trsm_kernel(double* A, long n, long koff, int m, long k, int BS, long nb) {
    __shared__ double sL[BLOCK_SIZE][BLOCK_SIZE];

    const long i = k + 1 + blockIdx.x;
    if (i >= nb) return;
    const long ioff = i * BS;
    const long remI = n - ioff;
    const int mi = static_cast<int>(remI < BS ? remI : BS);

    const int tx = threadIdx.x;

    for (int c = tx; c < m; c += blockDim.x) {
        for (int r = 0; r < m; ++r) {
            sL[r][c] = A[(koff + r) * n + (koff + c)];
        }
    }
    __syncthreads();

    if (tx < mi) {
        double row[BLOCK_SIZE];
        for (int c = 0; c < m; ++c) {
            row[c] = A[(ioff + tx) * n + (koff + c)];
        }
        for (int c = 0; c < m; ++c) {
            double sum = 0.0;
            for (int p = 0; p < c; ++p) {
                sum += row[p] * sL[c][p];
            }
            row[c] = (row[c] - sum) / sL[c][c];
        }
        for (int c = 0; c < m; ++c) {
            A[(ioff + tx) * n + (koff + c)] = row[c];
        }
    }
}

// Trailing submatrix update: A_ij -= A_ik * A_jk^T for k < j <= i < nb.
// One CUDA block per (i, j) block pair; blocks with j > i are skipped.
__global__ void syrk_kernel(double* A, long n, long koff, int m, long k, int BS, long nb) {
    __shared__ double sAik[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sAjk[BLOCK_SIZE][BLOCK_SIZE];

    const long ii = k + 1 + blockIdx.y;
    const long jj = k + 1 + blockIdx.x;
    if (ii >= nb || jj >= nb || jj > ii) return;

    const long ioff = ii * BS;
    const long joff = jj * BS;
    const long remI = n - ioff;
    const long remJ = n - joff;
    const int mi = static_cast<int>(remI < BS ? remI : BS);
    const int mj = static_cast<int>(remJ < BS ? remJ : BS);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (tx < m && ty < mi) {
        sAik[ty][tx] = A[(ioff + ty) * n + (koff + tx)];
    }
    if (tx < m && ty < mj) {
        sAjk[ty][tx] = A[(joff + ty) * n + (koff + tx)];
    }
    __syncthreads();

    if (ty < mi && tx < mj) {
        double sum = 0.0;
        for (int p = 0; p < m; ++p) {
            sum += sAik[ty][p] * sAjk[tx][p];
        }
        A[(ioff + ty) * n + (joff + tx)] -= sum;
    }
}

// Zeros the strictly upper triangular part of A (matches the reference
// algorithm, which never writes above the diagonal).
__global__ void zero_upper_kernel(double* A, long n) {
    const long col = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long row = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const long N = static_cast<long>(n);
    const int BS = BLOCK_SIZE;
    const long nb = (N + BS - 1) / BS;

    double* d_A = nullptr;
    int* d_error = nullptr;
    int* d_errorIdx = nullptr;
    const size_t bytes = n * n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errorIdx, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    bool success = true;

    for (long k = 0; k < nb; ++k) {
        const long koff = k * BS;
        const int m = static_cast<int>(std::min<long>(BS, N - koff));

        const int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_error, &zero, sizeof(int), cudaMemcpyHostToDevice));

        potf2_kernel<<<1, dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_A, N, koff, m, d_error, d_errorIdx);
        CUDA_CHECK(cudaGetLastError());

        int hErr = 0, hErrIdx = 0;
        CUDA_CHECK(cudaMemcpy(&hErr, d_error, sizeof(int), cudaMemcpyDeviceToHost));
        if (hErr) {
            CUDA_CHECK(cudaMemcpy(&hErrIdx, d_errorIdx, sizeof(int), cudaMemcpyDeviceToHost));
            printf("Error: Matrix is not positive definite at diagonal element %d\n", hErrIdx);
            success = false;
            break;
        }

        const long nPanels = nb - k - 1;
        if (nPanels > 0) {
            trsm_kernel<<<static_cast<unsigned int>(nPanels), BLOCK_SIZE>>>(d_A, N, koff, m, k, BS, nb);
            CUDA_CHECK(cudaGetLastError());

            dim3 grid(static_cast<unsigned int>(nPanels), static_cast<unsigned int>(nPanels));
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            syrk_kernel<<<grid, block>>>(d_A, N, koff, m, k, BS, nb);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (success) {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(static_cast<unsigned int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE),
                   static_cast<unsigned int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE));
        zero_upper_kernel<<<grid, block>>>(d_A, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(d_A);
    cudaFree(d_error);
    cudaFree(d_errorIdx);

    return success;
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
