#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 32
#define SYRK_TILE 16

// ============================================================
// Kernel 1: Factor a diagonal block (up to BLOCK_SIZE x BLOCK_SIZE)
// One thread block with BLOCK_SIZE threads, column-by-column
// ============================================================
__global__ void factor_diag_block(double* __restrict__ A, int n, int jb, int bs, int* error_flag) {
    __shared__ double s[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tid = threadIdx.x;

    if (tid < bs) {
        for (int c = 0; c < bs; c++)
            s[tid][c] = A[(jb + tid) * n + (jb + c)];
    }
    __syncthreads();

    for (int j = 0; j < bs; j++) {
        if (tid == j) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += s[j][k] * s[j][k];
            double val = s[j][j] - sum;
            if (val <= 0.0) {
                *error_flag = 1;
                s[j][j] = 1.0;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();

        if (tid > j && tid < bs) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += s[tid][k] * s[j][k];
            s[tid][j] = (s[tid][j] - sum) / s[j][j];
        }
        __syncthreads();
    }

    // Write back, zeroing upper triangular part within this block
    if (tid < bs) {
        for (int c = 0; c < bs; c++)
            A[(jb + tid) * n + (jb + c)] = (c <= tid) ? s[tid][c] : 0.0;
    }
}

// ============================================================
// Kernel 2: TRSM - solve panel below diagonal block
// L21 * L11^T = A21, each thread handles one row independently
// ============================================================
__global__ void trsm_lower(double* __restrict__ A, int n, int jb, int bs) {
    __shared__ double L11[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tid = threadIdx.x;

    // Cooperatively load the factored diagonal block into shared memory
    for (int idx = tid; idx < bs * bs; idx += blockDim.x) {
        int r = idx / bs;
        int c = idx % bs;
        L11[r][c] = A[(jb + r) * n + (jb + c)];
    }
    __syncthreads();

    int i = jb + bs + blockIdx.x * blockDim.x + tid;
    if (i >= n) return;

    // Load panel row into registers
    double row[BLOCK_SIZE];
    for (int j = 0; j < bs; j++)
        row[j] = A[i * n + (jb + j)];

    // Forward substitution
    for (int j = 0; j < bs; j++) {
        double sum = 0.0;
        for (int k = 0; k < j; k++)
            sum += row[k] * L11[j][k];
        row[j] = (row[j] - sum) / L11[j][j];
    }

    // Write back
    for (int j = 0; j < bs; j++)
        A[i * n + (jb + j)] = row[j];
}

// ============================================================
// Kernel 3: SYRK - update trailing lower-triangular submatrix
// A22 -= L21 * L21^T, with shared memory tiling
// ============================================================
__global__ void syrk_lower(double* __restrict__ A, int n, int jb, int bs) {
    int row_start = jb + bs;

    int brow = blockIdx.y;
    int bcol = blockIdx.x;
    if (brow < bcol) return; // only lower triangular blocks

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = row_start + brow * SYRK_TILE + ty;
    int j = row_start + bcol * SYRK_TILE + tx;

    double sum = 0.0;

    __shared__ double sA[SYRK_TILE][SYRK_TILE + 1];
    __shared__ double sB[SYRK_TILE][SYRK_TILE + 1];

    for (int kk = 0; kk < bs; kk += SYRK_TILE) {
        int k_len = bs - kk;
        if (k_len > SYRK_TILE) k_len = SYRK_TILE;

        // Load tile of L21 for row block (coalesced: consecutive tx = consecutive columns)
        sA[ty][tx] = (i < n && tx < k_len) ? A[i * n + (jb + kk + tx)] : 0.0;

        // Load tile of L21 for col block (transposed store for coalesced reads)
        int j_ld = row_start + bcol * SYRK_TILE + ty;
        sB[tx][ty] = (j_ld < n && tx < k_len) ? A[j_ld * n + (jb + kk + tx)] : 0.0;

        __syncthreads();

        for (int k = 0; k < k_len; k++)
            sum += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }

    if (i < n && j < n && i >= j)
        A[i * n + j] -= sum;
}

// ============================================================
// Kernel 4: Zero upper triangular part
// ============================================================
__global__ void zero_upper(double* __restrict__ A, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i)
        A[i * n + j] = 0.0;
}

// ============================================================
// GPU Blocked Cholesky Decomposition
// ============================================================
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int N = static_cast<int>(n);
    size_t matrix_bytes = n * n * sizeof(double);

    double* d_A;
    int* d_error;
    CHECK_CUDA(cudaMalloc(&d_A, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_error, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_error, 0, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), matrix_bytes, cudaMemcpyHostToDevice));

    for (int jb = 0; jb < N; jb += BLOCK_SIZE) {
        int bs = BLOCK_SIZE;
        if (N - jb < bs) bs = N - jb;

        // 1. Factor diagonal block
        factor_diag_block<<<1, BLOCK_SIZE>>>(d_A, N, jb, bs, d_error);

        int panel_rows = N - jb - bs;
        if (panel_rows > 0) {
            // 2. TRSM: solve panel below diagonal block
            int trsm_threads = 128;
            int trsm_blocks = (panel_rows + trsm_threads - 1) / trsm_threads;
            trsm_lower<<<trsm_blocks, trsm_threads>>>(d_A, N, jb, bs);

            // 3. SYRK: update trailing submatrix
            int grid_dim = (panel_rows + SYRK_TILE - 1) / SYRK_TILE;
            dim3 syrk_grid(grid_dim, grid_dim);
            dim3 syrk_block(SYRK_TILE, SYRK_TILE);
            syrk_lower<<<syrk_grid, syrk_block>>>(d_A, N, jb, bs);
        }
    }

    // Zero upper triangular part
    dim3 zero_block(16, 16);
    dim3 zero_grid((N + 15) / 16, (N + 15) / 16);
    zero_upper<<<zero_grid, zero_block>>>(d_A, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    int h_error = 0;
    CHECK_CUDA(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(A.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_error));

    if (h_error) {
        printf("Error: Matrix is not positive definite\n");
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

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

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    // Warm up CUDA context before timing
    CHECK_CUDA(cudaFree(0));

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

    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(A, "CholeskyL");
    }

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
