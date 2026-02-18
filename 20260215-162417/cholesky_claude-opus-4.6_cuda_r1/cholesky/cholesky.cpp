#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int CHOL_BLOCK = 64;
constexpr int TILE_DIM = 16;

// Kernel: Cholesky factorization of a small diagonal block in shared memory
__global__ void chol_diag_kernel(double* __restrict__ A, int n, int jb, int bs) {
    extern __shared__ double sA[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    for (int idx = tid; idx < bs * bs; idx += nthreads) {
        int r = idx / bs, c = idx % bs;
        sA[idx] = A[(jb + r) * n + (jb + c)];
    }
    __syncthreads();

    for (int j = 0; j < bs; j++) {
        if (tid == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += sA[j * bs + k] * sA[j * bs + k];
            sA[j * bs + j] = sqrt(sA[j * bs + j] - sum);
        }
        __syncthreads();

        double diag_val = sA[j * bs + j];
        for (int i = j + 1 + tid; i < bs; i += nthreads) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += sA[i * bs + k] * sA[j * bs + k];
            sA[i * bs + j] = (sA[i * bs + j] - sum) / diag_val;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < bs * bs; idx += nthreads) {
        int r = idx / bs, c = idx % bs;
        A[(jb + r) * n + (jb + c)] = (c <= r) ? sA[idx] : 0.0;
    }
}

// Kernel: TRSM — solve panel rows below the diagonal block
// Each thread handles one full row, reading the factored diagonal block from shared memory
__global__ void trsm_kernel(double* __restrict__ A, int n, int jb, int bs, int trailing_rows) {
    extern __shared__ double sL[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Cooperatively load the diagonal block into shared memory
    for (int idx = tid; idx < bs * bs; idx += nthreads) {
        int r = idx / bs, c = idx % bs;
        sL[idx] = (c <= r) ? A[(jb + r) * n + (jb + c)] : 0.0;
    }
    __syncthreads();

    int row_idx = blockIdx.x * nthreads + tid;
    if (row_idx >= trailing_rows) return;
    int row = jb + bs + row_idx;

    for (int j = 0; j < bs; j++) {
        double sum = 0.0;
        for (int k = 0; k < j; k++)
            sum += A[row * n + (jb + k)] * sL[j * bs + k];
        A[row * n + (jb + j)] = (A[row * n + (jb + j)] - sum) / sL[j * bs + j];
    }
}

// Kernel: SYRK — tiled symmetric rank-k update of the trailing submatrix
// Computes C -= A_panel * A_panel^T  (lower triangular only)
__global__ void syrk_kernel(double* __restrict__ A, int n, int jb, int bs, int trailing_rows) {
    int tile_i = blockIdx.y;
    int tile_j = blockIdx.x;
    if (tile_j > tile_i) return;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = tile_i * TILE_DIM + ty;
    int col = tile_j * TILE_DIM + tx;
    int base = jb + bs;

    double val = 0.0;

    __shared__ double sA[TILE_DIM][TILE_DIM + 1];
    __shared__ double sB[TILE_DIM][TILE_DIM + 1];

    for (int kb = 0; kb < bs; kb += TILE_DIM) {
        int kk = kb + tx;
        sA[ty][tx] = (row < trailing_rows && kk < bs)
            ? A[(base + row) * n + jb + kk] : 0.0;

        kk = kb + ty;
        sB[tx][ty] = (col < trailing_rows && kk < bs)
            ? A[(base + col) * n + jb + kk] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            val += sA[ty][k] * sB[tx][k];

        __syncthreads();
    }

    if (row < trailing_rows && col < trailing_rows && col <= row)
        A[(base + row) * n + (base + col)] -= val;
}

// Kernel: zero the strict upper triangle
__global__ void zero_upper_kernel(double* __restrict__ A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int r = idx / n, c = idx % n;
    if (c > r) A[idx] = 0.0;
}

// Host driver for the blocked GPU Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    int N = (int)n;

    for (int jb = 0; jb < N; jb += CHOL_BLOCK) {
        int bs = (jb + CHOL_BLOCK <= N) ? CHOL_BLOCK : (N - jb);
        int trailing = N - jb - bs;

        // 1. Factor diagonal block
        int diag_threads = (bs < 256) ? bs : 256;
        int smem_diag = bs * bs * (int)sizeof(double);
        chol_diag_kernel<<<1, diag_threads, smem_diag>>>(d_A, N, jb, bs);

        if (trailing > 0) {
            // 2. TRSM: solve the panel below the diagonal block
            int trsm_threads = 128;
            int trsm_blocks = (trailing + trsm_threads - 1) / trsm_threads;
            int smem_trsm = bs * bs * (int)sizeof(double);
            trsm_kernel<<<trsm_blocks, trsm_threads, smem_trsm>>>(d_A, N, jb, bs, trailing);

            // 3. SYRK: update the trailing submatrix
            int grid_dim = (trailing + TILE_DIM - 1) / TILE_DIM;
            dim3 syrk_grid(grid_dim, grid_dim);
            dim3 syrk_block(TILE_DIM, TILE_DIM);
            syrk_kernel<<<syrk_grid, syrk_block>>>(d_A, N, jb, bs, trailing);
        }
    }

    // Zero the upper triangle for clean output
    {
        int total = N * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        zero_upper_kernel<<<blocks, threads>>>(d_A, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
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
