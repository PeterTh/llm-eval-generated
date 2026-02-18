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
} while(0)

#define CHOL_BLOCK 32
#define TILE_DIM 16
#define PANEL_THREADS 256

// Kernel: Factor the diagonal block using a single thread block
__global__ void factorDiagBlock(double* __restrict__ A, const int n,
                                const int jb, const int bs, int* __restrict__ errflag) {
    __shared__ double s[CHOL_BLOCK][CHOL_BLOCK + 1];
    const int tx = threadIdx.x;

    if (tx < bs) {
        for (int c = 0; c < bs; c++)
            s[tx][c] = A[(jb + tx) * n + (jb + c)];
    }
    __syncthreads();

    for (int j = 0; j < bs; j++) {
        if (tx == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += s[j][k] * s[j][k];
            double val = s[j][j] - sum;
            if (val <= 0.0) {
                *errflag = 1;
                s[j][j] = 1.0;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();

        if (tx > j && tx < bs) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += s[tx][k] * s[j][k];
            s[tx][j] = (s[tx][j] - sum) / s[j][j];
        }
        __syncthreads();
    }

    if (tx < bs) {
        for (int c = 0; c <= tx; c++)
            A[(jb + tx) * n + (jb + c)] = s[tx][c];
        for (int c = tx + 1; c < bs; c++)
            A[(jb + tx) * n + (jb + c)] = 0.0;
    }
}

// Kernel: Solve panel below the diagonal block via forward substitution
__global__ void solvePanel(double* __restrict__ A, const int n,
                           const int jb, const int bs) {
    __shared__ double Ld[CHOL_BLOCK][CHOL_BLOCK + 1];
    const int tid = threadIdx.x;
    const int row = jb + bs + blockIdx.x * blockDim.x + tid;

    for (int idx = tid; idx < bs * bs; idx += blockDim.x) {
        int r = idx / bs, c = idx % bs;
        Ld[r][c] = A[(jb + r) * n + (jb + c)];
    }
    __syncthreads();

    if (row >= n) return;

    double panel[CHOL_BLOCK];
    for (int j = 0; j < bs; j++)
        panel[j] = A[row * n + (jb + j)];

    for (int j = 0; j < bs; j++) {
        double sum = 0.0;
        for (int k = 0; k < j; k++)
            sum += panel[k] * Ld[j][k];
        panel[j] = (panel[j] - sum) / Ld[j][j];
    }

    for (int j = 0; j < bs; j++)
        A[row * n + (jb + j)] = panel[j];
}

// Kernel: Rank-bs symmetric update of trailing submatrix
__global__ void updateTrailing(double* __restrict__ A, const int n,
                               const int jb, const int bs) {
    const int bRow = blockIdx.y, bCol = blockIdx.x;
    if (bCol > bRow) return;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = (jb + bs) + bRow * TILE_DIM + ty;
    const int col = (jb + bs) + bCol * TILE_DIM + tx;

    __shared__ double sA[TILE_DIM][TILE_DIM + 1];
    __shared__ double sB[TILE_DIM][TILE_DIM + 1];

    double sum = 0.0;
    for (int kk = 0; kk < bs; kk += TILE_DIM) {
        int chunk = bs - kk;
        if (chunk > TILE_DIM) chunk = TILE_DIM;

        sA[ty][tx] = (row < n && tx < chunk) ? A[row * n + (jb + kk + tx)] : 0.0;
        sB[tx][ty] = (col < n && ty < chunk) ? A[col * n + (jb + kk + ty)] : 0.0;
        __syncthreads();

        for (int k = 0; k < chunk; k++)
            sum += sA[ty][k] * sB[tx][k];
        __syncthreads();
    }

    if (row < n && col < n && col <= row)
        A[row * n + col] -= sum;
}

// Kernel: Zero upper triangular part
__global__ void zeroUpper(double* __restrict__ A, const int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row)
        A[row * n + col] = 0.0;
}

// GPU-accelerated blocked Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int N = static_cast<int>(n);

    double* d_A;
    int* d_err;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));

    for (int jb = 0; jb < N; jb += CHOL_BLOCK) {
        int bs = N - jb;
        if (bs > CHOL_BLOCK) bs = CHOL_BLOCK;

        factorDiagBlock<<<1, CHOL_BLOCK>>>(d_A, N, jb, bs, d_err);

        int remaining = N - jb - bs;
        if (remaining > 0) {
            int panelBlocks = (remaining + PANEL_THREADS - 1) / PANEL_THREADS;
            solvePanel<<<panelBlocks, PANEL_THREADS>>>(d_A, N, jb, bs);

            int trailBlocks = (remaining + TILE_DIM - 1) / TILE_DIM;
            dim3 grid(trailBlocks, trailBlocks);
            dim3 block(TILE_DIM, TILE_DIM);
            updateTrailing<<<grid, block>>>(d_A, N, jb, bs);
        }
    }

    {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
        zeroUpper<<<grid, block>>>(d_A, N);
    }

    int h_err = 0;
    CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_err));

    if (h_err) {
        printf("Error: Matrix is not positive definite\n");
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
