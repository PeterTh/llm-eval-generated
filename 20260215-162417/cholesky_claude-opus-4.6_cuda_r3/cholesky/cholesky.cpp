#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define BLOCK_SIZE 32

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Panel factorization: right-looking Cholesky on a small diagonal block
__global__ void panelFactor(double* A, int n, int jb, int bs) {
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;

    if (tx < bs && ty < bs)
        tile[ty][tx] = A[(jb + ty) * n + (jb + tx)];
    __syncthreads();

    for (int j = 0; j < bs; j++) {
        if (tx == 0 && ty == 0)
            tile[j][j] = sqrt(tile[j][j]);
        __syncthreads();

        if (tx == 0 && ty > j && ty < bs)
            tile[ty][j] /= tile[j][j];
        __syncthreads();

        if (ty > j && tx > j && ty < bs && tx < bs && ty >= tx)
            tile[ty][tx] -= tile[ty][j] * tile[tx][j];
        __syncthreads();
    }

    if (tx < bs && ty < bs) {
        if (ty >= tx)
            A[(jb + ty) * n + (jb + tx)] = tile[ty][tx];
        else
            A[(jb + ty) * n + (jb + tx)] = 0.0;
    }
}

// TRSM: solve column panel below diagonal block via forward substitution
__global__ void trsmPanel(double* A, int n, int jb, int bs) {
    __shared__ double L[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int i = threadIdx.x; i < bs * bs; i += blockDim.x) {
        int r = i / bs, c = i % bs;
        L[r][c] = A[(jb + r) * n + (jb + c)];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x + jb + bs;
    if (row >= n) return;

    int base = row * n + jb;
    for (int j = 0; j < bs; j++) {
        double val = A[base + j];
        for (int k = 0; k < j; k++)
            val -= A[base + k] * L[j][k];
        A[base + j] = val / L[j][j];
    }
}

// Trailing submatrix rank-bs update with shared memory tiling
__global__ void updateTrailing(double* A, int n, int jb, int bs) {
    __shared__ double sRow[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ double sCol[BLOCK_SIZE][BLOCK_SIZE + 1];

    int startIdx = jb + bs;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x + startIdx;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y + startIdx;

    if (blockIdx.y < blockIdx.x) return;

    // Load row data (coalesced: warp threads access consecutive addresses)
    if (row < n && threadIdx.x < bs)
        sRow[threadIdx.y][threadIdx.x] = A[row * n + jb + threadIdx.x];
    else
        sRow[threadIdx.y][threadIdx.x] = 0.0;

    // Load col data (coalesced via cooperative loading with swapped indices)
    int colForLoad = blockIdx.x * BLOCK_SIZE + threadIdx.y + startIdx;
    if (colForLoad < n && threadIdx.x < bs)
        sCol[threadIdx.y][threadIdx.x] = A[colForLoad * n + jb + threadIdx.x];
    else
        sCol[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();

    if (row < n && col < n && row >= col) {
        double val = 0.0;
        for (int k = 0; k < bs; k++)
            val += sRow[threadIdx.y][k] * sCol[threadIdx.x][k];
        A[row * n + col] -= val;
    }
}

// Zero upper triangular part
__global__ void zeroUpper(double* A, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row)
        A[row * n + col] = 0.0;
}

// GPU-accelerated blocked right-looking Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int N = static_cast<int>(n);
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    for (int jb = 0; jb < N; jb += BLOCK_SIZE) {
        int bs = (N - jb < BLOCK_SIZE) ? (N - jb) : BLOCK_SIZE;

        // Factor diagonal block
        dim3 panelBlock(bs, bs);
        panelFactor<<<1, panelBlock>>>(d_A, N, jb, bs);

        int remaining = N - jb - bs;
        if (remaining > 0) {
            // TRSM on column panel
            int tpb = 256;
            int nb = (remaining + tpb - 1) / tpb;
            trsmPanel<<<nb, tpb>>>(d_A, N, jb, bs);

            // Rank-bs update of trailing submatrix
            int gd = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 grid(gd, gd);
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            updateTrailing<<<grid, block>>>(d_A, N, jb, bs);
        }
    }

    // Zero upper triangular
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    zeroUpper<<<grid, block>>>(d_A, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

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
