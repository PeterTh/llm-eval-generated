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
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLK 64
#define TILE 16
#define STILE 32

// Factor diagonal block with unblocked Cholesky (single thread block)
__global__ void kernel_potrf(double* __restrict__ A, int n, int j, int jb) {
    for (int col = 0; col < jb; col++) {
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int k = 0; k < col; k++) {
                double v = A[(j + col) * n + (j + k)];
                sum += v * v;
            }
            A[(j + col) * n + (j + col)] = sqrt(A[(j + col) * n + (j + col)] - sum);
        }
        __syncthreads();

        int row = threadIdx.x;
        if (row > col && row < jb) {
            double sum = 0.0;
            for (int k = 0; k < col; k++) {
                sum += A[(j + row) * n + (j + k)] * A[(j + col) * n + (j + k)];
            }
            A[(j + row) * n + (j + col)] =
                (A[(j + row) * n + (j + col)] - sum) / A[(j + col) * n + (j + col)];
        }
        __syncthreads();
    }
}

// TRSM: solve panel below diagonal block
__global__ void kernel_trsm(double* __restrict__ A, int n, int j, int jb) {
    int row = j + jb + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    for (int c = 0; c < jb; c++) {
        double sum = 0.0;
        for (int k = 0; k < c; k++) {
            sum += A[row * n + (j + k)] * A[(j + c) * n + (j + k)];
        }
        A[row * n + (j + c)] = (A[row * n + (j + c)] - sum) / A[(j + c) * n + (j + c)];
    }
}

// SYRK: update trailing submatrix with 2x2 thread coarsening
__global__ void kernel_syrk(double* __restrict__ A, int n, int j, int jb) {
    if (blockIdx.x > blockIdx.y) return;

    __shared__ double As[STILE][TILE + 1];
    __shared__ double Bs[TILE][STILE + 1];

    int startRC = j + jb;
    int row0 = startRC + blockIdx.y * STILE + threadIdx.y;
    int row1 = row0 + TILE;
    int col0 = startRC + blockIdx.x * STILE + threadIdx.x;
    int col1 = col0 + TILE;

    double c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    for (int t = 0; t < jb; t += TILE) {
        int tk = jb - t;
        if (tk > TILE) tk = TILE;

        // Load As: 32 rows × TILE cols
        As[threadIdx.y][threadIdx.x] =
            (row0 < n && threadIdx.x < tk) ? A[row0 * n + (j + t + threadIdx.x)] : 0.0;
        As[threadIdx.y + TILE][threadIdx.x] =
            (row1 < n && threadIdx.x < tk) ? A[row1 * n + (j + t + threadIdx.x)] : 0.0;

        // Load Bs: TILE rows × 32 cols (transposed panel access)
        Bs[threadIdx.y][threadIdx.x] =
            (col0 < n && threadIdx.y < tk) ? A[col0 * n + (j + t + threadIdx.y)] : 0.0;
        Bs[threadIdx.y][threadIdx.x + TILE] =
            (col1 < n && threadIdx.y < tk) ? A[col1 * n + (j + t + threadIdx.y)] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            double a0 = As[threadIdx.y][k];
            double a1 = As[threadIdx.y + TILE][k];
            double b0 = Bs[k][threadIdx.x];
            double b1 = Bs[k][threadIdx.x + TILE];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < n && col0 < n && col0 <= row0) A[row0 * n + col0] -= c00;
    if (row0 < n && col1 < n && col1 <= row0) A[row0 * n + col1] -= c01;
    if (row1 < n && col0 < n && col0 <= row1) A[row1 * n + col0] -= c10;
    if (row1 < n && col1 < n && col1 <= row1) A[row1 * n + col1] -= c11;
}

// Zero upper triangular part
__global__ void kernel_zero_upper(double* __restrict__ A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / n;
    int col = idx % n;
    if (row < n && col > row) {
        A[idx] = 0.0;
    }
}

// GPU-accelerated blocked Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A;
    size_t bytes = n * n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    int N = static_cast<int>(n);

    for (int j = 0; j < N; j += BLK) {
        int jb = (N - j < BLK) ? (N - j) : BLK;

        // 1. Factor diagonal block
        kernel_potrf<<<1, BLK>>>(d_A, N, j, jb);

        if (j + jb < N) {
            // 2. TRSM: update panel below diagonal block
            int panelRows = N - (j + jb);
            kernel_trsm<<<(panelRows + 255) / 256, 256>>>(d_A, N, j, jb);

            // 3. SYRK: update trailing submatrix
            int remaining = N - (j + jb);
            dim3 grid((remaining + STILE - 1) / STILE, (remaining + STILE - 1) / STILE);
            dim3 block(TILE, TILE);
            kernel_syrk<<<grid, block>>>(d_A, N, j, jb);
        }
    }

    // Zero upper triangular
    {
        int total = N * N;
        kernel_zero_upper<<<(total + 255) / 256, 256>>>(d_A, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
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
    
    // Warm up CUDA runtime before timing
    cudaFree(0);
    
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
