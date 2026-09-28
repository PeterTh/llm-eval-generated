#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition parallelized with CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is processed in panels of BLOCK_SIZE columns. For each panel:
//   1. The small diagonal block is factorized in-place (potf2-style kernel).
//   2. The panel below the diagonal block is solved via a blocked triangular
//      solve (trsm-style kernel), with one CUDA thread per matrix row.
//   3. The trailing submatrix is updated with a tiled, shared-memory
//      rank-BLOCK_SIZE update (syrk-style kernel) that exploits symmetry by
//      only computing the lower-triangular block tiles.
// This keeps all the O(n^3) work on the GPU and parallelizes across rows,
// columns, and tiles.

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                           \
            exit(1);                                                                     \
        }                                                                                 \
    } while (0)

namespace {

constexpr size_t BLOCK_SIZE = 128;
constexpr int TILE = 32;

// Factorizes the kb x kb diagonal block A[k:k+kb, k:k+kb] in-place into its
// lower-triangular Cholesky factor. One thread per row of the block; columns
// are processed sequentially (each column depends on the previous ones), but
// all rows below the current column are updated in parallel.
__global__ void potf2Kernel(double* A, size_t n, size_t k, size_t kb, int* failFlag) {
    extern __shared__ double diagCol[]; // holds column j values shared across threads
    const int tid = threadIdx.x;

    for (size_t j = 0; j < kb; ++j) {
        double val = 0.0;
        if (static_cast<size_t>(tid) == j) {
            double sum = 0.0;
            double* rowJ = &A[(k + j) * n + k];
            for (size_t kk = 0; kk < j; ++kk) {
                sum += rowJ[kk] * rowJ[kk];
            }
            val = rowJ[j] - sum;
            if (val <= 0.0) {
                *failFlag = 1;
                val = 0.0; // avoid NaNs from sqrt of negative; failure already flagged
            }
            val = sqrt(val);
            rowJ[j] = val;
            diagCol[0] = val;
        }
        __syncthreads();
        const double ljj = diagCol[0];

        if (static_cast<size_t>(tid) > j && static_cast<size_t>(tid) < kb) {
            double sum = 0.0;
            double* rowI = &A[(k + tid) * n + k];
            const double* rowJ = &A[(k + j) * n + k];
            for (size_t kk = 0; kk < j; ++kk) {
                sum += rowI[kk] * rowJ[kk];
            }
            rowI[j] = (ljj > 0.0) ? (rowI[j] - sum) / ljj : 0.0;
        }
        __syncthreads();
    }
}

// Solves L_ik * L_kk^T = A_ik for each row i in [mstart, n) (the panel below
// the diagonal block), i.e. computes L_ik via forward substitution against
// the already-factorized kb x kb lower-triangular block L_kk. Each CUDA
// thread owns one full row of the panel and walks the kb columns
// sequentially (independent across rows, so fully parallel across the panel).
__global__ void trsmPanelKernel(double* A, size_t n, size_t k, size_t kb, size_t mstart) {
    const size_t i = mstart + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double* rowI = &A[i * n + k];
    for (size_t c = 0; c < kb; ++c) {
        const double* rowC = &A[(k + c) * n + k];
        double sum = 0.0;
        for (size_t kk = 0; kk < c; ++kk) {
            sum += rowI[kk] * rowC[kk];
        }
        rowI[c] = (rowI[c] - sum) / rowC[c];
    }
}

// Updates the trailing submatrix A[i, j] -= sum_c L[i, k+c] * L[j, k+c] for
// i, j in [mstart, n), i >= j (lower triangle only, since A stays symmetric
// and only the lower triangle is maintained/read by later steps). Uses a
// standard tiled shared-memory matmul pattern, with whole tiles that lie
// entirely in the strictly-upper triangle skipped entirely.
__global__ void trailingUpdateKernel(double* A, size_t n, size_t k, size_t kb, size_t mstart) {
    __shared__ double tileI[TILE][TILE];
    __shared__ double tileJ[TILE][TILE];

    const size_t bi = mstart + static_cast<size_t>(blockIdx.y) * TILE;
    const size_t bj = mstart + static_cast<size_t>(blockIdx.x) * TILE;

    // Skip whole tiles that fall entirely in the strictly-upper triangle.
    if (bj > bi + TILE - 1) return;

    const size_t i = bi + threadIdx.y;
    const size_t j = bj + threadIdx.x;

    double acc = 0.0;
    for (size_t c0 = 0; c0 < kb; c0 += TILE) {
        const size_t ci = c0 + threadIdx.x;
        const size_t cj = c0 + threadIdx.y;

        tileI[threadIdx.y][threadIdx.x] = (i < n && ci < kb) ? A[i * n + k + ci] : 0.0;
        tileJ[threadIdx.y][threadIdx.x] = (j < n && cj < kb) ? A[j * n + k + cj] : 0.0;
        __syncthreads();

#pragma unroll
        for (int cc = 0; cc < TILE; ++cc) {
            acc += tileI[threadIdx.y][cc] * tileJ[cc][threadIdx.x];
        }
        __syncthreads();
    }

    if (i < n && j < n && i >= j) {
        A[i * n + j] -= acc;
    }
}

__global__ void zeroUpperTriangleKernel(double* A, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* d_A = nullptr;
    int* d_fail = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));

    bool ok = true;

    for (size_t k = 0; k < n && ok; k += BLOCK_SIZE) {
        const size_t kb = std::min(BLOCK_SIZE, n - k);

        potf2Kernel<<<1, static_cast<int>(kb), sizeof(double)>>>(d_A, n, k, kb, d_fail);
        CUDA_CHECK(cudaGetLastError());

        int hostFail = 0;
        CUDA_CHECK(cudaMemcpy(&hostFail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if (hostFail) {
            printf("Error: Matrix is not positive definite near block starting at %zu\n", k);
            ok = false;
            break;
        }

        const size_t mstart = k + kb;
        if (mstart < n) {
            const size_t rows = n - mstart;
            const int threads = 256;
            const int blocks = static_cast<int>((rows + threads - 1) / threads);
            trsmPanelKernel<<<blocks, threads>>>(d_A, n, k, kb, mstart);
            CUDA_CHECK(cudaGetLastError());

            const size_t trailing = n - mstart;
            const int tiles = static_cast<int>((trailing + TILE - 1) / TILE);
            dim3 grid(tiles, tiles);
            dim3 block(TILE, TILE);
            trailingUpdateKernel<<<grid, block>>>(d_A, n, k, kb, mstart);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (ok) {
        dim3 block(32, 32);
        dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                  static_cast<unsigned int>((n + block.y - 1) / block.y));
        zeroUpperTriangleKernel<<<grid, block>>>(d_A, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_A);
    cudaFree(d_fail);

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
