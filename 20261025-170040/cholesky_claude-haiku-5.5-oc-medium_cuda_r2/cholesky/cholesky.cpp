#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (left-looking, column-oriented).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The factorization runs on the GPU. The working copy W is column-major:
// W[c * n + r] holds element (r, c), so the entries of one column are contiguous
// across rows and warp accesses are coalesced.
//
// For each column j (one kernel launch per column):
//   - L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2). The running sum of squares
//     dacc[j] already holds sum_{k<j} L(j,k)^2, accumulated in k order, so every
//     block evaluates this in O(1).
//   - For all rows i > j in parallel:
//       L(i,j) = (A(i,j) - sum_{k<j} L(i,k) * L(j,k)) / L(j,j)
//     and dacc[i] += L(i,j)^2.
// Each dot product is accumulated in the same k = 0..j-1 order as the original
// sequential code. The diagonal is written back once all columns are processed.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_), #call);                             \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

namespace {

constexpr int kBlock = 256;             // threads per block for the column kernel
constexpr int kTile = 32;               // transpose tile edge
constexpr size_t kCheckInterval = 128;  // columns between host checks of the failure flag
constexpr int kUnroll = 4;              // independent loads in flight per thread

// dst[c * rows + r] = src[r * cols + c]  (src is rows x cols, dst is cols x rows).
// If lowerOnly is set, elements with r > c are written as zero (upper triangle).
__global__ void transposeKernel(const double* __restrict__ src, double* __restrict__ dst,
                                size_t rows, size_t cols, bool lowerOnly) {
    __shared__ double tile[kTile][kTile + 1];
    const size_t c0 = (size_t)blockIdx.x * kTile;
    const size_t r0 = (size_t)blockIdx.y * kTile;

    for (int dy = threadIdx.y; dy < kTile; dy += blockDim.y) {
        const size_t r = r0 + dy;
        const size_t c = c0 + threadIdx.x;
        if (r < rows && c < cols) {
            tile[dy][threadIdx.x] = src[r * cols + c];
        }
    }
    __syncthreads();

    for (int dy = threadIdx.y; dy < kTile; dy += blockDim.y) {
        const size_t c = c0 + dy;            // src column == dst row
        const size_t r = r0 + threadIdx.x;   // src row == dst column
        if (c < cols && r < rows) {
            double v = tile[threadIdx.x][dy];
            if (lowerOnly && r > c) {
                v = 0.0;
            }
            dst[c * rows + r] = v;
        }
    }
}

// Processes column j: computes L(j,j) (redundantly in every block) and all
// L(i,j) for i > j in parallel.
__global__ void columnKernel(double* __restrict__ W, double* __restrict__ dacc, size_t n,
                             size_t j, int* failCol) {
    __shared__ double sh[kBlock];

    // L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2). Diagonal of W still holds A(j,j).
    const double diagVal = W[j * n + j] - dacc[j];
    if (diagVal <= 0.0) {
        // Matrix is not positive definite; record the smallest failing index.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicMin(failCol, (int)j);
        }
    }
    const double ljj = sqrt(diagVal);

    const size_t i = j + 1 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;

    double sum = 0.0;
    for (size_t base = 0; base < j; base += kBlock) {
        const size_t cnt = (j - base < (size_t)kBlock) ? (j - base) : (size_t)kBlock;
        if ((size_t)threadIdx.x < cnt) {
            sh[threadIdx.x] = W[(base + threadIdx.x) * n + j];  // L(j, base + t)
        }
        __syncthreads();
        if (active) {
            // Issue several independent loads before the dependent accumulation
            // chain to hide global memory latency. Terms are still added in k order.
            size_t t = 0;
            for (; t + kUnroll <= cnt; t += kUnroll) {
                double w[kUnroll];
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    w[u] = W[(base + t + u) * n + i];           // L(i, k)
                }
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    sum += w[u] * sh[t + u];                    // L(i,k) * L(j,k)
                }
            }
            for (; t < cnt; ++t) {
                sum += W[(base + t) * n + i] * sh[t];
            }
        }
        __syncthreads();
    }

    if (active) {
        const double lij = (W[j * n + i] - sum) / ljj;
        W[j * n + i] = lij;
        dacc[i] += lij * lij;
    }
}

// Writes the final diagonal L(j,j) = sqrt(A(j,j) - dacc[j]) into W.
__global__ void finalizeDiagonalKernel(double* __restrict__ W, const double* __restrict__ dacc,
                                       size_t n) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        W[j * n + j] = sqrt(W[j * n + j] - dacc[j]);
    }
}

}  // namespace

// Initializes the CUDA context so that its one-time cost is not counted in timing.
void initCuda() {
    CUDA_CHECK(cudaFree(nullptr));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) {
        return true;
    }

    const size_t bytes = n * n * sizeof(double);
    const size_t vecBytes = n * sizeof(double);
    double* dA = nullptr;     // row-major input / final output
    double* dW = nullptr;     // column-major working copy
    double* dAcc = nullptr;   // dAcc[i] = sum_{k<j} L(i,k)^2 at the start of column j
    int* dFail = nullptr;     // smallest failing diagonal index (INT_MAX if none)
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dW, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, vecBytes));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));

    const int noFail = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAcc, 0, vecBytes));

    const dim3 tBlock(kTile, 8);
    const unsigned tiles = (unsigned)((n + kTile - 1) / kTile);
    const dim3 tGrid(tiles, tiles);

    // dW[c * n + r] = A(r, c)
    transposeKernel<<<tGrid, tBlock>>>(dA, dW, n, n, false);

    int failCol = noFail;
    for (size_t j = 0; j < n; ++j) {
        const size_t rows = n - j - 1;
        // At least one block is launched so that the diagonal check always runs.
        const unsigned blocks = (rows > 0) ? (unsigned)((rows + kBlock - 1) / kBlock) : 1u;
        columnKernel<<<blocks, kBlock>>>(dW, dAcc, n, j, dFail);
        if ((j + 1) % kCheckInterval == 0 || j + 1 == n) {
            CUDA_CHECK(cudaMemcpy(&failCol, dFail, sizeof(int), cudaMemcpyDeviceToHost));
            if (failCol != noFail) {
                break;
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());

    if (failCol != noFail) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failCol);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dW));
        CUDA_CHECK(cudaFree(dAcc));
        CUDA_CHECK(cudaFree(dFail));
        return false;
    }

    finalizeDiagonalKernel<<<(unsigned)((n + kBlock - 1) / kBlock), kBlock>>>(dW, dAcc, n);

    // dA[i * n + j] = L(i, j) for j <= i, zero otherwise
    transposeKernel<<<tGrid, tBlock>>>(dW, dA, n, n, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dAcc));
    CUDA_CHECK(cudaFree(dFail));
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
    initCuda();  // exclude one-time CUDA context creation from the timing
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
