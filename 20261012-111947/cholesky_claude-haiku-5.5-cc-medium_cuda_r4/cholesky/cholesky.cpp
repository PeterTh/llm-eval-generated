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

// Blocked CUDA Cholesky decomposition (right-looking, lower triangular, row-major storage)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Each block column of width TILE is processed in three GPU phases:
//   1. factor the diagonal tile (one thread block)
//   2. solve the panel below it (one thread per row)
//   3. update the trailing lower-triangular tiles (one thread block per tile)

constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Factor the diagonal tile (k0, k0) in place. Runs as a single thread block.
// Records the smallest failing diagonal index in failIdx.
__global__ void potrfDiagKernel(double* A, int n, int k0, int nb, int* failIdx) {
    __shared__ double T[TILE][TILE + 1];
    __shared__ bool failed;

    const int t = threadIdx.x;
    if (t == 0) failed = false;
    if (t < nb) {
        for (int c = 0; c < nb; ++c) {
            T[t][c] = A[(size_t)(k0 + t) * n + k0 + c];
        }
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        // Diagonal element
        if (t == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += T[j][k] * T[j][k];
            }
            const double val = T[j][j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                atomicMin(failIdx, k0 + j);
                failed = true;
            } else {
                T[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (failed) break;

        // Off-diagonal elements in column j
        const int i = j + 1 + t;
        if (i < nb) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += T[i][k] * T[j][k];
            }
            T[i][j] = (T[i][j] - sum) / T[j][j];
        }
        __syncthreads();
    }

    if (t < nb) {
        for (int c = 0; c <= t; ++c) {
            A[(size_t)(k0 + t) * n + k0 + c] = T[t][c];
        }
    }
}

// Solve L(i,K) * L(K,K)^T = A(i,K) for every row i below the diagonal tile.
// Each thread handles one row; the diagonal tile is staged in shared memory.
__global__ void trsmPanelKernel(double* A, int n, int k0, int nb) {
    __shared__ double D[TILE][TILE + 1];

    const int t = threadIdx.x;
    for (int idx = t; idx < nb * nb; idx += blockDim.x) {
        const int r = idx / nb;
        const int c = idx % nb;
        D[r][c] = A[(size_t)(k0 + r) * n + k0 + c];
    }
    __syncthreads();

    const int i = k0 + nb + blockIdx.x * blockDim.x + t;
    if (i >= n) return;

    double* row = A + (size_t)i * n + k0;
    for (int j = 0; j < nb; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            sum += row[k] * D[j][k];
        }
        row[j] = (row[j] - sum) / D[j][j];
    }
}

// Trailing update: A(r,c) -= sum_{k in panel} L(r,k) * L(c,k) for the lower-triangular
// tiles (bi >= bj) of the trailing submatrix starting at t0. Diagonal tiles only touch r >= c.
__global__ void syrkUpdateKernel(double* A, int n, int k0, int nb, int t0) {
    const int bj = blockIdx.x;
    const int bi = blockIdx.y;
    if (bj > bi) return;

    __shared__ double Ra[TILE][TILE + 1];
    __shared__ double Rb[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    for (int r = ty; r < TILE; r += blockDim.y) {
        const int gi = t0 + bi * TILE + r;
        const int gj = t0 + bj * TILE + r;
        Ra[r][tx] = (gi < n && tx < nb) ? A[(size_t)gi * n + k0 + tx] : 0.0;
        Rb[r][tx] = (gj < n && tx < nb) ? A[(size_t)gj * n + k0 + tx] : 0.0;
    }
    __syncthreads();

    for (int r = ty; r < TILE; r += blockDim.y) {
        const int gr = t0 + bi * TILE + r;
        const int gc = t0 + bj * TILE + tx;
        if (gr < n && gc < n && gr >= gc) {
            double acc = 0.0;
            for (int k = 0; k < nb; ++k) {
                acc += Ra[r][k] * Rb[tx][k];
            }
            A[(size_t)gr * n + gc] -= acc;
        }
    }
}

// C = X * X^T (+ diagAdd on the diagonal). Each dot product is accumulated in increasing k
// order, the same order as the sequential reference loops.
__global__ void gramKernel(const double* X, double* C, int n, double diagAdd) {
    __shared__ double Xa[TILE][TILE + 1];
    __shared__ double Xb[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i0 = blockIdx.y * TILE;
    const int j0 = blockIdx.x * TILE;

    double acc = 0.0;
    for (int kc = 0; kc < n; kc += TILE) {
        const int k = kc + tx;
        const int gi = i0 + ty;
        const int gj = j0 + ty;
        Xa[ty][tx] = (gi < n && k < n) ? X[(size_t)gi * n + k] : 0.0;
        Xb[ty][tx] = (gj < n && k < n) ? X[(size_t)gj * n + k] : 0.0;
        __syncthreads();

        const int kmax = min(TILE, n - kc);
        for (int kk = 0; kk < kmax; ++kk) {
            acc += Xa[ty][kk] * Xb[tx][kk];
        }
        __syncthreads();
    }

    const int i = i0 + ty;
    const int j = j0 + tx;
    if (i < n && j < n) {
        C[(size_t)i * n + j] = (i == j) ? acc + diagAdd : acc;
    }
}

// Per-row maxima of |R - A| and |R - A| / (|A| + 1e-10). One block per row.
__global__ void errorKernel(const double* R, const double* A, int n, double* rowAbs, double* rowRel) {
    __shared__ double sAbs[256];
    __shared__ double sRel[256];

    const int tid = threadIdx.x;
    const int i = blockIdx.x;

    double mAbs = 0.0;
    double mRel = 0.0;
    for (int j = tid; j < n; j += blockDim.x) {
        const size_t idx = (size_t)i * n + j;
        const double error = fabs(R[idx] - A[idx]);
        mAbs = fmax(mAbs, error);
        const double rel = error / (fabs(A[idx]) + 1e-10);
        mRel = fmax(mRel, rel);
    }
    sAbs[tid] = mAbs;
    sRel[tid] = mRel;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sAbs[tid] = fmax(sAbs[tid], sAbs[tid + s]);
            sRel[tid] = fmax(sRel[tid], sRel[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        rowAbs[i] = sAbs[0];
        rowRel[i] = sRel[0];
    }
}

// Zero the strict upper triangle (the sequential algorithm leaves it zero)
__global__ void zeroStrictUpperKernel(double* A, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) {
        A[(size_t)i * n + j] = 0.0;
    }
}

// Device-side C = X * X^T (+ diagAdd on the diagonal)
void gramDevice(const double* dX, double* dC, int n, double diagAdd) {
    if (n == 0) return;
    const int T = (n + TILE - 1) / TILE;
    gramKernel<<<dim3(T, T), dim3(TILE, TILE)>>>(dX, dC, n, diagAdd);
    CUDA_CHECK(cudaGetLastError());
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    if (n == 0) return;

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential, so the sequence is reproducible)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T on the GPU, adding n to the diagonal for diagonal dominance
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    gramDevice(dB, dA, static_cast<int>(n), static_cast<double>(n));
    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const int N = static_cast<int>(n);
    double* dA = nullptr;
    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    const int noFailure = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &noFailure, sizeof(int), cudaMemcpyHostToDevice));

    for (int k0 = 0; k0 < N; k0 += TILE) {
        const int nb = std::min(TILE, N - k0);
        const int t0 = k0 + nb;
        const int rows = N - t0;

        potrfDiagKernel<<<1, TILE>>>(dA, N, k0, nb, dFail);
        if (rows == 0) break;

        trsmPanelKernel<<<(rows + 127) / 128, 128>>>(dA, N, k0, nb);

        const int T = (rows + TILE - 1) / TILE;
        syrkUpdateKernel<<<dim3(T, T), dim3(TILE, 8)>>>(dA, N, k0, nb, t0);
    }
    CUDA_CHECK(cudaGetLastError());

    zeroStrictUpperKernel<<<dim3((N + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    int failIdx = noFailure;
    CUDA_CHECK(cudaMemcpy(&failIdx, dFail, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dFail));

    if (failIdx != noFailure) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", static_cast<size_t>(failIdx));
        return false;
    }

    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    double maxError = 0.0;
    double relError = 0.0;

    if (n > 0) {
        const int N = static_cast<int>(n);
        double* dL = nullptr;
        double* dO = nullptr;
        double* dR = nullptr;
        double* dRowAbs = nullptr;
        double* dRowRel = nullptr;
        CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dO, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dRowAbs, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dRowRel, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dO, A_orig.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

        // Compute L * L^T
        gramDevice(dL, dR, N, 0.0);

        // Compare with original
        errorKernel<<<N, 256>>>(dR, dO, N, dRowAbs, dRowRel);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> rowAbs(n);
        std::vector<double> rowRel(n);
        CUDA_CHECK(cudaMemcpy(rowAbs.data(), dRowAbs, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rowRel.data(), dRowRel, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dO));
        CUDA_CHECK(cudaFree(dR));
        CUDA_CHECK(cudaFree(dRowAbs));
        CUDA_CHECK(cudaFree(dRowRel));

        for (size_t i = 0; i < n; ++i) {
            maxError = std::max(maxError, rowAbs[i]);
            relError = std::max(relError, rowRel[i]);
        }
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

    // Initialize the CUDA context so device setup is not counted in the timing
    CUDA_CHECK(cudaFree(nullptr));

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
