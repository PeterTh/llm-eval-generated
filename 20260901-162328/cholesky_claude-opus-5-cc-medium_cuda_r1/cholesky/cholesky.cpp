#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition, CUDA implementation (blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The factorization is performed entirely on the GPU: for every panel of NB
// columns the diagonal block is factored (POTF2), the block column below it is
// updated by a triangular solve (TRSM) and the remaining trailing submatrix
// receives a symmetric rank-NB update (SYRK).  The auxiliary matrix products of
// the benchmark (matrix generation and validation) run on the GPU as well.

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, __LINE__);  \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

static constexpr int NB = 32;       // panel width == k-tile of the trailing update
static constexpr int TILE = 32;     // rows/columns computed per thread block
static constexpr int RPT = 4;       // output rows per thread (block is 32 x 8)
static constexpr int TRSM_T = 128;   // rows handled per TRSM block

// ---------------------------------------------------------------------------
// Cholesky kernels
// ---------------------------------------------------------------------------

// Factor the nb x nb diagonal block at (J, J) with a single warp.  Lane r holds
// row r of the block in registers; the rows of the block are exchanged with
// warp shuffles, so no barrier is needed on this latency critical path.  Rows
// outside of a partial panel are padded with the identity.
__global__ void potf2Kernel(double* __restrict__ A, const int n, const int J, const int nb,
                            int* __restrict__ info) {
    const unsigned mask = 0xffffffffu;
    const int r = threadIdx.x;  // row within the block

    double reg[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        reg[c] = (r < nb && c < nb && c <= r) ? A[(size_t)(J + r) * n + (J + c)]
                                              : ((c == r) ? 1.0 : 0.0);
    }

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        // Four partial sums keep the (long latency) double precision pipeline
        // busy on this strictly sequential part of the factorization.
        double s[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int k = 0; k < j; ++k) {
            s[k & 3] += reg[k] * __shfl_sync(mask, reg[k], j);
        }
        const double val = reg[j] - ((s[0] + s[1]) + (s[2] + s[3]));

        if (r == j) {
            if (val <= 0.0 && r < nb) {
                atomicMin(info, J + j);
            }
            reg[j] = sqrt(val);
        }
        const double diag = __shfl_sync(mask, reg[j], j);
        if (r > j) {
            reg[j] = val / diag;
        }
    }

#pragma unroll
    for (int c = 0; c < NB; ++c) {
        if (r < nb && c < nb && c <= r) {
            A[(size_t)(J + r) * n + (J + c)] = reg[c];
        }
    }
}

// Solve A[i][J:J+NB] * L11^T = A[i][J:J+NB] for all rows below the panel.
// A panel that has rows below it is always NB columns wide, so the triangular
// solve of every row is fully unrolled and kept in registers.
__global__ void trsmKernel(double* __restrict__ A, const int n, const int J) {
    __shared__ double Ls[NB][NB + 1];
    __shared__ double rinv[NB];

    const int t = threadIdx.x;
    for (int idx = t; idx < NB * NB; idx += TRSM_T) {
        const int r = idx >> 5;
        const int c = idx & (NB - 1);
        Ls[r][c] = (c <= r) ? A[(size_t)(J + r) * n + (J + c)] : 0.0;
    }
    if (t < NB) {
        rinv[t] = 1.0 / A[(size_t)(J + t) * n + (J + t)];
    }
    __syncthreads();

    const int i = J + NB + blockIdx.x * TRSM_T + t;
    if (i >= n) {
        return;
    }

    double* __restrict__ row = A + (size_t)i * n + J;
    double reg[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        reg[c] = row[c];
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double sum = 0.0;
#pragma unroll
        for (int k = 0; k < j; ++k) {
            sum += reg[k] * Ls[j][k];
        }
        reg[j] = (reg[j] - sum) * rinv[j];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        row[c] = reg[c];
    }
}

// Symmetric rank-NB update of the trailing submatrix: A -= L21 * L21^T.
__global__ void syrkKernel(double* __restrict__ A, const int n, const int J, const int base) {
    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bi < bj) {
        return;  // strictly upper triangular tile, not referenced
    }

    __shared__ double As[TILE][NB + 1];
    __shared__ double Bs[TILE][NB + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = base + bi * TILE;
    const int col0 = base + bj * TILE;

    for (int r = ty; r < TILE; r += blockDim.y) {
        const int gr = row0 + r;
        const int gc = col0 + r;
        As[r][tx] = (gr < n) ? A[(size_t)gr * n + J + tx] : 0.0;
        Bs[r][tx] = (gc < n) ? A[(size_t)gc * n + J + tx] : 0.0;
    }
    __syncthreads();

    double acc[RPT] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        const double b = Bs[tx][k];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            acc[r] += As[ty + r * 8][k] * b;
        }
    }

    const int j = col0 + tx;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = row0 + ty + r * 8;
        if (i < n && j <= i) {
            A[(size_t)i * n + j] -= acc[r];
        }
    }
}

// Zero the strictly upper triangular part of the rows [J, J + nb), which are
// final once the diagonal block of the panel has been factored.
__global__ void zeroUpperRowsKernel(double* __restrict__ A, const int n, const int J,
                                    const int nb) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    const int i = J + r;
    if (r < nb && j < n && j > i) {
        A[(size_t)i * n + j] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Auxiliary kernels (matrix generation / validation)
// ---------------------------------------------------------------------------

// C = X * Y^T for row-major n x n matrices.
__global__ void gemmNTKernel(const double* __restrict__ X, const double* __restrict__ Y,
                             double* __restrict__ C, const int n) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;

    double acc[RPT] = {0.0, 0.0, 0.0, 0.0};

    for (int kt = 0; kt < n; kt += TILE) {
        const int k = kt + tx;
        for (int r = ty; r < TILE; r += blockDim.y) {
            const int gr = row0 + r;
            const int gc = col0 + r;
            As[r][tx] = (gr < n && k < n) ? X[(size_t)gr * n + k] : 0.0;
            Bs[r][tx] = (gc < n && k < n) ? Y[(size_t)gc * n + k] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            const double b = Bs[tx][kk];
#pragma unroll
            for (int r = 0; r < RPT; ++r) {
                acc[r] += As[ty + r * 8][kk] * b;
            }
        }
        __syncthreads();
    }

    const int j = col0 + tx;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = row0 + ty + r * 8;
        if (i < n && j < n) {
            C[(size_t)i * n + j] = acc[r];
        }
    }
}

__global__ void addDiagonalKernel(double* __restrict__ A, const int n, const double value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[(size_t)i * n + i] += value;
    }
}

// Reduce max absolute / relative error between the reconstruction and the original.
__global__ void errorKernel(const double* __restrict__ R, const double* __restrict__ Aorig,
                            const size_t count, unsigned long long* __restrict__ out) {
    double maxAbs = 0.0;
    double maxRel = 0.0;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(R[idx] - Aorig[idx]);
        maxAbs = fmax(maxAbs, error);
        maxRel = fmax(maxRel, error / (fabs(Aorig[idx]) + 1e-10));
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        maxAbs = fmax(maxAbs, __shfl_down_sync(0xffffffffu, maxAbs, offset));
        maxRel = fmax(maxRel, __shfl_down_sync(0xffffffffu, maxRel, offset));
    }

    if ((threadIdx.x & 31) == 0) {
        // Both values are non-negative, so their bit patterns compare like integers.
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(maxAbs));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(maxRel));
    }
}

// ---------------------------------------------------------------------------

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* dA = nullptr;
    int* dInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));

    // Finished rows are streamed back to the host on a second stream while the
    // factorization keeps working on the trailing submatrix.
    cudaStream_t computeStream, copyStream;
    cudaEvent_t rowsDone;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    CUDA_CHECK(cudaEventCreateWithFlags(&rowsDone, cudaEventDisableTiming));

    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, computeStream));
    CUDA_CHECK(cudaMemcpyAsync(dInfo, &ni, sizeof(int), cudaMemcpyHostToDevice, computeStream));

    for (int J = 0; J < ni; J += NB) {
        const int nb = std::min(NB, ni - J);

        potf2Kernel<<<1, 32, 0, computeStream>>>(dA, ni, J, nb, dInfo);

        // Rows of this panel are final now: clear their upper part and stream
        // them back to the host while the trailing submatrix is being updated.
        zeroUpperRowsKernel<<<dim3((ni + 255) / 256, nb), 256, 0, computeStream>>>(dA, ni, J, nb);
        CUDA_CHECK(cudaEventRecord(rowsDone, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(copyStream, rowsDone, 0));
        CUDA_CHECK(cudaMemcpyAsync(A.data() + (size_t)J * n, dA + (size_t)J * n,
                                   (size_t)nb * n * sizeof(double), cudaMemcpyDeviceToHost,
                                   copyStream));

        const int base = J + nb;
        const int m = ni - base;
        if (m > 0) {
            // A panel with rows below it always spans the full NB columns
            trsmKernel<<<(m + TRSM_T - 1) / TRSM_T, TRSM_T, 0, computeStream>>>(dA, ni, J);

            const int nt = (m + TILE - 1) / TILE;
            syrkKernel<<<dim3(nt, nt), dim3(32, 8), 0, computeStream>>>(dA, ni, J, base);
        }
    }

    int info = ni;
    CUDA_CHECK(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(copyStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(rowsDone));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dInfo));

    if (info < ni) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info);
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

    if (n == 0) {
        return;
    }

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    const dim3 block(32, 8);
    const dim3 grid((ni + TILE - 1) / TILE, (ni + TILE - 1) / TILE);
    gemmNTKernel<<<grid, block>>>(dB, dB, dA, ni);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonalKernel<<<(ni + 255) / 256, 256>>>(dA, ni, (double)n);

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    if (n == 0) {
        printf("Max absolute error: %.10e\n", 0.0);
        printf("Max relative error: %.10e\n", 0.0);
        return true;
    }

    const int ni = (int)n;
    const size_t bytes = n * n * sizeof(double);

    double* dL = nullptr;
    double* dR = nullptr;
    double* dOrig = nullptr;
    unsigned long long* dErr = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, bytes));
    CUDA_CHECK(cudaMalloc(&dR, bytes));
    CUDA_CHECK(cudaMalloc(&dOrig, bytes));
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOrig, A_orig.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));

    // Compute L * L^T
    const dim3 block(32, 8);
    const dim3 grid((ni + TILE - 1) / TILE, (ni + TILE - 1) / TILE);
    gemmNTKernel<<<grid, block>>>(dL, dL, dR, ni);

    // Compare with original
    const size_t count = n * n;
    const int blocks = (int)std::min<size_t>((count + 255) / 256, 4096);
    errorKernel<<<blocks, 256>>>(dR, dOrig, count, dErr);

    unsigned long long err[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(err, dErr, sizeof(err), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dOrig));
    CUDA_CHECK(cudaFree(dErr));

    double maxError, relError;
    memcpy(&maxError, &err[0], sizeof(double));
    memcpy(&relError, &err[1], sizeof(double));

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

    // Initialize the CUDA context before any timed section
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

    // Page-locked host memory speeds up the transfers of the timed section
    const bool pinned = (n > 0) && (cudaHostRegister(A.data(), n * n * sizeof(double),
                                                     cudaHostRegisterDefault) == cudaSuccess);

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (pinned) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
    }

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
