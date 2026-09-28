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

// Cholesky decomposition (blocked right-looking algorithm, CUDA)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The matrix is processed in NB x NB column blocks. For every block k:
//   1. POTRF  : factorize the diagonal block (single CUDA block, shared memory)
//               and invert the resulting triangular factor
//   2. TRSM   : solve the panel below the diagonal block, expressed as the GEMM
//               P = P * L_kk^-T so that it parallelizes over all panel rows
//   3. SYRK   : rank-NB update of the remaining trailing submatrix (lower part)
// Step 3 carries the bulk of the O(n^3/3) work and is parallel over the whole
// trailing submatrix. Finished block rows are streamed back to the host on a
// second stream, overlapping the device-to-host transfer with the computation.

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);  \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// Blocking parameters. TILE must match NB so that a single trailing-update tile
// covers exactly one block column of the panel.
static constexpr int NB = 64;         // algorithmic block size
static constexpr int TILE = 64;       // GEMM tile (rows/cols per CUDA block)
static constexpr int BK = 16;         // GEMM tile depth
static constexpr int GEMM_THREADS = 256;

// C = alpha * A * B^T + beta * C   (all matrices row-major)
// Each thread owns a 4x4 register tile with a strided (stride 16) mapping, which
// keeps both the shared memory reads and the global stores conflict free.
// Accumulation runs over k in ascending order with a single accumulator per
// output element, i.e. the same summation order as the sequential reference.
// If lowerOnly is set, only elements with j <= i are computed/stored.
__global__ __launch_bounds__(GEMM_THREADS) void gemmNT(const int M, const int N, const int K,
                                                       const double alpha,
                                                       const double* A, const int lda,
                                                       const double* B, const int ldb,
                                                       const double beta, double* C,
                                                       const int ldc, const bool lowerOnly) {
    const int bi = blockIdx.y * TILE;
    const int bj = blockIdx.x * TILE;
    if (lowerOnly && bj > bi) {
        return;  // tile lies strictly in the upper triangle
    }

    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE + 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15;   // column direction of the register tile
    const int ty = tid >> 4;   // row direction of the register tile
    const int ll = tid & 15;   // k index used while loading
    const int lr = tid >> 4;   // row index used while loading

    double acc[4][4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            acc[p][q] = 0.0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int gc = k0 + ll;
        const bool kValid = gc < K;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = lr + 16 * u;
            const int gra = bi + r;
            const int grb = bj + r;
            As[ll][r] = (kValid && gra < M) ? A[(size_t)gra * lda + gc] : 0.0;
            Bs[ll][r] = (kValid && grb < N) ? B[(size_t)grb * ldb + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int l = 0; l < BK; ++l) {
            double a[4], b[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                a[e] = As[l][ty + 16 * e];
                b[e] = Bs[l][tx + 16 * e];
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[p][q] += a[p] * b[q];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int i = bi + ty + 16 * p;
        if (i >= M) {
            continue;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int j = bj + tx + 16 * q;
            if (j >= N || (lowerOnly && j > i)) {
                continue;
            }
            double* dst = &C[(size_t)i * ldc + j];
            *dst = (beta == 0.0) ? alpha * acc[p][q] : beta * (*dst) + alpha * acc[p][q];
        }
    }
}

// Unblocked Cholesky of the nb x nb diagonal block at (k, k), executed by a
// single CUDA block entirely in shared memory (right-looking).
// Additionally inverts the resulting triangular factor into Linv, which turns
// the subsequent panel solve into a plain (fully parallel) GEMM.
// The inverse is kept in the otherwise unused strictly-upper half of the shared
// tile in transposed form: inv[r][c] (r > c) lives in s[c][r].
__global__ __launch_bounds__(256) void potrfDiag(double* __restrict__ A, const size_t n, const int k,
                                                 const int nb, double* __restrict__ Linv,
                                                 int* __restrict__ info) {
    __shared__ double s[NB][NB + 1];
    __shared__ double idiag[NB];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb;
        const int c = idx - r * nb;
        s[r][c] = (c <= r) ? A[(size_t)(k + r) * n + (k + c)] : 0.0;
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (tid == 0) {
            const double val = s[j][j];
            if (!(val > 0.0)) {
                atomicMin(info, k + j);
            }
            s[j][j] = sqrt(val);
        }
        __syncthreads();

        const double d = s[j][j];
        for (int i = j + 1 + tid; i < nb; i += nthreads) {
            s[i][j] /= d;
        }
        __syncthreads();

        // rank-1 update of the trailing lower part of the block
        const int m = nb - j - 1;
        for (int idx = tid; idx < m * m; idx += nthreads) {
            const int r = j + 1 + idx / m;
            const int c = j + 1 + idx % m;
            if (r >= c) {
                s[r][c] -= s[r][j] * s[c][j];
            }
        }
        __syncthreads();
    }

    // Store the factor before the strictly-upper half gets used for the inverse
    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb;
        const int c = idx - r * nb;
        if (c <= r) {
            A[(size_t)(k + r) * n + (k + c)] = s[r][c];
        }
    }

    // Invert the lower triangular factor: one thread per column, forward
    // substitution on L * X = I. Every thread only writes its own column of X,
    // so no synchronization is needed inside the loop.
    if (tid < nb) {
        const int c = tid;
        const double dc = 1.0 / s[c][c];
        idiag[c] = dc;
        for (int i = c + 1; i < nb; ++i) {
            double sum = s[i][c] * dc;  // l == c uses inv[c][c]
            for (int l = c + 1; l < i; ++l) {
                sum += s[i][l] * s[c][l];
            }
            s[c][i] = -sum / s[i][i];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < nb * nb; idx += nthreads) {
        const int r = idx / nb;
        const int c = idx - r * nb;
        Linv[r * NB + c] = (c > r) ? 0.0 : ((c == r) ? idiag[r] : s[c][r]);
    }
}

// Zero the strictly upper triangular part of the block rows [k, k + nb).
__global__ void zeroUpperRows(double* __restrict__ A, const size_t n, const int k, const int nb) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t width = n - k;  // columns k .. n-1 hold the part that can be non-zero
    if (idx >= (size_t)nb * width) {
        return;
    }
    const size_t r = idx / width;
    const size_t c = idx - r * width;  // relative to column k
    if (c > r) {
        A[(size_t)(k + r) * n + (k + c)] = 0.0;
    }
}

__global__ void addDiagonal(double* __restrict__ A, const size_t n, const double v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * n + i] += v;
    }
}

static inline unsigned int ceilDiv(const size_t a, const unsigned int b) {
    return (unsigned int)((a + b - 1) / b);
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) {
        return true;
    }

    double* dA = nullptr;
    double* dLinv = nullptr;
    int* dInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dLinv, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));

    cudaStream_t sMain, sCopy;
    CUDA_CHECK(cudaStreamCreate(&sMain));
    CUDA_CHECK(cudaStreamCreate(&sCopy));
    cudaEvent_t rowsReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&rowsReady, cudaEventDisableTiming));

    const int infoInit = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(dInfo, &infoInit, sizeof(int), cudaMemcpyHostToDevice, sMain));
    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, sMain));

    for (size_t k = 0; k < n; k += NB) {
        const int nb = (int)std::min((size_t)NB, n - k);
        const size_t rows = n - (k + nb);

        // 1. factorize the diagonal block (and invert it)
        potrfDiag<<<1, 256, 0, sMain>>>(dA, n, (int)k, nb, dLinv, dInfo);

        // The block rows [k, k+nb) are final now; zero their upper part and
        // stream them back to the host while the trailing update runs.
        zeroUpperRows<<<ceilDiv((size_t)nb * (n - k), 256), 256, 0, sMain>>>(dA, n, (int)k, nb);
        CUDA_CHECK(cudaEventRecord(rowsReady, sMain));

        if (rows > 0) {
            // 2. panel solve, expressed as P = P * Linv^T (in-place: every block
            //    reads exactly the rows it later writes, separated by a barrier)
            double* panel = dA + (k + nb) * n + k;
            gemmNT<<<dim3(1, ceilDiv(rows, TILE)), GEMM_THREADS, 0, sMain>>>(
                (int)rows, nb, nb, 1.0, panel, (int)n, dLinv, NB, 0.0, panel, (int)n, false);

            // 3. trailing update: A22 -= P * P^T  (lower triangle only)
            double* trailing = dA + (k + nb) * n + (k + nb);
            const unsigned int tiles = ceilDiv(rows, TILE);
            gemmNT<<<dim3(tiles, tiles), GEMM_THREADS, 0, sMain>>>(
                (int)rows, (int)rows, nb, -1.0, panel, (int)n, panel, (int)n, 1.0, trailing,
                (int)n, true);
        }

        CUDA_CHECK(cudaStreamWaitEvent(sCopy, rowsReady, 0));
        CUDA_CHECK(cudaMemcpyAsync(A.data() + k * n, dA + k * n, (size_t)nb * n * sizeof(double),
                                   cudaMemcpyDeviceToHost, sCopy));
    }

    int info = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, sMain));
    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaStreamSynchronize(sCopy));

    CUDA_CHECK(cudaEventDestroy(rowsReady));
    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dLinv));
    CUDA_CHECK(cudaFree(dInfo));

    if (info != INT_MAX) {
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

    // Compute A = B * B^T on the GPU
    double* dB = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int tiles = ceilDiv(n, TILE);
    gemmNT<<<dim3(tiles, tiles), GEMM_THREADS>>>((int)n, (int)n, (int)n, 1.0, dB, (int)n, dB,
                                                 (int)n, 0.0, dA, (int)n, false);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonal<<<ceilDiv(n, 256), 256>>>(dA, n, (double)n);

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU
    double* dL = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int tiles = ceilDiv(n, TILE);
    gemmNT<<<dim3(tiles, tiles), GEMM_THREADS>>>((int)n, (int)n, (int)n, 1.0, dL, (int)n, dL,
                                                 (int)n, 0.0, dR, (int)n, false);

    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));

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
