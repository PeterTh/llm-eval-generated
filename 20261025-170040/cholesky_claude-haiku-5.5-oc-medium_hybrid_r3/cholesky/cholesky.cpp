// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Algorithm: blocked left-looking Cholesky (A = L * L^T, row-major, lower L).
// For every element L(i,j) the dot product over k < j is accumulated by a single
// sequential fused-multiply-add chain, as in the sequential reference. The blocked
// formulation keeps that chain intact: the k < J0 part for a block of columns J is
// computed by a k-ascending GPU GEMM, and the in-block remainder continues the same
// accumulator on the host. The output follows the reference's operation order; the
// remaining differences are rounding-level (the reference's own rounding varies with
// its vectorization, so it is not bit-reproducible across compiler flags).
//
// Parallelization:
//   MPI    - the rows below each column block are split across ranks; every rank
//            receives the finished block column (MPI_Allgatherv) and keeps a full
//            copy of L on the host and on its GPU.
//   CUDA   - each rank drives one GPU (round-robin over local ranks). The
//            dominant O(n^3) work, the GEMM over finished columns, runs on the GPU.
//   OpenMP - host-side work: diagonal-block factorization, triangular solves of
//            the rows below the block, matrix generation and validation.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// Width of a column block (panel). Also the GPU GEMM's N dimension.
constexpr size_t kPanel = 128;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

struct Dist {
    int rank = 0;
    int size = 1;
    int localRank = 0;
    int localSize = 1;
};

// Index of the first entry of packed lower-triangular row i (row i has i+1 entries).
inline size_t tri(size_t i) { return i * (i + 1) / 2; }

// Balanced contiguous partition of [lo, hi) into `parts` pieces; returns piece `part`.
inline void partition(size_t lo, size_t hi, int parts, int part, size_t& b, size_t& e) {
    const size_t len = hi - lo;
    b = lo + len * static_cast<size_t>(part) / parts;
    e = lo + len * static_cast<size_t>(part + 1) / parts;
}

// ---------------------------------------------------------------------------
// GPU GEMM:  C[i][jj] = sum_{k<K} A[i][k] * B[jj][k]   (C row-major, ldc)
// Each output is accumulated by one register chain with k strictly ascending,
// so the result is bit-identical to a sequential fma loop over k.
// TR is the per-thread register tile edge: the CTA computes a (16*TR)^2 tile.
// Large TR gives throughput for big products; TR=1 gives enough CTAs for the
// small diagonal-block products.
// ---------------------------------------------------------------------------
template <int TR>
__global__ void __launch_bounds__(256)
gemmNTKernel(int M, int N, int K,
             const double* __restrict__ A, int lda,
             const double* __restrict__ B, int ldb,
             double* __restrict__ C, int ldc) {
    constexpr int TILE = 16 * TR;  // output tile edge
    constexpr int TK = 16;         // k tile
    constexpr int LOADS = TILE * TK / 256;
    __shared__ double As[TK][TILE + 1];
    __shared__ double Bs[TK][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;

    double acc[TR][TR];
#pragma unroll
    for (int r = 0; r < TR; ++r)
#pragma unroll
        for (int c = 0; c < TR; ++c) acc[r][c] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TK) {
        const int kmax = min(TK, K - k0);
#pragma unroll
        for (int t = 0; t < LOADS; ++t) {
            const int idx = tid + t * 256;
            const int r = idx / TK;
            const int kk = idx % TK;
            const int gr = row0 + r;
            const int gc = col0 + r;
            As[kk][r] = (kk < kmax && gr < M) ? A[static_cast<size_t>(gr) * lda + k0 + kk] : 0.0;
            Bs[kk][r] = (kk < kmax && gc < N) ? B[static_cast<size_t>(gc) * ldb + k0 + kk] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < kmax; ++kk) {
            double a[TR], b[TR];
#pragma unroll
            for (int r = 0; r < TR; ++r) {
                a[r] = As[kk][ty + 16 * r];
                b[r] = Bs[kk][tx + 16 * r];
            }
#pragma unroll
            for (int r = 0; r < TR; ++r)
#pragma unroll
                for (int c = 0; c < TR; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TR; ++r) {
        const int gr = row0 + ty + 16 * r;
#pragma unroll
        for (int c = 0; c < TR; ++c) {
            const int gc = col0 + tx + 16 * c;
            if (gr < M && gc < N) C[static_cast<size_t>(gr) * ldc + gc] = acc[r][c];
        }
    }
}

// Launches C = A * B^T (k-ascending accumulation) on the device.
template <int TR>
void gemmNT(int M, int N, int K, const double* A, int lda, const double* B, int ldb,
            double* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    constexpr int TILE = 16 * TR;
    const dim3 block(16, 16);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemmNTKernel<TR><<<grid, block>>>(M, N, K, A, lda, B, ldb, C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Diagonal block of column block [J0, J0+w): finishes L for those columns on the
// host. Pd holds the partial sums over k < J0 (ld = ldp). Every rank runs this on
// identical inputs, so all ranks reach the same result (and the same failure).
// ---------------------------------------------------------------------------
bool factorDiagonalBlock(std::vector<double>& A, const size_t n, const size_t J0,
                         const size_t w, const double* Pd, const size_t ldp, const bool report) {
    for (size_t j = 0; j < w; ++j) {
        const size_t jg = J0 + j;
        for (size_t i = j; i < w; ++i) {
            const size_t ig = J0 + i;
            double sum = Pd[i * ldp + j];
            for (size_t k = 0; k < j; ++k) {
                sum = std::fma(A[ig * n + J0 + k], A[jg * n + J0 + k], sum);
            }
            if (i == j) {
                const double val = A[jg * n + jg] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    if (report) printf("Error: Matrix is not positive definite at diagonal element %zu\n", jg);
                    return false;
                }
                A[jg * n + jg] = sqrt(val);
            } else {
                A[ig * n + jg] = (A[ig * n + jg] - sum) / A[jg * n + jg];
            }
        }
    }
    return true;
}

// Blocked left-looking Cholesky. On success the lower triangle of A holds L and the
// upper triangle is zero. `A` must hold the input matrix on entry (on every rank).
bool choleskyDecomposition(std::vector<double>& A, const size_t n, const Dist& d,
                           double* dL, double* dP,
                           std::vector<double>& Pd, std::vector<double>& Pr,
                           std::vector<double>& pack) {
    constexpr size_t nb = kPanel;

    for (size_t J0 = 0; J0 < n; J0 += nb) {
        const size_t J1 = std::min(n, J0 + nb);
        const size_t w = J1 - J0;

        // Rows below the block split across ranks; the block rows are computed by all.
        size_t rs = 0, re = 0;
        partition(J1, n, d.size, d.rank, rs, re);
        const size_t rc = re - rs;

        // (a) GPU: partial sums over finished columns k < J0, ascending k.
        gemmNT<1>(static_cast<int>(w), static_cast<int>(w), static_cast<int>(J0),
                  dL + J0 * n, static_cast<int>(n), dL + J0 * n, static_cast<int>(n),
                  dP, static_cast<int>(nb));
        gemmNT<4>(static_cast<int>(rc), static_cast<int>(w), static_cast<int>(J0),
               dL + rs * n, static_cast<int>(n), dL + J0 * n, static_cast<int>(n),
               dP + nb * nb, static_cast<int>(nb));
        CUDA_CHECK(cudaMemcpy2D(Pd.data(), nb * sizeof(double), dP, nb * sizeof(double),
                                w * sizeof(double), w, cudaMemcpyDeviceToHost));
        if (rc > 0) {
            CUDA_CHECK(cudaMemcpy2D(Pr.data(), nb * sizeof(double), dP + nb * nb, nb * sizeof(double),
                                    w * sizeof(double), rc, cudaMemcpyDeviceToHost));
        }

        // (b) Host: finish the diagonal block (identical on every rank).
        if (!factorDiagonalBlock(A, n, J0, w, Pd.data(), nb, d.rank == 0)) return false;

        // (c) Host (OpenMP): solve for the rows below the block assigned to this rank.
        // The original entries A(i, J) are read from A; the results are written straight
        // into the exchange buffer (row t of this rank's share, width w).
#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < rc; ++t) {
            const size_t i = rs + t;
            const double* Ai_orig = &A[i * n];
            double* Li = &pack[(rs - J1 + t) * w];
            for (size_t j = 0; j < w; ++j) {
                const double* Ljj_row = &A[(J0 + j) * n];
                double sum = Pr[t * nb + j];
                for (size_t k = 0; k < j; ++k) {
                    sum = std::fma(Li[k], Ljj_row[J0 + k], sum);
                }
                Li[j] = (Ai_orig[J0 + j] - sum) / Ljj_row[J0 + j];
            }
        }

        // (d) MPI: exchange the rows below the block so every rank has the full panel.
        const size_t rowsBelow = n - J1;
        if (rowsBelow > 0) {
            std::vector<int> counts(d.size), displs(d.size);
            for (int p = 0; p < d.size; ++p) {
                size_t pb = 0, pe = 0;
                partition(J1, n, d.size, p, pb, pe);
                counts[p] = static_cast<int>((pe - pb) * w);
                displs[p] = static_cast<int>((pb - J1) * w);
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pack.data(), counts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
            for (size_t t = 0; t < rowsBelow; ++t) {
                for (size_t j = 0; j < w; ++j) {
                    A[(J1 + t) * n + J0 + j] = pack[t * w + j];
                }
            }
        }

        // (e) Upload the finished block column (rows J0..n) to the GPU for later GEMMs.
        CUDA_CHECK(cudaMemcpy2D(dL + J0 * n + J0, n * sizeof(double), A.data() + J0 * n + J0,
                                n * sizeof(double), w * sizeof(double), n - J0,
                                cudaMemcpyHostToDevice));
    }

    // Zero out upper triangular part
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
    }

    return true;
}

// Generate a symmetric positive definite matrix (identical values to the sequential
// generator: B from rand_r, A = B * B^T with a sequential k-sum, plus n on the diagonal).
// Rows of the lower triangle are split across ranks and exchanged.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const Dist& d) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Row boundaries balancing the triangular work (row i costs i+1).
    std::vector<size_t> bnd(d.size + 1);
    for (int p = 0; p <= d.size; ++p) {
        bnd[p] = (p == d.size) ? n
                               : static_cast<size_t>(std::floor(n * std::sqrt(static_cast<double>(p) / d.size)));
    }
    const size_t r0 = bnd[d.rank];
    const size_t r1 = bnd[d.rank + 1];

    std::vector<double> packed(tri(n));
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = r0; i < r1; ++i) {
        const double* Bi = &B[i * n];
        double* out = &packed[tri(i)];
        for (size_t j = 0; j <= i; ++j) {
            const double* Bj = &B[j * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum = std::fma(Bi[k], Bj[k], sum);
            }
            out[j] = sum;
        }
        // Add diagonal dominance to ensure positive definiteness
        out[i] += n;
    }

    std::vector<int> counts(d.size), displs(d.size);
    for (int p = 0; p < d.size; ++p) {
        counts[p] = static_cast<int>(tri(bnd[p + 1]) - tri(bnd[p]));
        displs[p] = static_cast<int>(tri(bnd[p]));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, packed.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) A[i * n + j] = packed[tri(i) + j];
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) A[i * n + j] = A[j * n + i];
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
#pragma omp parallel for schedule(dynamic, 4)
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

#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
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

int runBenchmark(int argc, char** argv, const Dist& d) {
    const bool root = (d.rank == 0);
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // The packed lower triangle is exchanged with MPI counts of type int.
    if (n > 65535) {
        if (root) printf("Matrix size %zu exceeds the supported maximum of 65535\n", n);
        return 1;
    }

    // Each rank drives one GPU; this benchmark requires CUDA.
    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices == 0) {
        if (root) fprintf(stderr, "No CUDA devices available; this benchmark requires a GPU\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(d.localRank % numDevices));

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, d);

    if (validate && root) {
        A_orig = A; // Save original for validation
    }

    // Workspace (allocated outside the timed region)
    constexpr size_t nb = kPanel;
    double* dL = nullptr;
    double* dP = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(1, n * n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dP, std::max<size_t>(1, (n + nb) * nb) * sizeof(double)));
    std::vector<double> Pd(nb * nb), Pr(std::max<size_t>(1, n * nb)), pack(std::max<size_t>(1, n * nb));

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, d, dL, dP, Pd, Pr, pack);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dP));

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", max_ms);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation (root only; the result is identical on every rank)
    if (validate) {
        int status = 0;
        if (root) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        return status;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    Dist d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &d.localRank);
    MPI_Comm_size(localComm, &d.localSize);
    MPI_Comm_free(&localComm);

    // Split the node's cores between the ranks sharing it, unless the user set OMP_NUM_THREADS.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        const long perRank = std::max(1L, online / std::max(1, d.localSize));
        omp_set_num_threads(static_cast<int>(perRank));
    }

    int status = runBenchmark(argc, argv, d);

    // All ranks exit with the same status.
    int globalStatus = 0;
    MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalStatus;
}
