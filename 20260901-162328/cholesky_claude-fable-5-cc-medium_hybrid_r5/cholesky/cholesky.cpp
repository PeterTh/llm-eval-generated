#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked right-looking Cholesky decomposition.
//
// - Block columns of width NB are distributed cyclically over the MPI ranks;
//   each rank drives one GPU (selected by local rank on the node).
// - The owner of the current block column factors the NB x NB diagonal block
//   on the CPU with OpenMP, solves the panel below it with a CUDA TRSM kernel,
//   and broadcasts the finished panel to all ranks.
// - Every rank then applies the trailing SYRK/GEMM update to the block columns
//   it owns using a tiled CUDA kernel, so the O(n^3) work is split across all
//   GPUs in the cluster.
// - Since every panel is broadcast, each rank ends up with the complete factor
//   L on its GPU; no final gather is required.

constexpr size_t NB = 256;  // distribution / panel block size
constexpr int TS = 32;      // CUDA tile size

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// A[r][c] -= sum_t A[r][pcol+t] * A[c][pcol+t]
// for r in [rowStart, rowStart+m), c in [colStart, colStart+w), t in [0, kw).
// Used both for the trailing update (panel * panel^T) and for the blocked
// TRSM update inside a panel. Each element accumulates in increasing t order,
// matching the summation order of the original sequential algorithm.
__global__ void gemmNTKernel(double* __restrict__ A, size_t n, size_t rowStart,
                             size_t m, size_t colStart, size_t w, size_t pcol,
                             size_t kw) {
    __shared__ double Xs[TS][TS + 1];
    __shared__ double Ys[TS][TS + 1];

    const size_t r = rowStart + blockIdx.y * TS + threadIdx.y;
    const size_t c = colStart + blockIdx.x * TS + threadIdx.x;
    const size_t yr = colStart + blockIdx.x * TS + threadIdx.y;

    double acc = 0.0;
    for (size_t t0 = 0; t0 < kw; t0 += TS) {
        const size_t t = t0 + threadIdx.x;
        Xs[threadIdx.y][threadIdx.x] =
            (r < rowStart + m && t < kw) ? A[r * n + pcol + t] : 0.0;
        Ys[threadIdx.y][threadIdx.x] =
            (yr < colStart + w && t < kw) ? A[yr * n + pcol + t] : 0.0;
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TS; ++tt) {
            acc += Xs[threadIdx.y][tt] * Ys[threadIdx.x][tt];
        }
        __syncthreads();
    }
    if (r < rowStart + m && c < colStart + w) {
        A[r * n + c] -= acc;
    }
}

// A[r][c] = sum_k B[r][k] * B[k][c]^T (+ n on the diagonal); computes the
// symmetric positive definite input matrix A = B * B^T + n * I on the GPU.
__global__ void genMatKernel(double* __restrict__ A,
                             const double* __restrict__ B, size_t n) {
    __shared__ double Xs[TS][TS + 1];
    __shared__ double Ys[TS][TS + 1];

    const size_t r = blockIdx.y * TS + threadIdx.y;
    const size_t c = blockIdx.x * TS + threadIdx.x;
    const size_t yr = blockIdx.x * (size_t)TS + threadIdx.y;

    double acc = 0.0;
    for (size_t t0 = 0; t0 < n; t0 += TS) {
        const size_t t = t0 + threadIdx.x;
        Xs[threadIdx.y][threadIdx.x] = (r < n && t < n) ? B[r * n + t] : 0.0;
        Ys[threadIdx.y][threadIdx.x] = (yr < n && t < n) ? B[yr * n + t] : 0.0;
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TS; ++tt) {
            acc += Xs[threadIdx.y][tt] * Ys[threadIdx.x][tt];
        }
        __syncthreads();
    }
    if (r < n && c < n) {
        A[r * n + c] = acc + (r == c ? (double)n : 0.0);
    }
}

// Solve B := B * L^{-T} for one TS-wide sub-panel: rows [rowStart, rowStart+m)
// of columns [k0+s, k0+s+sw), against the already-factored TS x TS diagonal
// sub-block at (k0+s, k0+s). One thread per row; the diagonal sub-block is
// padded to the identity beyond sw so the solve can stay fully unrolled.
__global__ void trsmPanelKernel(double* __restrict__ A, size_t n, size_t k0,
                                size_t s, size_t sw, size_t rowStart,
                                size_t m) {
    __shared__ double Ls[TS][TS + 1];

    for (int idx = threadIdx.x; idx < TS * TS; idx += blockDim.x) {
        const int rr = idx / TS;
        const int cc = idx % TS;
        double v = (rr == cc) ? 1.0 : 0.0;
        if (rr < (int)sw && cc < (int)sw && cc <= rr) {
            v = A[(k0 + s + rr) * n + (k0 + s + cc)];
        }
        Ls[rr][cc] = v;
    }
    __syncthreads();

    const size_t r = rowStart + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rowStart + m) return;

    double b[TS];
#pragma unroll
    for (int j = 0; j < TS; ++j) {
        b[j] = (j < (int)sw) ? A[r * n + k0 + s + j] : 0.0;
    }
#pragma unroll
    for (int j = 0; j < TS; ++j) {
        double v = b[j];
#pragma unroll
        for (int t = 0; t < TS; ++t) {
            if (t < j) v -= b[t] * Ls[j][t];
        }
        b[j] = v / Ls[j][j];
    }
#pragma unroll
    for (int j = 0; j < TS; ++j) {
        if (j < (int)sw) A[r * n + k0 + s + j] = b[j];
    }
}

// Unblocked Cholesky of the nb x nb diagonal block on the CPU (OpenMP over
// the rows of each column). Zeroes the upper triangle of the block. Returns
// the failing column index if the block is not positive definite, else -1.
static long long potrfHost(double* D, const size_t nb) {
    // The block is small (nb <= NB); cap the team size so the parallel loop
    // is not swamped by fork/join overhead when many cores are available.
    const int nthreads = std::max(1, std::min(omp_get_max_threads(), (int)(nb / 16)));
    for (size_t j = 0; j < nb; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += D[j * nb + k] * D[j * nb + k];
        }
        const double val = D[j * nb + j] - sum;
        if (val <= 0.0) {
            return (long long)j;
        }
        const double d = sqrt(val);
        D[j * nb + j] = d;
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (size_t i = j + 1; i < nb; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += D[i * nb + k] * D[j * nb + k];
            }
            D[i * nb + j] = (D[i * nb + j] - s) / d;
        }
        for (size_t c = j + 1; c < nb; ++c) {
            D[j * nb + c] = 0.0;
        }
    }
    return -1;
}

// Factor block column k in place on the owner's GPU/CPU: CPU (OpenMP) POTRF
// of the diagonal block, CUDA TRSM of the rows below, then stage the finished
// panel into the pinned host buffer h_panel for the broadcast. Returns the
// global index of the failing diagonal element, or -1 on success.
static long long computePanel(double* d_A, const size_t n, const size_t k,
                              double* h_panel, double* h_diag) {
    const size_t dbl = sizeof(double);
    const size_t k0 = k * NB;
    const size_t kb = std::min(NB, n - k0);
    const size_t mrows = n - (k0 + kb);  // panel rows below the diag block

    CUDA_CHECK(cudaMemcpy2D(h_diag, kb * dbl, d_A + k0 * n + k0, n * dbl,
                            kb * dbl, kb, cudaMemcpyDeviceToHost));
    const long long status = potrfHost(h_diag, kb);
    if (status >= 0) {
        return status + (long long)k0;  // global failing diagonal index
    }
    CUDA_CHECK(cudaMemcpy2D(d_A + k0 * n + k0, n * dbl, h_diag, kb * dbl,
                            kb * dbl, kb, cudaMemcpyHostToDevice));
    // Blocked TRSM of the panel below the diagonal block.
    if (mrows > 0) {
        for (size_t s = 0; s < kb; s += TS) {
            const size_t sw = std::min((size_t)TS, kb - s);
            const int bt = 128;
            trsmPanelKernel<<<(unsigned)((mrows + bt - 1) / bt), bt>>>(
                d_A, n, k0, s, sw, k0 + kb, mrows);
            if (s + sw < kb) {
                dim3 grid((unsigned)((kb - s - sw + TS - 1) / TS),
                          (unsigned)((mrows + TS - 1) / TS));
                gemmNTKernel<<<grid, dim3(TS, TS)>>>(d_A, n, k0 + kb, mrows,
                                                     k0 + s + sw, kb - s - sw,
                                                     k0 + s, sw);
            }
        }
    }
    // Stage the finished panel (diag block + rows below) for the broadcast;
    // the copy also synchronizes the kernels above.
    CUDA_CHECK(cudaMemcpy2D(h_panel, kb * dbl, d_A + k0 * n + k0, n * dbl,
                            kb * dbl, n - k0, cudaMemcpyDeviceToHost));
    return -1;
}

// Distributed blocked right-looking Cholesky on the device matrix d_A with
// one-step lookahead: while all ranks apply the trailing update for step k,
// the owner of block column k+1 has already factored that panel (its column
// was updated first) and its broadcast is in flight via MPI_Ibcast, hiding
// the panel factorization and communication on the critical path.
// h_panel (n*NB) and h_diag (NB*NB) are pinned host staging buffers.
bool choleskyDecomposition(double* d_A, const size_t n, const int rank,
                           const int nranks, double* h_panel, double* h_diag) {
    const size_t nblocks = (n + NB - 1) / NB;
    const size_t dbl = sizeof(double);

    auto reportFailure = [rank](long long idx) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal "
                   "element %lld\n", idx);
        }
    };

    // Prologue: factor and broadcast panel 0.
    long long status = -1;
    if (rank == 0) {
        status = computePanel(d_A, n, 0, h_panel, h_diag);
    }
    MPI_Bcast(&status, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (status >= 0) {
        reportFailure(status);
        return false;
    }
    MPI_Bcast(h_panel, (int)(n * std::min(NB, n)), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    if (rank != 0) {
        CUDA_CHECK(cudaMemcpy2D(d_A, n * dbl, h_panel, std::min(NB, n) * dbl,
                                std::min(NB, n) * dbl, n,
                                cudaMemcpyHostToDevice));
    }

    for (size_t k = 0; k < nblocks; ++k) {
        const size_t k0 = k * NB;
        const size_t kb = std::min(NB, n - k0);
        const bool haveNext = (k + 1 < nblocks);
        const int ownerNext = (int)((k + 1) % (size_t)nranks);
        long long nextStatus = -1;
        MPI_Request reqs[2];

        if (haveNext) {
            const size_t c0 = (k + 1) * NB;
            const size_t nkb = std::min(NB, n - c0);
            if (rank == ownerNext) {
                // Lookahead: update block column k+1 with panel k, then
                // factor it right away so its broadcast overlaps the bulk
                // trailing update below.
                dim3 grid((unsigned)((nkb + TS - 1) / TS),
                          (unsigned)((n - c0 + TS - 1) / TS));
                gemmNTKernel<<<grid, dim3(TS, TS)>>>(d_A, n, c0, n - c0, c0,
                                                     nkb, k0, kb);
                nextStatus = computePanel(d_A, n, k + 1, h_panel, h_diag);
            }
            MPI_Ibcast(&nextStatus, 1, MPI_LONG_LONG, ownerNext,
                       MPI_COMM_WORLD, &reqs[0]);
            MPI_Ibcast(h_panel, (int)((n - c0) * nkb), MPI_DOUBLE, ownerNext,
                       MPI_COMM_WORLD, &reqs[1]);
        }

        // Trailing update of the block columns owned by this rank (column
        // k+1 was already handled by its owner above).
        for (size_t j = k + 2 - (rank == ownerNext ? 0 : 1); j < nblocks;
             ++j) {
            if ((int)(j % (size_t)nranks) != rank) continue;
            const size_t c0 = j * NB;
            const size_t w = std::min(NB, n - c0);
            const size_t m = n - c0;
            dim3 grid((unsigned)((w + TS - 1) / TS),
                      (unsigned)((m + TS - 1) / TS));
            gemmNTKernel<<<grid, dim3(TS, TS)>>>(d_A, n, c0, m, c0, w, k0, kb);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        if (haveNext) {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            if (nextStatus >= 0) {
                reportFailure(nextStatus);
                return false;
            }
            const size_t c0 = (k + 1) * NB;
            const size_t nkb = std::min(NB, n - c0);
            if (rank != ownerNext) {
                CUDA_CHECK(cudaMemcpy2D(d_A + c0 * n + c0, n * dbl, h_panel,
                                        nkb * dbl, nkb * dbl, n - c0,
                                        cudaMemcpyHostToDevice));
            }
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix A = B * B^T + n * I directly
// on the GPU. B is drawn from the same deterministic rand_r(42) stream as the
// original sequential code, so the input matrix is reproducible on all ranks.
void generatePositiveDefiniteMatrix(double* d_A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    dim3 grid((unsigned)((n + TS - 1) / TS), (unsigned)((n + TS - 1) / TS));
    genMatKernel<<<grid, dim3(TS, TS)>>>(d_A, d_B, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_B));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (OpenMP)
#pragma omp parallel for schedule(static)
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

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks,
               omp_get_max_threads());
    }

    // Bind each rank to a GPU by local (per-node) rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    CUDA_CHECK(cudaSetDevice(localRank % devCount));

    // Allocate the matrix on the GPU plus pinned host staging buffers.
    double* d_A = nullptr;
    double* h_panel = nullptr;
    double* h_diag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_panel, n * NB * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_diag, NB * NB * sizeof(double)));

    // Generate positive definite matrix (redundantly on every rank's GPU;
    // the deterministic seed and fixed accumulation order make it identical
    // everywhere, so no broadcast of the O(n^2) matrix is needed).
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d_A, n);

    std::vector<double> A;
    std::vector<double> A_orig;
    if (rank == 0 && validate) {
        A_orig.resize(n * n);  // Save original for validation
        CUDA_CHECK(cudaMemcpy(A_orig.data(), d_A, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n, rank, nranks, h_panel, h_diag);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if (rank == 0) {
        // Every rank holds the full factor; fetch it once on rank 0 and zero
        // the (untouched) upper triangular part like the original code.
        A.resize(n * n);
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }

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
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFreeHost(h_panel));
    CUDA_CHECK(cudaFreeHost(h_diag));

    MPI_Finalize();
    return exitCode;
}
