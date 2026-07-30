#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr int    BLOCK_SIZE = 64;     // Block size for blocked Cholesky
constexpr int    TILE_DIM   = 16;     // Tile dimension for CUDA kernel
constexpr size_t COL_CHUNK  = 4096;   // GPU column chunk (limits GPU memory)

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                                    \
    cudaError_t err_ = call;                                                     \
    if (err_ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(err_));                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                            \
} while (0)

#define MPI_CHECK(call) do {                                                     \
    int err_ = call;                                                             \
    if (err_ != MPI_SUCCESS) {                                                   \
        fprintf(stderr, "MPI error at %s:%d: err=%d\n",                          \
                __FILE__, __LINE__, err_);                                       \
        MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                            \
} while (0)

// ---------------------------------------------------------------------------
// CUDA kernel:  C = C - A * B^T
//   A   : M x K   (local panel, row-major, leading dim = K)
//   B_T : K x N   (full panel, transposed, leading dim = N)
//   C   : M x N   (trailing submatrix, row-major, leading dim = N)
// ---------------------------------------------------------------------------
__global__ void trailing_update_kernel(double*       C,
                                       const double* A,
                                       const double* B_T,
                                       int M, int N, int K)
{
    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    double sum = 0.0;

    for (int k = 0; k < K; k += TILE_DIM) {
        // Load tile of A (row-major, leading dim K)
        if (row < M && k + tx < K)
            As[ty][tx] = A[static_cast<size_t>(row) * K + k + tx];
        else
            As[ty][tx] = 0.0;

        // Load tile of B_T (row-major, leading dim N)
        if (col < N && k + ty < K)
            Bs[tx][ty] = B_T[static_cast<size_t>(k + ty) * N + col];
        else
            Bs[tx][ty] = 0.0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i)
            sum += As[ty][i] * Bs[tx][i];

        __syncthreads();
    }

    if (row < M && col < N)
        C[static_cast<size_t>(row) * N + col] = C[static_cast<size_t>(row) * N + col] - sum;
}

// ---------------------------------------------------------------------------
// GPU-accelerated trailing matrix update
//   A_local  : local portion of the global matrix (local_nrows x n, row-major)
// ---------------------------------------------------------------------------
void gpu_trailing_update(std::vector<double>& A_local, size_t n,
                         size_t row_start, size_t local_nrows,
                         size_t j, size_t b,
                         const std::vector<double>& full_panel,
                         size_t remaining_rows)
{
    size_t upd_start = std::max(row_start, j + b);
    if (upd_start >= row_start + local_nrows) return;   // nothing to update

    size_t upd_end   = row_start + local_nrows;
    int M    = static_cast<int>(upd_end - upd_start);   // local rows in trailing matrix
    int fullN = static_cast<int>(remaining_rows);        // total rows in trailing matrix
    int K    = static_cast<int>(b);

    if (M == 0 || fullN == 0) return;

    // ---------- Extract local panel (M x K) ----------
    std::vector<double> panel_local(static_cast<size_t>(M) * K);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            panel_local[static_cast<size_t>(i) * K + k] =
                A_local[(static_cast<size_t>(i) + upd_start - row_start) * n + j + k];
        }
    }

    // ---------- Transpose full_panel: NxK -> KxN ----------
    // B_T[kk][l] = full_panel[l * K + kk]
    std::vector<double> B_T(static_cast<size_t>(K) * fullN);
    #pragma omp parallel for collapse(2)
    for (int kk = 0; kk < K; ++kk) {
        for (int l = 0; l < fullN; ++l) {
            B_T[static_cast<size_t>(kk) * fullN + l] =
                full_panel[static_cast<size_t>(l) * K + kk];
        }
    }

    // ---------- Copy to GPU ----------
    double *d_A = nullptr, *d_B_T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,   static_cast<size_t>(M) * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B_T, static_cast<size_t>(K) * fullN * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A,   panel_local.data(),
                          static_cast<size_t>(M) * K * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_T, B_T.data(),
                          static_cast<size_t>(K) * fullN * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---------- Process C in column chunks ----------
    for (int l_start = 0; l_start < fullN; l_start += static_cast<int>(COL_CHUNK)) {
        int chunk = std::min(static_cast<int>(COL_CHUNK), fullN - l_start);

        // Extract C chunk from A_local
        std::vector<double> C_chunk(static_cast<size_t>(M) * chunk);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int l = 0; l < chunk; ++l) {
                C_chunk[static_cast<size_t>(i) * chunk + l] =
                    A_local[(static_cast<size_t>(i) + upd_start - row_start) * n +
                            (j + b + l_start + l)];
            }
        }

        double *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(M) * chunk * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_C, C_chunk.data(),
                              static_cast<size_t>(M) * chunk * sizeof(double),
                              cudaMemcpyHostToDevice));

        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((chunk + TILE_DIM - 1) / TILE_DIM,
                  (M    + TILE_DIM - 1) / TILE_DIM);

        // Point into B_T at column l_start (kernel uses leading dim = fullN)
        trailing_update_kernel<<<grid, block>>>(d_C, d_A,
                                                d_B_T + l_start,
                                                M, chunk, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back
        CUDA_CHECK(cudaMemcpy(C_chunk.data(), d_C,
                              static_cast<size_t>(M) * chunk * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // Write back to A_local
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int l = 0; l < chunk; ++l) {
                A_local[(static_cast<size_t>(i) + upd_start - row_start) * n +
                        (j + b + l_start + l)] = C_chunk[static_cast<size_t>(i) * chunk + l];
            }
        }

        CUDA_CHECK(cudaFree(d_C));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_T));
}

// ---------------------------------------------------------------------------
// Row-distribution helpers (deterministic on every rank)
// ---------------------------------------------------------------------------
void compute_row_distribution(size_t n, int nprocs,
                              std::vector<size_t>& local_nrows,
                              std::vector<size_t>& starts)
{
    local_nrows.resize(nprocs);
    starts.resize(nprocs);
    size_t base = n / nprocs;
    size_t extra = n % nprocs;
    for (int p = 0; p < nprocs; ++p) {
        local_nrows[p] = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        starts[p] = (p == 0) ? 0 : starts[p-1] + local_nrows[p-1];
    }
}

// ---------------------------------------------------------------------------
// Blocked Cholesky decomposition with hybrid MPI + OpenMP + CUDA.
// Uses a right-looking block algorithm with 1D block-row distribution.
//
//   A_local : local_nrows x n matrix, row-major (in-place, output L)
//   row_start / row_end : global row range owned by this rank
// ---------------------------------------------------------------------------
bool choleskyDecomposition_hybrid(std::vector<double>& A_local, size_t n,
                                  int rank, int nprocs,
                                  size_t row_start, size_t row_end,
                                  const std::vector<size_t>& starts,
                                  const std::vector<size_t>& local_nrows)
{
    const size_t bsize = BLOCK_SIZE;

    for (size_t j = 0; j < n; j += bsize) {
        size_t b = std::min(bsize, n - j);

        // ----- 1. Factor diagonal block A[j:j+b, j:j+b] -----
        // Find rank that owns the first row of the block
        int diag_rank = 0;
        for (int p = 0; p < nprocs; ++p) {
            if (starts[p] <= j && j < starts[p] + local_nrows[p]) {
                diag_rank = p;
                break;
            }
        }

        // Gather diagonal block from owners to diag_rank
        std::vector<double> L_jj(b * b, 0.0);

        // Each rank sends its rows that overlap [j, j+b)
        size_t ol_start = std::max(j, row_start);
        size_t ol_end   = std::min(j + b, row_end);
        size_t ol_rows  = (ol_end > ol_start) ? (ol_end - ol_start) : 0;
        std::vector<double> send_buf(ol_rows * b);
        for (size_t ii = 0; ii < ol_rows; ++ii) {
            size_t g_row = ol_start + ii;
            size_t l_row = g_row - row_start;
            for (size_t kk = 0; kk < b; ++kk) {
                size_t col = j + kk;
                if (col <= g_row)   // lower triangle
                    send_buf[ii * b + kk] = A_local[l_row * n + col];
                else
                    send_buf[ii * b + kk] = 0.0;   // upper triangle should be 0
            }
        }

        // Compute recv counts / displs for diag_rank
        std::vector<int> recv_counts(nprocs, 0);
        std::vector<int> recv_displs(nprocs, 0);
        if (rank == diag_rank) {
            int offset = 0;
            for (int p = 0; p < nprocs; ++p) {
                size_t p_ol_start = std::max(j, starts[p]);
                size_t p_ol_end   = std::min(j + b, starts[p] + local_nrows[p]);
                size_t p_ol_rows  = (p_ol_end > p_ol_start) ? (p_ol_end - p_ol_start) : 0;
                recv_counts[p] = static_cast<int>(p_ol_rows * b);
                recv_displs[p] = offset;
                offset += recv_counts[p];
            }
        }

        MPI_Gatherv(send_buf.data(), static_cast<int>(ol_rows * b), MPI_DOUBLE,
                    L_jj.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                    diag_rank, MPI_COMM_WORLD);

        // Factor L_jj on diag_rank (unblocked Cholesky)
        if (rank == diag_rank) {
            for (size_t ii = 0; ii < b; ++ii) {
                for (size_t kk = 0; kk <= ii; ++kk) {
                    double sum = 0.0;
                    for (size_t ss = 0; ss < kk; ++ss)
                        sum += L_jj[ii * b + ss] * L_jj[kk * b + ss];

                    if (ii == kk) {
                        double val = L_jj[ii * b + ii] - sum;
                        if (val <= 0.0) {
                            fprintf(stderr, "[rank %d] Matrix not positive definite at diag %zu\n",
                                    rank, j + ii);
                            return false;
                        }
                        L_jj[ii * b + ii] = sqrt(val);
                        // Zero upper triangle of this row in L_jj
                        for (size_t ss = ii + 1; ss < b; ++ss)
                            L_jj[ii * b + ss] = 0.0;
                    } else {
                        L_jj[ii * b + kk] = (L_jj[ii * b + kk] - sum) / L_jj[kk * b + kk];
                    }
                }
            }
        }

        // Broadcast factored L_jj from diag_rank to all
        MPI_Bcast(L_jj.data(), static_cast<int>(b * b), MPI_DOUBLE, diag_rank, MPI_COMM_WORLD);

        // Write L_jj back to A_local for owned rows
        size_t write_start = std::max(j, row_start);
        size_t write_end   = std::min(j + b, row_end);
        #pragma omp parallel for
        for (size_t ii = write_start; ii < write_end; ++ii) {
            size_t ii_in_blk = ii - j;
            size_t l_row = ii - row_start;
            for (size_t kk = 0; kk <= ii_in_blk; ++kk)
                A_local[l_row * n + (j + kk)] = L_jj[ii_in_blk * b + kk];
            // Zero the remainder of this row in the upper-tri part of the block
            for (size_t kk = ii_in_blk + 1; kk < b; ++kk)
                A_local[l_row * n + (j + kk)] = 0.0;
        }

        // ----- 2. TRSM: compute L[local_rows, j:j+b] for rows >= j+b -----
        // For each owned row i >= j+b:
        //   L_ik = (A_ik - sum_{s<k} L_is * L_jj_ks) / L_jj_kk
        size_t trsm_start = std::max(row_start, j + b);
        #pragma omp parallel for
        for (size_t i = trsm_start; i < row_end; ++i) {
            size_t l_row = i - row_start;
            for (size_t k = 0; k < b; ++k) {
                double sum = 0.0;
                for (size_t s = 0; s < k; ++s)
                    sum += A_local[l_row * n + (j + s)] * L_jj[k * b + s];
                A_local[l_row * n + (j + k)] =
                    (A_local[l_row * n + (j + k)] - sum) / L_jj[k * b + k];
            }
        }

        // ----- 3. Allgather the full panel L[j+b:n, j:j+b] -----
        size_t remaining = (j + b >= n) ? 0 : (n - j - b);
        if (remaining > 0) {
            // Compute panel counts / displs for allgatherv
            std::vector<int> panel_counts(nprocs);
            std::vector<int> panel_displs(nprocs);
            int total_send = 0;
            for (int p = 0; p < nprocs; ++p) {
                size_t p_start = starts[p];
                size_t p_end   = starts[p] + local_nrows[p];
                size_t p_ol    = (p_end > j + b) ? (p_end - std::max(p_start, j + b)) : 0;
                panel_counts[p] = static_cast<int>(p_ol * b);
                panel_displs[p] = total_send;
                total_send += panel_counts[p];
            }

            // Build local panel contribution
            size_t local_panel_rows = (row_end > j + b)
                                        ? (row_end - std::max(row_start, j + b))
                                        : 0;
            std::vector<double> local_panel(local_panel_rows * b);
            if (local_panel_rows > 0) {
                #pragma omp parallel for collapse(2)
                for (size_t ii = 0; ii < local_panel_rows; ++ii) {
                    for (size_t kk = 0; kk < b; ++kk) {
                        size_t g_row = std::max(row_start, j + b) + ii;
                        local_panel[ii * b + kk] =
                            A_local[(g_row - row_start) * n + (j + kk)];
                    }
                }
            }

            std::vector<double> full_panel(remaining * b);
            MPI_Allgatherv(local_panel.data(),
                           static_cast<int>(local_panel_rows * b), MPI_DOUBLE,
                           full_panel.data(),
                           panel_counts.data(), panel_displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);

            // ----- 4. GPU-accelerated trailing matrix update -----
            // A_local[my_remaining_rows, j+b:n] -= L_local * L_full^T
            gpu_trailing_update(A_local, n, row_start, local_nrows[rank],
                                j, b, full_panel, remaining);
        }
    }

    // ----- Final: zero upper triangular part for all local rows -----
    #pragma omp parallel for
    for (size_t i = row_start; i < row_end; ++i) {
        size_t l_row = i - row_start;
        for (size_t jj = i + 1; jj < n; ++jj)
            A_local[l_row * n + jj] = 0.0;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generate a symmetric positive definite matrix (on rank 0)
// ---------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Validation: L * L^T == A_orig  (called on rank 0 with full matrices)
// ---------------------------------------------------------------------------
bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double err = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, err);
        double rel = err / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Set device for this MPI rank (round-robin GPU assignment)
    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus > 0) {
        int dev = rank % ngpus;
        CUDA_CHECK(cudaSetDevice(dev));
    } else {
        fprintf(stderr, "No CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---------- Parse arguments ----------
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
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
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices available: %d\n", ngpus);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---------- Row distribution ----------
    std::vector<size_t> local_nrows, starts;
    compute_row_distribution(n, nprocs, local_nrows, starts);
    size_t row_start = starts[rank];
    size_t row_end   = row_start + local_nrows[rank];

    // ---------- Generate matrix on rank 0 and scatter ----------
    std::vector<double> A_local(local_nrows[rank] * n);
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        std::vector<double> A_full(n * n);
        generatePositiveDefiniteMatrix(A_full, n);

        if (validate) A_orig = A_full;

        // Build MPI_Scatterv parameters
        std::vector<int> scounts(nprocs), sdispls(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            scounts[p] = static_cast<int>(local_nrows[p] * n);
            sdispls[p] = static_cast<int>(starts[p] * n);
        }
        MPI_Scatterv(A_full.data(), scounts.data(), sdispls.data(), MPI_DOUBLE,
                     A_local.data(), static_cast<int>(local_nrows[rank] * n),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     A_local.data(), static_cast<int>(local_nrows[rank] * n),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ---------- Timed hybrid Cholesky decomposition ----------
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition_hybrid(
                        A_local, n, rank, nprocs,
                        row_start, row_end, starts, local_nrows);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_ms = static_cast<long long>(duration.count());
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (!success) {
            printf("Cholesky decomposition failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Computation time: %lld ms\n", max_ms);
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (static_cast<double>(max_ms) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---------- Gather L to rank 0 ----------
    std::vector<double> L_full;
    if (rank == 0) L_full.resize(n * n);

    std::vector<int> gcounts(nprocs), gdispls(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        gcounts[p] = static_cast<int>(local_nrows[p] * n);
        gdispls[p] = static_cast<int>(starts[p] * n);
    }
    MPI_Gatherv(A_local.data(), static_cast<int>(local_nrows[rank] * n), MPI_DOUBLE,
                L_full.data(), gcounts.data(), gdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---------- Print results (rank 0) ----------
    if (printResults && rank == 0) {
        print_results(L_full, "CholeskyL");
    }

    // ---------- Validation (rank 0) ----------
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(L_full, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Finalize();
    return 0;
}
