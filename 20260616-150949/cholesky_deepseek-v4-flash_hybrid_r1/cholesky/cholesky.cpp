#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>

#include <omp.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int BLOCK_SIZE = 64;  // block size for right-looking algorithm

// ---------------------------------------------------------------------------
// CUDA kernel: trailing submatrix update
//   C(m, n) -= A(m, k) * B(n, k)^T
//   All matrices are row-major.
//   A and B are contiguous (leading dimension = k).
//   C has leading dimension ldC (may be > n).
// ---------------------------------------------------------------------------
template <int BS = 16>
__global__ void gemm_sub_kernel(double* __restrict__ C,
                                 const double* __restrict__ A,
                                 const double* __restrict__ B,
                                 int m, int n, int k, int ldC) {
    // Use padded shared memory to avoid bank conflicts
    __shared__ double sA[BS * (BLOCK_SIZE + 1)];
    __shared__ double sB[BS * (BLOCK_SIZE + 1)];

    int tx = threadIdx.x; // 0 .. BS-1
    int ty = threadIdx.y; // 0 .. BS-1

    int row_base = blockIdx.y * BS;
    int col_base = blockIdx.x * BS;

    // --- load A tile (coalesced) ---
    // Thread (tx, ty) loads A[row_base + ty][t] for t = tx, tx+BS, ...
    for (int t = tx; t < k; t += BS) {
        int r = row_base + ty;
        if (r < m) {
            sA[ty * (BLOCK_SIZE + 1) + t] = A[r * k + t];
        } else {
            sA[ty * (BLOCK_SIZE + 1) + t] = 0.0;
        }
    }

    // --- load B tile (coalesced) ---
    for (int t = tx; t < k; t += BS) {
        int c = col_base + ty;
        if (c < n) {
            sB[ty * (BLOCK_SIZE + 1) + t] = B[c * k + t];
        } else {
            sB[ty * (BLOCK_SIZE + 1) + t] = 0.0;
        }
    }

    __syncthreads();

    // --- compute one output element ---
    int row = row_base + ty;
    int col = col_base + tx;
    if (row < m && col < n) {
        double sum = 0.0;
#pragma unroll
        for (int t = 0; t < k; ++t) {
            sum += sA[ty * (BLOCK_SIZE + 1) + t] * sB[tx * (BLOCK_SIZE + 1) + t];
        }
        C[row * ldC + col] -= sum;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Unblocked Cholesky of a small nb × nb diagonal block (in-place, row-major)
// diag is nb × nb with leading dimension ld (row stride).
static bool diag_cholesky(double* diag, size_t nb, size_t ld) {
    for (size_t j = 0; j < nb; ++j) {
        double sum = 0.0;
        for (size_t t = 0; t < j; ++t) {
            double v = diag[j * ld + t];
            sum += v * v;
        }
        double val = diag[j * ld + j] - sum;
        if (val <= 0.0) {
            fprintf(stderr, "Error: not positive definite at diag %zu\n", j);
            return false;
        }
        diag[j * ld + j] = sqrt(val);

        for (size_t i = j + 1; i < nb; ++i) {
            sum = 0.0;
            for (size_t t = 0; t < j; ++t) {
                sum += diag[i * ld + t] * diag[j * ld + t];
            }
            diag[i * ld + j] = (diag[i * ld + j] - sum) / diag[j * ld + j];
        }
    }
    return true;
}

// TRSM:  L_below * L_diag^T = A_below
// panel = (nb + m) × nb contiguous row-major buffer.
// Rows [0, nb) = diagonal block (L_diag).
// Rows [nb, nb+m) = below-diagonal part (overwritten with L_below).
static void trsm_panel(double* panel, size_t nb, size_t m) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        double* row_i = panel + (nb + i) * nb;
        for (size_t j = 0; j < nb; ++j) {
            double sum = 0.0;
            for (size_t t = 0; t < j; ++t) {
                sum += row_i[t] * panel[j * nb + t];
            }
            row_i[j] = (row_i[j] - sum) / panel[j * nb + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel Cholesky decomposition (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
static bool parallelCholesky(std::vector<double>& h_A,  // host buffer (n × local_n)
                              double* d_A,               // device buffer (n × local_n)
                              size_t n,
                              size_t local_n,
                              size_t col_start,
                              int rank, int size,
                              MPI_Comm comm,
                              cudaStream_t stream) {
    // Pre‑compute column ownership
    std::vector<int> col_owner(size + 1);
    for (int i = 0; i <= size; ++i)
        col_owner[i] = static_cast<int>((n * i) / size);

    // Device panel buffer – large enough for the largest panel (n × BLOCK_SIZE)
    const size_t max_block = static_cast<size_t>(BLOCK_SIZE);
    double* d_panel = nullptr;
    if (cudaMalloc(&d_panel, n * max_block * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[%d] cudaMalloc(d_panel) failed\n", rank);
        return false;
    }

    // Host panel buffer for MPI communication
    std::vector<double> h_panel(n * max_block);

    for (size_t k = 0; k < n; ) {
        // Find which rank owns column k
        int p_rank = 0;
        while (col_owner[p_rank + 1] <= static_cast<int>(k)) ++p_rank;

        // Actual block size: limited by the constant BLOCK_SIZE, the
        // remaining matrix width, and the number of columns the panel
        // rank still holds from position k onwards.
        size_t avail = static_cast<size_t>(col_owner[p_rank + 1]) - k;
        size_t act_nb = std::min({max_block, n - k, avail});

        size_t m = (k + act_nb < n) ? (n - k - act_nb) : 0;
        size_t panel_rows = n - k;
        size_t p_col_off = k - static_cast<size_t>(col_owner[p_rank]);

        // ---------------------------------------------------------------
        // Phase 1 : Panel factorization (done by owning rank on CPU)
        // ---------------------------------------------------------------
        if (rank == p_rank) {
            // Copy the panel from GPU to host (packed: contiguous in host buffer)
            cudaMemcpy2DAsync(
                h_panel.data(), act_nb * sizeof(double),
                d_A + k * local_n + p_col_off, local_n * sizeof(double),
                act_nb * sizeof(double), panel_rows,
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Factor the diagonal block
            if (!diag_cholesky(h_panel.data(), act_nb, act_nb)) {
                cudaFree(d_panel);
                return false;
            }

            // TRSM for the below‑diagonal part
            if (m > 0)
                trsm_panel(h_panel.data(), act_nb, m);
        }

        // ---------------------------------------------------------------
        // Phase 2 : Broadcast the panel to all ranks
        // ---------------------------------------------------------------
        MPI_Bcast(h_panel.data(), static_cast<int>(panel_rows * act_nb),
                  MPI_DOUBLE, p_rank, comm);

        // ---------------------------------------------------------------
        // Phase 3 : Copy panel to device
        // ---------------------------------------------------------------
        cudaMemcpyAsync(d_panel, h_panel.data(),
                        panel_rows * act_nb * sizeof(double),
                        cudaMemcpyHostToDevice, stream);

        // If this rank owns part of the panel columns, update d_A
        size_t local_p_start = std::max(col_start, k);
        size_t local_p_end   = std::min(col_start + local_n, k + act_nb);
        if (local_p_start < local_p_end) {
            size_t l_off     = local_p_start - col_start;
            size_t p_off     = local_p_start - k;
            cudaMemcpy2DAsync(
                d_A + k * local_n + l_off, local_n * sizeof(double),
                h_panel.data() + p_off * act_nb, act_nb * sizeof(double),
                act_nb * sizeof(double), panel_rows,
                cudaMemcpyHostToDevice, stream);
        }

        // ---------------------------------------------------------------
        // Phase 4 : Trailing sub‑matrix update on GPU
        // ---------------------------------------------------------------
        if (m > 0) {
            size_t trail_start = std::max(col_start, k + act_nb);
            size_t trail_end   = col_start + local_n;
            if (trail_start < trail_end) {
                size_t n_local = trail_end - trail_start;
                size_t l_trail = trail_start - col_start;
                size_t p_trail = trail_start - k;

                double* d_C = d_A + (k + act_nb) * local_n + l_trail;
                double* d_A_mat = d_panel + act_nb * act_nb;          // row act_nb
                double* d_B_mat = d_panel + p_trail * act_nb;         // row p_trail

                dim3 block(16, 16);
                dim3 grid(static_cast<unsigned>((n_local + 15) / 16),
                          static_cast<unsigned>((m + 15) / 16));

                gemm_sub_kernel<<<grid, block, 0, stream>>>(
                    d_C, d_A_mat, d_B_mat,
                    static_cast<int>(m), static_cast<int>(n_local),
                    static_cast<int>(act_nb), static_cast<int>(local_n));
            }
        }

        cudaStreamSynchronize(stream);

        k += act_nb;   // advance – act_nb may vary per iteration
    }

    cudaFree(d_panel);
    return true;
}

// ---------------------------------------------------------------------------
// Matrix generation – each rank generates only its local columns
// ---------------------------------------------------------------------------
static void generateLocalMatrix(std::vector<double>& A_local,
                                 size_t n, size_t local_n, size_t col_start) {
    // Method: A = B * B^T + n*I, where B is random.
    // Each rank independently builds the full B (same seed) so all columns
    // are consistent, but only stores the columns it owns.

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (static_cast<double>(rand_r(&seed)) / RAND_MAX) - 0.5;
    }

#pragma omp parallel for
    for (size_t jj = 0; jj < local_n; ++jj) {
        size_t global_j = col_start + jj;
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t t = 0; t < n; ++t) {
                sum += B[i * n + t] * B[global_j * n + t];
            }
            A_local[i * local_n + jj] = sum;
            if (i == global_j)
                A_local[i * local_n + jj] += static_cast<double>(n);
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (rank 0 only, after gathering L)
// ---------------------------------------------------------------------------
static bool validateCholesky(const std::vector<double>& L,
                              const std::vector<double>& A_orig,
                              size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
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

// ---------------------------------------------------------------------------
// Gather the distributed L back to rank 0 as a full n×n row-major matrix
// ---------------------------------------------------------------------------
static std::vector<double> gatherMatrix(const std::vector<double>& h_A,
                                         size_t n, size_t local_n,
                                         const std::vector<int>& col_owner,
                                         int rank, int size,
                                         MPI_Comm comm) {
    std::vector<int> sendcnt(size), displ(size);
    for (int i = 0; i < size; ++i) {
        size_t ln = static_cast<size_t>(col_owner[i + 1] - col_owner[i]);
        sendcnt[i] = static_cast<int>(n * ln);
        displ[i]   = (i == 0) ? 0 : displ[i - 1] + sendcnt[i - 1];
    }

    int total = sendcnt[size - 1] + ((size > 1) ? displ[size - 1] : 0);
    std::vector<double> gathered(total);
    MPI_Gatherv(h_A.data(), static_cast<int>(n * local_n), MPI_DOUBLE,
                gathered.data(), sendcnt.data(), displ.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        std::vector<double> full(n * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                int r = 0;
                while (col_owner[r + 1] <= static_cast<int>(j)) ++r;
                size_t rn = static_cast<size_t>(col_owner[r + 1] - col_owner[r]);
                size_t off = static_cast<size_t>(displ[r]);
                size_t lj  = j - static_cast<size_t>(col_owner[r]);
                full[i * n + j] = gathered[off + i * rn + lj];
            }
        }
        return full;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* progName) {
    fprintf(stderr, "Usage: %s [options]\n", progName);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <num>     Matrix size (default: 512)\n");
    fprintf(stderr, "  -v           Enable validation\n");
    fprintf(stderr, "  -r           Print results for external validation\n");
    fprintf(stderr, "  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------- argument parsing on rank 0, then broadcast ----------
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // ---------- column distribution ----------
    std::vector<int> col_owner(size + 1);
    for (int i = 0; i <= size; ++i)
        col_owner[i] = static_cast<int>((n * i) / size);

    size_t col_start = static_cast<size_t>(col_owner[rank]);
    size_t local_n   = static_cast<size_t>(col_owner[rank + 1] - col_owner[rank]);

    // ---------- GPU setup ----------
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        fprintf(stderr, "[%d] No CUDA-capable device found.\n", rank);
        MPI_Finalize();
        return 1;
    }
    // Simple mapping: each MPI rank uses GPU (rank % devCount)
    cudaSetDevice(rank % devCount);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---------- host & device memory ----------
    std::vector<double> h_A(n * local_n);
    double* d_A = nullptr;
    if (cudaMalloc(&d_A, n * local_n * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "[%d] cudaMalloc failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    // ---------- generate matrix ----------
    double tGen = -MPI_Wtime();
    generateLocalMatrix(h_A, n, local_n, col_start);
    tGen += MPI_Wtime();

    // Copy to device
    cudaMemcpyAsync(d_A, h_A.data(), n * local_n * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Save original for validation (rank 0)
    std::vector<double> A_orig;
    if (validate && rank == 0) {
        std::vector<double> full_A = gatherMatrix(h_A, n, local_n, col_owner,
                                                   rank, size, MPI_COMM_WORLD);
        A_orig = full_A;
    } else if (validate) {
        // Non-rank 0 still needs to participate in gatherMatrix
        gatherMatrix(h_A, n, local_n, col_owner, rank, size, MPI_COMM_WORLD);
    }

    // ---------- Cholesky factorization ----------
    double tStart = MPI_Wtime();

    bool success = parallelCholesky(h_A, d_A, n, local_n, col_start,
                                     rank, size, MPI_COMM_WORLD, stream);
    double tEnd = MPI_Wtime();
    double elapsed = tEnd - tStart;

    if (!success) {
        fprintf(stderr, "[%d] Cholesky decomposition failed\n", rank);
        cudaFree(d_A);
        cudaStreamDestroy(stream);
        MPI_Finalize();
        return 1;
    }

    // Copy result back to host
    cudaMemcpyAsync(h_A.data(), d_A, n * local_n * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ---------- output (rank 0) ----------
    if (rank == 0) {
        // Gather full L for output / validation
        std::vector<double> full_L = gatherMatrix(h_A, n, local_n, col_owner,
                                                   rank, size, MPI_COMM_WORLD);

        // The Cholesky factor only defines the lower triangle (including diagonal).
        // Zero out the upper triangular part (which still carries stale data) so
        // that L * L^T reconstruction is correct.
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                full_L[i * n + j] = 0.0;

        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("GPU devices: %d\n", devCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generation time: %.3f s\n", tGen);
        printf("Computing Cholesky decomposition (hybrid MPI+OpenMP+CUDA)...\n");
        printf("Computation time: %.0f ms\n", elapsed * 1000.0);

        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(full_L, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(full_L, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                cudaFree(d_A);
                cudaStreamDestroy(stream);
                MPI_Finalize();
                return 1;
            }
        }
    } else {
        // Non‑root ranks still need to participate
        if (printResults || validate) {
            gatherMatrix(h_A, n, local_n, col_owner, rank, size, MPI_COMM_WORLD);
        }
    }

    cudaFree(d_A);
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}
