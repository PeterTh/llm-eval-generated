#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Validate by computing L * L^T and comparing with original matrix
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
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

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// Find which rank owns a given global row
static inline int row_to_rank(size_t global_row, int num_ranks,
                              const std::vector<int>& displacements,
                              const std::vector<int>& block_sizes) {
    for (int r = 0; r < num_ranks; ++r) {
        if (global_row >= (size_t)displacements[r] &&
            global_row < (size_t)(displacements[r] + block_sizes[r])) {
            return r;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // ── Parse arguments on rank 0, broadcast to all ──────────────────────
    size_t n = 512;
    int validate_flag = 0, print_results_flag = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_results_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_results_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const bool validate = validate_flag != 0;
    const bool printResults = print_results_flag != 0;

    // ── 1-D block row distribution ───────────────────────────────────────
    const size_t base_block = n / (size_t)num_ranks;
    const size_t remainder  = n % (size_t)num_ranks;
    const size_t my_block   = base_block + (rank < (int)remainder ? 1 : 0);
    const size_t my_start   = (size_t)rank * base_block + std::min((size_t)rank, remainder);

    std::vector<int> block_sizes(num_ranks), displacements(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        block_sizes[r]   = (int)(base_block + (r < (int)remainder ? 1 : 0));
        displacements[r] = (int)((size_t)r * base_block + std::min((size_t)r, remainder));
    }

    // Pre-compute row-owner lookup table
    std::vector<int> row_owners(n);
    for (int r = 0; r < num_ranks; ++r) {
        for (int row = displacements[r]; row < displacements[r] + block_sizes[r]; ++row)
            row_owners[row] = r;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ── Generate positive-definite matrix on rank 0, scatter to all ──────
    // A = B * B^T + n*I  (deterministic with seed 42)
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k)
                    s += B[i * n + k] * B[j * n + k];
                A_full[i * n + j] = s;
            }
        }
        for (size_t i = 0; i < n; ++i)
            A_full[i * n + i] += (double)n;
    }

    if (rank == 0) printf("Generating positive definite matrix...\n");

    std::vector<double> A_local(my_block * n);
    std::vector<int> send_counts(num_ranks), send_disps(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        send_counts[r] = block_sizes[r] * (int)n;
        send_disps[r]  = displacements[r] * (int)n;
    }
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 send_counts.data(), send_disps.data(), MPI_DOUBLE,
                 my_block > 0 ? A_local.data() : nullptr,
                 (int)(my_block * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> A_orig_local;
    if (validate) A_orig_local = A_local;

    // ── Blocked left-looking Cholesky decomposition (MPI) ────────────────
    // 1-D row distribution; block size 64 for BLAS-3 trailing-submatrix
    // update in each panel.
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    constexpr size_t BLOCK = 64;
    bool success = true;

    // Reusable buffers (avoid per-panel allocation)
    std::vector<double> send_buf;
    std::vector<double> panel_data;
    std::vector<double> inner_row;
    inner_row.reserve(n);

    for (size_t ps = 0; ps < n && success; ps += BLOCK) {
        const size_t pe = std::min(ps + BLOCK, n);   // panel end
        const size_t pw = pe - ps;                   // panel width

        // ─ Phase 1: BLAS-3 update of panel columns using previous panels ─
        if (ps > 0) {
            // Determine how many rows of each rank fall inside [ps, pe)
            std::vector<int> recv_counts(num_ranks), recv_disps(num_ranks);

            const int my_o_start = std::max((int)ps, (int)my_start);
            const int my_o_end   = std::min((int)(my_start + my_block), (int)pe);
            const int my_o_rows  = std::max(0, my_o_end - my_o_start);
            const int my_sc      = my_o_rows * (int)ps;

            for (int r = 0; r < num_ranks; ++r) {
                const int rs = displacements[r], re = rs + block_sizes[r];
                const int os = std::max(rs, (int)ps);
                const int oe = std::min(re, (int)pe);
                const int orows = std::max(0, oe - os);
                recv_counts[r] = orows * (int)ps;
                recv_disps[r]  = (os - (int)ps) * (int)ps;
            }

            panel_data.assign(pw * ps, 0.0);

            // Pack send buffer (contiguous columns 0..ps-1 for each row)
            if (my_sc > 0) {
                send_buf.resize(my_sc);
                for (int i = 0; i < my_o_rows; ++i) {
                    const int lr = my_o_start + i - (int)my_start;
                    std::memcpy(send_buf.data() + i * ps,
                                A_local.data() + lr * n,
                                ps * sizeof(double));
                }
            }

            MPI_Allgatherv(my_sc > 0 ? send_buf.data() : nullptr,
                           my_sc, MPI_DOUBLE,
                           panel_data.data(),
                           recv_counts.data(), recv_disps.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);

            // A_local[:, ps:pe) -= A_local[:, 0:ps) * panel_data^T
            for (int lr = 0; lr < (int)my_block; ++lr) {
                const size_t gr = lr + my_start;
                double* ar = A_local.data() + lr * n;
                for (size_t pc = 0; pc < pw; ++pc) {
                    const size_t col = ps + pc;
                    if (gr < col) continue;
                    const double* pr = panel_data.data() + pc * ps;
                    double sum = 0.0;
                    for (size_t k = 0; k < ps; ++k)
                        sum += ar[k] * pr[k];
                    ar[col] -= sum;
                }
            }
        }

        // ─ Phase 2: Factor panel columns sequentially (BLAS-2) ───────────
        for (size_t col = ps; col < pe && success; ++col) {
            const int owner = row_owners[(int)col];
            double diag_val;

            // Update column col using columns within the same panel
            if (col > ps) {
                const size_t iw = col - ps;
                inner_row.resize(iw);

                if (rank == owner) {
                    const int lr = (int)(col - my_start);
                    for (size_t k = ps; k < col; ++k)
                        inner_row[k - ps] = A_local[lr * n + k];
                }

                MPI_Bcast(inner_row.data(), (int)iw, MPI_DOUBLE, owner,
                          MPI_COMM_WORLD);

                for (int lr = 0; lr < (int)my_block; ++lr) {
                    const size_t gr = lr + my_start;
                    if (gr < col) continue;
                    double* ar = A_local.data() + lr * n;
                    double sum = 0.0;
                    for (size_t k = ps; k < col; ++k)
                        sum += ar[k] * inner_row[k - ps];
                    ar[col] -= sum;
                }
            }

            // Factor diagonal element
            if (rank == owner) {
                const int lr = (int)(col - my_start);
                const double val = A_local[lr * n + col];
                if (val <= 0.0) {
                    if (rank == 0)
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                               col);
                    success = false;
                    diag_val = -1.0;
                } else {
                    diag_val = sqrt(val);
                    A_local[lr * n + col] = diag_val;
                }
            }

            MPI_Bcast(&diag_val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            if (!success) break;

            // Off-diagonal: L(i,col) = A(i,col) / L(col,col)
            for (int lr = 0; lr < (int)my_block; ++lr) {
                const size_t gr = lr + my_start;
                if (gr <= col) continue;
                A_local[lr * n + col] /= diag_val;
            }

            // Zero upper triangular part of row col
            if (rank == owner) {
                const int lr = (int)(col - my_start);
                for (size_t j = col + 1; j < n; ++j)
                    A_local[lr * n + j] = 0.0;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        if (global_duration_ms > 0) {
            const double ops    = (double)n * n * n / 3.0;
            const double gflops = ops / ((double)global_duration_ms / 1000.0) / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }
    }

    // ── Gather full L to rank 0 ──────────────────────────────────────────
    std::vector<double> L_full;
    std::vector<int> rcv(num_ranks), rdp(num_ranks);
    L_full.resize(n * n);
    for (int r = 0; r < num_ranks; ++r) {
        rcv[r] = block_sizes[r] * (int)n;
        rdp[r] = displacements[r] * (int)n;
    }

    MPI_Gatherv(my_block * n > 0 ? A_local.data() : nullptr,
                (int)(my_block * n), MPI_DOUBLE,
                rank == 0 ? L_full.data() : nullptr,
                rcv.data(), rdp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && rank == 0)
        print_results(L_full, "CholeskyL");

   // ── Validation ────────────────────────────────────────────────────────
    if (validate) {
        std::vector<double> A_orig_full(n * n);
        MPI_Gatherv(my_block * n > 0 ? A_orig_local.data() : nullptr,
                    (int)(my_block * n), MPI_DOUBLE,
                    rank == 0 ? A_orig_full.data() : nullptr,
                    rcv.data(), rdp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            if (validateCholesky(L_full, A_orig_full, n))
                printf("Validation: PASSED\n");
            else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
