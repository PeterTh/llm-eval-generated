#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse args on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int vi = static_cast<int>(validate), pri = static_cast<int>(printResults);
    MPI_Bcast(&vi, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = static_cast<bool>(vi);
    MPI_Bcast(&pri, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = static_cast<bool>(pri);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- 2-D process grid: p rows x q columns, p*q == num_procs, p <= q ----
    int p = 1, q = 1;
    for (int i = 1; i <= num_procs; ++i) {
        if (num_procs % i == 0 && i <= num_procs / i) {
            p = i;
            q = num_procs / i;
        }
    }
    int my_row = rank / q;
    int my_col = rank % q;

    if (rank == 0) {
        printf("Process grid: %d x %d\n", p, q);
    }

    // Block sizes / offsets (handles non-even division)
    std::vector<int> row_sizes(p), row_offs(p + 1, 0);
    int base_r = static_cast<int>(N) / p, extra_r = static_cast<int>(N) % p;
    for (int i = 0; i < p; ++i) {
        row_sizes[i] = base_r + (i < extra_r ? 1 : 0);
        row_offs[i + 1] = row_offs[i] + row_sizes[i];
    }
    std::vector<int> col_sizes(q), col_offs(q + 1, 0);
    int base_c = static_cast<int>(N) / q, extra_c = static_cast<int>(N) % q;
    for (int i = 0; i < q; ++i) {
        col_sizes[i] = base_c + (i < extra_c ? 1 : 0);
        col_offs[i + 1] = col_offs[i] + col_sizes[i];
    }

    int my_rs = row_sizes[my_row];          // my row-block height
    int my_cs = col_sizes[my_col];          // my col-block width
    int N_int = static_cast<int>(N);

    // Local tiles
    std::vector<double> C_local(static_cast<size_t>(my_rs) * my_cs, 0.0);
    std::vector<double> A_local(static_cast<size_t>(my_rs) * my_cs);
    std::vector<double> B_local(static_cast<size_t>(my_rs) * my_cs);

    // ---- Initialise local tiles of A and B ----
    if (rank == 0) printf("Initializing matrices...\n");
    int r_off = row_offs[my_row], c_off = col_offs[my_col];
    for (int i = 0; i < my_rs; ++i) {
        for (int j = 0; j < my_cs; ++j) {
            size_t idx = static_cast<size_t>(i) * my_cs + j;
            A_local[idx] = getPseudoRndValue(N, r_off + i, c_off + j);
            B_local[idx] = getPseudoRndValue(N, r_off + i, c_off + j);
        }
    }

    // Row / column communicators for the 2-D grid
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    if (rank == 0) printf("Computing matrix multiplication...\n");

    // ---- Synchronise and start wall-clock timing ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // ---- Gather A's full row-strips (Allgatherv along rows) ----
    // After this: A_full[i*N + k] == A[r_off+i, k]   for i in [0,my_rs), k in [0,N)
    std::vector<double> A_full(static_cast<size_t>(my_rs) * N);
    {
        std::vector<int> rcnt(q), disp(q);
        int cum = 0;
        for (int k = 0; k < q; ++k) {
            rcnt[k] = my_rs * col_sizes[k];
            disp[k] = cum;
            cum += rcnt[k];
        }
        std::vector<double> flat(cum);
        MPI_Allgatherv(A_local.data(), my_rs * my_cs, MPI_DOUBLE,
                       flat.data(), rcnt.data(), disp.data(), MPI_DOUBLE, row_comm);
        // Rearrange from block-concatenated layout to row-major
        for (int k = 0; k < q; ++k) {
            int csz = col_sizes[k], coff = col_offs[k];
            for (int i = 0; i < my_rs; ++i)
                for (int j = 0; j < csz; ++j)
                    A_full[static_cast<size_t>(i) * N + coff + j] =
                        flat[static_cast<size_t>(disp[k] + i * csz + j)];
        }
    }

    // ---- Gather B's full column-strips (Allgatherv along columns) ----
    // After this: B_full[k*my_cs + j] == B[k, c_off+j]   for k in [0,N), j in [0,my_cs)
    std::vector<double> B_full(static_cast<size_t>(N) * my_cs);
    {
        std::vector<int> rcnt(p), disp(p);
        int cum = 0;
        for (int k = 0; k < p; ++k) {
            rcnt[k] = row_sizes[k] * my_cs;
            disp[k] = cum;
            cum += rcnt[k];
        }
        std::vector<double> flat(cum);
        MPI_Allgatherv(B_local.data(), my_rs * my_cs, MPI_DOUBLE,
                       flat.data(), rcnt.data(), disp.data(), MPI_DOUBLE, col_comm);
        // Rearrange: B_full[k*my_cs + j] == B[k, c_off+j]
        for (int k = 0; k < p; ++k) {
            int rsz = row_sizes[k], roff = row_offs[k];
            for (int i = 0; i < rsz; ++i)
                for (int j = 0; j < my_cs; ++j)
                    B_full[static_cast<size_t>(roff + i) * my_cs + j] =
                        flat[static_cast<size_t>(disp[k] + i * my_cs + j)];
        }
    }

    // ---- Local blocked multiply (i-k-j loop order, cache-friendly) ----
    for (int i = 0; i < my_rs; ++i) {
        for (int k = 0; k < N_int; ++k) {
            double a_ik = A_full[static_cast<size_t>(i) * N + k];
            for (int j = 0; j < my_cs; ++j) {
                C_local[static_cast<size_t>(i) * my_cs + j] +=
                    a_ik * B_full[static_cast<size_t>(k) * my_cs + j];
            }
        }
    }

    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_time = t1 - t0;

    // ---- Gather C to rank 0 ----
    std::vector<int> recv_counts(num_procs), recv_displs(num_procs);
    int flat_total = 0;
    for (int r = 0; r < num_procs; ++r) {
        int rr = r / q, rc = r % q;
        recv_counts[r] = row_sizes[rr] * col_sizes[rc];
        recv_displs[r] = flat_total;
        flat_total += recv_counts[r];
    }

    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    {
        std::vector<double> C_flat(flat_total);
        MPI_Gatherv(C_local.data(), my_rs * my_cs, MPI_DOUBLE,
                    C_flat.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            int idx = 0;
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j < q; ++j) {
                    int rs = row_sizes[i], cs = col_sizes[j];
                    int ro = row_offs[i], co = col_offs[j];
                    for (int ii = 0; ii < rs; ++ii)
                        for (int jj = 0; jj < cs; ++jj)
                            C[static_cast<size_t>(ro + ii) * N + (co + jj)] = C_flat[idx++];
                }
            }
        }
    }

    // ---- Output (rank 0) ----
    if (rank == 0) {
        long long ms = static_cast<long long>(comp_time * 1000.0);
        printf("Computation time: %lld ms\n", ms);
        double gflops = (2.0 * static_cast<double>(N) * N * N) / (comp_time * 1e9);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
            for (size_t pi = 0; pi < 5 && valid; ++pi) {
                for (size_t pj = 0; pj < 5 && valid; ++pj) {
                    size_t ci = checkPoints[pi] % N;
                    size_t cj = checkPoints[pj] % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, ci, k) * getPseudoRndValue(N, k, cj);
                    double actual = C[ci * N + cj];
                    double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               ci, cj, expected, actual, relError);
                        valid = false;
                    }
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Comm_free(&row_comm);
            MPI_Comm_free(&col_comm);
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
