#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky with MPI row distribution.
// Rows are distributed blockwise among ranks. Each panel step
// broadcasts the factored column panel, then all ranks perform
// a local trailing-matrix update (rank-NB GEMM).

static constexpr int NB = 128;

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int nn = (int)n;

    // Block row distribution: process p owns rows [rstart[p], rstart[p]+rcount[p])
    std::vector<int> rstart(nprocs), rcount(nprocs);
    for (int p = 0; p < nprocs; p++) {
        rstart[p] = (int)((size_t)p * n / nprocs);
        rcount[p] = (int)((size_t)(p + 1) * n / nprocs) - rstart[p];
    }
    int my_start = rstart[rank];
    int my_rows  = rcount[rank];

    // Full matrix lives on rank 0 for generation, scatter, and gather
    std::vector<double> A;
    std::vector<double> A_orig;
    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }

    // Scatter / gather counts
    std::vector<int> scounts(nprocs), sdispls(nprocs);
    for (int p = 0; p < nprocs; p++) {
        scounts[p] = rcount[p] * nn;
        sdispls[p] = rstart[p] * nn;
    }

    std::vector<double> local_A((size_t)my_rows * nn);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_DOUBLE,
                 local_A.data(), my_rows * nn, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Row-owner lookup
    std::vector<int> row_owner(nn);
    for (int p = 0; p < nprocs; p++)
        for (int r = rstart[p]; r < rstart[p] + rcount[p]; r++)
            row_owner[r] = p;

    // Pre-allocated work buffers
    std::vector<double> bcast_buf(NB);
    std::vector<double> full_panel((size_t)nn * NB);
    std::vector<double> my_panel_buf((size_t)my_rows * NB);
    std::vector<int> ag_counts(nprocs), ag_displs(nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    int fail_flag = 0;

    for (int k = 0; k < nn && !fail_flag; k += NB) {
        int kb = std::min(NB, nn - k);

        // ---- Panel factorization (columns k..k+kb-1) ----
        for (int j = k; j < k + kb && !fail_flag; j++) {
            int owner = row_owner[j];

            if (rank == owner) {
                int lj = j - my_start;
                double sum = 0.0;
                for (int c = k; c < j; c++) {
                    double v = local_A[(size_t)lj * nn + c];
                    sum += v * v;
                }
                double val = local_A[(size_t)lj * nn + j] - sum;
                if (val <= 0.0) {
                    fail_flag = 1;
                } else {
                    local_A[(size_t)lj * nn + j] = sqrt(val);
                    for (int c = k; c <= j; c++)
                        bcast_buf[c - k] = local_A[(size_t)lj * nn + c];
                }
            }

            MPI_Bcast(&fail_flag, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (fail_flag) break;

            MPI_Bcast(bcast_buf.data(), j - k + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            double diag = bcast_buf[j - k];

            // Update local rows below row j
            for (int li = 0; li < my_rows; li++) {
                int gi = my_start + li;
                if (gi <= j) continue;
                double s = 0.0;
                const double* rp = &local_A[(size_t)li * nn + k];
                for (int c = 0; c < j - k; c++)
                    s += rp[c] * bcast_buf[c];
                local_A[(size_t)li * nn + j] = (local_A[(size_t)li * nn + j] - s) / diag;
            }
        }
        if (fail_flag) break;

        // ---- Trailing-matrix update ----
        int ts = k + kb;
        if (ts >= nn) continue;
        // Allgather the panel L[ts:n, k:k+kb]
        int total_rows = 0;
        for (int p = 0; p < nprocs; p++) {
            int cs = std::max(rstart[p], ts);
            int ce = std::min(rstart[p] + rcount[p], nn);
            int nr = std::max(0, ce - cs);
            ag_counts[p] = nr * kb;
            ag_displs[p] = total_rows * kb;
            total_rows += nr;
        }

        int my_cs = std::max(my_start, ts);
        int my_ce = std::min(my_start + my_rows, nn);
        int my_pr = std::max(0, my_ce - my_cs);

        for (int r = 0; r < my_pr; r++) {
            int lr = (my_cs - my_start) + r;
            std::memcpy(&my_panel_buf[(size_t)r * kb],
                        &local_A[(size_t)lr * nn + k],
                        kb * sizeof(double));
        }

        MPI_Allgatherv(my_panel_buf.data(), my_pr * kb, MPI_DOUBLE,
                       full_panel.data(), ag_counts.data(), ag_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Local rank-kb update: A[gi][j] -= L[gi][k:k+kb] · L[j][k:k+kb]
        for (int li = 0; li < my_rows; li++) {
            int gi = my_start + li;
            if (gi < ts) continue;
            const double* my_prow = &local_A[(size_t)li * nn + k];
            double* a_row = &local_A[(size_t)li * nn];
            for (int j = ts; j <= gi; j++) {
                const double* fp_row = &full_panel[(size_t)(j - ts) * kb];
                double dot = 0.0;
                for (int c = 0; c < kb; c++)
                    dot += my_prow[c] * fp_row[c];
                a_row[j] -= dot;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    if (fail_flag) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Zero upper triangle
    for (int li = 0; li < my_rows; li++) {
        int gi = my_start + li;
        for (int j = gi + 1; j < nn; j++)
            local_A[(size_t)li * nn + j] = 0.0;
    }

    // Gather full result to rank 0
    if (rank == 0) A.resize(n * n);
    MPI_Gatherv(local_A.data(), my_rows * nn, MPI_DOUBLE,
                rank == 0 ? A.data() : nullptr,
                scounts.data(), sdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
