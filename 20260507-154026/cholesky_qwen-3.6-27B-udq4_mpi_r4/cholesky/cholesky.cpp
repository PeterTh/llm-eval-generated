#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Row distribution: 1D block distribution of matrix rows across MPI processes
struct RowDist {
    size_t local_start;
    size_t local_nrows;

    static RowDist compute(size_t n, int rank, int size) {
        RowDist dist;
        if (size <= 0) {
            dist.local_start = 0;
            dist.local_nrows = n;
            return dist;
        }
        size_t base = n / size;
        size_t rem = n % size;
        if (rank < (int)rem) {
            dist.local_start = (size_t)rank * (base + 1);
            dist.local_nrows = base + 1;
        } else {
            dist.local_start = (size_t)rank * base + rem;
            dist.local_nrows = base;
        }
        return dist;
    }

    // Find which process owns a given global row
    static int findProc(size_t global_row, size_t n, int size) {
        if (size <= 0) return 0;
        size_t base = n / size;
        size_t rem = n % size;
        for (int p = 0; p < size; ++p) {
            size_t start, nrows;
            if (p < (int)rem) {
                start = (size_t)p * (base + 1);
                nrows = base + 1;
            } else {
                start = (size_t)p * base + rem;
                nrows = base;
            }
            if (global_row >= start && global_row < start + nrows) return p;
        }
        return 0;
    }
};

// Generate a symmetric positive definite matrix (runs on rank 0)
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

// MPI-parallelized right-looking blocked Cholesky decomposition
// local_A stores dist.local_nrows x n in row-major order (rows owned by this rank)
bool choleskyDecomposition(std::vector<double>& local_A, const size_t n,
                           const RowDist& dist, const int num_procs, const int my_rank) {
    const size_t base_block = 64;

    for (size_t k = 0; k < n; k += base_block) {
        // Dynamic block size: diagonal block must fit within one process
        int diag_proc = RowDist::findProc(k, n, num_procs);
        RowDist diag_dist = RowDist::compute(n, diag_proc, num_procs);
        size_t diag_local_k = k - diag_dist.local_start;
        size_t max_b = diag_dist.local_nrows - diag_local_k;
        size_t b = std::min({base_block, n - k, max_b});

        // --- Step 1: Compute diagonal block L(k:k+b, k:k+b) on diag_proc ---
        if (my_rank == diag_proc) {
            size_t local_k = k - dist.local_start;
            for (size_t i = 0; i < b; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    if (i == j) {
                        for (size_t l = 0; l < j; ++l)
                            sum += local_A[(local_k + i) * n + k + l] *
                                   local_A[(local_k + i) * n + k + l];
                        double val = local_A[(local_k + i) * n + k + i] - sum;
                        if (val <= 0.0) {
                            if (my_rank == 0)
                                printf("Error: not positive definite at diagonal %zu\n", k + i);
                            return false;
                        }
                        local_A[(local_k + i) * n + k + i] = sqrt(val);
                    } else {
                        for (size_t l = 0; l < j; ++l)
                            sum += local_A[(local_k + i) * n + k + l] *
                                   local_A[(local_k + j) * n + k + l];
                        local_A[(local_k + i) * n + k + j] =
                            (local_A[(local_k + i) * n + k + j] - sum) /
                            local_A[(local_k + j) * n + k + j];
                    }
                }
            }
        }

        // --- Step 2: Broadcast diagonal block to all processes ---
        std::vector<double> diag_block(b * b);
        if (my_rank == diag_proc) {
            size_t local_k = k - dist.local_start;
            for (size_t i = 0; i < b; ++i)
                for (size_t j = 0; j < b; ++j)
                    diag_block[i * b + j] = local_A[(local_k + i) * n + k + j];
        }
        MPI_Bcast(diag_block.data(), (int)(b * b), MPI_DOUBLE, diag_proc, MPI_COMM_WORLD);

        // --- Steps 3–5: Off-diagonal triangular solve, gather, SYRK ---
        if (k + b < n) {
            size_t remaining = n - k - b;

            // Determine local rows that fall in the off-diagonal region [k+b, n)
            size_t off_start = std::max(k + b, dist.local_start);
            size_t off_end   = dist.local_start + dist.local_nrows;
            size_t m_local   = (off_start < off_end) ? (off_end - off_start) : 0;

            std::vector<double> local_off(m_local * b);
            if (m_local > 0) {
                // Extract A(k+b:n, k:k+b) for local rows
                for (size_t i = 0; i < m_local; ++i) {
                    size_t lr = (off_start + i) - dist.local_start;
                    for (size_t j = 0; j < b; ++j)
                        local_off[i * b + j] = local_A[lr * n + k + j];
                }

                // Step 3: Forward substitution  local_off = local_off * diag_block^{-T}
                for (size_t i = 0; i < m_local; ++i) {
                    for (size_t j = 0; j < b; ++j) {
                        double sum = 0.0;
                        for (size_t l = 0; l < j; ++l)
                            sum += diag_block[j * b + l] * local_off[i * b + l];
                        local_off[i * b + j] =
                            (local_off[i * b + j] - sum) / diag_block[j * b + j];
                    }
                }

                // Write solved off-diagonal block back
                for (size_t i = 0; i < m_local; ++i) {
                    size_t lr = (off_start + i) - dist.local_start;
                    for (size_t j = 0; j < b; ++j)
                        local_A[lr * n + k + j] = local_off[i * b + j];
                }
            }

            // Step 4: Allgatherv to form full off-diagonal block on every rank
            std::vector<int> recvcounts(num_procs), displs(num_procs);
            int disp = 0;
            for (int p = 0; p < num_procs; ++p) {
                RowDist pd = RowDist::compute(n, p, num_procs);
                size_t ps = std::max(k + b, pd.local_start);
                size_t pe = pd.local_start + pd.local_nrows;
                size_t pm = (ps < pe) ? (pe - ps) : 0;
                recvcounts[p] = (int)(pm * b);
                displs[p] = disp;
                disp += (int)(pm * b);
            }

            std::vector<double> full_off(remaining * b);
            MPI_Allgatherv(local_off.data(), (int)(m_local * b), MPI_DOUBLE,
                           full_off.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);

            // Step 5: SYRK update of trailing submatrix (lower triangle only)
            //   A(k+b:n, k+b:n) -= L(k+b:n, k:k+b) * L(k+b:n, k:k+b)^T
            if (m_local > 0) {
                for (size_t i = 0; i < m_local; ++i) {
                    size_t ig   = off_start + i;          // global row
                    size_t lr   = ig - dist.local_start;   // local row
                    size_t io   = ig - k - b;              // row index in full_off
                    size_t jmax = std::min(ig - k - b, remaining - 1);

                    for (size_t j = 0; j <= jmax; ++j) {
                        double sum = 0.0;
                        for (size_t l = 0; l < b; ++l)
                            sum += full_off[io * b + l] * full_off[j * b + l];
                        local_A[lr * n + k + b + j] -= sum;
                    }
                }
            }
        }
    }

    // Zero out upper triangle on local rows
    for (size_t i = 0; i < dist.local_nrows; ++i) {
        size_t gr = dist.local_start + i;
        for (size_t j = gr + 1; j < n; ++j)
            local_A[i * n + j] = 0.0;
    }

    return true;
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

    int num_procs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
            if (my_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (my_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (my_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    RowDist dist = RowDist::compute(n, my_rank, num_procs);
    std::vector<double> local_A(dist.local_nrows * n);

    // Generate positive definite matrix on rank 0
    std::vector<double> full_A;
    if (my_rank == 0) {
        printf("Generating positive definite matrix...\n");
        full_A.resize(n * n);
        generatePositiveDefiniteMatrix(full_A, n);
    }

    // Scatter matrix rows to all processes
    std::vector<int> sendcounts(num_procs), displs(num_procs);
    int disp = 0;
    for (int p = 0; p < num_procs; ++p) {
        RowDist pd = RowDist::compute(n, p, num_procs);
        sendcounts[p] = (int)(pd.local_nrows * n);
        displs[p] = disp;
        disp += sendcounts[p];
    }

    MPI_Scatterv(full_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), (int)(dist.local_nrows * n), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    std::vector<double> A_orig;
    if (validate && my_rank == 0) A_orig = full_A;

    // Timed parallel Cholesky decomposition
    if (my_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(local_A, n, dist, num_procs, my_rank);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration = duration.count();
    long max_duration = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (my_rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        printf("Computation time: %ld ms\n", max_duration);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full result to rank 0
    std::vector<double> full_L;
    if (my_rank == 0) full_L.resize(n * n);

    MPI_Gatherv(local_A.data(), (int)(dist.local_nrows * n), MPI_DOUBLE,
                full_L.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && my_rank == 0) {
        print_results(full_L, "CholeskyL");
    }

    if (validate && my_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(full_L, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return success ? 0 : 1;
}
