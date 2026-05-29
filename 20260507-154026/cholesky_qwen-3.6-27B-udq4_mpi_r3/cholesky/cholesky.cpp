#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static int num_procs, my_rank;

// Row-block distribution helper
struct RowDist {
    int N, nprocs, rank;
    int local_start, local_count;

    RowDist(int n, int r, int np) : N(n), nprocs(np), rank(r) {
        int base = N / np;
        int rem = N % np;
        local_start = r * base + std::min(r, rem);
        local_count = base + (r < rem ? 1 : 0);
    }

    int local_idx(int g) const {
        if (g < local_start || g >= local_start + local_count) return -1;
        return g - local_start;
    }

    int owner(int g) const {
        int base = N / nprocs;
        int rem = N % nprocs;
        if (base == 0) return std::min(g, nprocs - 1);
        if (g < rem * (base + 1)) {
            return g / (base + 1);
        } else {
            return rem + (g - rem * (base + 1)) / base;
        }
    }
};

// MPI-parallel blocked Cholesky with row-block distribution
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int N = static_cast<int>(n);
    const int B = std::min(64, N);

    RowDist dist(N, my_rank, num_procs);
    const int lr = dist.local_count;

    // Local matrix: lr rows × N columns (row-major)
    std::vector<double> local_A(lr * N);
    if (lr > 0) {
        std::memcpy(local_A.data(), A.data() + dist.local_start * N,
                    lr * N * sizeof(double));
    }

    // Precompute MPI receive counts and displacements
    std::vector<int> recvcounts(num_procs), displs(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        RowDist rd(N, r, num_procs);
        recvcounts[r] = rd.local_count;
        displs[r] = rd.local_start;
    }

    // Reusable buffers
    std::vector<double> W_full;
    std::vector<double> send_buf;
    std::vector<double> W_T;
    std::vector<int> rc, dp;
    std::vector<double> row_k_panel(B);
    std::vector<double> local_sums(std::max(lr, 1));

    int prev_pw = 0;
    int j = 0;

    while (j < N) {
        int j_end = std::min(j + B, N);
        int pw = j_end - j;

        // === SYRK update: A(j:end, j:end) -= W * W^T ===
        if (prev_pw > 0) {
            W_full.resize(N * prev_pw, 0.0);

            // Prepare send buffer for Allgatherv
            send_buf.resize(lr * prev_pw);
            if (lr > 0) {
                for (int i = 0; i < lr; ++i) {
                    std::memcpy(send_buf.data() + i * prev_pw,
                                local_A.data() + i * N + (j - prev_pw),
                                prev_pw * sizeof(double));
                }
            }

            // Gather full previous panel from all ranks
            rc.assign(num_procs, 0);
            dp.assign(num_procs, 0);
            for (int r = 0; r < num_procs; ++r) {
                rc[r] = recvcounts[r] * prev_pw;
                dp[r] = displs[r] * prev_pw;
            }

            MPI_Allgatherv(send_buf.data(), lr * prev_pw, MPI_DOUBLE,
                           W_full.data(), rc.data(), dp.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);

            if (lr > 0) {
                int tcols = N - j;
                // Transpose W_full columns [j:N) for cache-friendly update
                W_T.resize(prev_pw * tcols);
                for (int p = 0; p < prev_pw; ++p)
                    for (int l = j; l < N; ++l)
                        W_T[p * tcols + (l - j)] = W_full[l * prev_pw + p];

                // Local SYRK update: A_local -= W_local * W^T
                for (int i = 0; i < lr; ++i) {
                    int gi = dist.local_start + i;
                    if (gi < j) continue;  // skip rows above trailing submatrix
                    double* a_row = local_A.data() + i * N + j;
                    for (int p = 0; p < prev_pw; ++p) {
                        double wip = W_full[gi * prev_pw + p];
                        const double* w_row = W_T.data() + p * tcols;
                        for (int l = 0; l < tcols; ++l)
                            a_row[l] -= wip * w_row[l];
                    }
                }
            }
        }

        // === Factor panel columns ===
        for (int k = j; k < j_end; ++k) {
            int k_owner = dist.owner(k);
            int k_local = dist.local_idx(k);
            int pcols = k - j;

            // Broadcast L(k, j:k-1) from the rank owning row k
            if (pcols > 0) {
                if (k_local >= 0) {
                    std::memcpy(row_k_panel.data(),
                                local_A.data() + k_local * N + j,
                                pcols * sizeof(double));
                }
                MPI_Bcast(row_k_panel.data(), pcols, MPI_DOUBLE,
                          k_owner, MPI_COMM_WORLD);
            }

            // Compute dot products for local rows (or zero if no panel columns yet)
            if (lr > 0) {
                if (pcols > 0) {
                    for (int i = 0; i < lr; ++i) {
                        double s = 0.0;
                        const double* li = local_A.data() + i * N + j;
                        for (int p = 0; p < pcols; ++p)
                            s += li[p] * row_k_panel[p];
                        local_sums[i] = s;
                    }
                } else {
                    std::fill(local_sums.begin(), local_sums.begin() + lr, 0.0);
                }
            }

            // Compute diagonal element
            double diag_val = 0.0;
            if (k_local >= 0) {
                diag_val = local_A[k_local * N + k] - local_sums[k_local];
                if (diag_val <= 0.0) {
                    if (my_rank == 0)
                        printf("Error: Matrix is not positive definite at diagonal element %d\n", k);
                    return false;
                }
                diag_val = sqrt(diag_val);
                local_A[k_local * N + k] = diag_val;
            }

            // Broadcast diagonal value
            double diag_bc;
            if (my_rank == k_owner) diag_bc = diag_val;
            MPI_Bcast(&diag_bc, 1, MPI_DOUBLE, k_owner, MPI_COMM_WORLD);

            // Compute off-diagonal elements and zero upper triangular
            if (lr > 0) {
                for (int i = 0; i < lr; ++i) {
                    int gi = dist.local_start + i;
                    if (gi > k)
                        local_A[i * N + k] = (local_A[i * N + k] - local_sums[i]) / diag_bc;
                    else if (gi < k)
                        local_A[i * N + k] = 0.0;
                }
            }
        }

        j = j_end;
        prev_pw = pw;
    }

    // Gather result back to rank 0
    std::vector<int> rc_A(num_procs), dp_A(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        RowDist rd(N, r, num_procs);
        rc_A[r] = rd.local_count * N;
        dp_A[r] = rd.local_start * N;
    }

    MPI_Gatherv(local_A.data(), lr * N, MPI_DOUBLE,
               A.data(), rc_A.data(), dp_A.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return true;
}

// Generate a symmetric positive definite matrix (on rank 0, broadcast to all)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Rank 0 generates the full matrix, then broadcasts to all ranks
    if (my_rank == 0) {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        
        // Generate random matrix B
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        
        // Compute A = B * B^T
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                A[i * n + j] = sum;
            }
        }
        
        // Add diagonal dominance to ensure positive definiteness
        for (size_t i = 0; i < n; ++i) {
            A[i * n + i] += n;
        }
    }
    
    // Broadcast full matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    // Only runs on rank 0
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, only rank 0 prints)
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
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("MPI processes: %d\n", num_procs);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix (full on every rank)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (my_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (my_rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (my_rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
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
