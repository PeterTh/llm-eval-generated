#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition using right-looking algorithm
// with 1D block-column distribution.
// Each process owns a contiguous block of columns stored in column-major
// order for cache-friendly rank-1 updates.

// Generate a symmetric positive definite matrix (deterministic, identical on all ranks)
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

// Block-column distribution helpers
inline int col_start_of(int rank, int size, int n) {
    return (int)((long long)rank * n / size);
}

inline int col_end_of(int rank, int size, int n) {
    return (int)((long long)(rank + 1) * n / size);
}

// Find which rank owns column j
inline int col_owner_of(int j, int size, int n) {
    int p = (int)((long long)j * size / n);
    if (p >= size) p = size - 1;
    while (p < size - 1 && col_start_of(p + 1, size, n) <= j) ++p;
    while (p > 0 && col_start_of(p, size, n) > j) --p;
    return p;
}

// Right-looking Cholesky with 1D block-column distribution.
// A_local is column-major: A_local[ci * n + i] = element (row i, global col col_start+ci).
// On exit the upper triangle of local columns is zeroed.
bool choleskyMPI(double* A_local, int n, int my_col_start, int ncols, int rank, int size) {
    std::vector<double> panel(n);

    for (int j = 0; j < n; ++j) {
        int owner = col_owner_of(j, size, n);

        if (rank == owner) {
            int lj = j - my_col_start; // local column index of column j

            // Diagonal element (column-major access)
            double diag_val = A_local[(size_t)lj * n + j];
            if (diag_val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", j);
                return false;
            }
            double ljj = sqrt(diag_val);
            A_local[(size_t)lj * n + j] = ljj;

            // Division step: L[i,j] = A[i,j] / L[j,j] for i > j
            double* col_j = A_local + (size_t)lj * n;
            for (int i = j + 1; i < n; ++i) {
                col_j[i] /= ljj;
            }

            // Fill panel buffer for broadcast
            panel[j] = ljj;
            for (int i = j + 1; i < n; ++i) {
                panel[i] = col_j[i];
            }
        }

        // Broadcast panel L[j:n, j] from owner to all
        MPI_Bcast(panel.data() + j, n - j, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Rank-1 update of trailing submatrix on local columns
        // For each local column c > j:  A[c:n, c] -= L[c,j] * L[c:n, j]
        for (int ci = 0; ci < ncols; ++ci) {
            int c = my_col_start + ci;
            if (c <= j) continue;

            const double lcj = panel[c];
            double* col = A_local + (size_t)ci * n;
            for (int i = c; i < n; ++i) {
                col[i] -= panel[i] * lcj;
            }
        }
    }

    // Zero out upper triangular part of local columns
    for (int ci = 0; ci < ncols; ++ci) {
        int c = my_col_start + ci;
        double* col = A_local + (size_t)ci * n;
        for (int i = 0; i < c; ++i) {
            col[i] = 0.0;
        }
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            printUsage(argv[0]);
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
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate positive definite matrix (identical on all ranks)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    std::vector<double> A(n * n);
    generatePositiveDefiniteMatrix(A, n);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = A;
    }

    // Extract local columns into column-major layout
    const int nn = (int)n;
    int cs = col_start_of(rank, size, nn);
    int ce = col_end_of(rank, size, nn);
    int ncols = ce - cs;

    std::vector<double> A_local((size_t)ncols * n);
    for (int ci = 0; ci < ncols; ++ci) {
        int c = cs + ci;
        for (size_t i = 0; i < n; ++i) {
            A_local[(size_t)ci * n + i] = A[i * n + c];
        }
    }

    // Free the full matrix copy to save memory
    { std::vector<double>().swap(A); }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyMPI(A_local.data(), nn, cs, ncols, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Check global success
    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!global_ok) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Gather local columns to rank 0 using MPI_Gatherv
    // The gathered buffer on rank 0 is the full matrix in column-major order
    std::vector<int> recvcounts, displs;
    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(n * n);
        recvcounts.resize(size);
        displs.resize(size);
        for (int p = 0; p < size; ++p) {
            int pcs = col_start_of(p, size, nn);
            int pce = col_end_of(p, size, nn);
            recvcounts[p] = (pce - pcs) * (int)n;
            displs[p] = pcs * (int)n;
        }
    }

    MPI_Gatherv(A_local.data(), ncols * (int)n, MPI_DOUBLE,
                gathered.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Convert from column-major gathered buffer to row-major L
        std::vector<double> L(n * n);
        for (int c = 0; c < nn; ++c) {
            for (size_t i = 0; i < n; ++i) {
                L[i * n + c] = gathered[(size_t)c * n + i];
            }
        }
        { std::vector<double>().swap(gathered); }

        printf("Computation time: %lld ms\n", duration_ms);

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(L, A_orig, n);

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
