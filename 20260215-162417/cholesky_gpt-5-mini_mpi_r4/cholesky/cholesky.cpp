#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition using simple row-block distribution and column broadcasts.
// The algorithm preserves the semantics of the original unblocked algorithm but distributes
// rows across MPI ranks. Root generates the matrix and scatters rows to all ranks.

// Generate a symmetric positive definite matrix on root
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Same as original validation
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
    }
    return relError <= 1e-6;
}

void printUsage(const char* progName) {
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Usage: %s [options]\n", progName);
        printf("Options:\n");
        printf("  -n <num>     Matrix size (default: 512)\n");
        printf("  -v           Enable validation\n");
        printf("  -r           Print results for external validation\n");
        printf("  -h           Show this help message\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse arguments only on rank 0, then broadcast
    if (rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast options
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int vi = validate ? 1 : 0;
    int ri = printResults ? 1 : 0;
    MPI_Bcast(&vi, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ri, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (vi != 0);
    printResults = (ri != 0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution (contiguous blocks)
    std::vector<int> counts(size), displs(size);
    size_t base = n / size;
    size_t rem = n % size;
    for (int r = 0; r < size; ++r) {
        size_t rows = base + (size_t)(r < (int)rem);
        counts[r] = (int)(rows * n); // number of doubles
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];

    // Root generates full matrix then scatters rows
    std::vector<double> A_local(counts[rank]);
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, keep original on root
    std::vector<double> A_orig;
    if (rank == 0 && validate) A_orig = A_full;

    // Helper to map global row index -> owner and local index
    std::vector<int> rowOwner(n); // compute owners by ranges
    {
        for (int r = 0; r < size; ++r) {
            int start = displs[r] / (int)n;
            int cnt = counts[r] / (int)n;
            for (int i = 0; i < cnt; ++i) rowOwner[start + i] = r;
        }
    }

    // Precompute prefix rows to get local row start global index
    int local_row_count = counts[rank] / (int)n;
    int local_row_start = displs[rank] / (int)n;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Main column-wise Cholesky
    std::vector<double> colbuf; // will hold L[k,0..k]
    for (int k = 0; k < (int)n; ++k) {
        int owner = rowOwner[k];
        if (rank == owner) {
            // owner computes L[k,0..k]
            int local_k = k - local_row_start;
            // compute diagonal
            double sum = 0.0;
            for (int t = 0; t < k; ++t) {
                double val = A_local[local_k * n + t];
                sum += val * val;
            }
            double val = A_local[local_k * n + k] - sum;
            if (val <= 0.0) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal %d\n", k);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            double Lkk = sqrt(val);
            A_local[local_k * n + k] = Lkk;

            // compute L[k,j] for j<k (already stored) -- keep as-is
            // prepare buffer
            colbuf.resize(k + 1);
            for (int t = 0; t <= k; ++t) colbuf[t] = A_local[local_k * n + t];
            // broadcast to others
            MPI_Bcast(colbuf.data(), k + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        } else {
            // non-owner receives column
            colbuf.resize(k + 1);
            MPI_Bcast(colbuf.data(), k + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }

        // All ranks update their local rows i where global i > k
        for (int local_i = 0; local_i < local_row_count; ++local_i) {
            int gi = local_row_start + local_i;
            if (gi <= k) continue;
            // compute dot = sum_{t=0..k-1} A_local[local_i,n,t] * colbuf[t]
            double dot = 0.0;
            double* row = &A_local[local_i * n];
            for (int t = 0; t < k; ++t) dot += row[t] * colbuf[t];
            row[k] = (row[k] - dot) / colbuf[k];
        }

        // owners for later columns keep upper triangular zeroing as original did
        if (rank == owner) {
            int local_k = k - local_row_start;
            // zero upper part in that row for columns > k
            for (int j = k + 1; j < (int)n; ++j) A_local[local_k * n + j] = 0.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather result to root
    if (rank == 0) A_full.assign(n * n, 0.0);
    MPI_Gatherv(A_local.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A_full, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) { printf("Validation: PASSED\n"); MPI_Finalize(); return 0; }
            else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
