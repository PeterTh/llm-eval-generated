#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition using MPI (row-block distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Generate a symmetric positive definite matrix (kept for rank 0 generation)
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
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
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

// Helper to compute block row start and count for a rank
static void block_range(size_t n, int rank, int size, size_t &start, size_t &count) {
    size_t base = n / size;
    size_t rem = n % size;
    if ((size_t)rank < rem) {
        count = base + 1;
        start = rank * count;
    } else {
        count = base;
        start = rem * (base + 1) + (rank - rem) * base;
    }
}

bool distributedCholesky(size_t n, std::vector<double>& A_full, bool validate, bool printResults) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // compute local block
    size_t row_start, local_rows;
    block_range(n, rank, size, row_start, local_rows);

    // prepare scatter counts for rows
    std::vector<int> sendcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t s, cnt; block_range(n, r, size, s, cnt);
        sendcounts[r] = static_cast<int>(cnt * n);
        displs[r] = static_cast<int>(s * n);
    }

    // allocate local matrix storage (rows owned by this rank)
    std::vector<double> A_local(local_rows * n);

    // Scatter rows from rank 0 (A_full) to local A_local
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(local_rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Keep a copy of original on rank 0 for validation if needed
    std::vector<double> A_orig;
    if (rank == 0 && validate) A_orig = A_full;

    // Local L stored in same layout as A_local (overwrite lower triangle)
    std::vector<double> L_local(local_rows * n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // main loop
    std::vector<double> pivot_buf; // buffer for L[k,0..k]
    for (size_t k = 0; k < n; ++k) {
        // determine owner of row k
        int owner = 0;
        // compute owner by inverse of block_range: find r such that k in [start, start+count)
        // simple loop over ranks (size small relative to n)
        for (int r = 0; r < size; ++r) {
            size_t s, cnt; block_range(n, r, size, s, cnt);
            if (k >= s && k < s + cnt) { owner = r; break; }
        }

        // owner computes diagonal and its row entries
        if (rank == owner) {
            size_t local_k = k - row_start; // index within local rows
            double sum = 0.0;
            for (size_t s = 0; s < k; ++s) {
                double v = L_local[local_k * n + s];
                sum += v * v;
            }
            double diag = A_local[local_k * n + k] - sum;
            if (diag <= 0.0) {
                if (rank == 0) printf("Error: Matrix not positive definite at diagonal %zu\n", k);
                return false;
            }
            double Lkk = sqrt(diag);
            L_local[local_k * n + k] = Lkk;

            // compute L[i,k] for local rows i > k
            for (size_t i = local_k + 1; i < local_rows; ++i) {
                double ssum = 0.0;
                for (size_t t = 0; t < k; ++t) ssum += L_local[i * n + t] * L_local[local_k * n + t];
                L_local[i * n + k] = (A_local[i * n + k] - ssum) / Lkk;
            }

            // prepare pivot buffer to broadcast L[k,0..k]
            pivot_buf.resize(k + 1);
            for (size_t s = 0; s <= k; ++s) pivot_buf[s] = L_local[local_k * n + s];
        }

        // Broadcast pivot_buf of length k+1 from owner
        if (rank != owner) pivot_buf.resize(k + 1);
        MPI_Bcast(pivot_buf.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double Lkk = pivot_buf[k];

        // other ranks compute their local rows' L[i,k]
        if (rank != owner) {
            // rows local_rows with global index i = row_start + local_i
            for (size_t local_i = 0; local_i < local_rows; ++local_i) {
                size_t global_i = row_start + local_i;
                if (global_i <= k) continue;
                double ssum = 0.0;
                for (size_t t = 0; t < k; ++t) ssum += L_local[local_i * n + t] * pivot_buf[t];
                L_local[local_i * n + k] = (A_local[local_i * n + k] - ssum) / Lkk;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / max_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather L_local back to rank 0 if needed for output/validation
    if (rank == 0) {
        // reuse A_full to gather L
        std::vector<int> recvcounts(size), rdispls(size);
        for (int r = 0; r < size; ++r) {
            size_t s, cnt; block_range(n, r, size, s, cnt);
            recvcounts[r] = static_cast<int>(cnt * n);
            rdispls[r] = static_cast<int>(s * n);
        }
        MPI_Gatherv(L_local.data(), static_cast<int>(local_rows * n), MPI_DOUBLE,
                    A_full.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (printResults) print_results(A_full, "CholeskyL");

        if (validate) {
            bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            return valid;
        }
    } else {
        // non-root ranks send data
        MPI_Gatherv(L_local.data(), static_cast<int>(local_rows * n), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return true;
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (same on all ranks when launched under mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // Prepare full matrix on rank 0 and call distributed routine
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
    }

    bool success = distributedCholesky(n, A_full, validate, printResults);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
