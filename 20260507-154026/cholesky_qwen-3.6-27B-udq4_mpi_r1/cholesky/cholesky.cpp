#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Row-block distribution helpers
static void computeRowDist(size_t n, int np, int rank, size_t& s, size_t& c) {
    size_t base = n / (size_t)np, rem = n % (size_t)np;
    s = base * (size_t)rank + (rank < (int)rem ? (size_t)rank : rem);
    c = base + (rank < (int)rem ? 1 : 0);
}

static void buildCountsDispls(size_t n, int np,
                              std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(np); displs.resize(np);
    for (int p = 0; p < np; ++p) {
        size_t s, c; computeRowDist(n, np, p, s, c);
        counts[p] = (int)(c * n); displs[p] = (int)(s * n);
    }
}

// MPI-parallel Cholesky with row-block distribution
// Input:  A contains local rows [my_start, my_start+my_count) of the original matrix
// Output: A contains local rows of L (lower triangular factor)
//
// Algorithm: column-cyclic Cholesky with in-place WAXPYB update.
//   for j = 0..n-1:
//     for i = j..n-1:   A(i,j) already updated by prior WAXPYB steps
//       if i==j: L(j,j) = sqrt(A(j,j))
//       else:    L(i,j) = A(i,j) / L(j,j)
//     broadcast column j to all processes
//     for i = j+1..n-1: for k = j+1..n-1: A(i,k) -= L(i,j)*L(k,j)
bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int np) {
    size_t my_start, my_count;
    computeRowDist(n, np, rank, my_start, my_count);

    // Per-column communication buffers
    std::vector<double> local_vals(my_count);
    std::vector<double> col_j(n);

    // Row counts/displacements for Allgatherv
    std::vector<int> row_counts(np), row_displs(np);
    for (int p = 0; p < np; ++p) {
        size_t s, c; computeRowDist(n, np, p, s, c);
        row_counts[p] = (int)c; row_displs[p] = (int)s;
    }

    for (size_t j = 0; j < n; ++j) {
        // Extract A(i,j) for local rows i >= j; for i < j, send 0
        size_t li_lo = 0, li_hi = my_count;
        if (my_start + my_count <= j) { li_lo = my_count; }
        else if (my_start < j) { li_lo = j - my_start; }

        for (size_t li = 0; li < li_lo; ++li)
            local_vals[li] = 0.0;
        for (size_t li = li_lo; li < li_hi; ++li)
            local_vals[li] = A[li * n + j];

        // Gather column j values from all processes
        MPI_Allgatherv(local_vals.data(), (int)my_count, MPI_DOUBLE,
                       col_j.data(), row_counts.data(), row_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        double val = col_j[j];
        if (val <= 0.0) {
            if (rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        col_j[j] = sqrt(val);

        // Off-diagonal: L(i,j) = A(i,j) / L(j,j)
        double ljj = col_j[j];
        for (size_t i = j + 1; i < n; ++i)
            col_j[i] /= ljj;

        // Write factorized column back to A for local rows
        for (size_t li = li_lo; li < li_hi; ++li) {
            A[li * n + j] = col_j[my_start + li];
        }

        // --- WAXPYB update of trailing submatrix ---
        // A(i,k) -= L(i,j) * L(k,j)   for local rows i > j, columns k > j
        size_t uli = li_lo;
        if (uli < my_count && my_start + uli <= j) uli = j + 1 - my_start;
        for (size_t li = uli; li < my_count; ++li) {
            double lij = col_j[my_start + li];
            if (lij == 0.0) continue;
            for (size_t k = j + 1; k < n; ++k)
                A[li * n + k] -= lij * col_j[k];
        }
    }

    // Zero out upper triangular part for local rows
    for (size_t li = 0; li < my_count; ++li) {
        size_t i = my_start + li;
        for (size_t j2 = i + 1; j2 < n; ++j2)
            A[li * n + j2] = 0.0;
    }
    return true;
}

// Generate symmetric positive definite matrix: A = B*B^T + n*I
// Parallelized: rank 0 generates B, broadcasts, each process computes its rows
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n, int rank, int np) {
    size_t my_start, my_count;
    computeRowDist(n, np, rank, my_start, my_count);

    std::vector<double> full_B(n * n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            full_B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    MPI_Bcast(full_B.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (size_t li = 0; li < my_count; ++li) {
        size_t i = my_start + li;
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += full_B[i * n + k] * full_B[j * n + k];
            A[li * n + j] = s;
        }
        A[li * n + i] += (double)n;
    }
}

// Validate L*L^T == A_orig (runs on rank 0 only)
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = s;
        }

    double maxErr = 0.0, relErr = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double e = fabs(reconstructed[i] - A_orig[i]);
        maxErr = std::max(maxErr, e);
        relErr = std::max(relErr, e / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxErr);
    printf("Max relative error: %.10e\n", relErr);
    if (relErr > 1e-6) { printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    size_t n = 512;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", np);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t my_start, my_count;
    computeRowDist(n, np, rank, my_start, my_count);
    std::vector<double> local_A(my_count * n);

    if (rank == 0) printf("Generating positive definite matrix...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    generatePositiveDefiniteMatrix(local_A, n, rank, np);

    // Gather original A to rank 0 for validation
    std::vector<int> counts, displs;
    buildCountsDispls(n, np, counts, displs);
    std::vector<double> A_orig;
    if (validate && rank == 0) A_orig.resize(n * n);
    MPI_Gatherv(local_A.data(), (int)(my_count * n), MPI_DOUBLE,
                rank == 0 && validate ? A_orig.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    bool ok = choleskyDecomposition(local_A, n, rank, np);
    double elapsed = MPI_Wtime() - t0;
    MPI_Barrier(MPI_COMM_WORLD);

    if (!ok) { if (rank == 0) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }

    // Gather full L to rank 0
    std::vector<double> full_L(n * n);
    MPI_Gatherv(local_A.data(), (int)(my_count * n), MPI_DOUBLE,
                full_L.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", elapsed);
        double ops = (double)n * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / elapsed / 1e9);
        if (printResults) print_results(full_L, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(full_L, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
