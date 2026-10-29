#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            int value = atoi(argv[++i]);
            if (value < 0) { if (rank == 0) printf("Invalid matrix size\n"); status = 1; break; }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } status = 1; break; }
    }
    if (status) { MPI_Finalize(); return status; }
    // MPI counts are int; the optional full matrix collection also uses int displacements.
    if (n > INT_MAX || (n && n > static_cast<size_t>(INT_MAX) / n)) {
        if (rank == 0) printf("Matrix size exceeds MPI count limits\n");
        MPI_Finalize(); return 1;
    }
    const size_t localRows = n / ranks + (static_cast<size_t>(rank) < n % ranks);
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    // Each rank generates the same B, preserving the original rand_r sequence.
    std::vector<double> B(n * n), A(localRows * n), original;
    unsigned int seed = 42;
    for (double &x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    for (size_t local = 0; local < localRows; ++local) {
        size_t i = local * ranks + rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[local*n+j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    B.clear(); B.shrink_to_fit();
    if (validate) original = A;
    std::vector<double> pivot(n);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = true;
    for (size_t j = 0; j < n; ++j) {
        int owner = static_cast<int>(j % ranks);
        if (rank == owner) {
            double* row = A.data() + (j / ranks) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            double val = row[j] - sum;
            row[j] = val <= 0.0 ? -1.0 : std::sqrt(val);
            std::copy(row, row + j + 1, pivot.begin());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (pivot[j] < 0.0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            success = false; break;
        }
        for (size_t local = 0; local < localRows; ++local) {
            size_t i = local * ranks + rank;
            if (i <= j) continue;
            double* row = A.data() + local*n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (rank == 0) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    for (size_t local = 0; local < localRows; ++local) {
        size_t i = local * ranks + rank;
        std::fill(A.begin() + local*n + i + 1, A.begin() + (local+1)*n, 0.0);
    }
    if (rank == 0) {
        long duration = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration);
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / (duration / 1000.0) / 1e9);
    }
    if (validate || printResults) {
        std::vector<int> counts(ranks), displs(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = static_cast<int>((n / ranks + (static_cast<size_t>(p) < n % ranks)) * n);
            displs[p] = p ? displs[p-1] + counts[p-1] : 0;
        }
        std::vector<double> packed(n*n), L(n*n);
        MPI_Allgatherv(A.data(), static_cast<int>(localRows*n), MPI_DOUBLE, packed.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int p = 0; p < ranks; ++p)
            for (size_t l = 0; l < static_cast<size_t>(counts[p]) / (n ? n : 1); ++l)
                std::copy_n(packed.data() + displs[p] + l*n, n, L.data() + (l*ranks+p)*n);
        if (printResults && rank == 0) print_results(L, "CholeskyL");
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            double errors[2] = {0.0, 0.0}, global[2];
            for (size_t local = 0; local < localRows; ++local) {
                size_t i = local*ranks + rank;
                for (size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += L[i*n+k] * L[j*n+k];
                    double error = std::fabs(sum - original[local*n+j]);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[local*n+j]) + 1e-10));
                }
            }
            MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
                if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
                printf("Validation: %s\n", global[1] > 1e-6 ? "FAILED" : "PASSED");
            }
            if (global[1] > 1e-6) status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
