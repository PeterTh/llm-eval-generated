#include <algorithm>
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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI collective counts and displacements are represented by int.
    if (n > static_cast<size_t>(INT_MAX) || (n && n > static_cast<size_t>(INT_MAX) / n)) {
        if (rank == 0) fprintf(stderr, "Matrix size exceeds MPI count limits\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    const size_t localRows = n > static_cast<size_t>(rank) ? (n - 1 - rank) / ranks + 1 : 0;
    std::vector<double> B(n * n), A(localRows * n);
    unsigned int seed = 42;
    for (double& x : B) x = rand_r(&seed) / (double)RAND_MAX - 0.5;
    for (size_t row = 0; row < localRows; ++row) {
        const size_t i = row * ranks + rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[row*n+j] = sum + (i == j ? n : 0);
        }
    }
    B.clear(); B.shrink_to_fit();
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    std::vector<double> pivot(n);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool success = true;
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % ranks);
        if (rank == owner) {
            double* row = A.data() + (j / ranks) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            const double val = row[j] - sum;
            row[j] = val <= 0.0 ? -1.0 : std::sqrt(val);
            std::copy(row, row + j + 1, pivot.begin());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (pivot[j] < 0.0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            success = false;
            break;
        }
        for (size_t row = 0; row < localRows; ++row) {
            const size_t i = row * ranks + rank;
            if (i <= j) continue;
            double* a = A.data() + row * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += a[k] * pivot[k];
            a[j] = (a[j] - sum) / pivot[j];
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    for (size_t row = 0; row < localRows; ++row) {
        size_t i = row * ranks + rank;
        std::fill(A.begin() + row*n + i + 1, A.begin() + (row+1)*n, 0.0);
    }
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000);
        printf("Computation time: %ld ms\n", ms);
        double ops = (double)n * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
    }
    if (validate || printResults) {
        std::vector<int> counts(ranks), offsets(ranks);
        int offset = 0;
        for (int p = 0; p < ranks; ++p) {
            size_t rows = n > static_cast<size_t>(p) ? (n - 1 - p) / ranks + 1 : 0;
            counts[p] = static_cast<int>(rows * n);
            offsets[p] = offset;
            offset += counts[p];
        }
        // Validation needs every row on every rank; output alone only needs rank 0.
        std::vector<double> packed((validate || rank == 0) ? n*n : 0);
        if (validate) MPI_Allgatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        else MPI_Gatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> full((validate || rank == 0) ? n*n : 0);
        if (validate || rank == 0) {
            for (size_t i = 0; i < n; ++i)
                std::copy_n(packed.data() + offsets[i % ranks] + (i / ranks)*n, n, full.data() + i*n);
        }
        if (printResults && rank == 0) print_results(full, "CholeskyL");
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            double errors[2] = {0.0, 0.0}, global[2];
            for (size_t row = 0; row < localRows; ++row) {
                const double* a = A.data() + row*n;
                for (size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += a[k] * full[j*n+k];
                    double error = std::fabs(sum - original[row*n+j]);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[row*n+j]) + 1e-10));
                }
            }
            MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
                if (global[1] > 1e-6) printf("Validation failed: relative error too large\nValidation: FAILED\n");
                else printf("Validation: PASSED\n");
            }
            if (global[1] > 1e-6) { MPI_Finalize(); return 1; }
        }
    }
    MPI_Finalize();
    return 0;
}
