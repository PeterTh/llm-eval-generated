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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t n = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed > INT_MAX) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
    // MPI's variable-count collectives use int counts and displacements.
    if (n && n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) printf("Matrix size exceeds MPI count limit\n");
        MPI_Finalize();
        return 1;
    }
    const size_t localRows = n / processes + (static_cast<size_t>(rank) < n % processes);
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    // Every rank generates the same B, avoiding communication and preserving rand_r's sequence.
    std::vector<double> B(n * n), A(localRows * n), original;
    unsigned int seed = 42;
    for (double& value : B) value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    for (size_t r = 0; r < localRows; ++r) {
        size_t i = r * processes + rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[r * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    std::vector<double>().swap(B);
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    std::vector<double> pivot(n);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = true;
    for (size_t j = 0; j < n; ++j) {
        int owner = static_cast<int>(j % processes);
        if (rank == owner) {
            double* row = A.data() + (j / processes) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            double val = row[j] - sum;
            row[j] = val <= 0.0 ? -1.0 : sqrt(val);
            std::copy(row, row + j + 1, pivot.begin());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (pivot[j] < 0.0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            success = false;
            break;
        }
        for (size_t r = 0; r < localRows; ++r) {
            size_t i = r * processes + rank;
            if (i <= j) continue;
            double* row = A.data() + r * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    for (size_t r = 0; r < localRows; ++r) {
        size_t i = r * processes + rank;
        std::fill(A.begin() + r * n + i + 1, A.begin() + (r + 1) * n, 0.0);
    }
    if (rank == 0) {
        long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0);
    }
    if (validate || printResults) {
        std::vector<int> counts(processes), displacements(processes);
        int offset = 0;
        for (int p = 0; p < processes; ++p) {
            counts[p] = static_cast<int>((n / processes + (static_cast<size_t>(p) < n % processes)) * n);
            displacements[p] = offset;
            offset += counts[p];
        }
        std::vector<double> packed(n * n), L(n * n);
        MPI_Allgatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int p = 0; p < processes; ++p)
            for (size_t r = 0; r < static_cast<size_t>(counts[p]) / (n ? n : 1); ++r)
                std::copy_n(packed.data() + displacements[p] + r * n, n, L.data() + (r * processes + p) * n);
        if (printResults && rank == 0) print_results(L, "CholeskyL");
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            double errors[2] = {0.0, 0.0}, globalErrors[2];
            for (size_t r = 0; r < localRows; ++r) {
                size_t i = r * processes + rank;
                for (size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
                    double error = fabs(sum - original[r * n + j]);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (fabs(original[r * n + j]) + 1e-10));
                }
            }
            MPI_Allreduce(errors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", globalErrors[0], globalErrors[1]);
                if (globalErrors[1] > 1e-6) printf("Validation failed: relative error too large\n");
                printf("Validation: %s\n", globalErrors[1] > 1e-6 ? "FAILED" : "PASSED");
            }
            exitCode = globalErrors[1] > 1e-6 ? 1 : 0;
        }
    }
    MPI_Finalize();
    return exitCode;
}
