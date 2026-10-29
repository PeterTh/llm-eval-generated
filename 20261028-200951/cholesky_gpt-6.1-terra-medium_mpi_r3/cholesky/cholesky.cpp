#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > INT_MAX) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(value);
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
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    const size_t localRows = n / ranks + (static_cast<size_t>(rank) < n % ranks);
    std::vector<double> A(localRows * n), B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = rank + li * ranks;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[li * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    std::vector<double>().swap(B);
    std::vector<double> original;
    if (validate) original = A;
    std::vector<double> pivot(n);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
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
            if (val <= 0.0) {
                pivot[j] = -1.0;
            } else {
                row[j] = std::sqrt(val);
                std::copy(row, row + j + 1, pivot.begin());
            }
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (pivot[j] < 0.0) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            success = false;
            break;
        }
        for (size_t li = 0; li < localRows; ++li) {
            const size_t i = rank + li * ranks;
            if (i <= j) continue;
            double* row = A.data() + li * n;
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
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = rank + li * ranks;
        std::fill(A.begin() + li * n + i + 1, A.begin() + (li + 1) * n, 0.0);
    }
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / (ms / 1000.0) / 1e9);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) result.resize(n * n);
        for (size_t i = 0; i < n; ++i) {
            const int owner = static_cast<int>(i % ranks);
            if (owner == 0 && rank == 0)
                std::copy_n(A.data() + (i / ranks) * n, n, result.data() + i * n);
            else if (rank == owner)
                MPI_Send(A.data() + (i / ranks) * n, static_cast<int>(n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            else if (rank == 0)
                MPI_Recv(result.data() + i * n, static_cast<int>(n), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == 0) print_results(result, "CholeskyL");
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        double errors[2] = {0.0, 0.0}, globalErrors[2];
        for (size_t j = 0; j < n; ++j) {
            const int owner = static_cast<int>(j % ranks);
            if (rank == owner) std::copy_n(A.data() + (j / ranks) * n, n, pivot.data());
            MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
            for (size_t li = 0; li < localRows; ++li) {
                const double* row = A.data() + li * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += row[k] * pivot[k];
                const double orig = original[li * n + j];
                const double error = std::fabs(sum - orig);
                errors[0] = std::max(errors[0], error);
                errors[1] = std::max(errors[1], error / (std::fabs(orig) + 1e-10));
            }
        }
        MPI_Allreduce(errors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        valid = globalErrors[1] <= 1e-6;
        if (rank == 0) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", globalErrors[0], globalErrors[1]);
            if (!valid) printf("Validation failed: relative error too large\n");
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
