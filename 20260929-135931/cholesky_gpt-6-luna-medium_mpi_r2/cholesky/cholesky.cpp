#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

// Generate the same deterministic positive definite input as the serial
// benchmark. Only rank zero needs the complete input; rows are then scattered.
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t end = std::min(i, j);
            for (size_t k = 0; k <= end; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(original[i * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
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
    bool validate = false, printResults = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0) badArgs = true;
            else n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else badArgs = true;
    }
    // MPI collective counts are int-sized; reject impossible or overflowing layouts.
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) / n ||
        n * n > static_cast<size_t>(std::numeric_limits<int>::max())) badArgs = true;
    int anyBad;
    MPI_Allreduce(&badArgs, &anyBad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (anyBad) {
        if (rank == 0) { printf("Invalid options or matrix size\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return 1;
    }

    const size_t base = n / processes, extra = n % processes;
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    std::vector<int> counts(processes), displacements(processes);
    for (int p = 0; p < processes; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        const size_t first = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), extra);
        counts[p] = static_cast<int>(rows * n);
        displacements[p] = static_cast<int>(first * n);
    }
    std::vector<double> A(localRows * n);
    std::vector<double> fullA, original;
    if (rank == 0) {
        fullA.resize(n * n);
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", validate ? "enabled" : "disabled");
        generatePositiveDefiniteMatrix(fullA, n);
        if (validate) original = fullA;
    }
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 A.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<double> pivotRow(n), column(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k < (base + 1) * extra
            ? k / (base + 1)
            : extra + (k - (base + 1) * extra) / base);
        const size_t ownerFirst = static_cast<size_t>(owner) * base + std::min(static_cast<size_t>(owner), extra);
        const size_t ownerLocal = k - ownerFirst;
        double diagonal = 0.0;
        if (rank == owner) {
            double value = A[ownerLocal * n + k];
            if (value <= 0.0) success = false;
            else diagonal = sqrt(value);
            A[ownerLocal * n + k] = diagonal;
            for (size_t j = k + 1; j < n; ++j)
                pivotRow[j] = A[ownerLocal * n + j] / diagonal;
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, owner, MPI_COMM_WORLD);
        if (!success) break;
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        pivotRow[k] = diagonal;
        MPI_Bcast(pivotRow.data() + k + 1, static_cast<int>(n - k - 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (size_t i = 0; i < localRows; ++i) {
            const size_t globalI = firstRow + i;
            if (globalI > k) column[globalI] = A[i * n + k];
        }
        // Each process owns distinct entries in the column; an all-reduction
        // distributes them while avoiding a serial root gather/broadcast.
        MPI_Allreduce(MPI_IN_PLACE, column.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i = 0; i < localRows; ++i) {
            const size_t globalI = firstRow + i;
            if (globalI <= k) continue;
            const double lik = column[globalI] / diagonal;
            A[i * n + k] = lik;
            for (size_t j = k + 1; j < n; ++j)
                A[i * n + j] -= lik * pivotRow[j];
        }
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int localSuccess = success ? 1 : 0, globalSuccess = 0;
    MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) fullA.resize(n * n);
    MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? fullA.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        const long ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (static_cast<double>(n) * n * n / 3.0) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(fullA, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(fullA, original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
