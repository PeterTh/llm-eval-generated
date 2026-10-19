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

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t n = 512;
    bool validate = false, printResults = false;
    int parseError = 0, showHelp = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) showHelp = 1;
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); parseError = 1; }
    }
    int globalError;
    MPI_Allreduce(&parseError, &globalError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (showHelp || globalError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return globalError ? 1 : 0;
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Invalid matrix size\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = n / processes, remainder = n % processes;
    const size_t localRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = rank * base + std::min(static_cast<size_t>(rank), remainder);
    std::vector<int> counts(processes), displacements(processes);
    for (int p = 0; p < processes; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < remainder ? 1 : 0);
        const size_t offset = (static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), remainder)) * n;
        if (rows * n > INT_MAX || offset > INT_MAX) {
            if (rank == 0) printf("Matrix exceeds MPI count limits\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[p] = static_cast<int>(rows * n);
        displacements[p] = static_cast<int>(offset);
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> fullA;
    if (rank == 0) {
        fullA.resize(n * n);
        generatePositiveDefiniteMatrix(fullA, n);
    }
    std::vector<double> original;
    if (rank == 0 && validate) original = fullA;
    std::vector<double> local(localRows * n);
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    fullA.clear();
    fullA.shrink_to_fit();

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<double> pivot(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k < remainder * (base + 1) ? k / (base + 1) : remainder + (k - remainder * (base + 1)) / base);
        if (rank == owner) {
            const size_t row = k - firstRow;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += local[row * n + j] * local[row * n + j];
            const double diagonal = local[row * n + k] - sum;
            if (diagonal <= 0.0) success = false;
            else local[row * n + k] = sqrt(diagonal);
            std::copy_n(local.data() + row * n, n, pivot.data());
        }
        int pivotSuccess = rank == owner && success ? 1 : 0;
        MPI_Bcast(&pivotSuccess, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!pivotSuccess) {
            success = false;
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            break;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        const double diagonal = pivot[k];
        for (size_t r = 0; r < localRows; ++r) {
            const size_t i = firstRow + r;
            if (i <= k) continue;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += local[r * n + j] * pivot[j];
            local[r * n + k] = (local[r * n + k] - sum) / diagonal;
        }
    }
    if (success) {
        for (size_t r = 0; r < localRows; ++r)
            for (size_t j = firstRow + r + 1; j < n; ++j) local[r * n + j] = 0.0;
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> result;
    if (rank == 0) result.resize(n * n);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                rank == 0 ? result.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (!success) printf("Cholesky decomposition failed\n");
        else {
            printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
            const double ops = static_cast<double>(n) * n * n / 3.0;
            printf("Performance: %.3f GFLOPS\n", ops / elapsed / 1e9);
            if (printResults) print_results(result, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(result, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                success = valid;
            }
        }
    }
    int exitStatus = (rank == 0 && !success) ? 1 : 0;
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
