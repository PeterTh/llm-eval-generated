#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are block distributed.  The matrix is kept in row-major order locally;
// this makes the trailing update a streaming, vectorizable operation.
static void distribution(size_t n, int rank, int ranks, size_t& first, size_t& rows) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    rows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
}

static void make_counts(size_t n, int ranks, std::vector<int>& counts,
                        std::vector<int>& displacements) {
    counts.resize(ranks);
    displacements.resize(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        size_t first, rows;
        distribution(n, r, ranks, first, rows);
        (void)first;
        if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "Matrix is too large for MPI counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[r] = static_cast<int>(rows);
        displacements[r] = static_cast<int>(offset);
        offset += rows;
    }
}

static int owner_of_row(size_t row, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t longRows = base + 1;
    if (row < longRows * extra) return static_cast<int>(row / longRows);
    return static_cast<int>(extra + (row - longRows * extra) / base);
}

// Generate exactly the same matrix as the original benchmark, but retain only
// the rows owned by this rank after the initial distributed construction.
static void generatePositiveDefiniteMatrix(std::vector<double>& localA, size_t n,
                                           size_t first, size_t rows,
                                           int rank, int ranks) {
    std::vector<double> localB(rows * n);
    if (rank == 0) {
        std::vector<double> fullB(n * n);
        unsigned int seed = 42;
        for (double& value : fullB)
            value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

        std::vector<int> counts, displacements;
        make_counts(n, ranks, counts, displacements);
        for (int r = 0; r < ranks; ++r) {
            size_t rfirst, rrows;
            distribution(n, r, ranks, rfirst, rrows);
            if (r == 0) {
                std::copy_n(fullB.data() + rfirst * n, rrows * n, localB.data());
            } else {
                MPI_Send(fullB.data() + rfirst * n, counts[r] * static_cast<int>(n),
                         MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(localB.data(), static_cast<int>(rows * n), MPI_DOUBLE, 0, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<double> fullB(n * n);
    std::vector<int> rowCounts, rowDisplacements;
    make_counts(n, ranks, rowCounts, rowDisplacements);
    for (int& count : rowCounts) count *= static_cast<int>(n);
    for (int& displacement : rowDisplacements) displacement *= static_cast<int>(n);
    MPI_Allgatherv(localB.data(), static_cast<int>(rows * n), MPI_DOUBLE,
                   fullB.data(), rowCounts.data(), rowDisplacements.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    localA.assign(rows * n, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalI = first + i;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += fullB[globalI * n + k] * fullB[j * n + k];
            localA[i * n + j] = sum;
        }
        localA[i * n + globalI] += static_cast<double>(n);
    }
}

static bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                                   size_t first, size_t rows, int rank, int ranks) {
    std::vector<int> rowCounts, rowDisplacements;
    make_counts(n, ranks, rowCounts, rowDisplacements);
    std::vector<double> column(n);

    for (size_t k = 0; k < n; ++k) {
        const int pivotOwner = owner_of_row(k, n, ranks);

        double diagonal = 0.0;
        if (rank == pivotOwner)
            diagonal = std::sqrt(localA[(k - first) * n + k]);
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, pivotOwner, MPI_COMM_WORLD);
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;

        // Form this column locally, then make it available to every row owner.
        for (size_t i = 0; i < rows; ++i) {
            const size_t globalI = first + i;
            if (globalI > k)
                localA[i * n + k] /= diagonal;
            column[globalI] = (globalI < k) ? 0.0 : localA[i * n + k];
        }
        MPI_Allgatherv(column.data() + first, static_cast<int>(rows), MPI_DOUBLE,
                       column.data(), rowCounts.data(), rowDisplacements.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == pivotOwner) localA[(k - first) * n + k] = diagonal;
        for (size_t i = 0; i < rows; ++i) {
            const size_t globalI = first + i;
            if (globalI <= k) continue;
            const double lik = column[globalI];
            double* row = localA.data() + i * n;
            for (size_t j = k + 1; j < n; ++j)
                row[j] -= lik * column[j];
        }
    }

    for (size_t i = 0; i < rows; ++i)
        std::fill(localA.begin() + i * n + first + i + 1,
                  localA.begin() + i * n + n, 0.0);
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n == 0) { if (rank == 0) std::printf("Matrix size must be positive\n"); MPI_Finalize(); return 1; }

    size_t first, rows;
    distribution(n, rank, ranks, first, rows);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark (MPI, %d ranks)\n", ranks);
        std::printf("Matrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A;
    generatePositiveDefiniteMatrix(A, n, first, rows, rank, ranks);
    std::vector<double> original = validate ? A : std::vector<double>();

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool localSuccess = choleskyDecomposition(A, n, first, rows, rank, ranks);
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int ok = localSuccess ? 1 : 0, allOk = 0;
    MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk) { if (rank == 0) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }

    if (rank == 0) {
        std::printf("Computation time: %.0f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (static_cast<double>(n) * n * n / 3.0) / elapsed / 1e9);
    }

    std::vector<int> counts, displacements;
    make_counts(n, ranks, counts, displacements);
    for (int& count : counts) count *= static_cast<int>(n);
    for (int& displacement : displacements) displacement *= static_cast<int>(n);
    std::vector<double> gathered;
    std::vector<double> gatheredOriginal;
    if (validate || printResults) {
        if (rank == 0) gathered.resize(n * n);
        MPI_Gatherv(A.data(), static_cast<int>(rows * n), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            if (rank == 0) gatheredOriginal.resize(n * n);
            MPI_Gatherv(original.data(), static_cast<int>(rows * n), MPI_DOUBLE,
                        rank == 0 ? gatheredOriginal.data() : nullptr, counts.data(), displacements.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    int result = 0;
    if (rank == 0 && printResults) print_results(gathered, "CholeskyL");
    if (rank == 0 && validate) {
        double maxError = 0.0, relError = 0.0;
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += gathered[i*n+k] * gathered[j*n+k];
            const double error = std::fabs(sum - gatheredOriginal[i*n+j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(gatheredOriginal[i*n+j]) + 1e-10));
        }
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
        result = relError > 1e-6 ? 1 : 0;
        std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
