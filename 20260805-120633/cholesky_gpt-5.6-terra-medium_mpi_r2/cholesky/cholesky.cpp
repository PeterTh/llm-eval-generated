#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are assigned cyclically: row i belongs to rank i % nranks.  This is a
// better load balance than contiguous rows for the triangular trailing work.
static size_t localRowCount(size_t n, int rank, int ranks) {
    if (static_cast<size_t>(rank) >= n) return 0;
    return (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(ranks) + 1;
}

// Generate the same SPD matrix as the original program, retaining only rows
// owned by this process.  B is generated once and distributed before timing,
// avoiding redundant random-number generation on every rank.
static void generatePositiveDefiniteMatrix(std::vector<double>& localA, size_t n,
                                           int rank, int ranks) {
    std::vector<double> B(n * n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    MPI_Bcast(B.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const size_t rows = localRowCount(n, rank, ranks);
    for (size_t localI = 0; localI < rows; ++localI) {
        const size_t i = static_cast<size_t>(rank) + localI * ranks;
        double* arow = localA.data() + localI * n;
        const double* brow = B.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* bcol = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += brow[k] * bcol[k];
            arow[j] = sum;
        }
        arow[i] += n;
    }
}

// Distributed right-looking unblocked Cholesky.  The pivot row is broadcast
// after its diagonal is computed, allowing all ranks to independently update
// their owned rows without further synchronization in that column.
static bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                                  int rank, int ranks) {
    std::vector<double> pivot(n);
    const size_t localRows = localRowCount(n, rank, ranks);

    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k % static_cast<size_t>(ranks));
        int positive = 1;
        if (rank == owner) {
            double* row = localA.data() + ((k - static_cast<size_t>(rank)) /
                                           static_cast<size_t>(ranks)) * n;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += row[j] * row[j];
            const double value = row[k] - sum;
            if (value <= 0.0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                positive = 0;
            } else {
                row[k] = std::sqrt(value);
                std::copy_n(row, k + 1, pivot.data());
            }
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positive) return false;
        MPI_Bcast(pivot.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (size_t localI = 0; localI < localRows; ++localI) {
            const size_t i = static_cast<size_t>(rank) + localI * ranks;
            if (i <= k) continue;
            double* row = localA.data() + localI * n;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += row[j] * pivot[j];
            row[k] = (row[k] - sum) / pivot[k];
        }
    }

    for (size_t localI = 0; localI < localRows; ++localI) {
        const size_t i = static_cast<size_t>(rank) + localI * ranks;
        std::fill(localA.begin() + localI * n + i + 1,
                  localA.begin() + (localI + 1) * n, 0.0);
    }
    return true;
}

static std::vector<double> gatherRows(const std::vector<double>& localA, size_t n,
                                      int rank, int ranks) {
    const int localCount = static_cast<int>(localA.size());
    std::vector<int> counts, displacements;
    std::vector<double> packed, global;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
        int offset = 0;
        for (int p = 0; p < ranks; ++p) {
            counts[p] = static_cast<int>(localRowCount(n, p, ranks) * n);
            displacements[p] = offset;
            offset += counts[p];
        }
        packed.resize(n * n);
    }
    MPI_Gatherv(localA.data(), localCount, MPI_DOUBLE, packed.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        global.resize(n * n);
        for (int p = 0; p < ranks; ++p) {
            const size_t rows = localRowCount(n, p, ranks);
            for (size_t r = 0; r < rows; ++r) {
                const size_t globalRow = static_cast<size_t>(p) + r * ranks;
                std::copy_n(packed.data() + displacements[p] + r * n, n,
                            global.data() + globalRow * n);
            }
        }
    }
    return global;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relativeError = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k <= std::min(i, j); ++k) sum += L[i * n + k] * L[j * n + k];
        const double error = std::fabs(sum - original[i * n + j]);
        maxError = std::max(maxError, error);
        relativeError = std::max(relativeError, error / (std::fabs(original[i * n + j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relativeError);
    if (relativeError > 1e-6) { std::printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n > static_cast<size_t>(std::sqrt(std::numeric_limits<int>::max()))) {
        if (!rank) std::printf("Matrix size is too large for MPI counts\n");
        MPI_Finalize(); return 1;
    }
    if (!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n"
                           "MPI ranks: %d\nGenerating positive definite matrix...\n", n, n,
                           validate ? "enabled" : "disabled", ranks);
    std::vector<double> localA(localRowCount(n, rank, ranks) * n);
    generatePositiveDefiniteMatrix(localA, n, rank, ranks);
    std::vector<double> original;
    if (validate) original = localA;
    if (!rank) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (!rank) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (!rank) {
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double gflops = static_cast<double>(n) * n * n / 3.0 / maxElapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults || validate) {
        std::vector<double> result = gatherRows(localA, n, rank, ranks);
        std::vector<double> originalGlobal;
        if (validate) originalGlobal = gatherRows(original, n, rank, ranks);
        if (!rank && printResults) print_results(result, "CholeskyL");
        if (!rank && validate) {
            const bool valid = validateCholesky(result, originalGlobal, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
