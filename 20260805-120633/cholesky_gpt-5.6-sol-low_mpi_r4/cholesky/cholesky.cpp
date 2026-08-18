#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are assigned cyclically.  Unlike a block distribution, this remains
// balanced when only the bottom part of the matrix is active late in the
// factorization.
static size_t localRowCount(size_t n, int rank, int processes) {
    if (static_cast<size_t>(rank) >= n) return 0;
    return (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(processes) + 1;
}

static inline double dot(const double* a, const double* b, size_t count) {
    double sum = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
    for (size_t k = 0; k < count; ++k) sum += a[k] * b[k];
    return sum;
}

// Distributed left-looking Cholesky.  The owner finishes a pivot row and
// broadcasts it; all ranks then independently compute their rows below it.
static bool choleskyDecomposition(std::vector<double>& rows, size_t n,
                                  int rank, int processes) {
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(processes));
        if (rank == owner) {
            double* row = rows.data() + (j / static_cast<size_t>(processes)) * n;
            const double value = row[j] - dot(row, row, j);
            pivot[0] = value > 0.0 ? std::sqrt(value) : -1.0;
            row[j] = pivot[0];
            std::copy_n(row, j + 1, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (pivot[j] <= 0.0 || !std::isfinite(pivot[j])) {
            if (rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }

        const size_t first = j + 1 +
            (static_cast<size_t>(rank) + static_cast<size_t>(processes) -
             ((j + 1) % static_cast<size_t>(processes))) % static_cast<size_t>(processes);
        for (size_t i = first; i < n; i += static_cast<size_t>(processes)) {
            double* row = rows.data() + (i / static_cast<size_t>(processes)) * n;
            row[j] = (row[j] - dot(row, pivot.data(), j)) / pivot[j];
        }
    }

    // Entries above the diagonal were input values and are not part of L.
    for (size_t li = 0; li < localRowCount(n, rank, processes); ++li) {
        const size_t global = li * static_cast<size_t>(processes) + static_cast<size_t>(rank);
        std::fill(rows.begin() + static_cast<std::ptrdiff_t>(li * n + global + 1),
                  rows.begin() + static_cast<std::ptrdiff_t>((li + 1) * n), 0.0);
    }
    return true;
}

// Retain only cyclically owned rows of B.  All ranks advance the same random
// stream, preserving the original matrix exactly without replicating B.
static void generatePositiveDefiniteMatrix(std::vector<double>& rows, size_t n,
                                           int rank, int processes) {
    const size_t count = localRowCount(n, rank, processes);
    std::vector<double> localB(count * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool owned = static_cast<int>(i % static_cast<size_t>(processes)) == rank;
        for (size_t k = 0; k < n; ++k) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (owned) localB[(i / static_cast<size_t>(processes)) * n + k] = value;
        }
    }

    std::vector<double> bRow(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(processes));
        if (rank == owner)
            std::copy_n(localB.data() + (j / static_cast<size_t>(processes)) * n,
                        n, bRow.data());
        MPI_Bcast(bRow.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (size_t li = 0; li < count; ++li)
            rows[li * n + j] = dot(localB.data() + li * n, bRow.data(), n);
    }
    for (size_t li = 0; li < count; ++li) {
        const size_t i = li * static_cast<size_t>(processes) + static_cast<size_t>(rank);
        rows[li * n + i] += static_cast<double>(n);
    }
}

static std::vector<double> gatherRows(const std::vector<double>& rows, size_t n,
                                      int rank, int processes) {
    std::vector<int> counts(processes), displacements(processes);
    size_t total = 0;
    for (int r = 0; r < processes; ++r) {
        const size_t elements = localRowCount(n, r, processes) * n;
        counts[r] = static_cast<int>(elements);
        displacements[r] = static_cast<int>(total);
        total += elements;
    }
    std::vector<double> packed(rank == 0 ? total : 0);
    MPI_Gatherv(rows.data(), static_cast<int>(rows.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> matrix(rank == 0 ? n * n : 0);
    if (rank == 0) {
        for (int r = 0; r < processes; ++r)
            for (size_t li = 0; li < localRowCount(n, r, processes); ++li) {
                const size_t global = li * static_cast<size_t>(processes) + static_cast<size_t>(r);
                std::copy_n(packed.data() + static_cast<size_t>(displacements[r]) + li * n,
                            n, matrix.data() + global * n);
            }
    }
    return matrix;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const size_t terms = std::min(i, j) + 1;
            const double actual = dot(L.data() + i * n, L.data() + j * n, terms);
            const double error = std::fabs(actual - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) std::printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t n = 512;
    bool validate = false, printResultsFlag = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            badArgs = badArgs || *end != '\0' || value == 0;
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResultsFlag = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); badArgs = true; }
    }
    // MPI uses int element counts in the collectives used here.
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (n != 0 && n > std::numeric_limits<size_t>::max() / n)) badArgs = true;
    if (help || badArgs) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("MPI processes: %d\nValidation: %s\n", processes, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    std::vector<double> rows(localRowCount(n, rank, processes) * n);
    generatePositiveDefiniteMatrix(rows, n, rank, processes);
    std::vector<double> original;
    if (validate) original = gatherRows(rows, n, rank, processes);

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(rows, n, rank, processes);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<double> result;
    if (validate || printResultsFlag) result = gatherRows(rows, n, rank, processes);
    int exitCode = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = elapsed > 0.0 ? (static_cast<double>(n) * n * n / 3.0) / elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResultsFlag) print_results(result, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(result, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
