#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are assigned cyclically.  This is important for Cholesky: the number of
// active rows decreases every iteration, and a block distribution leaves some
// processes idle much earlier than others.
static size_t localRowCount(size_t n, int rank, int processes) {
    return rank < static_cast<int>(n) ? (n - static_cast<size_t>(rank) + processes - 1) / processes : 0;
}

static bool distributedCholesky(std::vector<double>& rows, size_t n,
                                int rank, int processes) {
    std::vector<double> pivot(n);
    int ok = 1;

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % processes);
        if (rank == owner) {
            double* row = rows.data() + (j / processes) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            const double value = row[j] - sum;
            if (!(value > 0.0)) {
                ok = 0;
            } else {
                row[j] = std::sqrt(value);
                std::copy_n(row, j + 1, pivot.data());
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) {
            if (rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        const size_t first = j < static_cast<size_t>(rank)
                                 ? 0
                                 : (j - static_cast<size_t>(rank)) / processes + 1;
        const size_t count = localRowCount(n, rank, processes);
        for (size_t li = first; li < count; ++li) {
            double* row = rows.data() + li * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }

    // Match the original representation exactly: the upper triangle is zero.
    for (size_t li = 0; li < localRowCount(n, rank, processes); ++li) {
        const size_t global = static_cast<size_t>(rank) + li * processes;
        std::fill(rows.begin() + li * n + global + 1, rows.begin() + (li + 1) * n, 0.0);
    }
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t end = std::min(i, j);
            for (size_t k = 0; k <= end; ++k) sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) std::printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t n = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0) parseStatus = 1;
            else n = static_cast<size_t>(value);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) parseStatus = 2;
        else parseStatus = 1;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (n != 0 && n > std::numeric_limits<size_t>::max() / n)) parseStatus = 1;
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nMPI processes: %d\n",
                    n, n, validate ? "enabled" : "disabled", processes);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<int> counts(processes), displacements(processes);
    size_t total = 0;
    for (int r = 0; r < processes; ++r) {
        const size_t elems = localRowCount(n, r, processes) * n;
        if (elems > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            total > static_cast<size_t>(std::numeric_limits<int>::max())) parseStatus = 1;
        counts[r] = static_cast<int>(elems);
        displacements[r] = static_cast<int>(total);
        total += elems;
    }
    MPI_Allreduce(MPI_IN_PLACE, &parseStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (parseStatus) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI collectives\n");
        MPI_Finalize(); return 1;
    }

    std::vector<double> packed;
    std::vector<double> original;
    if (rank == 0) {
        std::vector<double> full(n * n);
        generatePositiveDefiniteMatrix(full, n);
        if (validate) original = full;
        packed.resize(n * n);
        for (int r = 0; r < processes; ++r)
            for (size_t li = 0; li < localRowCount(n, r, processes); ++li) {
                const size_t gr = static_cast<size_t>(r) + li * processes;
                std::copy_n(full.data() + gr * n, n, packed.data() + displacements[r] + li * n);
            }
    }
    std::vector<double> rows(static_cast<size_t>(counts[rank]));
    MPI_Scatterv(rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 rows.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(rows, n, rank, processes);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { MPI_Finalize(); return 1; }

    const bool needFull = validate || printResults;
    std::vector<double> result;
    if (needFull) {
        MPI_Gatherv(rows.data(), counts[rank], MPI_DOUBLE,
                    rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            result.resize(n * n);
            for (int r = 0; r < processes; ++r)
                for (size_t li = 0; li < localRowCount(n, r, processes); ++li) {
                    const size_t gr = static_cast<size_t>(r) + li * processes;
                    std::copy_n(packed.data() + displacements[r] + li * n, n, result.data() + gr * n);
                }
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
        if (printResults) print_results(result, "CholeskyL");
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
