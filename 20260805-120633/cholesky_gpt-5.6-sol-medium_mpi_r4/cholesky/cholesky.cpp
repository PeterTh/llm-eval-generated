#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

size_t localRowCount(const size_t n, const int rank, const int ranks) {
    return n > static_cast<size_t>(rank)
               ? 1 + (n - 1 - static_cast<size_t>(rank)) /
                         static_cast<size_t>(ranks)
               : 0;
}

// Rows are distributed cyclically.  Besides balancing the initial matrix, this
// keeps the shrinking set of rows below the current pivot balanced as well.
void generatePositiveDefiniteMatrix(std::vector<double>& localA,
                                    const size_t n, const int rank,
                                    const int ranks) {
    const size_t localRows = localRowCount(n, rank, ranks);
    std::vector<double> localB(localRows * n);
    std::vector<double> otherRow(n);

    // Generate exactly the same B as the original serial implementation, but
    // retain only this rank's rows.
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool owned = static_cast<int>(i % ranks) == rank;
        double* const destination = owned ? localB.data() + (i / ranks) * n
                                          : nullptr;
        for (size_t k = 0; k < n; ++k) {
            const double value =
                (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (owned) destination[k] = value;
        }
    }

    // Regenerating one row of B at a time avoids replicating an n-by-n B on
    // every process.  Matrix generation is deliberately outside the timed
    // factorization, just as it was in the serial benchmark.
    seed = 42;
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            otherRow[k] =
                (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
        for (size_t li = 0; li < localRows; ++li) {
            const double* const b = localB.data() + li * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += b[k] * otherRow[k];
            const size_t globalRow = static_cast<size_t>(rank) + li * ranks;
            localA[li * n + j] = sum + (globalRow == j ? n : 0);
        }
    }
}

bool choleskyDecomposition(std::vector<double>& localA, const size_t n,
                           const int rank, const int ranks) {
    const size_t localRows = localRowCount(n, rank, ranks);
    std::vector<double> pivot(n);

    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k % ranks);
        int positiveDefinite = 1;

        if (rank == owner) {
            double* const row = localA.data() + (k / ranks) * n;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += row[j] * row[j];
            const double value = row[k] - sum;
            if (!(value > 0.0) || !std::isfinite(value)) {
                positiveDefinite = 0;
            } else {
                row[k] = std::sqrt(value);
                std::copy_n(row, k + 1, pivot.data());
                std::fill(row + k + 1, row + n, 0.0);
            }
        }

        MPI_Bcast(&positiveDefinite, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positiveDefinite) {
            if (rank == 0) {
                std::printf(
                    "Error: Matrix is not positive definite at diagonal "
                    "element %zu\n",
                    k);
            }
            return false;
        }

        MPI_Bcast(pivot.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        // Each rank updates only its rows.  Cyclic ownership makes this loop
        // nearly the same length on every rank at every pivot.
        for (size_t li = 0; li < localRows; ++li) {
            const size_t i = static_cast<size_t>(rank) + li * ranks;
            if (i <= k) continue;
            double* const row = localA.data() + li * n;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += row[j] * pivot[j];
            row[k] = (row[k] - sum) / pivot[k];
        }
    }
    return true;
}

// Gather rank-packed cyclic rows and restore ordinary row-major ordering.
std::vector<double> gatherMatrix(const std::vector<double>& localA,
                                 const size_t n, const int rank,
                                 const int ranks) {
    std::vector<int> counts(ranks), displacements(ranks);
    size_t total = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t count = localRowCount(n, r, ranks) * n;
        const size_t mpiLimit =
            static_cast<size_t>(std::numeric_limits<int>::max());
        if (count > mpiLimit || total > mpiLimit - count) {
            if (rank == 0)
                std::fprintf(stderr, "Matrix is too large for MPI_Gatherv\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(total);
        total += count;
    }

    std::vector<double> packed(rank == 0 ? n * n : 0);
    MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank != 0) return {};
    std::vector<double> matrix(n * n);
    for (int r = 0; r < ranks; ++r) {
        const size_t offset = static_cast<size_t>(displacements[r]);
        const size_t rows = localRowCount(n, r, ranks);
        for (size_t li = 0; li < rows; ++li) {
            const size_t globalRow = static_cast<size_t>(r) + li * ranks;
            std::copy_n(packed.data() + offset + li * n, n,
                        matrix.data() + globalRow * n);
        }
    }
    return matrix;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& original, const size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t limit = std::min(i, j);
            for (size_t k = 0; k <= limit && k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const size_t index = i * n + j;
            const double error = std::fabs(sum - original[index]);
            maxError = std::max(maxError, error);
            relativeError = std::max(
                relativeError,
                error / (std::fabs(original[index]) + 1.0e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1.0e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value > std::numeric_limits<size_t>::max())
                parseStatus = 1;
            else
                n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
        }
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) parseStatus = 1;
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    const size_t localRows = localRowCount(n, rank, ranks);
    std::vector<double> localA(localRows * n);
    generatePositiveDefiniteMatrix(localA, n, rank, ranks);
    std::vector<double> localOriginal;
    if (validate) localOriginal = localA;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, rank, ranks);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (validate || printResults)
        result = gatherMatrix(localA, n, rank, ranks);
    if (printResults && rank == 0) print_results(result, "CholeskyL");

    int valid = 1;
    if (validate) {
        std::vector<double> original =
            gatherMatrix(localOriginal, n, rank, ranks);
        if (rank == 0) {
            std::printf("Validating result...\n");
            valid = validateCholesky(result, original, n) ? 1 : 0;
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
