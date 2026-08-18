#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are assigned cyclically.  Unlike a contiguous partition, this remains
// balanced as the amount of work per row grows toward the bottom of the matrix.
static size_t localRowCount(size_t n, int rank, int ranks) {
    return static_cast<size_t>(rank) < n
               ? 1 + (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(ranks)
               : 0;
}

static bool mpiCountFits(size_t count) {
    return count <= static_cast<size_t>(std::numeric_limits<int>::max());
}

// Generate the same B as the original program, retaining only this rank's
// cyclic rows.  Cycling those rows around the ranks lets every rank form its
// own rows of A without replicating either dense matrix.
static bool generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n,
                                           int rank, int ranks, MPI_Comm comm) {
    const size_t localRows = localRowCount(n, rank, ranks);
    const size_t maxRows = (n + static_cast<size_t>(ranks) - 1) /
                           static_cast<size_t>(ranks);
    if (!mpiCountFits(maxRows * n)) return false;

    std::vector<double> B(localRows * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool mine = static_cast<int>(i % static_cast<size_t>(ranks)) == rank;
        double* row = mine ? B.data() + (i / static_cast<size_t>(ranks)) * n : nullptr;
        for (size_t k = 0; k < n; ++k) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (mine) row[k] = value;
        }
    }

    A.resize(localRows * n);
    std::vector<double> current(maxRows * n), incoming(maxRows * n);
    std::copy(B.begin(), B.end(), current.begin());
    int sourceRank = rank;

    for (int step = 0; step < ranks; ++step) {
        const size_t sourceRows = localRowCount(n, sourceRank, ranks);
        for (size_t li = 0; li < localRows; ++li) {
            const double* lhs = B.data() + li * n;
            double* out = A.data() + li * n;
            for (size_t sj = 0; sj < sourceRows; ++sj) {
                const double* rhs = current.data() + sj * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += lhs[k] * rhs[k];
                const size_t j = static_cast<size_t>(sourceRank) +
                                 sj * static_cast<size_t>(ranks);
                out[j] = sum;
            }
        }

        if (step + 1 < ranks) {
            const int nextSource = (sourceRank + ranks - 1) % ranks;
            const size_t nextRows = localRowCount(n, nextSource, ranks);
            MPI_Sendrecv(current.data(), static_cast<int>(sourceRows * n), MPI_DOUBLE,
                         (rank + 1) % ranks, 0,
                         incoming.data(), static_cast<int>(nextRows * n), MPI_DOUBLE,
                         (rank + ranks - 1) % ranks, 0, comm, MPI_STATUS_IGNORE);
            current.swap(incoming);
            sourceRank = nextSource;
        }
    }

    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = static_cast<size_t>(rank) + li * static_cast<size_t>(ranks);
        A[li * n + i] += static_cast<double>(n);
    }
    return true;
}

// Distributed left-looking Cholesky.  The operation order within every output
// element matches the sequential algorithm.  Each completed pivot row is
// broadcast once, after which all ranks independently advance their rows.
static bool choleskyDecomposition(std::vector<double>& A, size_t n,
                                  int rank, int ranks, MPI_Comm comm) {
    const size_t localRows = localRowCount(n, rank, ranks);
    std::vector<double> pivot(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(ranks));
        int positive = 1;
        if (rank == owner) {
            double* row = A.data() + (j / static_cast<size_t>(ranks)) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            const double value = row[j] - sum;
            if (value <= 0.0 || !std::isfinite(value)) {
                positive = 0;
            } else {
                row[j] = std::sqrt(value);
                std::copy_n(row, j + 1, pivot.data());
                std::fill(row + j + 1, row + n, 0.0);
            }
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, comm);
        if (!positive) {
            if (rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);

        for (size_t li = 0; li < localRows; ++li) {
            const size_t i = static_cast<size_t>(rank) + li * static_cast<size_t>(ranks);
            if (i <= j) continue;
            double* row = A.data() + li * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    return true;
}

static std::vector<double> gatherMatrix(const std::vector<double>& local, size_t n,
                                        int rank, int ranks, MPI_Comm comm) {
    std::vector<int> counts(ranks), displacements(ranks);
    size_t total = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t count = localRowCount(n, r, ranks) * n;
        if (!mpiCountFits(count) || !mpiCountFits(total)) return {};
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(total);
        total += count;
    }

    std::vector<double> packed(rank == 0 ? n * n : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
    if (rank != 0) return {};

    std::vector<double> matrix(n * n);
    for (int r = 0; r < ranks; ++r) {
        const double* src = packed.data() + displacements[r];
        const size_t rows = localRowCount(n, r, ranks);
        for (size_t li = 0; li < rows; ++li) {
            const size_t i = static_cast<size_t>(r) + li * static_cast<size_t>(ranks);
            std::copy_n(src + li * n, n, matrix.data() + i * n);
        }
    }
    return matrix;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t end = std::min(i, j);
            for (size_t k = 0; k <= end; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t n = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (argv[i][0] == '-' || !end || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max())
                parseStatus = 1;
            else
                n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
        } else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
        }
    }
    if (parseStatus) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    // Ranks in excess of the number of rows cannot contribute.  Excluding them
    // also avoids routing every pivot through idle processes for small inputs.
    const int activeSize = std::min(worldSize, static_cast<int>(
        std::min(n, static_cast<size_t>(std::numeric_limits<int>::max()))));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const int ranks = activeSize;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A;
    int generated = generatePositiveDefiniteMatrix(A, n, rank, ranks, comm) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &generated, 1, MPI_INT, MPI_MIN, comm);
    if (!generated) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI count limits\n");
        MPI_Comm_free(&comm);
        MPI_Finalize();
        return 1;
    }
    std::vector<double> original;
    if (validate) original = A;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, rank, ranks, comm);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Comm_free(&comm);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / elapsed / 1e9);
    }

    std::vector<double> fullA;
    std::vector<double> fullOriginal;
    if (printResults || validate) fullA = gatherMatrix(A, n, rank, ranks, comm);
    if (validate) fullOriginal = gatherMatrix(original, n, rank, ranks, comm);

    int valid = 1;
    if (rank == 0 && printResults) print_results(fullA, "CholeskyL");
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        valid = validateCholesky(fullA, fullOriginal, n) ? 1 : 0;
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
