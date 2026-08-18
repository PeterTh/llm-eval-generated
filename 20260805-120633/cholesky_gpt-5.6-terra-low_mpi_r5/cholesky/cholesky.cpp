#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed contiguously.  The factorization is the left-looking
// Cholesky algorithm expressed as rank-one trailing updates: after a column
// has been formed, its distributed part is all-gathered and every rank updates
// its own lower-triangular trailing rows.  Thus no rank owns the matrix.
struct RowDistribution {
    size_t begin, count;
    std::vector<int> counts, displacements;
};

static RowDistribution distributeRows(size_t n, int ranks, int rank) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    RowDistribution d{rank * base + std::min(static_cast<size_t>(rank), extra),
                      base + (static_cast<size_t>(rank) < extra), {}, {}};
    d.counts.resize(ranks);
    d.displacements.resize(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < extra);
        d.counts[r] = static_cast<int>(rows * n);
        d.displacements[r] = static_cast<int>(offset * n);
        offset += rows;
    }
    return d;
}

static int ownerOf(size_t row, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t large = (base + 1) * extra;
    return row < large ? static_cast<int>(row / (base + 1))
                       : static_cast<int>(extra + (row - large) / base);
}

// Generate exactly the same B and A = B*B^T + nI as the original program.
// B is replicated (O(n^2) storage), while A is generated only for local rows.
static void generatePositiveDefiniteMatrix(std::vector<double>& localA, size_t n,
                                           const RowDistribution& d) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    for (size_t li = 0; li < d.count; ++li) {
        const size_t i = d.begin + li;
        double* const arow = localA.data() + li * n;
        const double* const brow = B.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* const bcolrow = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += brow[k] * bcolrow[k];
            arow[j] = sum;
        }
        arow[i] += static_cast<double>(n);
    }
}

static bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                                  const RowDistribution& d, int ranks, int rank) {
    std::vector<double> column(n == 0 ? 1 : n);
    std::vector<double> localColumn(n == 0 ? 1 : n);
    std::vector<int> columnCounts(ranks), columnDisplacements(ranks);
    for (size_t j = 0; j < n; ++j) {
        const int owner = ownerOf(j, n, ranks);
        int valid = 1;
        double diagonal = 0.0;
        if (rank == owner) {
            diagonal = localA[(j - d.begin) * n + j];
            if (diagonal <= 0.0) valid = 0;
            else diagonal = std::sqrt(diagonal);
        }
        MPI_Bcast(&valid, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!valid) {
            if (rank == owner)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank == owner) localA[(j - d.begin) * n + j] = diagonal;

        const size_t first = std::max(d.begin, j + 1);
        const size_t last = d.begin + d.count;
        const size_t sendRows = first < last ? last - first : 0;
        for (size_t i = first; i < last; ++i) {
            localA[(i - d.begin) * n + j] /= diagonal;
            localColumn[i - first] = localA[(i - d.begin) * n + j];
        }

        for (int r = 0; r < ranks; ++r) {
            const size_t rbegin = static_cast<size_t>(r) * (n / ranks) +
                std::min(static_cast<size_t>(r), n % ranks);
            const size_t rcount = n / ranks + (static_cast<size_t>(r) < n % ranks);
            const size_t rfirst = std::max(rbegin, j + 1);
            columnCounts[r] = rfirst < rbegin + rcount ? static_cast<int>(rbegin + rcount - rfirst) : 0;
            columnDisplacements[r] = columnCounts[r] ? static_cast<int>(rfirst - (j + 1)) : 0;
        }
        MPI_Allgatherv(sendRows ? localColumn.data() : nullptr,
                       static_cast<int>(sendRows), MPI_DOUBLE, column.data(),
                       columnCounts.data(), columnDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        for (size_t i = first; i < last; ++i) {
            double* const row = localA.data() + (i - d.begin) * n;
            const double lij = row[j];
            for (size_t k = j + 1; k <= i; ++k) row[k] -= lij * column[k - (j + 1)];
        }
    }
    for (size_t li = 0; li < d.count; ++li) {
        double* const row = localA.data() + li * n;
        for (size_t j = d.begin + li + 1; j < n; ++j) row[j] = 0.0;
    }
    return true;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relativeError = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
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
    size_t n = 512; bool validate = false, printResults = false;
    int argumentError = 0, help = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = 1;
        else argumentError = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &argumentError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (help || argumentError || n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) { if (argumentError) std::printf("Unknown or incomplete option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return argumentError || n == 0 ? 1 : 0;
    }
    const RowDistribution d = distributeRows(n, ranks, rank);
    std::vector<double> A(d.count * n), original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",
                    n, n, validate ? "enabled" : "disabled");
    }
    generatePositiveDefiniteMatrix(A, n, d);
    if (validate) original = A;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = choleskyDecomposition(A, n, d, ranks, rank);
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int failed = success ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (failed) { if (rank == 0) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", milliseconds,
                    (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9);
    }
    std::vector<double> fullA, fullOriginal;
    if (rank == 0 && (validate || printResults)) fullA.resize(n * n);
    if (validate || printResults)
        MPI_Gatherv(A.data(), static_cast<int>(d.count * n), MPI_DOUBLE, fullA.data(), d.counts.data(), d.displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate) {
        if (rank == 0) fullOriginal.resize(n * n);
        MPI_Gatherv(original.data(), static_cast<int>(d.count * n), MPI_DOUBLE, fullOriginal.data(), d.counts.data(), d.displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int validationFailed = 0;
    if (rank == 0 && printResults) print_results(fullA, "CholeskyL");
    if (rank == 0 && validate) { validationFailed = validateCholesky(fullA, fullOriginal, n) ? 0 : 1; std::printf("Validation: %s\n", validationFailed ? "FAILED" : "PASSED"); }
    MPI_Bcast(&validationFailed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validationFailed;
}
