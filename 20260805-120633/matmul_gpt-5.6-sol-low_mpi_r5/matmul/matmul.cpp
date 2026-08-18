#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void rowRange(size_t N, int rank, int ranks, size_t& first, size_t& count) {
    const size_t q = N / static_cast<size_t>(ranks);
    const size_t r = N % static_cast<size_t>(ranks);
    count = q + (static_cast<size_t>(rank) < r);
    first = static_cast<size_t>(rank) * q +
            std::min(static_cast<size_t>(rank), r);
}

static void initRows(std::vector<double>& matrix, size_t N, size_t firstRow,
                     size_t rows) {
    for (size_t li = 0; li < rows; ++li) {
        const size_t i = firstRow + li;
        for (size_t j = 0; j < N; ++j)
            matrix[li * N + j] = getPseudoRndValue(N, i, j);
    }
}

// Store B transposed so the two vectors used by every dot product are contiguous.
static void initTransposedB(std::vector<double>& bt, size_t N) {
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < N; ++k)
            bt[j * N + k] = getPseudoRndValue(N, k, j);
}

static void matrixMultiply(const std::vector<double>& a,
                           const std::vector<double>& bt,
                           std::vector<double>& c, size_t N, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        const double* const ar = a.data() + i * N;
        double* const cr = c.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            const double* const br = bt.data() + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += ar[k] * br[k];
            cr[j] = sum;
        }
    }
}

static bool validateLocal(const std::vector<double>& a,
                          const std::vector<double>& bt,
                          const std::vector<double>& c, size_t N,
                          size_t firstRow, size_t rows) {
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = pi % N;
        if (i < firstRow || i >= firstRow + rows) continue;
        const size_t li = i - firstRow;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += a[li * N + k] * bt[j * N + k];
            const double actual = c[li * N + j];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseResult = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseResult = 2;
        else parseResult = 1;
    }
    if (parseResult) {
        if (rank == 0) {
            if (parseResult == 1) std::printf("Unknown or incomplete option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }
    if (N == 0 || N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) std::fprintf(stderr, "Matrix size must be positive and representable\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\nMPI processes: %d\n", validate ? "enabled" : "disabled", ranks);
        std::printf("Initializing matrices...\n");
    }

    size_t firstRow = 0, localRows = 0;
    rowRange(N, rank, ranks, firstRow, localRows);
    std::vector<double> a(localRows * N), bt(N * N), c(localRows * N);
    initRows(a, N, firstRow, localRows);
    initTransposedB(bt, N);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(a, bt, c, N, localRows);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(seconds * 1000.0);
        std::printf("Computation time: %ld ms\n", ms);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    if (printResults) {
        std::vector<int> counts(ranks), displacements(ranks);
        bool gatherable = N <= static_cast<size_t>(std::numeric_limits<int>::max());
        for (int p = 0; p < ranks; ++p) {
            size_t begin, rows;
            rowRange(N, p, ranks, begin, rows);
            gatherable = gatherable && rows * N <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                         begin * N <= static_cast<size_t>(std::numeric_limits<int>::max());
            counts[p] = gatherable ? static_cast<int>(rows * N) : 0;
            displacements[p] = gatherable ? static_cast<int>(begin * N) : 0;
        }
        if (!gatherable) {
            if (rank == 0) std::fprintf(stderr, "Result is too large for MPI_Gatherv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<double> fullC;
        if (rank == 0) fullC.resize(N * N);
        MPI_Gatherv(c.data(), static_cast<int>(c.size()), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }

    int localValid = !validate || validateLocal(a, bt, c, N, firstRow, localRows);
    int globallyValid = 0;
    MPI_Reduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    if (validate && rank == 0) {
        std::printf("Validating result...\nValidation: %s\n", globallyValid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return (rank == 0 && !globallyValid) ? 1 : 0;
}
