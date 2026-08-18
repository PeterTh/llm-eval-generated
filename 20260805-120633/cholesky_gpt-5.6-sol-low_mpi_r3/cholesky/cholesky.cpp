#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

struct RowDistribution {
    size_t n;
    int size;
    int rank;
    size_t first;
    size_t count;

    RowDistribution(size_t n_, int size_, int rank_)
        : n(n_), size(size_), rank(rank_) {
        const size_t base = n / static_cast<size_t>(size);
        const size_t extra = n % static_cast<size_t>(size);
        count = base + (static_cast<size_t>(rank) < extra);
        first = static_cast<size_t>(rank) * base +
                std::min(static_cast<size_t>(rank), extra);
    }

    int owner(size_t row) const {
        const size_t large = n % static_cast<size_t>(size);
        const size_t large_rows = large * (n / static_cast<size_t>(size) + 1);
        if (row < large_rows)
            return static_cast<int>(row / (n / static_cast<size_t>(size) + 1));
        const size_t base = n / static_cast<size_t>(size);
        return base == 0 ? static_cast<int>(row)
                         : static_cast<int>(large + (row - large_rows) / base);
    }
};

// Each rank creates the same B, but forms only its distributed rows of A.  This
// avoids communicating the input matrix and exactly retains the original RNG
// stream and summation order.
void generatePositiveDefiniteMatrix(std::vector<double>& localA,
                                    const RowDistribution& dist) {
    const size_t n = dist.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    for (size_t li = 0; li < dist.count; ++li) {
        const size_t i = dist.first + li;
        double* const arow = localA.data() + li * n;
        const double* const bi = B.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* const bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += bi[k] * bj[k];
            arow[j] = sum;
        }
        arow[i] += static_cast<double>(n);
    }
}

bool choleskyDecomposition(std::vector<double>& localA,
                           const RowDistribution& dist, MPI_Comm comm) {
    const size_t n = dist.n;
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int root = dist.owner(j);
        if (dist.rank == root) {
            double* const row = localA.data() + (j - dist.first) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += row[k] * row[k];
            const double val = row[j] - sum;
            if (val <= 0.0 || !std::isfinite(val)) {
                // A non-finite diagonal is an in-band failure indication,
                // avoiding a second collective on every pivot.
                pivot[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                row[j] = std::sqrt(val);
                std::copy_n(row, j + 1, pivot.data());
                std::fill(row + j + 1, row + n, 0.0);
            }
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, root, comm);
        if (!std::isfinite(pivot[j])) {
            if (dist.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }

        const size_t begin = std::max(dist.first, j + 1);
        for (size_t i = begin; i < dist.first + dist.count; ++i) {
            double* const row = localA.data() + (i - dist.first) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    return true;
}

std::vector<double> gatherMatrix(const std::vector<double>& local,
                                 const RowDistribution& dist, MPI_Comm comm) {
    std::vector<int> counts(dist.size), offsets(dist.size);
    for (int r = 0; r < dist.size; ++r) {
        RowDistribution d(dist.n, dist.size, r);
        counts[r] = static_cast<int>(d.count * dist.n);
        offsets[r] = static_cast<int>(d.first * dist.n);
    }
    std::vector<double> result;
    if (dist.rank == 0) result.resize(dist.n * dist.n);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                dist.rank == 0 ? result.data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, comm);
    return result;
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
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(original[i * n + j]) + 1e-10));
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

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else parseStatus = 1;
    }
    if (parseStatus || n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) / n) {
        if (rank == 0) {
            if (parseStatus == 1) std::printf("Unknown or invalid option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\nMPI processes: %d\n", validate ? "enabled" : "disabled", size);
        std::printf("Generating positive definite matrix...\n");
    }
    RowDistribution dist(n, size, rank);
    std::vector<double> localA(dist.count * n);
    generatePositiveDefiniteMatrix(localA, dist);
    std::vector<double> original;
    if (validate) original = localA;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, dist, MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long ms = static_cast<long>(elapsed * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Computation time: %ld ms\n", ms);
        std::printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
    }

    std::vector<double> A, A_orig;
    if (printResults || validate) A = gatherMatrix(localA, dist, MPI_COMM_WORLD);
    if (validate) A_orig = gatherMatrix(original, dist, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        if (printResults) print_results(A, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
