#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are assigned cyclically: row i belongs to rank i % nranks.  Unlike a
// contiguous distribution, this keeps the work in the shrinking trailing
// matrix balanced throughout the factorization.
static size_t localRowCount(size_t n, int rank, int nranks) {
    return n > static_cast<size_t>(rank)
               ? (n - 1 - static_cast<size_t>(rank)) / nranks + 1
               : 0;
}

static double randomValue(size_t index) {
    // Stateless deterministic generator: every rank can generate its rows
    // independently, with no serial random-number stream or root bottleneck.
    unsigned long long x = static_cast<unsigned long long>(index) + 42ULL;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n,
                                           int rank, int nranks) {
    const size_t localRows = localRowCount(n, rank, nranks);
    std::vector<double> localB(localRows * n);
    for (size_t lr = 0; lr < localRows; ++lr) {
        const size_t row = static_cast<size_t>(rank) + lr * nranks;
        for (size_t k = 0; k < n; ++k)
            localB[lr * n + k] = randomValue(row * n + k);
    }

    for (size_t lr = 0; lr < localRows; ++lr) {
        const double* bi = localB.data() + lr * n;
        double* ai = A.data() + lr * n;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += bi[k] * randomValue(j * n + k);
            ai[j] = sum;
        }
        const size_t row = static_cast<size_t>(rank) + lr * nranks;
        ai[row] += static_cast<double>(n);
    }
}

static bool choleskyDecomposition(std::vector<double>& A, size_t n,
                                  int rank, int nranks) {
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % nranks);
        int ok = 1;
        if (rank == owner) {
            double* row = A.data() + (j / nranks) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * row[k];
            const double val = row[j] - sum;
            if (val <= 0.0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                ok = 0;
            } else {
                row[j] = std::sqrt(val);
                std::copy_n(row, j + 1, pivot.data());
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) return false;
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        const size_t localRows = localRowCount(n, rank, nranks);
        for (size_t lr = 0; lr < localRows; ++lr) {
            const size_t i = static_cast<size_t>(rank) + lr * nranks;
            if (i <= j) continue;
            double* row = A.data() + lr * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
            row[j] = (row[j] - sum) / pivot[j];
        }
    }
    // Preserve the original output semantics: explicitly zero the upper half.
    const size_t localRows = localRowCount(n, rank, nranks);
    for (size_t lr = 0; lr < localRows; ++lr) {
        const size_t row = static_cast<size_t>(rank) + lr * nranks;
        std::fill(A.begin() + lr * n + row + 1, A.begin() + (lr + 1) * n, 0.0);
    }
    return true;
}

static std::vector<double> collectRows(const std::vector<double>& local, size_t n,
                                       int rank, int nranks, bool allRanks) {
    std::vector<int> counts(nranks), displs(nranks);
    int offset = 0;
    for (int r = 0; r < nranks; ++r) {
        counts[r] = static_cast<int>(localRowCount(n, r, nranks) * n);
        displs[r] = offset;
        offset += counts[r];
    }
    std::vector<double> packed(n * n), result;
    if (allRanks)
        MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, packed.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank == 0 ? packed.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (allRanks || rank == 0) {
        result.resize(n * n);
        for (int r = 0; r < nranks; ++r)
            for (size_t lr = 0; lr < localRowCount(n, r, nranks); ++lr)
                std::copy_n(packed.data() + displs[r] + lr * n, n,
                            result.data() + (static_cast<size_t>(r) + lr * nranks) * n);
    }
    return result;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& Aorig, size_t n,
                             int rank, int nranks) {
    const std::vector<double> fullL = collectRows(L, n, rank, nranks, true);
    double localMaxAbs = 0.0, localMaxRel = 0.0;
    const size_t localRows = localRowCount(n, rank, nranks);
    for (size_t lr = 0; lr < localRows; ++lr) {
        const double* ai = Aorig.data() + lr * n;
        const size_t i = static_cast<size_t>(rank) + lr * nranks;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k)
                sum += fullL[i * n + k] * fullL[j * n + k];
            const double error = std::fabs(sum - ai[j]);
            localMaxAbs = std::max(localMaxAbs, error);
            localMaxRel = std::max(localMaxRel, error / (std::fabs(ai[j]) + 1e-10));
        }
    }
    double maxAbs = 0.0, maxRel = 0.0;
    MPI_Reduce(&localMaxAbs, &maxAbs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMaxRel, &maxRel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxAbs, maxRel);
        valid = maxRel <= 1e-6;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) || n > std::sqrt(static_cast<double>(std::numeric_limits<int>::max()))) {
        if (!rank) std::printf("Matrix size must be between 1 and %d\n", static_cast<int>(std::sqrt(std::numeric_limits<int>::max())));
        MPI_Finalize(); return 1;
    }
    if (!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nMPI ranks: %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, nranks, validate ? "enabled" : "disabled");
    std::vector<double> A(localRowCount(n, rank, nranks) * n), Aorig;
    generatePositiveDefiniteMatrix(A, n, rank, nranks);
    if (validate) Aorig = A;
    if (!rank) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, rank, nranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (!rank) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (!rank) { std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0); std::printf("Performance: %.3f GFLOPS\n", (double(n) * n * n / 3.0) / maxElapsed / 1e9); }
    if (printResults) {
        std::vector<double> result = collectRows(A, n, rank, nranks, false);
        if (!rank) print_results(result, "CholeskyL");
    }
    if (validate && !validateCholesky(A, Aorig, n, rank, nranks)) exitCode = 1;
    MPI_Finalize();
    return exitCode;
}
