#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are assigned cyclically.  This is important for Cholesky: as the active
// trailing matrix shrinks, a block distribution leaves the high ranks idle.
static size_t localRowCount(size_t n, int rank, int ranks) {
    return n > static_cast<size_t>(rank)
               ? (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(ranks) + 1
               : 0;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n,
                                           int rank, int ranks) {
    // Recreate the original deterministic B on every rank.  Only the A rows
    // owned by this rank are formed, so the O(n^3) part is fully distributed.
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (double& x : b) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    const size_t rows = localRowCount(n, rank, ranks);
    for (size_t lr = 0; lr < rows; ++lr) {
        const size_t i = static_cast<size_t>(rank) + lr * static_cast<size_t>(ranks);
        double* ai = a.data() + lr * n;
        const double* bi = b.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* bj = b.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
            ai[j] = sum;
        }
        ai[i] += static_cast<double>(n);
    }
}

static bool choleskyDecomposition(std::vector<double>& a, size_t n,
                                  int rank, int ranks, MPI_Comm comm) {
    std::vector<double> column(n);
    std::vector<double> packed(localRowCount(n, rank, ranks));
    std::vector<double> gathered(n);
    std::vector<int> counts(ranks), displacements(ranks);
    const size_t rows = localRowCount(n, rank, ranks);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(ranks));
        double diagonal = 0.0;
        int positive = 1;
        if (rank == owner) {
            const size_t lr = j / static_cast<size_t>(ranks);
            double* row = a.data() + lr * n;
            positive = row[j] > 0.0 && std::isfinite(row[j]);
            if (positive) diagonal = row[j] = std::sqrt(row[j]);
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, comm);
        if (!positive) {
            if (rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, comm);

        size_t packed_count = 0;
        for (size_t lr = 0; lr < rows; ++lr) {
            const size_t i = static_cast<size_t>(rank) + lr * static_cast<size_t>(ranks);
            if (i >= j) {
                double* row = a.data() + lr * n;
                packed[packed_count++] = (i == j) ? diagonal : (row[j] /= diagonal);
            }
        }
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            const size_t rr = static_cast<size_t>(r);
            const size_t first = j + (rr + static_cast<size_t>(ranks) - j % static_cast<size_t>(ranks))
                                      % static_cast<size_t>(ranks);
            counts[r] = first < n ? static_cast<int>((n - 1 - first) / static_cast<size_t>(ranks) + 1) : 0;
            displacements[r] = total;
            total += counts[r];
        }
        MPI_Allgatherv(packed.data(), static_cast<int>(packed_count), MPI_DOUBLE,
                       gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, comm);
        for (int r = 0; r < ranks; ++r) {
            size_t i = j + (static_cast<size_t>(r) + static_cast<size_t>(ranks) - j % static_cast<size_t>(ranks))
                               % static_cast<size_t>(ranks);
            for (int q = 0; q < counts[r]; ++q, i += static_cast<size_t>(ranks))
                column[i] = gathered[static_cast<size_t>(displacements[r] + q)];
        }

        // Symmetric rank-1 update, retaining only the lower triangle.
        for (size_t lr = 0; lr < rows; ++lr) {
            const size_t i = static_cast<size_t>(rank) + lr * static_cast<size_t>(ranks);
            if (i <= j) continue;
            double* row = a.data() + lr * n;
            const double lij = column[i];
            for (size_t k = j + 1; k <= i; ++k) row[k] -= lij * column[k];
        }
    }

    for (size_t lr = 0; lr < rows; ++lr) {
        const size_t i = static_cast<size_t>(rank) + lr * static_cast<size_t>(ranks);
        std::fill(a.begin() + static_cast<std::ptrdiff_t>(lr * n + i + 1),
                  a.begin() + static_cast<std::ptrdiff_t>((lr + 1) * n), 0.0);
    }
    return true;
}

static std::vector<double> gatherMatrix(const std::vector<double>& local, size_t n,
                                        int rank, int ranks, MPI_Comm comm) {
    std::vector<double> full(rank == 0 ? n * n : 0);
    if (rank == 0) {
        for (size_t lr = 0; lr < localRowCount(n, 0, ranks); ++lr)
            std::copy_n(local.data() + lr * n, n, full.data() + lr * static_cast<size_t>(ranks) * n);
        for (int src = 1; src < ranks; ++src) {
            const size_t count = localRowCount(n, src, ranks);
            std::vector<double> packed(count * n);
            MPI_Recv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE, src, 7, comm, MPI_STATUS_IGNORE);
            for (size_t lr = 0; lr < count; ++lr) {
                const size_t row = static_cast<size_t>(src) + lr * static_cast<size_t>(ranks);
                std::copy_n(packed.data() + lr * n, n, full.data() + row * n);
            }
        }
    } else {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 7, comm);
    }
    return full;
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original,
                             size_t n, int rank, int ranks, MPI_Comm comm) {
    const std::vector<double> full = gatherMatrix(l, n, rank, ranks, comm);
    if (rank != 0) return true;
    double max_error = 0.0, rel_error = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        const size_t end = std::min(i, j);
        for (size_t k = 0; k <= end; ++k) sum += full[i*n+k] * full[j*n+k];
        const double error = std::fabs(sum - original[i*n+j]);
        max_error = std::max(max_error, error);
        rel_error = std::max(rel_error, error / (std::fabs(original[i*n+j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, rel_error);
    if (rel_error > 1e-6) std::printf("Validation failed: relative error too large\n");
    return rel_error <= 1e-6;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, print_results_requested = false;
    int exit_code = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::printf("Matrix size must be between 1 and INT_MAX\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nMPI processes: %d\nGenerating positive definite matrix...\n",
                               n, n, validate ? "enabled" : "disabled", ranks);
    std::vector<double> a(localRowCount(n, rank, ranks) * n);
    generatePositiveDefiniteMatrix(a, n, rank, ranks);
    std::vector<double> original;
    if (validate && rank == 0) original = gatherMatrix(a, n, rank, ranks, MPI_COMM_WORLD);
    else if (validate) (void)gatherMatrix(a, n, rank, ranks, MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, n, rank, ranks, MPI_COMM_WORLD);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (rank == 0) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / seconds / 1e9);
    }
    if (print_results_requested) {
        std::vector<double> full = gatherMatrix(a, n, rank, ranks, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "CholeskyL");
    }
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        bool valid = validateCholesky(a, original, n, rank, ranks, MPI_COMM_WORLD);
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", valid_int ? "PASSED" : "FAILED");
        exit_code = valid_int ? 0 : 1;
    }
    MPI_Finalize();
    return exit_code;
}
