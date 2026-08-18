#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

struct RowPartition {
    std::vector<int> counts;
    std::vector<int> displacements;
    int first_row;
    int local_rows;
};

RowPartition partitionRows(const size_t n, const int ranks, const int rank) {
    RowPartition result{{}, {}, 0, 0};
    result.counts.resize(ranks);
    result.displacements.resize(ranks);
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    int offset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        result.counts[r] = static_cast<int>(rows);
        result.displacements[r] = offset;
        if (r == rank) {
            result.first_row = offset;
            result.local_rows = static_cast<int>(rows);
        }
        offset += static_cast<int>(rows);
    }
    return result;
}

int ownerOfRow(const int row, const RowPartition& partition) {
    const auto it = std::upper_bound(partition.displacements.begin(),
                                     partition.displacements.end(), row);
    return static_cast<int>(it - partition.displacements.begin()) - 1;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool distributedCholesky(std::vector<double>& local_a, const size_t n,
                         const RowPartition& partition, const int rank,
                         const MPI_Comm comm) {
    std::vector<double> column(n, 0.0);
    std::vector<double> local_column(static_cast<size_t>(partition.local_rows));
    std::vector<int> element_counts(partition.counts.size());
    std::vector<int> element_displacements(partition.displacements.size());
    for (size_t r = 0; r < partition.counts.size(); ++r) {
        element_counts[r] = partition.counts[r];
        element_displacements[r] = partition.displacements[r];
    }

    for (int k = 0; k < static_cast<int>(n); ++k) {
        const int owner = ownerOfRow(k, partition);
        double diagonal = 0.0;
        int positive = 1;
        if (rank == owner) {
            diagonal = local_a[static_cast<size_t>(k - partition.first_row) * n + k];
            if (diagonal <= 0.0) positive = 0;
            else diagonal = std::sqrt(diagonal);
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, comm);
        if (!positive) return false;
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, comm);

        const int first = std::max(k + 1, partition.first_row);
        for (int i = first; i < partition.first_row + partition.local_rows; ++i)
            local_a[static_cast<size_t>(i - partition.first_row) * n + k] /= diagonal;
        if (rank == owner)
            local_a[static_cast<size_t>(k - partition.first_row) * n + k] = diagonal;

        // Every rank contributes its portion of L(:, k).  The all-gather lets
        // local trailing rows be updated without replicating the whole matrix.
        for (int local_i = 0; local_i < partition.local_rows; ++local_i)
            local_column[local_i] = local_a[static_cast<size_t>(local_i) * n + k];
        MPI_Allgatherv(local_column.data(), partition.local_rows, MPI_DOUBLE, column.data(), element_counts.data(),
                       element_displacements.data(), MPI_DOUBLE, comm);

        for (int i = first; i < partition.first_row + partition.local_rows; ++i) {
            double* const row = local_a.data() + static_cast<size_t>(i - partition.first_row) * n;
            const double lik = row[k];
            for (int j = k + 1; j <= i; ++j)
                row[j] -= lik * column[j];
        }
    }

    for (int local_i = 0; local_i < partition.local_rows; ++local_i) {
        double* const row = local_a.data() + static_cast<size_t>(local_i) * n;
        const int global_i = partition.first_row + local_i;
        std::fill(row + global_i + 1, row + n, 0.0);
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double max_error = 0.0, rel_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k) sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - A_orig[i * n + j]);
            max_error = std::max(max_error, error);
            rel_error = std::max(rel_error, error / (std::fabs(A_orig[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", rel_error);
    if (rel_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, print_results_requested = false;
    int argument_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else argument_error = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &argument_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (argument_error || n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) { std::printf("Invalid option or matrix size\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    const RowPartition partition = partitionRows(n, ranks, rank);
    std::vector<int> matrix_counts(ranks), matrix_displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        matrix_counts[r] = partition.counts[r] * static_cast<int>(n);
        matrix_displacements[r] = partition.displacements[r] * static_cast<int>(n);
    }
    std::vector<double> full_a, original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        full_a.resize(n * n);
        generatePositiveDefiniteMatrix(full_a, n);
        if (validate) original = full_a;
    }
    std::vector<double> local_a(static_cast<size_t>(partition.local_rows) * n);
    MPI_Scatterv(rank == 0 ? full_a.data() : nullptr, matrix_counts.data(), matrix_displacements.data(), MPI_DOUBLE,
                 local_a.data(), matrix_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    full_a.clear(); full_a.shrink_to_fit();

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(local_a, n, partition, rank, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int success_int = success ? 1 : 0, all_success = 0;
    MPI_Allreduce(&success_int, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }

    // Keep the factor distributed in the normal benchmark path.  Materialize
    // it on rank zero only for the existing validation/result-output options.
    std::vector<double> factor;
    if (validate || print_results_requested) {
        if (rank == 0) factor.resize(n * n);
        MPI_Gatherv(local_a.data(), matrix_counts[rank], MPI_DOUBLE, rank == 0 ? factor.data() : nullptr,
                    matrix_counts.data(), matrix_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int result = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = static_cast<double>(n) * n * n / 3.0 / max_elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (print_results_requested) print_results(factor, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateCholesky(factor, original, n) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
