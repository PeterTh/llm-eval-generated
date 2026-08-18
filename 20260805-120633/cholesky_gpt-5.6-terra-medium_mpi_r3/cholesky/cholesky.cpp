#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed in contiguous blocks.  A pivot row is broadcast after
// it has been completed, so every rank can update the part of the next column
// contained in its own rows without replicating the matrix.
struct RowDistribution {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<size_t> first_row;
};

RowDistribution makeDistribution(size_t n, int ranks) {
    RowDistribution distribution;
    distribution.counts.resize(ranks);
    distribution.displacements.resize(ranks);
    distribution.first_row.resize(ranks + 1);

    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    size_t row = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        const size_t rows = base + (static_cast<size_t>(rank) < remainder);
        distribution.first_row[rank] = row;
        distribution.counts[rank] = static_cast<int>(rows * n);
        distribution.displacements[rank] = static_cast<int>(row * n);
        row += rows;
    }
    distribution.first_row[ranks] = n;
    return distribution;
}

int ownerOfRow(size_t row, const RowDistribution& distribution) {
    return static_cast<int>(std::upper_bound(distribution.first_row.begin(),
                                             distribution.first_row.end(), row) -
                            distribution.first_row.begin()) - 1;
}

// Generate the same deterministic symmetric positive definite matrix as the
// original benchmark.  This is intentionally called only by rank zero.
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool distributedCholesky(std::vector<double>& local_a, size_t n,
                         size_t first_local_row, int rank,
                         const RowDistribution& distribution,
                         MPI_Comm communicator) {
    std::vector<double> pivot(n);
    const size_t local_rows = local_a.size() / n;
    for (size_t column = 0; column < n; ++column) {
        const int owner = ownerOfRow(column, distribution);
        const size_t local_pivot = column - first_local_row;
        int positive_definite = 1;

        if (owner == rank) {
            double* const row = local_a.data() + local_pivot * n;
            double sum = 0.0;
            for (size_t k = 0; k < column; ++k) {
                sum += row[k] * row[k];
            }
            const double value = row[column] - sum;
            if (value <= 0.0) {
                positive_definite = 0;
            } else {
                row[column] = std::sqrt(value);
                std::copy_n(row, column + 1, pivot.data());
            }
        }

        MPI_Bcast(&positive_definite, 1, MPI_INT, owner, communicator);
        if (!positive_definite) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", column);
            }
            return false;
        }
        MPI_Bcast(pivot.data(), static_cast<int>(column + 1), MPI_DOUBLE,
                  owner, communicator);

        // Only rows below the pivot are updated.  The inner product is kept in
        // a simple contiguous loop so compilers can vectorize it efficiently.
        for (size_t local_row = 0; local_row < local_rows; ++local_row) {
            const size_t global_row = first_local_row + local_row;
            if (global_row <= column) {
                continue;
            }
            double* const row = local_a.data() + local_row * n;
            double sum = 0.0;
            for (size_t k = 0; k < column; ++k) {
                sum += row[k] * pivot[k];
            }
            row[column] = (row[column] - sum) / pivot[column];
        }
    }

    for (size_t local_row = 0; local_row < local_rows; ++local_row) {
        double* const row = local_a.data() + local_row * n;
        const size_t global_row = first_local_row + local_row;
        std::fill(row + global_row + 1, row + n, 0.0);
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double rel_error = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        max_error = std::max(max_error, error);
        rel_error = std::max(rel_error, error / (std::fabs(A_orig[i]) + 1e-10));
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
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;
    int argument_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed == 0) argument_error = 1;
            else n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else argument_error = 1;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) / n) argument_error = 1;
    if (argument_error) {
        if (rank == 0) {
            std::printf("Invalid matrix size or option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const RowDistribution distribution = makeDistribution(n, ranks);
    const size_t first_local_row = distribution.first_row[rank];
    const size_t local_rows = distribution.first_row[rank + 1] - first_local_row;
    std::vector<double> local_a(local_rows * n);
    std::vector<double> original;
    std::vector<double> global_a;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        global_a.resize(n * n);
        generatePositiveDefiniteMatrix(global_a, n);
        if (validate) original = global_a;
    }
    MPI_Scatterv(rank == 0 ? global_a.data() : nullptr, distribution.counts.data(),
                 distribution.displacements.data(), MPI_DOUBLE, local_a.data(),
                 distribution.counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    global_a.clear();
    global_a.shrink_to_fit();

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(local_a, n, first_local_row, rank,
                                             distribution, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    const bool gather_result = validate || print_results_requested;
    if (gather_result) {
        if (rank == 0) global_a.resize(n * n);
        MPI_Gatherv(local_a.data(), distribution.counts[rank], MPI_DOUBLE,
                    rank == 0 ? global_a.data() : nullptr, distribution.counts.data(),
                    distribution.displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int result = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = max_elapsed > 0.0 ? ops / max_elapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (print_results_requested) print_results(global_a, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateCholesky(global_a, original, n)) std::printf("Validation: PASSED\n");
            else { std::printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
