#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are distributed cyclically.  Unlike a contiguous distribution, this
// keeps all ranks busy as the active lower-right part of the matrix shrinks.
struct DistributedMatrix {
    size_t n;
    int rank;
    int ranks;
    size_t local_rows;
    std::vector<double> data;

    DistributedMatrix(size_t n_, int rank_, int ranks_)
        : n(n_), rank(rank_), ranks(ranks_),
          local_rows(static_cast<size_t>(rank_) < n_
                         ? 1 + (n_ - 1 - static_cast<size_t>(rank_)) /
                                   static_cast<size_t>(ranks_)
                         : 0),
          data(local_rows * n_) {}

    double* row(size_t local_row) { return data.data() + local_row * n; }
    const double* row(size_t local_row) const {
        return data.data() + local_row * n;
    }
    size_t global_row(size_t local_row) const {
        return static_cast<size_t>(rank) + local_row * static_cast<size_t>(ranks);
    }
};

// MPI counts are ints.  Chunking also makes all bulk operations valid for
// matrices whose rows are larger than INT_MAX elements.
static void broadcastDoubles(double* values, size_t count, int root,
                             MPI_Comm comm) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_Bcast(values, chunk, MPI_DOUBLE, root, comm);
        values += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

// Generate the exact same pseudo-random B as the original benchmark, retaining
// only locally owned rows.  A = B B^T is then formed cooperatively.  Only A's
// lower triangle is needed by both factorization and validation.
static void generatePositiveDefiniteMatrix(DistributedMatrix& A,
                                           MPI_Comm comm) {
    std::vector<double> local_b(A.local_rows * A.n);
    unsigned int seed = 42;
    for (size_t i = 0; i < A.n; ++i) {
        const bool owned = static_cast<int>(i % static_cast<size_t>(A.ranks)) == A.rank;
        double* destination = owned ? local_b.data() + (i / A.ranks) * A.n : nullptr;
        for (size_t k = 0; k < A.n; ++k) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (owned) destination[k] = value;
        }
    }

    std::vector<double> b_row(A.n);
    for (size_t j = 0; j < A.n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(A.ranks));
        if (A.rank == owner) {
            std::copy_n(local_b.data() + (j / A.ranks) * A.n, A.n, b_row.data());
        }
        broadcastDoubles(b_row.data(), A.n, owner, comm);

        size_t first = 0;
        if (j >= static_cast<size_t>(A.rank))
            first = (j - static_cast<size_t>(A.rank) + A.ranks - 1) / A.ranks;
        for (size_t lr = first; lr < A.local_rows; ++lr) {
            const double* const own_b = local_b.data() + lr * A.n;
            double sum = 0.0;
            for (size_t k = 0; k < A.n; ++k) sum += own_b[k] * b_row[k];
            const size_t i = A.global_row(lr);
            A.row(lr)[j] = sum + (i == j ? static_cast<double>(A.n) : 0.0);
        }
    }
}

static bool choleskyDecomposition(DistributedMatrix& A, size_t& failed_diagonal,
                                  MPI_Comm comm) {
    std::vector<double> pivot(A.n);
    for (size_t j = 0; j < A.n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(A.ranks));
        int positive = 1;
        if (A.rank == owner) {
            double* const diagonal_row = A.row(j / A.ranks);
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += diagonal_row[k] * diagonal_row[k];
            const double value = diagonal_row[j] - sum;
            positive = value > 0.0;
            if (positive) {
                diagonal_row[j] = std::sqrt(value);
                std::copy_n(diagonal_row, j + 1, pivot.data());
            }
        }
        MPI_Bcast(&positive, 1, MPI_INT, owner, comm);
        if (!positive) {
            failed_diagonal = j;
            return false;
        }
        broadcastDoubles(pivot.data(), j + 1, owner, comm);

        size_t first = 0;
        if (j >= static_cast<size_t>(A.rank))
            first = (j - static_cast<size_t>(A.rank)) / A.ranks + 1;
        for (size_t lr = first; lr < A.local_rows; ++lr) {
            double* const current = A.row(lr);
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += current[k] * pivot[k];
            current[j] = (current[j] - sum) / pivot[j];
        }
    }

    // Preserve the original program's full row-major lower-triangular output.
    for (size_t lr = 0; lr < A.local_rows; ++lr) {
        const size_t i = A.global_row(lr);
        std::fill(A.row(lr) + i + 1, A.row(lr) + A.n, 0.0);
    }
    return true;
}

static bool validateCholesky(const DistributedMatrix& L,
                             const std::vector<double>& original,
                             MPI_Comm comm) {
    std::vector<double> pivot(L.n);
    double local_absolute = 0.0;
    double local_relative = 0.0;

    for (size_t j = 0; j < L.n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(L.ranks));
        if (L.rank == owner)
            std::copy_n(L.row(j / L.ranks), j + 1, pivot.data());
        broadcastDoubles(pivot.data(), j + 1, owner, comm);

        size_t first = 0;
        if (j >= static_cast<size_t>(L.rank))
            first = (j - static_cast<size_t>(L.rank) + L.ranks - 1) / L.ranks;
        for (size_t lr = first; lr < L.local_rows; ++lr) {
            const double* const row = L.row(lr);
            double reconstructed = 0.0;
            for (size_t k = 0; k <= j; ++k) reconstructed += row[k] * pivot[k];
            const double expected = original[lr * L.n + j];
            const double error = std::fabs(reconstructed - expected);
            local_absolute = std::max(local_absolute, error);
            local_relative = std::max(local_relative,
                                      error / (std::fabs(expected) + 1e-10));
        }
    }

    double global_absolute = 0.0;
    double global_relative = 0.0;
    MPI_Reduce(&local_absolute, &global_absolute, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_relative, &global_relative, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (L.rank == 0) {
        std::printf("Max absolute error: %.10e\n", global_absolute);
        std::printf("Max relative error: %.10e\n", global_relative);
    }
    int valid = 1;
    if (L.rank == 0) valid = global_relative <= 1e-6;
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

static std::vector<double> gatherMatrix(const DistributedMatrix& A,
                                        MPI_Comm comm) {
    constexpr int row_tag = 17;
    std::vector<double> result;
    if (A.rank == 0) {
        result.resize(A.n * A.n);
        for (size_t lr = 0; lr < A.local_rows; ++lr)
            std::copy_n(A.row(lr), A.n,
                        result.data() + A.global_row(lr) * A.n);
        for (int source = 1; source < A.ranks; ++source) {
            for (size_t i = static_cast<size_t>(source); i < A.n;
                 i += static_cast<size_t>(A.ranks)) {
                size_t remaining = A.n;
                double* destination = result.data() + i * A.n;
                while (remaining != 0) {
                    const int chunk = static_cast<int>(std::min(
                        remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(destination, chunk, MPI_DOUBLE, source, row_tag,
                             comm, MPI_STATUS_IGNORE);
                    destination += chunk;
                    remaining -= static_cast<size_t>(chunk);
                }
            }
        }
    } else {
        for (size_t lr = 0; lr < A.local_rows; ++lr) {
            size_t remaining = A.n;
            const double* source = A.row(lr);
            while (remaining != 0) {
                const int chunk = static_cast<int>(std::min(
                    remaining, static_cast<size_t>(INT_MAX)));
                MPI_Send(source, chunk, MPI_DOUBLE, 0, row_tag, comm);
                source += chunk;
                remaining -= static_cast<size_t>(chunk);
            }
        }
    }
    return result;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;
    int exit_code = 0;
    bool show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) std::printf("Invalid matrix size: %s\n", argv[i]);
                exit_code = 1;
            } else {
                n = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            exit_code = 1;
        }
    }
    if (n > std::numeric_limits<size_t>::max() / n) {
        if (rank == 0) std::printf("Matrix size is too large\n");
        exit_code = 1;
    }
    if (show_help || exit_code != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return show_help ? 0 : exit_code;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    DistributedMatrix A(n, rank, ranks);
    generatePositiveDefiniteMatrix(A, comm);
    std::vector<double> original;
    if (validate) original = A.data;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    size_t failed_diagonal = 0;
    const bool success = choleskyDecomposition(A, failed_diagonal, comm);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (!success) {
        if (rank == 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                        failed_diagonal);
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = seconds > 0.0 ? operations / seconds / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (print_results_requested) {
        std::vector<double> gathered = gatherMatrix(A, comm);
        if (rank == 0) print_results(gathered, "CholeskyL");
    }

    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateCholesky(A, original, comm);
        if (rank == 0)
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exit_code = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exit_code;
}
