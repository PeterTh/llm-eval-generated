#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {
constexpr size_t blockSize = 32;

// Block-cyclic rows balance both the shrinking factorization and generation.
// Each rank stores only its own rows; panel workspaces are O(n * blockSize).
struct Distribution {
    size_t n;
    int rank, ranks;
    std::vector<size_t> rows;
    std::vector<size_t> localIndex;

    Distribution(size_t size, int id, int count)
        : n(size), rank(id), ranks(count), localIndex(size) {
        for (size_t i = 0; i < n; ++i) {
            if (owner(i) == rank) {
                localIndex[i] = rows.size();
                rows.push_back(i);
            }
        }
    }
    int owner(size_t row) const {
        return static_cast<int>((row / blockSize) % ranks);
    }
};

void generatePositiveDefiniteMatrix(std::vector<double>& A,
                                   const Distribution& d) {
    const size_t n = d.n;
    std::vector<double> B(d.rows.size() * n);
    unsigned int seed = 42;
    // Advance the identical rand_r stream on each rank, retaining only local rows.
    // This preserves the original input without replicating the dense matrix.
    for (size_t i = 0; i < n; ++i) {
        double* row = d.owner(i) == d.rank ? B.data() + d.localIndex[i] * n : nullptr;
        for (size_t k = 0; k < n; ++k) {
            const double value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
            if (row) row[k] = value;
        }
    }
    std::vector<double> panel(blockSize * n);
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const int owner = d.owner(first);
        if (d.rank == owner) {
            std::copy_n(B.data() + d.localIndex[first] * n, width * n, panel.data());
        }
        MPI_Bcast(panel.data(), static_cast<int>(width * n), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            if (i < first) continue;
            const double* bi = B.data() + r * n;
            for (size_t j = first; j < std::min(first + width, i + 1); ++j) {
                const double* bj = panel.data() + (j - first) * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
                A[r * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const Distribution& d) {
    const size_t n = d.n;
    std::vector<double> diagonal(blockSize * blockSize);
    std::vector<double> packed(d.rows.size() * blockSize);
    std::vector<double> gathered(n * blockSize);
    // Transposing the panel makes the trailing updates contiguous and vectorizable.
    std::vector<double> panel(n * blockSize);
    std::vector<int> counts(d.ranks), offsets(d.ranks), cursors(d.ranks);

    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const size_t end = first + width;
        const int owner = d.owner(first);
        int failed = -1;
        if (d.rank == owner) {
            // Factor the diagonal tile, whose previous panels are already applied.
            for (size_t i = 0; i < width; ++i) {
                double* ai = A.data() + d.localIndex[first + i] * n + first;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += diagonal[i * width + k] * diagonal[j * width + k];
                    const double value = ai[j] - sum;
                    if (i == j) {
                        if (value <= 0.0) { failed = static_cast<int>(first + i); break; }
                        diagonal[i * width + j] = std::sqrt(value);
                    } else {
                        diagonal[i * width + j] = value / diagonal[j * width + j];
                    }
                    ai[j] = diagonal[i * width + j];
                }
                if (failed >= 0) break;
                std::fill(diagonal.data() + i * width + i + 1,
                          diagonal.data() + (i + 1) * width, 0.0);
            }
        }
        // All ranks leave together if a diagonal is not positive definite.
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed >= 0) {
            if (d.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
            return false;
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        // Solve L21 * L11^T = A21 independently for each locally owned row.
        size_t used = 0;
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            if (i < first) continue;
            double* ai = A.data() + r * n + first;
            if (i >= end) {
                for (size_t j = 0; j < width; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) sum += ai[k] * diagonal[j * width + k];
                    ai[j] = (ai[j] - sum) / diagonal[j * width + j];
                }
            }
            for (size_t j = 0; j < width; ++j)
                packed[used++] = first + j <= i ? ai[j] : 0.0;
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = first; i < n; ++i) counts[d.owner(i)] += static_cast<int>(width);
        int total = 0;
        for (int p = 0; p < d.ranks; ++p) {
            offsets[p] = total;
            total += counts[p];
        }
        MPI_Allgatherv(packed.data(), static_cast<int>(used), MPI_DOUBLE,
                       gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        cursors = offsets;
        for (size_t i = first; i < n; ++i) {
            int& cursor = cursors[d.owner(i)];
            for (size_t k = 0; k < width; ++k) panel[k * n + i] = gathered[cursor++];
        }

        // Right-looking rank-k update of the lower triangle. Column tiling keeps
        // each output segment in cache while all panel columns are applied.
        constexpr size_t columnTile = 256;
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            if (i < end) continue;
            for (size_t j = end; j <= i; j += columnTile) {
                const size_t length = std::min(columnTile, i - j + 1);
                double* out = A.data() + r * n + j;
                for (size_t k = 0; k < width; ++k) {
                    const double lik = panel[k * n + i];
                    const double* values = panel.data() + k * n + j;
                    for (size_t t = 0; t < length; ++t) out[t] -= lik * values[t];
                }
            }
        }
    }
    // The upper triangle was initialized to zero and never modified.
    return true;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& original, const Distribution& d) {
    const size_t n = d.n;
    std::vector<double> panel(blockSize * n);
    double errors[2] = {0.0, 0.0};
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const int owner = d.owner(first);
        if (d.rank == owner)
            std::copy_n(L.data() + d.localIndex[first] * n, width * n, panel.data());
        MPI_Bcast(panel.data(), static_cast<int>(width * n), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            if (i < first) continue;
            for (size_t j = first; j < std::min(first + width, i + 1); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k <= j; ++k)
                    sum += L[r * n + k] * panel[(j - first) * n + k];
                const double reference = original[r * n + j];
                const double error = std::fabs(sum - reference);
                errors[0] = std::max(errors[0], error);
                errors[1] = std::max(errors[1], error / (std::fabs(reference) + 1e-10));
            }
        }
    }
    // Both triangles have the same errors because A and L*L^T are symmetric.
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printDistributedResults(const std::vector<double>& A, const Distribution& d) {
    std::vector<double> result;
    if (d.rank == 0) result.resize(d.n * d.n);
    // Transfer one row block at a time, avoiding MPI's INT_MAX gather limit.
    for (size_t first = 0; first < d.n; first += blockSize) {
        const size_t width = std::min(blockSize, d.n - first);
        const int owner = d.owner(first);
        const int count = static_cast<int>(width * d.n);
        if (owner == 0) {
            if (d.rank == 0)
                std::copy_n(A.data() + d.localIndex[first] * d.n, width * d.n,
                            result.data() + first * d.n);
        } else if (d.rank == owner) {
            MPI_Send(A.data() + d.localIndex[first] * d.n, count, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
        } else if (d.rank == 0) {
            MPI_Recv(result.data() + first * d.n, count, MPI_DOUBLE, owner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if (d.rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(argument, &end, 10);
            if (errno || end == argument || *end || *argument == '-' ||
                parsed > static_cast<unsigned long long>(INT_MAX) / blockSize ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Distribution d(n, rank, ranks);
    std::vector<double> A(d.rows.size() * n);
    generatePositiveDefiniteMatrix(A, d);
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, d);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(A, d);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(A, original, d);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
