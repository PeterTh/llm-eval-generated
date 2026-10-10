#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

// Block-cyclic rows balance the shrinking trailing matrix. Only owned rows
// are stored; the O(n * block) communication buffers are reused each panel.
struct Distribution {
    size_t n, block;
    int rank, ranks;
    std::vector<size_t> rows;

    static size_t blockSize(size_t n, int ranks) {
        return std::min<size_t>(64, std::max<size_t>(1, n / (4 * size_t(ranks))));
    }
    Distribution(size_t size, int id, int count)
        : n(size), block(blockSize(size, count)),
          rank(id), ranks(count) {
        for (size_t first = size_t(rank) * block; first < n;
             first += size_t(ranks) * block) {
            for (size_t i = first; i < std::min(n, first + block); ++i)
                rows.push_back(i);
        }
    }
    int owner(size_t i) const { return int((i / block) % size_t(ranks)); }
    size_t local(size_t i) const {
        return (i / (block * size_t(ranks))) * block + i % block;
    }
};

// Preserve the original rand_r stream exactly, retaining only owned B rows.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Distribution& d) {
    std::vector<double> B(A.size());
    unsigned int seed = 42;
    for (size_t i = 0; i < d.n; ++i) {
        double* row = d.owner(i) == d.rank ? B.data() + d.local(i) * d.n : nullptr;
        for (size_t k = 0; k < d.n; ++k) {
            const double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
            if (row) row[k] = value;
        }
    }
    std::vector<double> panel(d.block * d.n);
    for (size_t first = 0; first < d.n; first += d.block) {
        const size_t width = std::min(d.block, d.n - first);
        const int root = d.owner(first);
        if (d.rank == root)
            std::copy_n(B.data() + d.local(first) * d.n, width * d.n, panel.data());
        MPI_Bcast(panel.data(), int(width * d.n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            const double* bi = B.data() + r * d.n;
            for (size_t j = first; j < std::min(first + width, i + 1); ++j) {
                const double* bj = panel.data() + (j - first) * d.n;
                double sum = 0.0;
                for (size_t k = 0; k < d.n; ++k) sum += bi[k] * bj[k];
                A[r * d.n + j] = sum + (i == j ? double(d.n) : 0.0);
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const Distribution& d) {
    std::vector<double> diagonal(d.block * d.block);
    std::vector<double> packed(d.n * d.block), panel(d.n * d.block);
    std::vector<int> counts(d.ranks), offsets(d.ranks), cursor(d.ranks);
    for (size_t first = 0; first < d.n; first += d.block) {
        const size_t width = std::min(d.block, d.n - first);
        const size_t end = first + width;
        const int root = d.owner(first);
        int success = 1;
        if (d.rank == root) {
            for (size_t i = 0; i < width; ++i)
                std::copy_n(A.data() + d.local(first + i) * d.n + first,
                            width, diagonal.data() + i * width);
            // Factor the small diagonal block of the current Schur complement.
            for (size_t j = 0; j < width; ++j) {
                double value = diagonal[j * width + j];
                for (size_t k = 0; k < j; ++k)
                    value -= diagonal[j * width + k] * diagonal[j * width + k];
                if (!(value > 0.0) || !std::isfinite(value)) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", first + j);
                    success = 0;
                    break;
                }
                diagonal[j * width + j] = std::sqrt(value);
                for (size_t i = j + 1; i < width; ++i) {
                    double v = diagonal[i * width + j];
                    for (size_t k = 0; k < j; ++k)
                        v -= diagonal[i * width + k] * diagonal[j * width + k];
                    diagonal[i * width + j] = v / diagonal[j * width + j];
                }
            }
            if (success)
                for (size_t i = 0; i < width; ++i)
                    std::copy_n(diagonal.data() + i * width, i + 1,
                                A.data() + d.local(first + i) * d.n + first);
        }
        MPI_Bcast(&success, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (!success) return false;
        MPI_Bcast(diagonal.data(), int(width * width), MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Solve A21 * L11^T = A21 independently on every owned trailing row.
        const size_t start = size_t(std::lower_bound(d.rows.begin(), d.rows.end(), end) - d.rows.begin());
        for (size_t r = start; r < d.rows.size(); ++r) {
            double* row = A.data() + r * d.n + first;
            for (size_t j = 0; j < width; ++j) {
                double value = row[j];
                for (size_t k = 0; k < j; ++k) value -= row[k] * diagonal[j * width + k];
                row[j] = value / diagonal[j * width + j];
            }
        }
        if (end == d.n) continue;
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < d.n; ++i) counts[d.owner(i)] += int(width);
        offsets[0] = 0;
        for (int p = 1; p < d.ranks; ++p) offsets[p] = offsets[p - 1] + counts[p - 1];
        // MPI_IN_PLACE avoids a separate send buffer. Rank segments contain
        // owned panel rows in increasing global row order.
        size_t pos = size_t(offsets[d.rank]);
        for (size_t r = start; r < d.rows.size(); ++r) {
            std::copy_n(A.data() + r * d.n + first, width, packed.data() + pos);
            pos += width;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, packed.data(), counts.data(),
                       offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        cursor = offsets;
        for (size_t i = end; i < d.n; ++i) {
            int& offset = cursor[d.owner(i)];
            for (size_t k = 0; k < width; ++k) panel[k * d.n + i] = packed[size_t(offset++)];
        }
        // Cache-tiled rank-k update of the lower trailing triangle. The inner
        // loop is contiguous and vectorizable without changing FP flags.
        constexpr size_t tile = 128;
        for (size_t r = start; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            double* row = A.data() + r * d.n;
            for (size_t j0 = end; j0 <= i; j0 += tile) {
                const size_t jend = std::min(i + 1, j0 + tile);
                for (size_t k = 0; k < width; ++k) {
                    const double value = row[first + k];
                    const double* column = panel.data() + k * d.n;
                    for (size_t j = j0; j < jend; ++j) row[j] -= value * column[j];
                }
            }
        }
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      const Distribution& d) {
    double errors[2] = {0.0, 0.0};
    std::vector<double> panel(d.block * d.n);
    for (size_t first = 0; first < d.n; first += d.block) {
        const size_t width = std::min(d.block, d.n - first);
        const int root = d.owner(first);
        if (d.rank == root)
            std::copy_n(L.data() + d.local(first) * d.n, width * d.n, panel.data());
        MPI_Bcast(panel.data(), int(width * d.n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        // Symmetry makes checking the lower triangle equivalent to checking
        // both halves, and avoids reconstructing a second dense matrix.
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            for (size_t j = first; j < std::min(first + width, i + 1); ++j) {
                const double* li = L.data() + r * d.n;
                const double* lj = panel.data() + (j - first) * d.n;
                double sum = 0.0;
                for (size_t k = 0; k <= j; ++k) sum += li[k] * lj[k];
                const double orig = original[r * d.n + j];
                const double error = std::fabs(sum - orig);
                if (!std::isfinite(error)) {
                    errors[0] = errors[1] = std::numeric_limits<double>::infinity();
                } else {
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(orig) + 1e-10));
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printDistributedResults(const std::vector<double>& A, const Distribution& d) {
    std::vector<double> full;
    if (d.rank == 0) full.resize(d.n * d.n);
    for (size_t first = 0; first < d.n; first += d.block) {
        const int owner = d.owner(first);
        const size_t count = std::min(d.block, d.n - first) * d.n;
        if (owner == 0 && d.rank == 0)
            std::copy_n(A.data() + d.local(first) * d.n, count, full.data() + first * d.n);
        else if (d.rank == owner)
            MPI_Send(A.data() + d.local(first) * d.n, int(count), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        else if (d.rank == 0)
            MPI_Recv(full.data() + first * d.n, int(count), MPI_DOUBLE, owner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (d.rank == 0) print_results(full, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            n = size_t(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    int result = 0;
    try {
        if (n > size_t(INT_MAX) / Distribution::blockSize(n, ranks) ||
            (n && n > std::numeric_limits<size_t>::max() / n / sizeof(double))) {
            if (rank == 0) printf("Matrix size exceeds supported storage or MPI count limits\n");
            MPI_Finalize();
            return 1;
        }
        const Distribution d(n, rank, ranks);
        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\n");
            printf("Matrix size: %zu x %zu\n", n, n);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Generating positive definite matrix...\n");
        }
        std::vector<double> A(d.rows.size() * n);
        generatePositiveDefiniteMatrix(A, d);
        std::vector<double> original;
        if (validate) original = A;
        if (rank == 0) printf("Computing Cholesky decomposition...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const bool success = choleskyDecomposition(A, d);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!success) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
            result = 1;
        } else {
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
                const double ops = double(n) * double(n) * double(n) / 3.0;
                printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
            }
            if (printResults) printDistributedResults(A, d);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateCholesky(A, original, d);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                result = valid ? 0 : 1;
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
