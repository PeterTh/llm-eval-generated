#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Square tiles are distributed cyclically over a two-dimensional process grid.
// Only the lower triangle is used. Padded edge tiles simplify panel packing.
struct DistributedMatrix {
    size_t n, b, nt, nr, nc;
    int rank, dims[2] = {0, 0}, row, col;
    MPI_Comm rows, cols;
    std::vector<double> data;

    explicit DistributedMatrix(size_t size) : n(size), b(std::max(size_t(1), std::min(size_t(64), n))),
                                             nt((n + b - 1) / b) {
        int processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols);
        nr = nt / dims[0] + (size_t(row) < nt % dims[0]);
        nc = nt / dims[1] + (size_t(col) < nt % dims[1]);
        data.resize(nr * nc * b * b, 0.0);
    }

    ~DistributedMatrix() {
        // Do not enter collective communicator cleanup while other ranks may
        // still be computing after an allocation failure on this rank.
        if (std::uncaught_exceptions()) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm_free(&rows);
        MPI_Comm_free(&cols);
    }

    size_t width(size_t i) const { return std::min(b, n - i * b); }
    int owner(size_t i, size_t j) const {
        return int(i % dims[0]) * dims[1] + int(j % dims[1]);
    }
    double* tile(size_t i, size_t j) {
        return data.data() + ((i / dims[0]) * nc + j / dims[1]) * b * b;
    }

    // All ranks traverse the original rand_r stream, but retain only the B rows
    // needed by their process row/column. No rank needs a replicated n-by-n B.
    void generate() {
        std::vector<double> br(nr * b * n), bc(nc * b * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            const size_t block = i / b;
            const bool keepRow = block % dims[0] == size_t(row);
            const bool keepCol = block % dims[1] == size_t(col);
            const size_t ri = (block / dims[0] * b + i % b) * n;
            const size_t ci = (block / dims[1] * b + i % b) * n;
            for (size_t k = 0; k < n; ++k) {
                const double value = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
                if (keepRow) br[ri + k] = value;
                if (keepCol) bc[ci + k] = value;
            }
        }
        for (size_t i = row; i < nt; i += dims[0]) {
            for (size_t j = col; j <= i; j += dims[1]) {
                double* a = tile(i, j);
                for (size_t ii = 0; ii < width(i); ++ii) {
                    const double* x = br.data() + (i / dims[0] * b + ii) * n;
                    for (size_t jj = 0; jj < width(j); ++jj) {
                        const double* y = bc.data() + (j / dims[1] * b + jj) * n;
                        double sum = 0.0;
                        for (size_t k = 0; k < n; ++k) sum += x[k] * y[k];
                        a[ii * b + jj] = sum + ((i == j && ii == jj) ? double(n) : 0.0);
                    }
                }
            }
        }
    }

    // Replicate L(i,k) only across process row i, and L(j,k)^T only down
    // process column j. Packing batches the panel into O(sqrt(P)) collectives.
    void exchangePanel(size_t k, size_t first, std::vector<double>& left,
                       std::vector<double>& right, std::vector<double>& message) {
        size_t begin = 0;
        while (begin < nr && size_t(row) + begin * dims[0] < first) ++begin;
        const size_t count = (nr - begin) * b * b;
        if (col == int(k % dims[1])) {
            for (size_t li = begin; li < nr; ++li) {
                const size_t i = row + li * dims[0];
                std::copy_n(tile(i, k), b * b, left.data() + li * b * b);
            }
        }
        if (count) {
            // Counts are bounded by the input-size check in run().
            MPI_Bcast(left.data() + begin * b * b, int(count), MPI_DOUBLE,
                      int(k % dims[1]), rows);
        }
        for (int r = 0; r < dims[0]; ++r) {
            size_t packed = 0;
            for (size_t lj = 0; lj < nc; ++lj) {
                const size_t j = col + lj * dims[1];
                if (j < first || j % dims[0] != size_t(r)) continue;
                if (row == r) {
                    std::copy_n(left.data() + (j / dims[0]) * b * b, b * b,
                                message.data() + packed);
                }
                packed += b * b;
            }
            if (!packed) continue;
            MPI_Bcast(message.data(), int(packed), MPI_DOUBLE, r, cols);
            packed = 0;
            for (size_t lj = 0; lj < nc; ++lj) {
                const size_t j = col + lj * dims[1];
                if (j < first || j % dims[0] != size_t(r)) continue;
                double* out = right.data() + lj * b * b;
                for (size_t x = 0; x < b; ++x)
                    for (size_t y = 0; y < b; ++y)
                        out[y * b + x] = message[packed + x * b + y];
                packed += b * b;
            }
        }
    }
};

// Cache-resident matrix product, with contiguous SIMD-friendly column updates.
// The right panel is already transposed; no BLAS dependency is required.
static void update(double* c, const double* a, const double* bt,
                   size_t m, size_t n, size_t depth, size_t stride) {
    for (size_t i = 0; i < m; ++i) {
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            double sums[8];
            for (size_t t = 0; t < 8; ++t) sums[t] = c[i * stride + j + t];
            for (size_t k = 0; k < depth; ++k) {
                const double x = a[i * stride + k];
                for (size_t t = 0; t < 8; ++t)
                    sums[t] -= x * bt[k * stride + j + t];
            }
            for (size_t t = 0; t < 8; ++t) c[i * stride + j + t] = sums[t];
        }
        for (; j < n; ++j) {
            double sum = c[i * stride + j];
            for (size_t k = 0; k < depth; ++k)
                sum -= a[i * stride + k] * bt[k * stride + j];
            c[i * stride + j] = sum;
        }
    }
}

bool choleskyDecomposition(DistributedMatrix& a) {
    const size_t b = a.b;
    std::vector<double> diagonal(b * b), left(a.nr * b * b),
                        right(a.nc * b * b), message(a.nc * b * b);
    for (size_t k = 0; k < a.nt; ++k) {
        const size_t w = a.width(k);
        int failed = 0;
        if (a.rank == a.owner(k, k)) {
            double* d = a.tile(k, k);
            for (size_t i = 0; i < w; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t t = 0; t < j; ++t) sum += d[i * b + t] * d[j * b + t];
                    const double value = d[i * b + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0)) { failed = int(i) + 1; break; }
                        d[i * b + j] = std::sqrt(value);
                    } else {
                        d[i * b + j] = value / d[j * b + j];
                    }
                }
                if (failed) break;
                std::fill(d + i * b + i + 1, d + (i + 1) * b, 0.0);
            }
            std::copy_n(d, b * b, diagonal.data());
        }
        MPI_Bcast(&failed, 1, MPI_INT, a.owner(k, k), MPI_COMM_WORLD);
        if (failed) {
            if (a.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       k * b + size_t(failed - 1));
            return false;
        }
        // Only the panel's process column needs the diagonal tile.
        if (a.col == int(k % a.dims[1])) {
            MPI_Bcast(diagonal.data(), int(b * b), MPI_DOUBLE, int(k % a.dims[0]), a.cols);
            for (size_t i = a.row; i < a.nt; i += a.dims[0]) {
                if (i <= k) continue;
                double* panel = a.tile(i, k);
                for (size_t ii = 0; ii < a.width(i); ++ii) {
                    for (size_t j = 0; j < w; ++j) {
                        double sum = 0.0;
                        for (size_t t = 0; t < j; ++t)
                            sum += panel[ii * b + t] * diagonal[j * b + t];
                        panel[ii * b + j] = (panel[ii * b + j] - sum) / diagonal[j * b + j];
                    }
                }
            }
        }
        a.exchangePanel(k, k + 1, left, right, message);
        for (size_t i = a.row; i < a.nt; i += a.dims[0]) {
            if (i <= k) continue;
            for (size_t j = a.col; j <= i; j += a.dims[1]) {
                if (j <= k) continue;
                update(a.tile(i, j), left.data() + (i / a.dims[0]) * b * b,
                       right.data() + (j / a.dims[1]) * b * b,
                       a.width(i), a.width(j), w, b);
            }
        }
    }
    return true;
}

bool validateCholesky(DistributedMatrix& a, const std::vector<double>& original) {
    const size_t b = a.b;
    std::vector<double> reconstructed(a.data.size(), 0.0), left(a.nr * b * b),
                        right(a.nc * b * b), message(a.nc * b * b);
    for (size_t k = 0; k < a.nt; ++k) {
        a.exchangePanel(k, k, left, right, message);
        for (size_t i = a.row; i < a.nt; i += a.dims[0]) {
            if (i < k) continue;
            for (size_t j = a.col; j <= i; j += a.dims[1]) {
                if (j < k) continue;
                const size_t offset = size_t(a.tile(i, j) - a.data.data());
                update(reconstructed.data() + offset,
                       left.data() + (i / a.dims[0]) * b * b,
                       right.data() + (j / a.dims[1]) * b * b,
                       a.width(i), a.width(j), a.width(k), b);
            }
        }
    }
    double errors[2] = {0.0, 0.0}, global[2];
    for (size_t i = a.row; i < a.nt; i += a.dims[0]) {
        for (size_t j = a.col; j <= i; j += a.dims[1]) {
            const size_t offset = size_t(a.tile(i, j) - a.data.data());
            for (size_t x = 0; x < a.width(i); ++x) {
                for (size_t y = 0; y < a.width(j); ++y) {
                    const size_t index = offset + x * b + y;
                    const double error = std::fabs(-reconstructed[index] - original[index]);
                    const double relative = error / (std::fabs(original[index]) + 1e-10);
                    errors[0] = std::max(errors[0], std::isfinite(error) ? error : INFINITY);
                    errors[1] = std::max(errors[1], std::isfinite(relative) ? relative : INFINITY);
                }
            }
        }
    }
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

// Gathering is needed only for the original, ordered external-results format.
void printDistributedResults(DistributedMatrix& a) {
    std::vector<double> result, buffer(a.b * a.b);
    if (a.rank == 0) result.resize(a.n * a.n, 0.0);
    for (size_t i = 0; i < a.nt; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (a.rank == owner && owner != 0)
                MPI_Send(a.tile(i, j), int(a.b * a.b), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (a.rank == 0) {
                const double* source;
                if (owner == 0) source = a.tile(i, j);
                else {
                    MPI_Recv(buffer.data(), int(a.b * a.b), MPI_DOUBLE, owner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    source = buffer.data();
                }
                for (size_t x = 0; x < a.width(i); ++x)
                    std::copy_n(source + x * a.b, a.width(j),
                                result.data() + (i * a.b + x) * a.n + j * a.b);
            }
        }
    }
    if (a.rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(argument, &end, 10);
            if (errno || argument == end || *end || argument[0] == '-' ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = size_t(value);
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
    // Bound packed panel counts to the MPI int-count interface, and guard all
    // n*n byte calculations (including the optional root-only output matrix).
    if (n > size_t(std::numeric_limits<int>::max()) / 64 - 63 ||
        (n && n > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
        if (rank == 0) printf("Matrix size is too large\n");
        return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix a(n);
    a.generate();
    std::vector<double> original;
    if (validate) original = a.data;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = double(n) * double(n) * double(n) / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(a);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, original);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    try {
        result = run(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
