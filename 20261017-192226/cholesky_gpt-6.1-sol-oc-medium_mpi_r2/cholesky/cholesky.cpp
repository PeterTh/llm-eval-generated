#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// A two-dimensional block-cyclic matrix. Only the lower triangle is used;
// edge tiles are padded with zeros to keep the update kernels contiguous.
struct Matrix {
    size_t n;
    int b, nt, rank, size, dims[2], row, col, nr, nc;
    MPI_Comm rows, cols;
    std::vector<double> data;

    Matrix(size_t order) : n(order) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols);
        // Retain enough tiles for load balance even on smaller problems.
        b = static_cast<int>(std::min<size_t>(64, std::max<size_t>(1,
            (n + 4 * dims[0] - 1) / (4 * dims[0]))));
        nt = static_cast<int>((n + b - 1) / b);
        nr = nt / dims[0] + (row < nt % dims[0]);
        nc = nt / dims[1] + (col < nt % dims[1]);
        data.resize(static_cast<size_t>(nr) * nc * b * b, 0.0);
    }

    ~Matrix() {
        MPI_Comm_free(&rows);
        MPI_Comm_free(&cols);
    }

    int owner(int i, int j) const { return (i % dims[0]) * dims[1] + j % dims[1]; }
    int extent(int i) const { return static_cast<int>(std::min<size_t>(b, n - size_t(i) * b)); }
    double* tile(int i, int j) {
        return data.data() + (size_t(i / dims[0]) * nc + j / dims[1]) * b * b;
    }
};

// Reproduce the original rand_r stream exactly, retaining only the B rows
// needed for this rank's matrix tiles. No rank stores the full input matrix.
void generatePositiveDefiniteMatrix(Matrix& a) {
    const int b = a.b;
    std::vector<double> br(size_t(a.nr) * b * a.n);
    std::vector<double> bc(size_t(a.nc) * b * a.n);
    unsigned int seed = 42;
    for (size_t i = 0; i < a.n; ++i) {
        const int block = static_cast<int>(i / b);
        double* r = block % a.dims[0] == a.row
            ? br.data() + (size_t(block / a.dims[0]) * b + i % b) * a.n : nullptr;
        // Store the column-side B panels transposed, so independent dot
        // products can be vectorized without reordering their summations.
        double* c = block % a.dims[1] == a.col
            ? bc.data() + size_t(block / a.dims[1]) * b * a.n + i % b : nullptr;
        for (size_t k = 0; k < a.n; ++k) {
            const double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
            if (r) r[k] = value;
            if (c) c[k * b] = value;
        }
    }
    for (int li = 0; li < a.nr; ++li) {
        const int i = a.row + li * a.dims[0];
        for (int lj = 0; lj < a.nc; ++lj) {
            const int j = a.col + lj * a.dims[1];
            if (j > i) continue;
            double* t = a.tile(i, j);
            const double* y = bc.data() + size_t(lj) * b * a.n;
            for (int r = 0; r < a.extent(i); ++r) {
                const double* x = br.data() + (size_t(li) * b + r) * a.n;
                const int width = i == j ? r + 1 : a.extent(j);
                for (size_t k = 0; k < a.n; ++k) {
                    const double value = x[k];
                    const double* in = y + k * b;
                    double* out = t + r * b;
                    for (int c = 0; c < width; ++c) out[c] += value * in[c];
                }
                if (i == j) t[r * b + r] += a.n;
            }
        }
    }
}

// Redistribute a factored block column into the two panel orientations needed
// by the local trailing tiles. Row broadcasts and column allgathers keep the
// communication on process-grid subcommunicators, not on every rank globally.
struct Panels {
    Matrix& a;
    std::vector<double> left, right, send, receive;
    std::vector<int> counts, offsets;

    explicit Panels(Matrix& matrix) : a(matrix),
        left(size_t(a.nr) * a.b * a.b), right(size_t(a.nc) * a.b * a.b),
        send(right.size()), receive(right.size()),
        counts(a.dims[0]), offsets(a.dims[0]) {}

    void exchange(int k) {
        const int area = a.b * a.b;
        const int first = std::max(0, (k - a.row + a.dims[0] - 1) / a.dims[0]);
        if (a.col == k % a.dims[1]) {
            for (int li = first; li < a.nr; ++li) {
                const int i = a.row + li * a.dims[0];
                std::copy_n(a.tile(i, k), area, left.data() + size_t(li) * area);
            }
        }
        const size_t start = size_t(first) * area;
        // Chunking also supports panels larger than MPI's int count range.
        for (size_t pos = start; pos < left.size();) {
            int count = static_cast<int>(std::min<size_t>(INT_MAX, left.size() - pos));
            MPI_Bcast(left.data() + pos, count, MPI_DOUBLE, k % a.dims[1], a.rows);
            pos += count;
        }
        int total = 0;
        for (int r = 0; r < a.dims[0]; ++r) {
            offsets[r] = total;
            int count = 0;
            for (int lj = 0; lj < a.nc; ++lj) {
                const int j = a.col + lj * a.dims[1];
                if (j >= k && j % a.dims[0] == r) {
                    if (r == a.row)
                        std::copy_n(left.data() + size_t(j / a.dims[0]) * area,
                                    area, send.data() + count);
                    count += area;
                }
            }
            counts[r] = count;
            total += count;
        }
        MPI_Allgatherv(send.data(), counts[a.row], MPI_DOUBLE, receive.data(),
                       counts.data(), offsets.data(), MPI_DOUBLE, a.cols);
        for (int r = 0; r < a.dims[0]; ++r) {
            int pos = offsets[r];
            for (int lj = 0; lj < a.nc; ++lj) {
                const int j = a.col + lj * a.dims[1];
                if (j >= k && j % a.dims[0] == r) {
                    std::copy_n(receive.data() + pos, area, right.data() + size_t(lj) * area);
                    pos += area;
                }
            }
        }
    }
};

// C -= X Y^T. Transpose the small right panel for contiguous, vectorizable
// inner loops; all three tiles fit in cache. Padding makes edge tiles safe.
void updateTile(double* c, const double* x, const double* y, int b,
                double sign = -1.0) {
    alignas(64) double transpose[64 * 64];
    for (int k = 0; k < b; ++k)
        for (int j = 0; j < b; ++j) transpose[k * b + j] = y[j * b + k];
    for (int i = 0; i < b; ++i) {
        double* out = c + i * b;
        for (int k = 0; k < b; ++k) {
            const double value = sign * x[i * b + k];
            const double* in = transpose + k * b;
            for (int j = 0; j < b; ++j) out[j] += value * in[j];
        }
    }
}

bool choleskyDecomposition(Matrix& a) {
    const int b = a.b;
    Panels panels(a);
    std::vector<double> diagonal(size_t(b) * b);
    for (int k = 0; k < a.nt; ++k) {
        const int owner = a.owner(k, k);
        int failed = -1;
        if (a.rank == owner) {
            double* d = a.tile(k, k);
            for (int i = 0; i < a.extent(k); ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (int q = 0; q < j; ++q) sum += d[i * b + q] * d[j * b + q];
                    const double value = d[i * b + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0)) { failed = k * b + j; break; }
                        d[i * b + j] = std::sqrt(value);
                    } else {
                        d[i * b + j] = value / d[j * b + j];
                    }
                }
                if (failed >= 0) break;
                // Trailing updates touch the upper half of diagonal tiles.
                std::fill(d + i * b + i + 1, d + (i + 1) * b, 0.0);
            }
            std::copy_n(d, b * b, diagonal.data());
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed >= 0) {
            if (a.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
            return false;
        }
        // Only the process column owning the panel needs the diagonal tile.
        if (a.col == k % a.dims[1]) {
            MPI_Bcast(diagonal.data(), b * b, MPI_DOUBLE, k % a.dims[0], a.cols);
            for (int li = 0; li < a.nr; ++li) {
                const int i = a.row + li * a.dims[0];
                if (i <= k) continue;
                double* t = a.tile(i, k);
                for (int r = 0; r < a.extent(i); ++r) {
                    for (int j = 0; j < a.extent(k); ++j) {
                        double sum = 0.0;
                        for (int q = 0; q < j; ++q)
                            sum += t[r * b + q] * diagonal[j * b + q];
                        t[r * b + j] = (t[r * b + j] - sum) / diagonal[j * b + j];
                    }
                }
            }
        }
        panels.exchange(k);
        for (int li = 0; li < a.nr; ++li) {
            const int i = a.row + li * a.dims[0];
            if (i <= k) continue;
            for (int lj = 0; lj < a.nc; ++lj) {
                const int j = a.col + lj * a.dims[1];
                if (j <= k || j > i) continue;
                updateTile(a.tile(i, j), panels.left.data() + size_t(li) * b * b,
                           panels.right.data() + size_t(lj) * b * b, b);
            }
        }
    }
    return true;
}

bool validateCholesky(Matrix& a, const std::vector<double>& original) {
    // Reconstruct only local lower tiles; symmetry gives the same error maxima
    // as checking both halves of the original full matrix.
    std::vector<double> reconstructed(a.data.size(), 0.0);
    Panels panels(a);
    const int b = a.b;
    for (int k = 0; k < a.nt; ++k) {
        panels.exchange(k);
        for (int li = 0; li < a.nr; ++li) {
            const int i = a.row + li * a.dims[0];
            for (int lj = 0; lj < a.nc; ++lj) {
                const int j = a.col + lj * a.dims[1];
                if (j < k || i < j) continue;
                updateTile(reconstructed.data() + (size_t(li) * a.nc + lj) * b * b,
                           panels.left.data() + size_t(li) * b * b,
                           panels.right.data() + size_t(lj) * b * b, b, 1.0);
            }
        }
    }
    double errors[2] = {0.0, 0.0};
    for (int li = 0; li < a.nr; ++li) {
        const int i = a.row + li * a.dims[0];
        for (int lj = 0; lj < a.nc; ++lj) {
            const int j = a.col + lj * a.dims[1];
            if (j > i) continue;
            const size_t base = (size_t(li) * a.nc + lj) * b * b;
            for (int r = 0; r < a.extent(i); ++r) {
                for (int c = 0; c < a.extent(j); ++c) {
                    if (i == j && c > r) continue;
                    const size_t index = base + r * b + c;
                    double error = std::fabs(reconstructed[index] - original[index]);
                    if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[index]) + 1e-10));
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printDistributedResults(Matrix& a) {
    std::vector<double> full, buffer(size_t(a.b) * a.b);
    if (a.rank == 0) full.resize(a.n * a.n, 0.0);
    for (int i = 0; i < a.nt; ++i) {
        for (int j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (a.rank == owner && owner != 0)
                MPI_Send(a.tile(i, j), a.b * a.b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (a.rank == 0) {
                const double* t;
                if (owner == 0) t = a.tile(i, j);
                else {
                    MPI_Recv(buffer.data(), a.b * a.b, MPI_DOUBLE, owner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    t = buffer.data();
                }
                for (int r = 0; r < a.extent(i); ++r)
                    for (int c = 0; c < a.extent(j); ++c)
                        if (i != j || c <= r)
                            full[(size_t(i) * a.b + r) * a.n + size_t(j) * a.b + c] = t[r * a.b + c];
            }
        }
    }
    if (a.rank == 0) print_results(full, "CholeskyL");
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
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long order = std::strtoull(value, &end, 10);
            // Panel Allgatherv uses int counts; also guard all size products.
            if (errno || value[0] == '-' || end == value || *end ||
                order > static_cast<unsigned long long>(INT_MAX / 64 - 64) ||
                order > std::numeric_limits<size_t>::max() / sizeof(double) / std::max(1ULL, order)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(order);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    Matrix a(n);
    generatePositiveDefiniteMatrix(a);
    std::vector<double> original;
    if (validate) original = a.data;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a);
    double elapsed = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double ops = double(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
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
