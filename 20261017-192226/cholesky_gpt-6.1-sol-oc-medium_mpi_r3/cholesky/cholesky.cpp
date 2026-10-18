#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {
constexpr int block = 64;
constexpr int tileSize = block * block;

struct Tile {
    int i, j;
    std::vector<double> a;
    Tile(int row, int col) : i(row), j(col), a(tileSize, 0.0) {}
};

// Only the lower triangle is stored, in a 2D block-cyclic distribution.
// Row/column panels require O(n * block / sqrt(P)) communication/storage.
class Matrix {
public:
    size_t n;
    int nt, rank, dims[2] = {0, 0}, row, col, nr, nc;
    MPI_Comm rows, cols;
    std::vector<Tile> tiles;
    std::vector<int> lookup, counts, displs, offsets;
    std::vector<double> rowPanel, colPanel, sendPanel;

    explicit Matrix(size_t size) : n(size), nt(static_cast<int>((n + block - 1) / block)) {
        int processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        row = rank / dims[1];
        col = rank % dims[1];
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols);
        nr = (nt + dims[0] - 1 - row) / dims[0];
        nc = (nt + dims[1] - 1 - col) / dims[1];
        lookup.assign(static_cast<size_t>(nr) * nc, -1);
        for (int i = row; i < nt; i += dims[0]) {
            for (int j = col; j <= i; j += dims[1]) {
                lookup[static_cast<size_t>(i / dims[0]) * nc + j / dims[1]] =
                    static_cast<int>(tiles.size());
                tiles.emplace_back(i, j);
            }
        }
        rowPanel.resize(static_cast<size_t>(nr) * tileSize);
        colPanel.resize(static_cast<size_t>(nc) * tileSize);
        sendPanel.reserve(static_cast<size_t>(std::min(nr, nc)) * tileSize);
        counts.resize(dims[0]);
        displs.resize(dims[0]);
        offsets.resize(nt);
    }

    ~Matrix() {
        MPI_Comm_free(&rows);
        MPI_Comm_free(&cols);
    }

    int extent(int t) const { return static_cast<int>(std::min<size_t>(block, n - size_t(t) * block)); }
    int owner(int i, int j) const { return (i % dims[0]) * dims[1] + j % dims[1]; }
    double* local(int i, int j) {
        return tiles[lookup[static_cast<size_t>(i / dims[0]) * nc + j / dims[1]]].a.data();
    }
    const double* left(int i) const { return rowPanel.data() + size_t(i / dims[0]) * tileSize; }
    const double* right(int j) const { return colPanel.data() + size_t(offsets[j]) * tileSize; }

    // First broadcast each process-row's panel from its owning column.
    // Then redistribute the appropriate panel tiles within process columns.
    // Both collectives are aggregated, rather than broadcasting individual rows.
    void panel(int k) {
        int first = k + (row - k % dims[0] + dims[0]) % dims[0];
        int start = first / dims[0];
        int length = std::max(0, nr - start) * tileSize;
        if (col == k % dims[1]) {
            for (int i = first; i < nt; i += dims[0])
                std::copy_n(local(i, k), tileSize, rowPanel.data() + size_t(i / dims[0]) * tileSize);
        }
        // Avoid forming an out-of-range pointer on ranks with no active rows.
        double* buffer = length ? rowPanel.data() + size_t(start) * tileSize : rowPanel.data();
        MPI_Bcast(buffer, length, MPI_DOUBLE, k % dims[1], rows);

        sendPanel.clear();
        int total = 0;
        for (int r = 0; r < dims[0]; ++r) {
            displs[r] = total;
            for (int j = col; j < nt; j += dims[1]) {
                if (j >= k && j % dims[0] == r) {
                    offsets[j] = total / tileSize;
                    total += tileSize;
                    if (r == row) {
                        const double* p = left(j);
                        sendPanel.insert(sendPanel.end(), p, p + tileSize);
                    }
                }
            }
            counts[r] = total - displs[r];
        }
        MPI_Allgatherv(sendPanel.data(), counts[row], MPI_DOUBLE,
                       colPanel.data(), counts.data(), displs.data(), MPI_DOUBLE, cols);
        // Transpose once per received tile, so all trailing updates read both
        // their output and the right panel contiguously in the SIMD loop.
        for (int offset = 0; offset < total; offset += tileSize) {
            double* p = colPanel.data() + offset;
            for (int i = 0; i < block; ++i)
                for (int j = 0; j < i; ++j)
                    std::swap(p[i * block + j], p[j * block + i]);
        }
    }

    void generate() {
        // Replay the original rand_r stream exactly. Keep only B rows needed
        // for this process row/column, never a replicated global dense matrix.
        std::vector<double> br(static_cast<size_t>(nr) * block * n);
        std::vector<double> bc(static_cast<size_t>(nc) * block * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            int t = static_cast<int>(i / block);
            double* r = t % dims[0] == row ? br.data() + (size_t(t / dims[0]) * block + i % block) * n : nullptr;
            double* c = t % dims[1] == col ? bc.data() + (size_t(t / dims[1]) * block + i % block) * n : nullptr;
            for (size_t j = 0; j < n; ++j) {
                double value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                if (r) r[j] = value;
                if (c) c[j] = value;
            }
        }
        for (Tile& t : tiles) {
            for (int i = 0; i < extent(t.i); ++i) {
                const double* x = br.data() + (size_t(t.i / dims[0]) * block + i) * n;
                int width = t.i == t.j ? i + 1 : extent(t.j);
                for (int j = 0; j < width; ++j) {
                    const double* y = bc.data() + (size_t(t.j / dims[1]) * block + j) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += x[k] * y[k];
                    t.a[i * block + j] = sum + (t.i == t.j && i == j ? n : 0);
                }
            }
        }
    }

    bool factor() {
        std::vector<double> diagonal(tileSize + 1);
        for (int k = 0; k < nt; ++k) {
            const int width = extent(k);
            if (rank == owner(k, k)) {
                double* d = local(k, k);
                diagonal[tileSize] = -1;
                for (int j = 0; j < width; ++j) {
                    double value = d[j * block + j];
                    for (int s = 0; s < j; ++s) value -= d[j * block + s] * d[j * block + s];
                    if (!(value > 0.0)) {
                        diagonal[tileSize] = size_t(k) * block + j;
                        break;
                    }
                    d[j * block + j] = std::sqrt(value);
                    for (int i = j + 1; i < width; ++i) {
                        double v = d[i * block + j];
                        for (int s = 0; s < j; ++s) v -= d[i * block + s] * d[j * block + s];
                        d[i * block + j] = v / d[j * block + j];
                    }
                }
                std::copy_n(d, tileSize, diagonal.data());
            }
            MPI_Bcast(diagonal.data(), tileSize + 1, MPI_DOUBLE, owner(k, k), MPI_COMM_WORLD);
            if (diagonal[tileSize] >= 0) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %.0f\n", diagonal[tileSize]);
                return false;
            }
            if (col == k % dims[1]) {
                for (Tile& t : tiles) {
                    if (t.j != k || t.i <= k) continue;
                    for (int i = 0; i < extent(t.i); ++i) {
                        for (int j = 0; j < width; ++j) {
                            double v = t.a[i * block + j];
                            for (int s = 0; s < j; ++s) v -= t.a[i * block + s] * diagonal[j * block + s];
                            t.a[i * block + j] = v / diagonal[j * block + j];
                        }
                    }
                }
            }
            panel(k);
            for (Tile& t : tiles) {
                if (t.j <= k) continue;
                update(t, left(t.i), right(t.j), width, -1.0);
            }
        }
        return true;
    }

    // A small, cache-resident rank-k update. Contiguous innermost loops let
    // the compiler vectorize without requiring an external BLAS library.
    void update(Tile& t, const double* x, const double* y, int width, double sign) const {
        for (int i = 0; i < extent(t.i); ++i) {
            const int columns = t.i == t.j ? i + 1 : extent(t.j);
            int j = 0;
            for (; j + 8 <= columns; j += 8) {
                double sums[8] = {};
                for (int s = 0; s < width; ++s) {
                    double v = x[i * block + s];
                    for (int q = 0; q < 8; ++q) sums[q] += v * y[s * block + j + q];
                }
                for (int q = 0; q < 8; ++q) t.a[i * block + j + q] += sign * sums[q];
            }
            for (; j < columns; ++j) {
                double sum = 0.0;
                for (int s = 0; s < width; ++s) sum += x[i * block + s] * y[s * block + j];
                t.a[i * block + j] += sign * sum;
            }
        }
    }

    bool validate(const std::vector<Tile>& original) {
        std::vector<Tile> reconstructed = original;
        for (Tile& t : reconstructed) std::fill(t.a.begin(), t.a.end(), 0.0);
        for (int k = 0; k < nt; ++k) {
            panel(k);
            for (Tile& t : reconstructed) {
                if (t.j >= k) update(t, left(t.i), right(t.j), extent(k), 1.0);
            }
        }
        double errors[2] = {0.0, 0.0}, global[2];
        for (size_t t = 0; t < tiles.size(); ++t) {
            for (int i = 0; i < extent(tiles[t].i); ++i) {
                int width = tiles[t].i == tiles[t].j ? i + 1 : extent(tiles[t].j);
                for (int j = 0; j < width; ++j) {
                    size_t index = i * block + j;
                    double a = original[t].a[index];
                    double error = std::fabs(reconstructed[t].a[index] - a);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(a) + 1e-10));
                }
            }
        }
        MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", global[0]);
            printf("Max relative error: %.10e\n", global[1]);
            if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
        }
        return global[1] <= 1e-6;
    }

    void print() {
        // Gather only when explicitly requested; the timed factorization never
        // stores the full matrix on any process.
        if (rank != 0) {
            for (const Tile& t : tiles)
                MPI_Send(t.a.data(), tileSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            return;
        }
        std::vector<double> result(n * n, 0.0), received(tileSize);
        for (int i = 0; i < nt; ++i) {
            for (int j = 0; j <= i; ++j) {
                int source = owner(i, j);
                const double* data;
                if (source == 0) data = local(i, j);
                else {
                    MPI_Recv(received.data(), tileSize, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    data = received.data();
                }
                for (int r = 0; r < extent(i); ++r)
                    std::copy_n(data + r * block, extent(j), result.data() + (size_t(i) * block + r) * n + size_t(j) * block);
            }
        }
        print_results(result, "CholeskyL");
    }
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            unsigned long long value = std::strtoull(arg, &end, 10);
            // Panel counts use MPI's int counts; also guard dense output size.
            if (errno || *arg == '-' || end == arg || *end ||
                value > static_cast<unsigned long long>(INT_MAX / block - block) ||
                value > std::sqrt(static_cast<long double>(std::numeric_limits<size_t>::max() / sizeof(double)))) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                return 1;
            }
            n = static_cast<size_t>(value);
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
    a.generate();
    std::vector<Tile> original;
    if (validate) original = a.tiles;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = a.factor();
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) a.print();
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = a.validate(original);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
