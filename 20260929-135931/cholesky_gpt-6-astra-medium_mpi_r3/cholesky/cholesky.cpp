#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
// Use the MPI C interface; deprecated MPI C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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

// Only lower-triangular tiles are stored. The process grid is row-major;
// separate row/column communicators keep panel traffic off unrelated ranks.
struct Matrix {
    int n, nt, rank, pr, pc, row, col;
    MPI_Comm rows, cols;
    std::vector<Tile> tiles;
    std::vector<double> left, right;
    Matrix(int size) : n(size), nt(size / block + (size % block != 0)) {
        int processes, dims[2] = {0, 0};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        pr = dims[0]; pc = dims[1]; row = rank / pc; col = rank % pc;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols);
        for (int i = row; i < nt; i += pr)
            for (int j = col; j <= i; j += pc) tiles.emplace_back(i, j);
        left.resize(static_cast<size_t>(nt / pr + 1) * tileSize);
        right.resize(static_cast<size_t>(nt / pc + 1) * tileSize);
    }
    ~Matrix() { MPI_Comm_free(&rows); MPI_Comm_free(&cols); }
    int owner(int i, int j) const { return (i % pr) * pc + j % pc; }
    int extent(int i) const { return std::min(block, n - i * block); }
    double* lpanel(int i) { return left.data() + static_cast<size_t>(i / pr) * tileSize; }
    double* rpanel(int j) { return right.data() + static_cast<size_t>(j / pc) * tileSize; }

    // First replicate L(i,k) along its process row, then transpose and
    // replicate along process column i. Every communicator sees the same order.
    void panels(int k, int first) {
        for (const auto& t : tiles)
            if (t.j == k && t.i >= first)
                std::copy(t.a.begin(), t.a.end(), lpanel(t.i));
        for (int i = first; i < nt; ++i) {
            if (row == i % pr)
                MPI_Bcast(lpanel(i), tileSize, MPI_DOUBLE, k % pc, rows);
            if (col == i % pc) {
                double* p = rpanel(i);
                if (row == i % pr) {
                    const double* q = lpanel(i);
                    for (int x = 0; x < block; ++x)
                        for (int y = 0; y < block; ++y) p[x * block + y] = q[y * block + x];
                }
                MPI_Bcast(p, tileSize, MPI_DOUBLE, i % pr, cols);
            }
        }
    }
};

// Cache-resident matrix multiply with contiguous, vectorizable inner loops.
// Four output rows share each loaded row of the transposed right panel.
void subtractProduct(double* a, const double* l, const double* rt,
                     int m, int n, int depth) {
    int i = 0;
    for (; i + 3 < m; i += 4) {
        for (int k = 0; k < depth; ++k) {
            const double x0 = l[i * block + k], x1 = l[(i + 1) * block + k];
            const double x2 = l[(i + 2) * block + k], x3 = l[(i + 3) * block + k];
            for (int j = 0; j < n; ++j) {
                const double y = rt[k * block + j];
                a[i * block + j] -= x0 * y;
                a[(i + 1) * block + j] -= x1 * y;
                a[(i + 2) * block + j] -= x2 * y;
                a[(i + 3) * block + j] -= x3 * y;
            }
        }
    }
    for (; i < m; ++i)
        for (int k = 0; k < depth; ++k)
            for (int j = 0; j < n; ++j) a[i * block + j] -= l[i * block + k] * rt[k * block + j];
}

void generatePositiveDefiniteMatrix(Matrix& a) {
    // Record exactly the original rand_r stream at each row boundary. Only
    // O(n) seeds and O(block*n) scratch space are needed on each process.
    std::vector<unsigned int> seeds(a.n);
    if (a.rank == 0) {
        unsigned int seed = 42;
        for (int i = 0; i < a.n; ++i) {
            seeds[i] = seed;
            for (int j = 0; j < a.n; ++j) rand_r(&seed);
        }
    }
    MPI_Bcast(seeds.data(), a.n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<double> bi(static_cast<size_t>(block) * a.n), bj(bi.size());
    auto generate = [&](int tile, std::vector<double>& b) {
        for (int i = 0; i < a.extent(tile); ++i) {
            unsigned int seed = seeds[tile * block + i];
            for (int k = 0; k < a.n; ++k)
                b[static_cast<size_t>(i) * a.n + k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
        }
    };
    int previous = -1;
    for (auto& t : a.tiles) {
        if (previous != t.i) { generate(t.i, bi); previous = t.i; }
        if (t.j != t.i) generate(t.j, bj);
        const auto& other = t.i == t.j ? bi : bj;
        for (int i = 0; i < a.extent(t.i); ++i)
            for (int j = 0; j < a.extent(t.j); ++j) {
                double sum = 0.0;
                for (int k = 0; k < a.n; ++k)
                    sum += bi[static_cast<size_t>(i) * a.n + k] * other[static_cast<size_t>(j) * a.n + k];
                t.a[i * block + j] = sum + (t.i == t.j && i == j ? a.n : 0);
            }
    }
}

bool choleskyDecomposition(Matrix& a) {
    std::vector<double> diagonal(tileSize);
    for (int k = 0; k < a.nt; ++k) {
        int bad = -1;
        const int width = a.extent(k), root = a.owner(k, k);
        if (a.rank == root) {
            for (auto& t : a.tiles) if (t.i == k && t.j == k) {
                for (int i = 0; i < width && bad < 0; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        double sum = 0.0;
                        for (int p = 0; p < j; ++p) sum += t.a[i * block + p] * t.a[j * block + p];
                        double value = t.a[i * block + j] - sum;
                        if (i == j) {
                            if (!(value > 0.0)) { bad = k * block + i; break; }
                            t.a[i * block + j] = std::sqrt(value);
                        } else t.a[i * block + j] = value / t.a[j * block + j];
                    }
                    for (int j = i + 1; j < block; ++j) t.a[i * block + j] = 0.0;
                }
                diagonal = t.a;
            }
        }
        MPI_Bcast(&bad, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (bad >= 0) {
            if (a.rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", bad);
            return false;
        }
        MPI_Bcast(diagonal.data(), tileSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (auto& t : a.tiles) if (t.j == k && t.i > k) {
            for (int i = 0; i < a.extent(t.i); ++i)
                for (int j = 0; j < width; ++j) {
                    double sum = 0.0;
                    for (int p = 0; p < j; ++p) sum += t.a[i * block + p] * diagonal[j * block + p];
                    t.a[i * block + j] = (t.a[i * block + j] - sum) / diagonal[j * block + j];
                }
        }
        a.panels(k, k + 1);
        for (auto& t : a.tiles) if (t.j > k)
            subtractProduct(t.a.data(), a.lpanel(t.i), a.rpanel(t.j), a.extent(t.i), a.extent(t.j), width);
    }
    return true;
}

bool validateCholesky(Matrix& a, const std::vector<Tile>& original) {
    auto residual = original;
    for (int k = 0; k < a.nt; ++k) {
        a.panels(k, k);
        for (auto& t : residual) if (t.j >= k)
            subtractProduct(t.a.data(), a.lpanel(t.i), a.rpanel(t.j), a.extent(t.i), a.extent(t.j), a.extent(k));
    }
    double errors[2] = {0.0, 0.0}, global[2];
    for (size_t t = 0; t < residual.size(); ++t)
        for (int i = 0; i < a.extent(residual[t].i); ++i)
            for (int j = 0; j < a.extent(residual[t].j); ++j) {
                double e = std::abs(residual[t].a[i * block + j]);
                if (!std::isfinite(e)) e = std::numeric_limits<double>::infinity();
                errors[0] = std::max(errors[0], e);
                errors[1] = std::max(errors[1], e / (std::abs(original[t].a[i * block + j]) + 1e-10));
            }
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printDistributedResults(const Matrix& a) {
    // The full row-major array is needed only for the existing output format.
    std::vector<double> full, buffer(tileSize);
    if (a.rank == 0) full.resize(static_cast<size_t>(a.n) * a.n, 0.0);
    size_t local = 0;
    for (int i = 0; i < a.nt; ++i) for (int j = 0; j <= i; ++j) {
        const int owner = a.owner(i, j);
        const double* data = buffer.data();
        if (a.rank == owner) {
            data = a.tiles[local++].a.data();
            if (owner != 0) MPI_Send(data, tileSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        if (a.rank == 0) {
            if (owner != 0) MPI_Recv(buffer.data(), tileSize, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int x = 0; x < a.extent(i); ++x)
                std::copy_n(data + x * block, a.extent(j), full.data() + static_cast<size_t>(i * block + x) * a.n + j * block);
        }
    }
    if (a.rank == 0) print_results(full, "CholeskyL");
}
} // namespace

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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512, status = 0;
    bool validate = false, printResults = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const long long value = std::strtoll(argv[++i], &end, 10);
            if (errno || *end || end == argv[i] || value < 0 || value > INT_MAX ||
                static_cast<unsigned long long>(value) > std::numeric_limits<size_t>::max() / sizeof(double) / std::max(1LL, value)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                status = 1; break;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1; break;
        }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize(); return status;
    }
    try {
        Matrix a(n);
        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Generating positive definite matrix...\n");
        }
        generatePositiveDefiniteMatrix(a);
        std::vector<Tile> original;
        if (validate) original = a.tiles;
        if (rank == 0) printf("Computing Cholesky decomposition...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const bool success = choleskyDecomposition(a);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!success) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
            status = 1;
        } else {
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
                printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? double(n) * n * n / (3e9 * seconds) : 0.0);
            }
            if (printResults) printDistributedResults(a);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateCholesky(a, original);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
