#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t first;
    size_t size;
};

Range partition(size_t n, int parts, int coordinate) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    const size_t pos = coordinate;
    return {pos * base + std::min(pos, extra), base + (pos < extra)};
}

constexpr size_t panelWidth = 8;

// Store each narrow column panel of B contiguously in reduction order.
void initialize(std::vector<double>& a, std::vector<double>& b, size_t n,
                Range rows, Range cols) {
    for (size_t i = 0; i < rows.size; ++i)
        for (size_t k = 0; k < n; ++k)
            a[i * n + k] = getPseudoRndValue(n, rows.first + i, k);

    for (size_t j = 0; j < cols.size; j += panelWidth)
        for (size_t k = 0; k < n; ++k)
            for (size_t t = 0; t < std::min(panelWidth, cols.size - j); ++t)
                b[j * n + k * panelWidth + t] = getPseudoRndValue(n, k, cols.first + j + t);
}

template<size_t height>
#if defined(__GNUC__) && !defined(__clang__)
// Use SLP across adjacent columns and keep the reduction loop scalar.
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
void multiplyTile(const double* a, const double* b, double* c, size_t n,
                  size_t stride, size_t width, size_t first, size_t end) {
    double sums[height][panelWidth]{};
    if (first != 0)
        for (size_t i = 0; i < height; ++i)
            std::copy_n(c + i * stride, width, sums[i]);

    // GCC vectorizes the original dot product in pairs, with a scalar
    // multiply-add for an odd final element on FMA-capable targets.
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
    const size_t pairedEnd = end - (end == n && n % 2);
#else
    const size_t pairedEnd = end;
#endif
    for (size_t k = first; k < pairedEnd; ++k)
        for (size_t i = 0; i < height; ++i)
            for (size_t j = 0; j < panelWidth; ++j)
                sums[i][j] += a[i * n + k] * b[k * panelWidth + j];

    for (size_t i = 0; i < height; ++i) {
        if (pairedEnd < end)
            for (size_t j = 0; j < width; ++j)
                sums[i][j] = std::fma(a[i * n + pairedEnd], b[pairedEnd * panelWidth + j], sums[i][j]);
        std::copy_n(sums[i], width, c + i * stride);
    }
}

void matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                    std::vector<double>& c, size_t n, size_t rows, size_t cols) {
    constexpr size_t depth = 192;
    for (size_t j = 0; j < cols; j += panelWidth) {
        const size_t width = std::min(cols - j, panelWidth);
        const double* panel = b.data() + j * n;
        for (size_t k = 0; k < n; k += depth) {
            const size_t end = std::min(n, k + depth);
            size_t i = 0;
            for (; i + 4 <= rows; i += 4)
                multiplyTile<4>(&a[i * n], panel, &c[i * cols + j], n, cols, width, k, end);
            for (; i < rows; ++i)
                multiplyTile<1>(&a[i * n], panel, &c[i * cols + j], n, cols, width, k, end);
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t n, Range rows, Range cols) {
    for (size_t i = rows.first; i < std::min<size_t>(5, rows.first + rows.size); ++i) {
        for (size_t j = cols.first; j < std::min<size_t>(5, cols.first + cols.size); ++j) {
            double expected = 0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            double actual = c[(i - rows.first) * cols.size + j - cols.first];
            double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

// Only external reporting needs the full row-major result.
void report(const std::vector<double>& local, size_t n, const int dims[2], int rank, int ranks) {
    if (ranks == 1) {
        print_results(local, "MatrixC");
        return;
    }
    if (rank != 0) {
        for (size_t pos = 0; pos < local.size();) {
            int count = std::min<size_t>(INT_MAX, local.size() - pos);
            MPI_Send(local.data() + pos, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            pos += count;
        }
        return;
    }
    std::vector<double> full(n * n), received;
    for (int source = 0; source < ranks; ++source) {
        Range rows = partition(n, dims[0], source / dims[1]);
        Range cols = partition(n, dims[1], source % dims[1]);
        if (!rows.size || !cols.size) continue;
        if (source != 0) {
            received.resize(rows.size * cols.size);
            for (size_t pos = 0; pos < received.size();) {
                int count = std::min<size_t>(INT_MAX, received.size() - pos);
                MPI_Recv(received.data() + pos, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pos += count;
            }
        }
        const auto& tile = source == 0 ? local : received;
        for (size_t i = 0; i < rows.size; ++i)
            std::copy_n(tile.data() + i * cols.size, cols.size, full.data() + (rows.first + i) * n + cols.first);
    }
    print_results(full, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    Range rows = partition(N, dims[0], rank / dims[1]);
    Range cols = partition(N, dims[1], rank % dims[1]);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> a, b, c(rows.size * cols.size);
    if (!c.empty()) {
        a.resize(rows.size * N);
        b.resize(((cols.size + panelWidth - 1) / panelWidth) * panelWidth * N);
        initialize(a, b, N, rows, cols);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (!c.empty()) matrixMultiply(a, b, c, N, rows.size, cols.size);
    double elapsed = MPI_Wtime() - start, duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / duration / 1e9);
    }

    if (printResults) report(c, N, dims, rank, ranks);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = validateResult(c, N, rows, cols);
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = benchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
