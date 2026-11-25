#include <mpi.h>

#include <algorithm>
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
    size_t start, count;
};

Range partition(size_t n, int parts, int index) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    return {index * base + std::min<size_t>(index, extra),
            base + (static_cast<size_t>(index) < extra)};
}

// Pack eight adjacent columns together so each inner iteration reads a
// contiguous SIMD-width group and reuses it across several output rows.
constexpr size_t lanes = 8;

template <size_t Rows>
#if defined(__GNUC__) && !defined(__clang__)
// Vectorize across the unrolled output columns, keeping k strictly sequential.
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
void multiplyTile(const double* a, const double* b, double* c, size_t n, size_t stride,
                  size_t width, size_t first, size_t end) {
    double sums[Rows][lanes] = {};
    if (first != 0) {
        for (size_t r = 0; r < Rows; ++r)
            std::copy_n(c + r * stride, width, sums[r]);
    }
    size_t limit = end;
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
    // GCC's original x86 dot product rounds paired products separately
    // and contracts the scalar tail for an odd inner dimension.
    if (end == n)
        limit -= n % 2;
#endif
    for (size_t k = first; k < limit; ++k) {
        for (size_t r = 0; r < Rows; ++r)
            for (size_t j = 0; j < lanes; ++j)
                sums[r][j] += a[r * n + k] * b[k * lanes + j];
    }
    if (limit < end) {
        for (size_t r = 0; r < Rows; ++r)
            for (size_t j = 0; j < lanes; ++j)
                sums[r][j] = std::fma(a[r * n + limit], b[limit * lanes + j], sums[r][j]);
    }
    for (size_t r = 0; r < Rows; ++r)
        std::copy_n(sums[r], width, c + r * stride);
}

void matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                    std::vector<double>& c, size_t n, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i += 64) {
        const size_t bottom = std::min(i + 64, rows);
        for (size_t k = 0; k < n; k += 256) {
            const size_t end = std::min(k + 256, n);
            for (size_t j = 0; j < cols; j += lanes) {
                const size_t width = std::min(lanes, cols - j);
                const double* panel = b.data() + j * n;
                size_t r = i;
                for (; r + 4 <= bottom; r += 4)
                    multiplyTile<4>(&a[r * n], panel, &c[r * cols + j], n, cols, width, k, end);
                for (; r < bottom; ++r)
                    multiplyTile<1>(&a[r * n], panel, &c[r * cols + j], n, cols, width, k, end);
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t n, Range rows, Range cols) {
    for (size_t i = rows.start; i < std::min<size_t>(5, rows.start + rows.count); ++i) {
        for (size_t j = cols.start; j < std::min<size_t>(5, cols.start + cols.count); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = c[(i - rows.start) * cols.count + j - cols.start];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                    i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

// A vector datatype describes an entire tile even when its total element
// count exceeds MPI's int count limit.
void reportResults(const std::vector<double>& c, size_t n, int rank, const int dims[2]) {
    if (dims[0] * dims[1] == 1) {
        print_results(c, "MatrixC");
        return;
    }
    std::vector<double> full(rank == 0 ? n * n : 0);
    for (int peer = rank; peer < dims[0] * dims[1]; ++peer) {
        const Range rows = partition(n, dims[0], peer / dims[1]);
        const Range cols = partition(n, dims[1], peer % dims[1]);
        if (rows.count && cols.count) {
            if (peer == 0) {
                for (size_t i = 0; i < rows.count; ++i)
                    std::copy_n(&c[i * cols.count], cols.count, &full[i * n]);
            } else {
                MPI_Datatype tile;
                MPI_Type_vector(rows.count, cols.count, rank == 0 ? n : cols.count, MPI_DOUBLE,
                                &tile);
                MPI_Type_commit(&tile);
                if (rank == 0)
                    MPI_Recv(&full[rows.start * n + cols.start], 1, tile, peer, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                else
                    MPI_Send(c.data(), 1, tile, 0, 0, MPI_COMM_WORLD);
                MPI_Type_free(&tile);
            }
        }
        if (rank != 0)
            return;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const int dimension = atoi(argv[++i]);
            if (dimension < 0) {
                if (rank == 0)
                    fprintf(stderr, "Matrix size must be nonnegative\n");
                MPI_Finalize();
                return 1;
            }
            N = dimension;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0)
                printUsage(argv[0]);
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

    int dims[2] = {};
    MPI_Dims_create(size, 2, dims);
    const Range rows = partition(N, dims[0], rank / dims[1]);
    const Range cols = partition(N, dims[1], rank % dims[1]);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> a(cols.count ? rows.count * N : 0);
    std::vector<double> b(rows.count ? ((cols.count + lanes - 1) / lanes) * lanes * N : 0);
    std::vector<double> c(rows.count * cols.count);
    if (!c.empty()) {
        for (size_t i = 0; i < rows.count; ++i)
            for (size_t k = 0; k < N; ++k)
                a[i * N + k] = getPseudoRndValue(N, rows.start + i, k);
        for (size_t j = 0; j < cols.count; j += lanes)
            for (size_t k = 0; k < N; ++k)
                for (size_t t = 0; t < std::min(lanes, cols.count - j); ++t)
                    b[j * N + k * lanes + t] = getPseudoRndValue(N, k, cols.start + j + t);
    }

    if (rank == 0)
        printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(a, b, c, N, rows.count, cols.count);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / seconds / 1e9);
    }

    if (printResults)
        reportResults(c, N, rank, dims);
    int valid = 1;
    if (validate) {
        if (rank == 0)
            printf("Validating result...\n");
        const int localValid = validateResult(c, N, rows, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
