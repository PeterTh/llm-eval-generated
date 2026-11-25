#include <algorithm>
#include <mpi.h>
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
    size_t first, size;
};

Range partition(size_t n, int parts, int index) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    const size_t i = index;
    return {i * base + std::min(i, extra), base + (i < extra)};
}

using Double4 = double __attribute__((vector_size(4 * sizeof(double))));
constexpr size_t panelWidth = 8;
constexpr size_t depthBlock = 128;

template<size_t Rows>
void multiplyPanel(const double* a, const double* b, double* c, size_t n,
                   size_t stride, size_t begin, size_t end) {
    Double4 sums[Rows][2];
    for (size_t r = 0; r < Rows; ++r)
        std::memcpy(sums[r], c + r * stride, sizeof(sums[r]));

    // Eight independent SIMD accumulators in the four-row kernel.
    for (size_t k = begin; k < end; ++k) {
        Double4 values[2];
        std::memcpy(values, b + k * panelWidth, sizeof(values));
        for (size_t r = 0; r < Rows; ++r) {
            const double left = a[r * n + k];
            sums[r][0] += left * values[0];
            sums[r][1] += left * values[1];
        }
    }
    for (size_t r = 0; r < Rows; ++r)
        std::memcpy(c + r * stride, sums[r], sizeof(sums[r]));
}

void matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                    std::vector<double>& c, size_t rows, size_t stride, size_t n) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
    // GCC's x86 reference loop uses separate multiply/add for pairs and
    // a fused multiply/add for the scalar remainder.
    const size_t depth = n - n % 2;
#else
    const size_t depth = n;
#endif
    for (size_t col = 0; col < stride; col += panelWidth) {
        const double* panel = b.data() + col * n;
        for (size_t k = 0; k < depth; k += depthBlock) {
            size_t end = std::min(k + depthBlock, depth);
            size_t row = 0;
            for (; row + 4 <= rows; row += 4)
                multiplyPanel<4>(&a[row * n], panel, &c[row * stride + col],
                                 n, stride, k, end);
            for (; row < rows; ++row)
                multiplyPanel<1>(&a[row * n], panel, &c[row * stride + col],
                                 n, stride, k, end);
        }
        if (depth < n) {
            for (size_t r = 0; r < rows; ++r)
                for (size_t j = 0; j < panelWidth; ++j) {
                    double& sum = c[r * stride + col + j];
                    sum = std::fma(a[r * n + depth], panel[depth * panelWidth + j], sum);
                }
        }
    }
}

bool validateTile(const std::vector<double>& c, Range rows, Range cols, size_t stride,
                  size_t n) {
    for (size_t i = rows.first; i < std::min<size_t>(5, rows.first + rows.size); ++i) {
        for (size_t j = cols.first; j < std::min<size_t>(5, cols.first + cols.size); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            double actual = c[(i - rows.first) * stride + j - cols.first];
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

// Describe an entire tile without narrowing its total element count to int.
MPI_Datatype tileType(Range rows, Range cols, size_t stride) {
    MPI_Datatype type;
    MPI_Type_create_hvector(rows.size, cols.size, stride * sizeof(double), MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

void reportResults(const std::vector<double>& c, Range rows, Range cols, size_t stride,
                   size_t n, int rank, int ranks, const int dims[2]) {
    if (rank == 0) {
        std::vector<double> result(n * n);
        for (int p = 0; p < ranks; ++p) {
            Range rr = partition(n, dims[0], p / dims[1]);
            Range cc = partition(n, dims[1], p % dims[1]);
            if (!rr.size || !cc.size) continue;
            double* dest = result.data() + rr.first * n + cc.first;
            if (p == 0) {
                for (size_t r = 0; r < rr.size; ++r)
                    std::copy_n(c.data() + r * stride, cc.size, dest + r * n);
            } else {
                MPI_Datatype tile = tileType(rr, cc, n);
                MPI_Recv(dest, 1, tile, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&tile);
            }
        }
        print_results(result, "MatrixC");
    } else if (rows.size && cols.size) {
        MPI_Datatype tile = tileType(rows, cols, stride);
        MPI_Send(c.data(), 1, tile, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&tile);
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    Range rows = partition(N, dims[0], rank / dims[1]);
    Range cols = partition(N, dims[1], rank % dims[1]);
    size_t stride = (cols.size + panelWidth - 1) / panelWidth * panelWidth;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> a(rows.size * N), b(stride * N), c(rows.size * stride);
    for (size_t r = 0; r < rows.size; ++r)
        for (size_t k = 0; k < N; ++k)
            a[r * N + k] = getPseudoRndValue(N, rows.first + r, k);
    // Store B in contiguous, zero-padded panels for the SIMD kernel.
    for (size_t col = 0; col < cols.size; col += panelWidth)
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < std::min(panelWidth, cols.size - col); ++j)
                b[col * N + k * panelWidth + j] = getPseudoRndValue(N, k, cols.first + col + j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    matrixMultiply(a, b, c, rows.size, stride, N);
    double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / duration / 1e9);
    }
    if (printResults) reportResults(c, rows, cols, stride, N, rank, ranks, dims);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = validateTile(c, rows, cols, stride, N);
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return !allValid;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
