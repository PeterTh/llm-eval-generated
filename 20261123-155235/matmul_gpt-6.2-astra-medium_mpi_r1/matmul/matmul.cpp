#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

struct Range {
    size_t first;
    size_t count;
};

Range partition(size_t n, int parts, int index) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    return {base * index + std::min<size_t>(index, extra), base + (size_t(index) < extra)};
}

constexpr size_t MR = 4, NR = 8;
constexpr size_t depthBlock = 128;

void matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                    std::vector<double>& c, size_t n, size_t rows, size_t stride) {
    // B is packed in strips of NR columns. Reuse each strip across local rows
    // while keeping a complete MR x NR output tile in registers.
    using Vec = double __attribute__((vector_size(4 * sizeof(double))));
    constexpr size_t vectors = NR / 4;
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__) && defined(__x86_64__)
    // The original GCC x86 dot product rounds paired products before adding
    // them and contracts the remaining scalar multiply-add for an odd depth.
    const size_t pairedDepth = n - n % 2;
#else
    const size_t pairedDepth = n;
#endif
    for (size_t k0 = 0; k0 < n; k0 += depthBlock) {
        const size_t end = std::min(pairedDepth, k0 + depthBlock);
        for (size_t j = 0; j < stride; j += NR) {
            for (size_t i = 0; i < rows; i += MR) {
                Vec sums[MR][vectors];
                for (size_t r = 0; r < MR; ++r)
                    std::memcpy(sums[r], &c[(i + r) * stride + j], sizeof(sums[r]));
                for (size_t k = k0; k < end; ++k) {
                    Vec strip[vectors];
                    std::memcpy(strip, &b[j * n + k * NR], sizeof(strip));
                    for (size_t r = 0; r < MR; ++r)
                        for (size_t v = 0; v < vectors; ++v)
                            sums[r][v] += a[(i + r) * n + k] * strip[v];
                }
                for (size_t r = 0; r < MR; ++r) {
                    double* out = &c[(i + r) * stride + j];
                    std::memcpy(out, sums[r], sizeof(sums[r]));
                    if (pairedDepth != n && k0 + depthBlock >= n) {
                        for (size_t s = 0; s < NR; ++s)
                            out[s] = std::fma(a[(i + r) * n + end], b[j * n + end * NR + s], out[s]);
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, Range rows, Range cols, size_t stride, size_t n) {
    for (size_t i = rows.first; i < std::min<size_t>(5, rows.first + rows.count); ++i) {
        for (size_t j = cols.first; j < std::min<size_t>(5, cols.first + cols.count); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = c[(i - rows.first) * stride + j - cols.first];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

MPI_Datatype tileType(Range rows, Range cols, size_t stride) {
    MPI_Datatype tile;
    MPI_Type_create_hvector(rows.count, cols.count, stride * sizeof(double), MPI_DOUBLE, &tile);
    MPI_Type_commit(&tile);
    return tile;
}

void printGlobalResults(const std::vector<double>& c, size_t stride, size_t n,
                        const int dims[2], int rank, int ranks) {
    const Range rows = partition(n, dims[0], rank / dims[1]);
    const Range cols = partition(n, dims[1], rank % dims[1]);
    if (rank != 0) {
        if (rows.count && cols.count) {
            MPI_Datatype tile = tileType(rows, cols, stride);
            MPI_Send(c.data(), 1, tile, 0, 0, MPI_COMM_WORLD);
            MPI_Type_free(&tile);
        }
        return;
    }

    std::vector<double> result(n * n);
    for (size_t i = 0; i < rows.count; ++i)
        std::copy_n(&c[i * stride], cols.count, &result[i * n]);
    for (int src = 1; src < ranks; ++src) {
        const Range rr = partition(n, dims[0], src / dims[1]);
        const Range cc = partition(n, dims[1], src % dims[1]);
        if (rr.count && cc.count) {
            MPI_Datatype tile = tileType(rr, cc, n);
            MPI_Recv(&result[rr.first * n + cc.first], 1, tile, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&tile);
        }
    }
    print_results(result, "MatrixC");
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int dimension = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            dimension = std::atoi(arg);
            if (dimension < 0) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    const size_t n = dimension;
    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    const Range rows = partition(n, dims[0], rank / dims[1]);
    const Range cols = partition(n, dims[1], rank % dims[1]);
    const size_t paddedRows = cols.count ? (rows.count + MR - 1) / MR * MR : 0;
    const size_t stride = rows.count ? (cols.count + NR - 1) / NR * NR : 0;
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    std::vector<double> a(paddedRows * n), b(stride * n), c(paddedRows * stride);
    if (paddedRows) {
        for (size_t i = 0; i < rows.count; ++i)
            for (size_t k = 0; k < n; ++k)
                a[i * n + k] = getPseudoRndValue(n, rows.first + i, k);
        for (size_t j = 0; j < cols.count; j += NR)
            for (size_t k = 0; k < n; ++k)
                for (size_t s = 0; s < std::min(NR, cols.count - j); ++s)
                    b[j * n + k * NR + s] = getPseudoRndValue(n, k, cols.first + j + s);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(a, b, c, n, paddedRows, stride);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2e-9 * n * n * n / seconds : 0.0);
    }
    if (printResults) printGlobalResults(c, stride, n, dims, rank, ranks);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = validateResult(c, rows, cols, stride, n);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
