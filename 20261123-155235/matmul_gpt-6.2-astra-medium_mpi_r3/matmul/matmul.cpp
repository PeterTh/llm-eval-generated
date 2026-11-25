#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t first;
    size_t count;
};

Range partition(size_t n, int parts, int index) {
    const size_t extra = n % parts;
    const size_t base = n / parts;
    return {index * base + std::min<size_t>(index, extra), base + (size_t(index) < extra)};
}

// Pack B in narrow column panels. A microkernel streams one panel and
// accumulates adjacent output columns in registers, in increasing k order.
constexpr size_t width = 16;
constexpr size_t height = 4;

template<size_t Rows>
inline void multiplyPanel(const double* a, const double* panel, double* output,
                          size_t n, size_t stride, size_t columns, size_t begin, size_t end) {
    double sums[Rows][width]{};
    if (begin != 0) {
        for (size_t r = 0; r < Rows; ++r)
            std::copy_n(output + r * stride, columns, sums[r]);
    }

    const size_t pairedEnd = end - (end % 2);
    for (size_t k = begin; k < pairedEnd; ++k) {
        for (size_t r = 0; r < Rows; ++r) {
            const double value = a[r * n + k];
            for (size_t c = 0; c < width; ++c)
                sums[r][c] += value * panel[k * width + c];
        }
    }
    for (size_t r = 0; r < Rows; ++r) {
        if (end != pairedEnd) {
            // The original dot product can contract its final unpaired term.
            for (size_t c = 0; c < columns; ++c) {
                const double value = a[r * n + pairedEnd];
                const double b = panel[pairedEnd * width + c];
#ifdef __FMA__
                sums[r][c] = std::fma(value, b, sums[r][c]);
#else
                sums[r][c] += value * b;
#endif
            }
        }
        std::copy_n(sums[r], columns, output + r * stride);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t n, size_t rows, size_t cols) {
    constexpr size_t depth = 128;
    for (size_t j = 0; j < cols; j += width) {
        const double* panel = B.data() + j * n;
        const size_t columns = std::min(width, cols - j);
        for (size_t k = 0; k < n; k += depth) {
            const size_t end = std::min(k + depth, n);
            size_t i = 0;
            for (; i + height <= rows; i += height)
                multiplyPanel<height>(&A[i * n], panel, &C[i * cols + j], n, cols, columns, k, end);
            for (; i < rows; ++i)
                multiplyPanel<1>(&A[i * n], panel, &C[i * cols + j], n, cols, columns, k, end);
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t n, Range rows, Range cols) {
    for (size_t i = rows.first; i < std::min<size_t>(5, rows.first + rows.count); ++i) {
        for (size_t j = cols.first; j < std::min<size_t>(5, cols.first + cols.count); ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = C[(i - rows.first) * cols.count + j - cols.first];
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

void printDistributedResults(const std::vector<double>& C, size_t n, int rank, const int dims[2]) {
    if (dims[0] * dims[1] == 1) {
        print_results(C, "MatrixC");
        return;
    }
    if (rank == 0) {
        std::vector<double> full(n * n);
        for (int p = 0; p < dims[0] * dims[1]; ++p) {
            const Range rows = partition(n, dims[0], p / dims[1]);
            const Range cols = partition(n, dims[1], p % dims[1]);
            if (!rows.count || !cols.count) continue;
            double* dest = full.data() + rows.first * n + cols.first;
            if (p == 0) {
                for (size_t i = 0; i < rows.count; ++i)
                    std::copy_n(C.data() + i * cols.count, cols.count, dest + i * n);
            } else {
                MPI_Datatype tile;
                MPI_Type_vector(rows.count, cols.count, n, MPI_DOUBLE, &tile);
                MPI_Type_commit(&tile);
                MPI_Recv(dest, 1, tile, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&tile);
            }
        }
        print_results(full, "MatrixC");
    } else if (!C.empty()) {
        const Range cols = partition(n, dims[1], rank % dims[1]);
        MPI_Datatype row;
        MPI_Type_contiguous(cols.count, MPI_DOUBLE, &row);
        MPI_Type_commit(&row);
        MPI_Send(C.data(), C.size() / cols.count, row, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&row);
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

int benchmark(int argc, char** argv, int rank, int size) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const int value = atoi(argv[++i]);
            if (value < 0) {
                if (rank == 0) printf("Matrix size must be nonnegative\n");
                return 1;
            }
            N = value;
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    int dims[2]{};
    MPI_Dims_create(size, 2, dims);
    const Range rows = partition(N, dims[0], rank / dims[1]);
    const Range cols = partition(N, dims[1], rank % dims[1]);
    std::vector<double> C(rows.count * cols.count);
    std::vector<double> A, B;
    if (!C.empty()) {
        A.resize(rows.count * N);
        B.resize(((cols.count + width - 1) / width) * width * N);
        for (size_t i = 0; i < rows.count; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, rows.first + i, k);
        for (size_t j = 0; j < cols.count; j += width)
            for (size_t k = 0; k < N; ++k)
                for (size_t c = 0; c < std::min(width, cols.count - j); ++c)
                    B[j * N + k * width + c] = getPseudoRndValue(N, k, cols.first + j + c);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (!C.empty()) matrixMultiply(A, B, C, N, rows.count, cols.count);
    const double localTime = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localTime, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * N * N * N / seconds / 1e9 : 0.0);
    }

    if (printResults) printDistributedResults(C, N, rank, dims);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(C, N, rows, cols);
        int valid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int status = benchmark(argc, argv, rank, size);
    MPI_Finalize();
    return status;
}
