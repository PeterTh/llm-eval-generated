#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    // Keep a strip of C in cache while traversing k in the original order.
    constexpr size_t block = 128;
    for (size_t jj = 0; jj < N; jj += block) {
        const size_t jend = std::min(N, jj + block);
        for (size_t i = 0; i < rows; ++i) {
            double* c = C.data() + i * N;
            for (size_t k = 0; k < N; ++k) {
                const double a = A[i * N + k];
                const double* b = B.data() + k * N;
                for (size_t j = jj; j < jend; ++j)
                    c[j] += a * b[j];
            }
        }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long n = strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || n == 0 ||
                n > std::numeric_limits<size_t>::max() ||
                n > std::numeric_limits<size_t>::max() / n) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(n);
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
    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N), C(rows * N);
    initMatrix(A, N, first, rows);
    initMatrix(B, N, 0, N);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(N * N);
            std::copy(C.begin(), C.end(), result.begin());
        }
        // Chunk transfers so MPI's int count does not limit the matrix size.
        for (int p = 1; p < ranks; ++p) {
            const size_t pRows = base + (static_cast<size_t>(p) < extra);
            const size_t pFirst = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), extra);
            const size_t count = pRows * N;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                if (rank == p) MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (rank == 0) MPI_Recv(result.data() + pFirst * N + offset, chunk, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(chunk);
            }
        }
        if (rank == 0) print_results(result, "MatrixC");
    }
    int localValid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (size_t pi = 0; pi < 5; ++pi) {
            const size_t i = pi % N;
            if (i < first || i >= first + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = pj % N;
                double expected = 0;
                for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                const double actual = C[(i - first) * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                    localValid = 0;
                }
            }
        }
    }
    int valid = 1;
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate && rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    MPI_Finalize();
    return valid ? 0 : 1;
}
