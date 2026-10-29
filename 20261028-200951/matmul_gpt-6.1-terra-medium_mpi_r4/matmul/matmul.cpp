#include <mpi.h>

#include <algorithm>
#include <cerrno>
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Each output element accumulates k in increasing order, as in the original.
// Blocking the columns and k keeps the active part of B and C in cache.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t rows, size_t N) {
    constexpr size_t tile = 64;
    for (size_t ii = 0; ii < rows; ii += tile) {
        for (size_t jj = 0; jj < N; jj += tile) {
            const size_t jend = std::min(jj + tile, N);
            for (size_t kk = 0; kk < N; kk += tile) {
                const size_t kend = std::min(kk + tile, N);
                for (size_t i = ii; i < std::min(ii + tile, rows); ++i) {
                    double* c = C.data() + i * N;
                    for (size_t k = kk; k < kend; ++k) {
                        const double a = A[i * N + k];
                        const double* b = B.data() + k * N;
                        for (size_t j = jj; j < jend; ++j)
                            c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t N = 512;
    bool validate = false, printResults = false;
    int status = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() ||
                parsed > std::numeric_limits<size_t>::max() / parsed) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                status = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    if (status || help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }
    const size_t base = N / static_cast<size_t>(processes);
    const size_t extra = N % static_cast<size_t>(processes);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t first = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), extra);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N), C(rows * N, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows, N);
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    }
    if (printResults) {
        // Point-to-point chunks avoid MPI's int count and displacement limits.
        constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
        if (rank == 0) {
            std::vector<double> result(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int p = 1; p < processes; ++p) {
                size_t pr = base + (static_cast<size_t>(p) < extra);
                size_t offset = (base * static_cast<size_t>(p) + std::min(static_cast<size_t>(p), extra)) * N;
                size_t count = pr * N;
                for (size_t pos = 0; pos < count; pos += std::min(chunk, count - pos))
                    MPI_Recv(result.data() + offset + pos, static_cast<int>(std::min(chunk, count - pos)), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(result, "MatrixC");
        } else {
            for (size_t pos = 0; pos < C.size(); pos += std::min(chunk, C.size() - pos))
                MPI_Send(C.data() + pos, static_cast<int>(std::min(chunk, C.size() - pos)), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = 1;
        for (size_t pi = 0; pi < 5 && valid; ++pi) {
            size_t i = pi % N;
            if (i < first || i >= first + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                double actual = C[(i - first) * N + j];
                double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                    valid = 0;
                    break;
                }
            }
        }
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        status = allValid ? 0 : 1;
    }
    MPI_Finalize();
    return status;
}
