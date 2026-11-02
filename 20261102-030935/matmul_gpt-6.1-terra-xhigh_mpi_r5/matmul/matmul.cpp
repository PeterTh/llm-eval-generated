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

// Generate the same values as the original benchmark, including size_t arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Each rank owns a contiguous range of rows.  Work on several rows at once so
// that each cache line of B is reused, while accumulating k in its original order.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t rows) {
    constexpr size_t rowTile = 32;
    constexpr size_t depthTile = 128;
    constexpr size_t rowBlock = 4;
    constexpr size_t colBlock = 32;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        const size_t rowEnd = std::min(ii + rowTile, rows);
        for (size_t j = 0; j < N; j += colBlock) {
            const size_t nc = std::min(colBlock, N - j);
            for (size_t kk = 0; kk < N; kk += depthTile) {
                const size_t depthEnd = std::min(kk + depthTile, N);
                for (size_t i = ii; i < rowEnd; i += rowBlock) {
                    const size_t nr = std::min(rowBlock, rowEnd - i);
                    double sums[rowBlock][colBlock] = {};
                    if (kk != 0) {
                        for (size_t r = 0; r < nr; ++r)
                            std::copy_n(C.data() + (i + r) * N + j, nc, sums[r]);
                    }
                    for (size_t k = kk; k < depthEnd; ++k) {
                        const double* b = B.data() + k * N + j;
                        for (size_t r = 0; r < nr; ++r) {
                            const double a = A[(i + r) * N + k];
                            for (size_t c = 0; c < nc; ++c)
                                sums[r][c] += a * b[c];
                        }
                    }
                    for (size_t r = 0; r < nr; ++r)
                        std::copy_n(sums[r], nc, C.data() + (i + r) * N + j);
                }
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

// MPI counts are int, so transfer large row ranges in chunks.
void transferDoubles(double* data, size_t count, int peer, bool send) {
    const size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, maxCount));
        if (send) {
            MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max() ||
                parsed > std::numeric_limits<size_t>::max() / parsed ||
                parsed * parsed > std::vector<double>().max_size()) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    const size_t p = static_cast<size_t>(processes);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rows = N / p + (rankIndex < N % p ? 1 : 0);
    const size_t firstRow = rankIndex * (N / p) + std::min(rankIndex, N % p);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
    if (rows != 0)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                B[i * N + j] = getPseudoRndValue(N, i, j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int source = 1; source < processes; ++source) {
                const size_t s = static_cast<size_t>(source);
                const size_t sourceRows = N / p + (s < N % p ? 1 : 0);
                const size_t sourceFirst = s * (N / p) + std::min(s, N % p);
                if (sourceRows != 0)
                    transferDoubles(result.data() + sourceFirst * N, sourceRows * N, source, false);
            }
            print_results(result, "MatrixC");
        } else if (!C.empty()) {
            transferDoubles(C.data(), C.size(), 0, true);
        }
    }

    int valid = 1;
    if (validate) {
        // Every sampled row has exactly one owner, so summing these values
        // collects the 25 checks without gathering the complete matrix.
        double local[25] = {}, actual[25] = {};
        for (size_t pi = 0; pi < 5; ++pi) {
            const size_t i = pi % N;
            if (i >= firstRow && i - firstRow < rows)
                for (size_t pj = 0; pj < 5; ++pj)
                    local[pi * 5 + pj] = C[(i - firstRow) * N + pj % N];
        }
        MPI_Reduce(local, actual, 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            for (size_t pi = 0; pi < 5 && valid; ++pi) {
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t i = pi % N, j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * B[k * N + j];
                    const double value = actual[pi * 5 + pj];
                    const double relError = std::abs((value - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, value, relError);
                        valid = 0;
                        break;
                    }
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
