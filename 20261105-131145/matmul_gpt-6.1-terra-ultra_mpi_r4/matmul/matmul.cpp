#include <mpi.h>

#include <algorithm>
#include <climits>
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

// Each rank owns a contiguous range of complete rows.
static size_t rowCount(size_t N, int size, int rank) {
    return N / size + (static_cast<size_t>(rank) < N % size);
}

static size_t rowOffset(size_t N, int size, int rank) {
    return (N / size) * rank + std::min(static_cast<size_t>(rank), N % size);
}

static void initRows(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

// Blocking keeps a B panel in cache. Each output is accumulated in increasing
// k order, while adjacent columns are computed together for SIMD execution.
static void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, size_t N, size_t rows) {
    if (rows == 0) return;
    constexpr size_t KC = 128, JC = 128, MR = 4, NR = 8;
    for (size_t jj = 0; jj < N; jj += JC) {
        const size_t jEnd = std::min(jj + JC, N);
        for (size_t kk = 0; kk < N; kk += KC) {
            const size_t kEnd = std::min(kk + KC, N);
            size_t i = 0;
            for (; i + MR <= rows; i += MR) {
                size_t j = jj;
                for (; j + NR <= jEnd; j += NR) {
                    double c0[NR], c1[NR], c2[NR], c3[NR];
                    for (size_t x = 0; x < NR; ++x) {
                        c0[x] = C[(i + 0) * N + j + x];
                        c1[x] = C[(i + 1) * N + j + x];
                        c2[x] = C[(i + 2) * N + j + x];
                        c3[x] = C[(i + 3) * N + j + x];
                    }
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a0 = A[(i + 0) * N + k];
                        const double a1 = A[(i + 1) * N + k];
                        const double a2 = A[(i + 2) * N + k];
                        const double a3 = A[(i + 3) * N + k];
                        for (size_t x = 0; x < NR; ++x) {
                            const double b = B[k * N + j + x];
                            c0[x] += a0 * b;
                            c1[x] += a1 * b;
                            c2[x] += a2 * b;
                            c3[x] += a3 * b;
                        }
                    }
                    for (size_t x = 0; x < NR; ++x) {
                        C[(i + 0) * N + j + x] = c0[x];
                        C[(i + 1) * N + j + x] = c1[x];
                        C[(i + 2) * N + j + x] = c2[x];
                        C[(i + 3) * N + j + x] = c3[x];
                    }
                }
                for (; j < jEnd; ++j)
                    for (size_t r = 0; r < MR; ++r) {
                        double sum = C[(i + r) * N + j];
                        for (size_t k = kk; k < kEnd; ++k)
                            sum += A[(i + r) * N + k] * B[k * N + j];
                        C[(i + r) * N + j] = sum;
                    }
            }
            for (; i < rows; ++i)
                for (size_t j = jj; j < jEnd; ++j) {
                    double sum = C[i * N + j];
                    for (size_t k = kk; k < kEnd; ++k)
                        sum += A[i * N + k] * B[k * N + j];
                    C[i * N + j] = sum;
                }
        }
    }
}

static void printUsage(const char* progName) {
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
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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
    if (N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t rows = rowCount(N, size, rank);
    const size_t firstRow = rowOffset(N, size, rank);
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    initRows(A, N, firstRow, rows);
    if (rows) initRows(B, N, 0, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double gflops = (2.0 * N * N * N) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        // Chunked point-to-point transfers avoid MPI's int count limit.
        std::vector<double> result;
        if (rank == 0) {
            result.resize(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int source = 1; source < size; ++source) {
                const size_t count = rowCount(N, size, source) * N;
                const size_t offset = rowOffset(N, size, source) * N;
                for (size_t pos = 0; pos < count;) {
                    const int chunk = static_cast<int>(std::min(count - pos, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset + pos, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += chunk;
                }
            }
            print_results(result, "MatrixC");
        } else {
            for (size_t pos = 0; pos < C.size();) {
                const int chunk = static_cast<int>(std::min(C.size() - pos, static_cast<size_t>(INT_MAX)));
                MPI_Send(C.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                pos += chunk;
            }
        }
    }

    int valid = 1;
    if (validate) {
        double localChecks[25] = {}, checks[25] = {};
        if (N) {
            for (size_t pi = 0; pi < 5; ++pi) {
                const size_t i = pi % N;
                if (i >= firstRow && i - firstRow < rows)
                    for (size_t pj = 0; pj < 5; ++pj)
                        localChecks[pi * 5 + pj] = C[(i - firstRow) * N + pj % N];
            }
        }
        MPI_Reduce(localChecks, checks, 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            for (size_t pi = 0; N && pi < 5 && valid; ++pi) {
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t i = pi % N, j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double actual = checks[pi * 5 + pj];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
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
