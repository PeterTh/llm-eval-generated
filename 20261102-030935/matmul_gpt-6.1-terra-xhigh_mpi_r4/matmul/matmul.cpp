#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The inputs are deterministic, so each rank can generate its own rows and B.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static size_t rowCount(size_t N, int rank, int ranks) {
    return N / ranks + (static_cast<size_t>(rank) < N % ranks);
}

static size_t rowStart(size_t N, int rank, int ranks) {
    return (N / ranks) * rank + std::min(static_cast<size_t>(rank), N % ranks);
}

// Process several rows together to reuse B, with contiguous columns in the
// innermost loop so the compiler can vectorize without changing k order.
static void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, size_t rows, size_t N) {
    constexpr size_t rowBlock = 4;
    constexpr size_t colBlock = 128;
    constexpr size_t kBlock = 128;
    for (size_t ii = 0; ii < rows; ii += rowBlock) {
        const size_t iend = std::min(ii + rowBlock, rows);
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t jend = std::min(jj + colBlock, N);
            for (size_t kk = 0; kk < N; kk += kBlock) {
                const size_t kend = std::min(kk + kBlock, N);
                for (size_t k = kk; k < kend; ++k) {
                    const double* b = B.data() + k * N;
                    for (size_t i = ii; i < iend; ++i) {
                        const double a = A[i * N + k];
                        double* c = C.data() + i * N;
                        for (size_t j = jj; j < jend; ++j) {
                            c[j] += a * b[j];
                        }
                    }
                }
            }
        }
    }
}

static bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                           const std::vector<double>& C, size_t first, size_t rows, size_t N) {
    if (N == 0) return true;
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = pi % N;
        if (i < first || i - first >= rows) continue;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[(i - first) * N + k] * B[k * N + j];
            }
            const double actual = C[(i - first) * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t first = rowStart(N, rank, ranks);
    const size_t rows = rowCount(N, rank, ranks);
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    for (size_t i = 0; i < N && rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
    if (rows) std::copy_n(B.data() + first * N, rows * N, A.data());

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows, N);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        // Chunk messages so result collection is not limited by MPI's int count.
        std::vector<double> result;
        if (rank == 0) {
            result.resize(N * N);
            std::copy(C.begin(), C.end(), result.begin());
        }
        for (int source = 1; source < ranks; ++source) {
            const size_t count = rowCount(N, source, ranks) * N;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                if (rank == source) {
                    MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                } else if (rank == 0) {
                    MPI_Recv(result.data() + rowStart(N, source, ranks) * N + offset,
                             chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                offset += chunk;
            }
        }
        if (rank == 0) print_results(result, "MatrixC");
    }

    int allValid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(A, B, C, first, rows, N) ? 1 : 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return allValid ? 0 : 1;
}
