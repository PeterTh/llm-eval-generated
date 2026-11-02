#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate the same values as the original benchmark, using global indices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// Each rank owns complete output rows. Blocking columns and several rows keeps
// the active portion of B and C in cache and makes the inner loop contiguous.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    constexpr size_t rowBlock = 4;
    constexpr size_t colBlock = 128;
    for (size_t ii = 0; ii < rows; ii += rowBlock) {
        const size_t iEnd = std::min(ii + rowBlock, rows);
        for (size_t jj = 0; jj < N; jj += colBlock) {
            const size_t jEnd = std::min(jj + colBlock, N);
            for (size_t k = 0; k < N; ++k) {
                const double* b = B.data() + k * N;
                for (size_t i = ii; i < iEnd; ++i) {
                    const double a = A[i * N + k];
                    double* c = C.data() + i * N;
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t firstRow, size_t rows) {
    if (N == 0) return true;
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi : checkPoints) {
        const size_t i = pi % N;
        if (i < firstRow || i - firstRow >= rows) continue;
        for (size_t pj : checkPoints) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[(i - firstRow) * N + k] * B[k * N + j];
            }
            const double actual = C[(i - firstRow) * N + j];
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// A contiguous, balanced row partition, including the case of more ranks than rows.
size_t rowCount(size_t N, int rank, int ranks) {
    return N / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < N % static_cast<size_t>(ranks));
}
size_t rowStart(size_t N, int rank, int ranks) {
    const size_t r = static_cast<size_t>(rank), p = static_cast<size_t>(ranks);
    return r * (N / p) + std::min(r, N % p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || end == arg || *end || *arg == '-' ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                exitCode = 1;
                break;
            }
            N = static_cast<size_t>(value);
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
            exitCode = 1;
            break;
        }
    }
    if (!exitCode && N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) printf("Matrix size is too large\n");
        exitCode = 1;
    }
    if (exitCode) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t rows = rowCount(N, rank, ranks);
    const size_t firstRow = rowStart(N, rank, ranks);
    try {
        std::vector<double> A(rows * N), B(N * N), C(rows * N);
        initMatrix(A, N, firstRow, rows);
        initMatrix(B, N, 0, N);

        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        matrixMultiply(A, B, C, N, rows);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
            const double gflops = (2.0 * N * N * N) / duration / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }

        if (printResults) {
            // Chunk point-to-point transfers to avoid MPI's int count limit.
            if (rank == 0) {
                std::vector<double> result(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int source = 1; source < ranks; ++source) {
                    size_t offset = rowStart(N, source, ranks) * N;
                    size_t remaining = rowCount(N, source, ranks) * N;
                    while (remaining) {
                        const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        offset += count;
                        remaining -= count;
                    }
                }
                print_results(result, "MatrixC");
            } else {
                size_t offset = 0, remaining = C.size();
                while (remaining) {
                    const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += count;
                    remaining -= count;
                }
            }
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const int localValid = validateResult(A, B, C, N, firstRow, rows) ? 1 : 0;
            int valid = 0;
            MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
