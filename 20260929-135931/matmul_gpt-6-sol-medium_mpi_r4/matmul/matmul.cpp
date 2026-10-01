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

// A and C use contiguous row ranges. Every rank generates B locally.
size_t firstRow(size_t N, int rank, int ranks) {
    return (N / static_cast<size_t>(ranks)) * static_cast<size_t>(rank)
         + std::min(N % static_cast<size_t>(ranks), static_cast<size_t>(rank));
}

void initRows(std::vector<double>& mat, size_t N, size_t first, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, first + i, j);
        }
    }
}

void initTransposedB(std::vector<double>& B, size_t N) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
            B[j * N + k] = getPseudoRndValue(N, k, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    // Compute four columns together, retaining a register accumulator for
    // each. Each result visits k in the same order as the serial benchmark.
    for (size_t i = 0; i < rows; ++i) {
        const double* a = A.data() + i * N;
        size_t j = 0;
        for (; j + 3 < N; j += 4) {
            const double* b0 = B.data() + (j + 0) * N;
            const double* b1 = B.data() + (j + 1) * N;
            const double* b2 = B.data() + (j + 2) * N;
            const double* b3 = B.data() + (j + 3) * N;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < N; ++k) {
                const double ak = a[k];
                s0 += ak * b0[k];
                s1 += ak * b1[k];
                s2 += ak * b2[k];
                s3 += ak * b3[k];
            }
            C[i * N + j + 0] = s0;
            C[i * N + j + 1] = s1;
            C[i * N + j + 2] = s2;
            C[i * N + j + 3] = s3;
        }
        for (; j < N; ++j) {
            const double* b = B.data() + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) sum += a[k] * b[k];
            C[i * N + j] = sum;
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t first, size_t rows) {
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t globalI = pi % N;
        if (globalI < first || globalI >= first + rows) continue;
        const size_t localI = globalI - first;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[localI * N + k] * B[j * N + k];
            }
            const double actual = C[localI * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            globalI, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void gatherResult(const std::vector<double>& local, std::vector<double>& full,
                  size_t N, int rank, int ranks) {
    // Chunk messages so MPI's int count limit does not constrain matrix size.
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    if (rank == 0) {
        std::copy(local.begin(), local.end(), full.begin());
        for (int source = 1; source < ranks; ++source) {
            size_t offset = firstRow(N, source, ranks) * N;
            size_t remaining = (firstRow(N, source + 1, ranks) - firstRow(N, source, ranks)) * N;
            while (remaining != 0) {
                const int count = static_cast<int>(std::min(remaining, chunk));
                MPI_Recv(full.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
                remaining -= count;
            }
        }
    } else {
        size_t offset = 0;
        while (offset < local.size()) {
            const int count = static_cast<int>(std::min(local.size() - offset, chunk));
            MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += count;
        }
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || value[0] == '-' ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                exitCode = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode == 0 && N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large\n");
        exitCode = 1;
    }
    if (exitCode != 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    const size_t first = firstRow(N, rank, ranks);
    const size_t rows = firstRow(N, rank + 1, ranks) - first;
    std::vector<double> A(rows * N), B(N * N), C(rows * N, 0.0);
    initRows(A, N, first, rows);
    initTransposedB(B, N);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double gflops = 2.0 * static_cast<double>(N) * N * N / seconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> full;
        if (rank == 0) full.resize(N * N);
        gatherResult(C, full, N, rank, ranks);
        if (rank == 0) print_results(full, "MatrixC");
    }

    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const int localFailure = !validateResult(A, B, C, N, first, rows);
        int failure = 0;
        MPI_Allreduce(&localFailure, &failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", failure ? "FAILED" : "PASSED");
        exitCode = failure;
    }
    MPI_Finalize();
    return exitCode;
}
