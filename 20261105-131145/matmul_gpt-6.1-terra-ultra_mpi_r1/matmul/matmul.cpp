#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Both inputs are deterministic, so every rank can construct the portions it needs.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t begin;
    size_t size;
};

Range partition(size_t n, int parts, int index) {
    const size_t p = static_cast<size_t>(parts);
    const size_t i = static_cast<size_t>(index);
    return {i * (n / p) + std::min(i, n % p), n / p + (i < n % p)};
}

// A is rows x N, B is N x columns, and C is rows x columns. Each
// output's k terms are accumulated in the same order as the original code.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows, size_t columns) {
    // The original GCC/native dot product uses separate operations for pairs
    // of k values and a fused operation for the final unpaired value.
    const size_t endK = N - N % 2;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < endK; ++k) {
            const double a = A[i * N + k];
            for (size_t j = 0; j < columns; ++j)
                C[i * columns + j] += a * B[k * columns + j];
        }
        if (endK != N) for (size_t j = 0; j < columns; ++j)
            C[i * columns + j] = std::fma(A[i * N + endK], B[endK * columns + j], C[i * columns + j]);
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

// MPI's traditional point-to-point interface uses int counts.
void transferBlock(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send)
            MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool stop = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || value[0] == '\0' || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                exitCode = 1;
                stop = true;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            stop = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            stop = true;
            break;
        }
    }
    if (!stop && N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) printf("Matrix size is too large\n");
        exitCode = 1;
        stop = true;
    }
    if (stop) {
        MPI_Finalize();
        return exitCode;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const Range rows = partition(N, dims[0], rank / dims[1]);
    const Range columns = partition(N, dims[1], rank % dims[1]);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    // Empty blocks need neither input, including when there are more ranks than rows or columns.
    const bool active = rows.size != 0 && columns.size != 0;
    std::vector<double> A(active ? rows.size * N : 0);
    std::vector<double> B(active ? N * columns.size : 0);
    std::vector<double> C(rows.size * columns.size);
    if (active) {
        for (size_t i = 0; i < rows.size; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, rows.begin + i, k);
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < columns.size; ++j)
                B[k * columns.size + j] = getPseudoRndValue(N, k, columns.begin + j);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) matrixMultiply(A, B, C, N, rows.size, columns.size);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
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
            for (int source = 0; source < ranks; ++source) {
                const Range sourceRows = partition(N, dims[0], source / dims[1]);
                const Range sourceCols = partition(N, dims[1], source % dims[1]);
                const size_t count = sourceRows.size * sourceCols.size;
                std::vector<double> received(source == 0 ? 0 : count);
                const double* block = source == 0 ? C.data() : received.data();
                if (source != 0) transferBlock(received.data(), count, source, false);
                if (sourceCols.size != 0) {
                    for (size_t i = 0; i < sourceRows.size; ++i)
                        std::copy_n(block + i * sourceCols.size, sourceCols.size,
                                    result.data() + (sourceRows.begin + i) * N + sourceCols.begin);
                }
            }
            print_results(result, "MatrixC");
        } else {
            transferBlock(C.data(), C.size(), 0, true);
        }
    }

    if (validate) {
        std::array<double, 25> localChecks{};
        std::array<double, 25> checks{};
        if (N != 0) {
            for (size_t pi = 0; pi < 5; ++pi) {
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t i = pi % N, j = pj % N;
                    if (i >= rows.begin && i - rows.begin < rows.size &&
                        j >= columns.begin && j - columns.begin < columns.size)
                        localChecks[pi * 5 + pj] = C[(i - rows.begin) * columns.size + j - columns.begin];
                }
            }
        }
        MPI_Reduce(localChecks.data(), checks.data(), 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = true;
            for (size_t pi = 0; pi < 5 && valid && N != 0; ++pi) {
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t i = pi % N, j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double actual = checks[pi * 5 + pj];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, relError);
                        valid = false;
                        break;
                    }
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return exitCode;
}
