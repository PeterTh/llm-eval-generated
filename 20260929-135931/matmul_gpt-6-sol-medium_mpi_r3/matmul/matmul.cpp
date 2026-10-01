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

size_t firstRow(const size_t N, const int rank, const int ranks) {
    return (N / static_cast<size_t>(ranks)) * static_cast<size_t>(rank)
         + std::min(static_cast<size_t>(rank), N % static_cast<size_t>(ranks));
}

size_t rowCount(const size_t N, const int rank, const int ranks) {
    return N / static_cast<size_t>(ranks)
         + (static_cast<size_t>(rank) < N % static_cast<size_t>(ranks));
}

void initRows(std::vector<double>& mat, const size_t N, const size_t first) {
    for (size_t i = 0; i < mat.size() / N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, first + i, j);
        }
    }
}

// Each dot product retains k order; four output rows share each loaded B row.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; i += 4) {
        const size_t blockRows = std::min(size_t{4}, rows - i);
        double* c0 = C.data() + i * N;
        double* c1 = blockRows > 1 ? c0 + N : nullptr;
        double* c2 = blockRows > 2 ? c0 + 2 * N : nullptr;
        double* c3 = blockRows > 3 ? c0 + 3 * N : nullptr;
        for (size_t k = 0; k < N; ++k) {
            const double* b = B.data() + k * N;
            const double a0 = A[i * N + k];
            if (blockRows == 4) {
                const double a1 = A[(i + 1) * N + k];
                const double a2 = A[(i + 2) * N + k];
                const double a3 = A[(i + 3) * N + k];
                for (size_t j = 0; j < N; ++j) {
                    const double bj = b[j];
                    c0[j] += a0 * bj;
                    c1[j] += a1 * bj;
                    c2[j] += a2 * bj;
                    c3[j] += a3 * bj;
                }
            } else {
                for (size_t r = 0; r < blockRows; ++r) {
                    double* c = C.data() + (i + r) * N;
                    const double a = A[(i + r) * N + k];
                    for (size_t j = 0; j < N; ++j) c[j] += a * b[j];
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N,
                    const size_t first, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        if (i < first || i >= first + A.size() / N) continue;
        const size_t localRow = i - first;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[localRow * N + k] * B[k * N + j];
            const double actual = C[localRow * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!(relError <= 1e-6)) {
                std::printf("Rank %d: validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            rank, i, j, expected, actual, relError);
                valid = false;
            }
        }
    }
    return valid;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*end != '\0' || parsed == 0 || parsed > std::numeric_limits<size_t>::max())
                exitCode = 1;
            else
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
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            exitCode = 1;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::vector<double>().max_size() ||
        (printResults && N * N > static_cast<size_t>(std::numeric_limits<int>::max())))
        exitCode = 1;
    if (exitCode) {
        if (rank == 0) {
            std::fprintf(stderr, "Invalid matrix size or arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t first = firstRow(N, rank, ranks);
    const size_t rows = rowCount(N, rank, ranks);
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rows * N, 0.0);
    initRows(A, N, first);
    initRows(B, N, 0);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        const double gflops = (2.0 * N * N * N) / seconds / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> offsets;
        std::vector<double> fullC;
        if (rank == 0) {
            counts.resize(ranks);
            offsets.resize(ranks);
            fullC.resize(N * N);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(rowCount(N, r, ranks) * N);
                offsets[r] = static_cast<int>(firstRow(N, r, ranks) * N);
            }
        }
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? offsets.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }

    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const int localValid = validateResult(A, B, C, N, first, rank) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        exitCode = allValid ? 0 : 1;
    }
    MPI_Finalize();
    return exitCode;
}
