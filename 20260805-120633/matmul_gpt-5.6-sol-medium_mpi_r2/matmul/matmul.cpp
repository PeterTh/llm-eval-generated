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

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const std::size_t N, const std::size_t i,
                                   const std::size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct RowRange {
    std::size_t first;
    std::size_t count;
};

RowRange rowsForRank(const std::size_t N, const int rank, const int ranks) {
    const std::size_t base = N / static_cast<std::size_t>(ranks);
    const std::size_t extra = N % static_cast<std::size_t>(ranks);
    const std::size_t r = static_cast<std::size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1U : 0U)};
}

void initLocalA(std::vector<double>& A, const std::size_t N,
                const RowRange rows) {
    for (std::size_t i = 0; i < rows.count; ++i) {
        const std::size_t global_i = rows.first + i;
        for (std::size_t j = 0; j < N; ++j) {
            A[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
}

void initMatrix(std::vector<double>& matrix, const std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            matrix[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-blocked i-k-j multiplication. Each rank owns complete rows, so no
// communication is needed in the arithmetic hot path. The k blocks are
// visited in increasing order, preserving the serial summation order.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B, std::vector<double>& C,
                    const std::size_t localRows, const std::size_t N) {
    constexpr std::size_t block = 64;
    std::fill(C.begin(), C.end(), 0.0);

    for (std::size_t ii = 0; ii < localRows; ii += block) {
        const std::size_t iEnd = std::min(ii + block, localRows);
        for (std::size_t kk = 0; kk < N; kk += block) {
            const std::size_t kEnd = std::min(kk + block, N);
            for (std::size_t jj = 0; jj < N; jj += block) {
                const std::size_t jEnd = std::min(jj + block, N);
                for (std::size_t i = ii; i < iEnd; ++i) {
                    double* const c = C.data() + i * N;
                    const double* const a = A.data() + i * N;
                    for (std::size_t k = kk; k < kEnd; ++k) {
                        const double aik = a[k];
                        const double* const b = B.data() + k * N;
                        for (std::size_t j = jj; j < jEnd; ++j) {
                            c[j] += aik * b[j];
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& B,
                    const std::vector<double>& localC,
                    const RowRange rows, const std::size_t N,
                    const int rank) {
    constexpr std::size_t points[] = {0, 1, 2, 3, 4};
    double localValues[25] = {};
    double values[25] = {};

    for (std::size_t pi = 0; pi < 5; ++pi) {
        const std::size_t i = points[pi] % N;
        if (i < rows.first || i >= rows.first + rows.count) continue;
        for (std::size_t pj = 0; pj < 5; ++pj) {
            const std::size_t j = points[pj] % N;
            localValues[pi * 5 + pj] = localC[(i - rows.first) * N + j];
        }
    }
    MPI_Reduce(localValues, values, 25, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank != 0) return true;
    for (std::size_t pi = 0; pi < 5; ++pi) {
        for (std::size_t pj = 0; pj < 5; ++pj) {
            const std::size_t i = points[pi] % N;
            const std::size_t j = points[pj] % N;
            double expected = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }
            const double actual = values[pi * 5 + pj];
            const double error = std::abs((actual - expected) /
                                          (expected + 1e-10));
            if (error > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end != '\0' || value == 0 ||
                value > std::numeric_limits<std::size_t>::max()) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                parseStatus = 1;
            } else {
                N = static_cast<std::size_t>(value);
            }
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
            parseStatus = 1;
        }
    }
    if (parseStatus != 0 || N > std::numeric_limits<std::size_t>::max() / N) {
        if (rank == 0 && parseStatus == 0)
            std::fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    const RowRange rows = rowsForRank(N, rank, ranks);
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> A(rows.count * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rows.count * N);
    initLocalA(A, N, rows);
    initMatrix(B, N);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows.count, N);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = elapsed > 0.0
            ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
               static_cast<double>(N)) / elapsed / 1e9
            : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        const std::size_t total = N * N;
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0)
                std::fprintf(stderr, "Result is too large for MPI_Gatherv\n");
            status = 1;
        } else {
            std::vector<int> counts(static_cast<std::size_t>(ranks));
            std::vector<int> offsets(static_cast<std::size_t>(ranks));
            for (int r = 0; r < ranks; ++r) {
                const RowRange rr = rowsForRank(N, r, ranks);
                counts[static_cast<std::size_t>(r)] = static_cast<int>(rr.count * N);
                offsets[static_cast<std::size_t>(r)] = static_cast<int>(rr.first * N);
            }
            std::vector<double> result;
            if (rank == 0) result.resize(total);
            MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                        rank == 0 ? result.data() : nullptr, counts.data(),
                        offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0) print_results(result, "MatrixC");
        }
    }

    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(B, C, rows, N, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
