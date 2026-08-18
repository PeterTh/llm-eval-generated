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
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// B is stored transposed so both operands are traversed contiguously in the
// innermost loop.  Every rank constructs it directly, avoiding a broadcast.
void initLocalMatrices(std::vector<double>& A, std::vector<double>& BT,
                       size_t N, size_t firstRow, size_t localRows) {
    for (size_t i = 0; i < localRows; ++i)
        for (size_t k = 0; k < N; ++k)
            A[i * N + k] = getPseudoRndValue(N, firstRow + i, k);

    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < N; ++j)
            BT[j * N + k] = getPseudoRndValue(N, k, j);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& BT,
                    std::vector<double>& C, size_t N, size_t localRows) {
    // Row/column tiles keep several independent dot products in flight and
    // improve cache reuse while retaining the original summation order.
    constexpr size_t TILE = 32;
    for (size_t ii = 0; ii < localRows; ii += TILE) {
        const size_t iEnd = std::min(ii + TILE, localRows);
        for (size_t jj = 0; jj < N; jj += TILE) {
            const size_t jEnd = std::min(jj + TILE, N);
            for (size_t i = ii; i < iEnd; ++i) {
                const double* __restrict a = A.data() + i * N;
                for (size_t j = jj; j < jEnd; ++j) {
                    const double* __restrict b = BT.data() + j * N;
                    double sum = 0.0;
                    for (size_t k = 0; k < N; ++k) sum += a[k] * b[k];
                    C[i * N + j] = sum;
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N, j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value == 0 || value > std::numeric_limits<size_t>::max()) exitCode = 1;
            else N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); exitCode = 1; }
    }
    if (exitCode) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return exitCode;
    }

    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("MPI processes: %d\nValidation: %s\nInitializing matrices...\n",
                    ranks, validate ? "enabled" : "disabled");
    }
    std::vector<double> A(localRows * N), BT(N * N), localC(localRows * N);
    initLocalMatrices(A, BT, N, firstRow, localRows);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, BT, localC, N, localRows);
    const double localTime = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localTime, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> C;
    if (validate || printResults) {
        if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result is too large for MPI_Gatherv counts\n");
            MPI_Finalize();
            return 1;
        }
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = base + (static_cast<size_t>(r) < extra);
            const size_t row = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
            counts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>(row * N);
        }
        if (rank == 0) C.resize(N * N);
        MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(duration * 1000.0);
        const double gflops = 2.0 * static_cast<double>(N) * N * N / duration / 1e9;
        std::printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n", milliseconds, gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(C, N);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
