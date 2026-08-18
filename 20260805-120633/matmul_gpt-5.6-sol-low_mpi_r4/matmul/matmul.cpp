#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// Every rank can deterministically generate B, avoiding a root bottleneck and
// a large broadcast before the timed computation.
void initLocalMatrices(std::vector<double>& A, std::vector<double>& B,
                       size_t N, size_t firstRow, size_t localRows) {
    for (size_t i = 0; i < localRows; ++i)
        for (size_t k = 0; k < N; ++k)
            A[i * N + k] = getPseudoRndValue(N, firstRow + i, k);

    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < N; ++j)
            B[k * N + j] = getPseudoRndValue(N, k, j);
}

void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B, std::vector<double>& C,
                    size_t N, size_t localRows) {
    constexpr size_t block = 64;
    for (size_t ii = 0; ii < localRows; ii += block) {
        const size_t iEnd = std::min(localRows, ii + block);
        for (size_t jj = 0; jj < N; jj += block) {
            const size_t jEnd = std::min(N, jj + block);
            for (size_t kk = 0; kk < N; kk += block) {
                const size_t kEnd = std::min(N, kk + block);
                for (size_t i = ii; i < iEnd; ++i) {
                    double* c = C.data() + i * N;
                    const double* a = A.data() + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aik = a[k];
                        const double* b = B.data() + k * N;
                        for (size_t j = jj; j < jEnd; ++j)
                            c[j] += aik * b[j];
                    }
                }
            }
        }
    }
}

bool validateLocal(const std::vector<double>& A,
                   const std::vector<double>& B,
                   const std::vector<double>& C, size_t N, size_t firstRow,
                   size_t localRows) {
    const size_t checks = std::min<size_t>(5, N);
    for (size_t i = 0; i < checks; ++i) {
        if (i < firstRow || i >= firstRow + localRows) continue;
        const size_t li = i - firstRow;
        for (size_t j = 0; j < checks; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[li * N + k] * B[k * N + j];
            const double actual = C[li * N + j];
            const double error = std::abs((actual - expected) /
                                          (expected + 1e-10));
            if (error > 1e-6) {
                std::fprintf(stderr,
                    "Validation failed at (%zu, %zu): expected %.10f, "
                    "got %.10f (error: %.10e)\n", i, j, expected, actual,
                    error);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) parseStatus = 1;
            else N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else parseStatus = 1;
    }
    if (parseStatus) {
        if (rank == 0) {
            if (parseStatus == 1) std::fprintf(stderr, "Invalid command line\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }
    const size_t base = N / static_cast<size_t>(ranks);
    const size_t extra = N % static_cast<size_t>(ranks);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N), B(N * N), C(localRows * N);
    initLocalMatrices(A, B, N, firstRow, localRows);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, localRows);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double n = static_cast<double>(N);
        std::printf("Performance: %.3f GFLOPS\n",
                    duration > 0.0 ? 2.0 * n * n * n / duration / 1e9 : 0.0);
    }

    if (printResults) {
        // MPI_Gatherv counts are int. Chunking rows keeps the implementation
        // correct even when the complete matrix contains more than INT_MAX
        // elements.
        std::vector<double> fullC;
        if (rank == 0) fullC.resize(N * N);
        const size_t rowsPerRound = std::max<size_t>(1,
            static_cast<size_t>(std::numeric_limits<int>::max()) / N);
        std::vector<int> counts(ranks), displacements(ranks);
        for (size_t begin = 0; begin < N; begin += rowsPerRound) {
            const size_t end = std::min(N, begin + rowsPerRound);
            for (int r = 0; r < ranks; ++r) {
                const size_t rr = static_cast<size_t>(r);
                const size_t rFirst = rr * base + std::min(rr, extra);
                const size_t rRows = base + (rr < extra);
                const size_t lo = std::max(begin, rFirst);
                const size_t hi = std::min(end, rFirst + rRows);
                counts[r] = static_cast<int>((hi > lo ? hi - lo : 0) * N);
                displacements[r] = hi > lo
                    ? static_cast<int>((lo - begin) * N) : 0;
            }
            const size_t sendBegin = std::max(begin, firstRow);
            const size_t sendEnd = std::min(end, firstRow + localRows);
            const int sendCount = static_cast<int>(
                (sendEnd > sendBegin ? sendEnd - sendBegin : 0) * N);
            const double* send = C.data() +
                (sendCount ? (sendBegin - firstRow) * N : 0);
            double* recv = rank == 0 ? fullC.data() + begin * N : nullptr;
            MPI_Gatherv(send, sendCount, MPI_DOUBLE, recv, counts.data(),
                        displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) print_results(fullC, "MatrixC");
    }

    int localValid = !validate || validateLocal(A, B, C, N, firstRow, localRows);
    int globallyValid = 0;
    MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (validate && rank == 0)
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");

    MPI_Finalize();
    return validate && !globallyValid ? 1 : 0;
}
