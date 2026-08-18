#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Keep initialization identical to the original benchmark.  Matrices are
// distributed by rows, so the global row index is passed to this function.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initRows(std::vector<double>& mat, const size_t N, const size_t firstRow) {
    const size_t rows = N == 0 ? 0 : mat.size() / N;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

void initTransposedMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[j * N + i] = getPseudoRndValue(N, i, j);
}

// Broadcast large buffers in MPI_INT-sized pieces.  This also avoids an
// unnecessary MPI datatype and makes the input scale past the count limit.
void broadcastDoubles(double* data, size_t count, const int root) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& BT,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    // BT is transposed once, making both operands contiguous in the inner
    // loop.  The k loop remains in increasing order, preserving the original
    // floating-point reduction order for each output element.
    constexpr size_t tile = 32;
    for (size_t ii = 0; ii < rows; ii += tile) {
        for (size_t jj = 0; jj < N; jj += tile) {
            const size_t iEnd = std::min(ii + tile, rows);
            const size_t jEnd = std::min(jj + tile, N);
            for (size_t i = ii; i < iEnd; ++i) {
                for (size_t j = jj; j < jEnd; ++j) {
                    double sum = 0.0;
                    const double* a = A.data() + i * N;
                    const double* b = BT.data() + j * N;
                    for (size_t k = 0; k < N; ++k)
                        sum += a[k] * b[k];
                    C[i * N + j] = sum;
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }
            const double actual = C[i * N + j];
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
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
            parseStatus = 1;
        }
    }
    if (parseStatus != 0 || N == 0) {
        MPI_Finalize();
        return parseStatus != 0 ? 1 : 0;
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min<size_t>(static_cast<size_t>(rank), extraRows);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N);
    std::vector<double> BT(N * N);
    std::vector<double> C(localRows * N);
    initRows(A, N, firstRow);
    if (rank == 0) initTransposedMatrix(BT, N);
    broadcastDoubles(BT.data(), BT.size(), 0);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, BT, C, localRows, N);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> fullC;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        for (int r = 0; r < worldSize; ++r) {
            const size_t rFirst = static_cast<size_t>(r) * baseRows +
                                  std::min<size_t>(static_cast<size_t>(r), extraRows);
            const size_t rRows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            counts[r] = static_cast<int>(rRows * N);
            displacements[r] = static_cast<int>(rFirst * N);
        }
        fullC.resize(N * N);
    }
    MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                rank == 0 ? fullC.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double seconds = maxElapsed > 0.0 ? maxElapsed : 1e-12;
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        if (printResults) print_results(fullC, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullC, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
