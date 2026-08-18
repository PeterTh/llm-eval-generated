#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate the same pseudo-random values as the serial benchmark.  Rows are
// generated from their global index, so no initialization communication is
// needed for A.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t firstRow) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat, N, 0);
}

// Broadcast in MPI_INT-sized pieces, keeping the code correct for large
// matrices as well as the usual benchmark sizes.
void broadcastDoubles(std::vector<double>& data, const int root) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < data.size(); offset += maxChunk) {
        const int count = static_cast<int>(std::min(maxChunk, data.size() - offset));
        MPI_Bcast(data.data() + offset, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }
}

// B is transposed once so the inner loop writes contiguous C elements.  Each
// element still accumulates k=0..N-1 in the same order as the original code.
void transpose(const std::vector<double>& matrix, std::vector<double>& transposed,
               const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            transposed[j * N + i] = matrix[i * N + j];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& Bt,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * Bt[j * N + k];
            }
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t firstRow,
                   const size_t rows, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t globalI = checkPoints[pi] % N;
        if (globalI < firstRow || globalI >= firstRow + rows) continue;
        const size_t i = globalI - firstRow;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                valid = false;
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       globalI, j, expected, actual, relError);
            }
        }
    }
    return valid;
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
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = std::strtoull(argv[++i], nullptr, 10);
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
            parseError = 1;
        }
    }
    if (N == 0) parseError = 1;
    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Bt(N * N);
    std::vector<double> C(localRows * N);
    initMatrixRows(A, N, firstRow);
    if (rank == 0) initMatrix(B, N);
    broadcastDoubles(B, 0);
    transpose(B, Bt, N);
    if (!validate) std::vector<double>().swap(B);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, Bt, C, localRows, N);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long localMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long milliseconds = 0;
    MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", milliseconds);
        const double seconds = milliseconds / 1000.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0);
    }

    std::vector<double> fullC;
    if (printResults) {
        if (rank == 0) fullC.resize(N * N);
        std::vector<int> counts(worldSize), displacements(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            counts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseRows +
                                                  std::min(static_cast<size_t>(r), extraRows)) * N);
        }
        MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, rank == 0 ? fullC.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }

    if (validate) {
        const bool localValid = validateResult(A, B, C, firstRow, localRows, N);
        int localStatus = localValid ? 1 : 0;
        int globalStatus = 0;
        MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validating result...\nValidation: %s\n", globalStatus ? "PASSED" : "FAILED");
        MPI_Finalize();
        return globalStatus ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
