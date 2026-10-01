#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initRows(std::vector<double>& mat, size_t N, size_t firstRow) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool validateLocal(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, size_t N, size_t firstRow) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    const size_t rows = A.size() / N;
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        if (i < firstRow || i - firstRow >= rows) continue;
        const size_t localRow = i - firstRow;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[localRow * N + k] * B[k * N + j];
            }
            const double actual = C[localRow * N + j];
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

// MPI counts are ints. Transfer large buffers in chunks so matrix size is
// limited by address space rather than the MPI count representation.
void broadcastMatrix(std::vector<double>& B) {
    for (size_t offset = 0; offset < B.size();) {
        const int count = static_cast<int>(std::min(B.size() - offset,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_Bcast(B.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
}

void gatherResult(const std::vector<double>& localC, std::vector<double>& C,
                  size_t N, int rank, int ranks) {
    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    if (rank == 0) {
        std::copy(localC.begin(), localC.end(), C.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t firstRow = static_cast<size_t>(source) * baseRows +
                                    std::min(static_cast<size_t>(source), extraRows);
            size_t remaining = (baseRows + (static_cast<size_t>(source) < extraRows)) * N;
            size_t offset = firstRow * N;
            while (remaining) {
                const int count = static_cast<int>(std::min(remaining,
                                                            static_cast<size_t>(INT_MAX)));
                MPI_Recv(C.data() + offset, count, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
                remaining -= count;
            }
        }
    } else {
        for (size_t offset = 0; offset < localC.size();) {
            const int count = static_cast<int>(std::min(localC.size() - offset,
                                                        static_cast<size_t>(INT_MAX)));
            MPI_Send(localC.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += count;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N, 0.0);
    initRows(A, N, firstRow);
    if (rank == 0) initRows(B, N, 0);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    broadcastMatrix(B);
    matrixMultiply(A, B, localC, N);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double gflops = (2.0 * N * N * N) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        gatherResult(localC, C, N, rank, ranks);
        if (rank == 0) print_results(C, "MatrixC");
    }

    int success = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localSuccess = validateLocal(A, B, localC, N, firstRow) ? 1 : 0;
        MPI_Allreduce(&localSuccess, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", success ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return success ? 0 : 1;
}
