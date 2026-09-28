#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a contiguous block of rows [rowStart, rowEnd) of an NxN matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t rowEnd) {
    for (size_t i = rowStart; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowStart) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Multiply a local block of rows of A ([rowStart, rowEnd) of the global matrix) by the full matrix B
void matrixMultiplyBlock(const std::vector<double>& Alocal, const std::vector<double>& B,
                          std::vector<double>& Clocal, const size_t N,
                          const size_t rowStart, const size_t rowEnd) {
    for (size_t i = rowStart; i < rowEnd; ++i) {
        const size_t li = i - rowStart;
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += Alocal[li * N + k] * B[k * N + j];
            }
            Clocal[li * N + j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare. A's values are reconstructed
// on the fly via getPseudoRndValue since A is distributed across ranks and not gathered.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank under mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Distribute rows of A (and C) as evenly as possible across ranks
    const size_t baseRows = N / static_cast<size_t>(numRanks);
    const size_t remainder = N % static_cast<size_t>(numRanks);
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowOffsets(numRanks);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            rowOffsets[r] = static_cast<int>(offset);
            rowCounts[r] = static_cast<int>(rows);
            offset += rows;
        }
    }
    const size_t rowStart = static_cast<size_t>(rowOffsets[rank]);
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t rowEnd = rowStart + localRows;

    // Allocate matrices: B is fully replicated (needed in its entirety by every rank),
    // A and C are held only as the local row block owned by this rank.
    std::vector<double> Alocal(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Clocal(localRows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(Alocal, N, rowStart, rowEnd);
    initMatrixRows(B, N, 0, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiplyBlock(Alocal, B, Clocal, N, rowStart, rowEnd);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full result matrix on rank 0 for validation / output
    std::vector<double> C;
    std::vector<int> recvCounts(numRanks);
    std::vector<int> recvDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        recvCounts[r] = rowCounts[r] * static_cast<int>(N);
        recvDispls[r] = rowOffsets[r] * static_cast<int>(N);
    }
    if (rank == 0) {
        C.resize(N * N);
    }
    MPI_Gatherv(Clocal.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
