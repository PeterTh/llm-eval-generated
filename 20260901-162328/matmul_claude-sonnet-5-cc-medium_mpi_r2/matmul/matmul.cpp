#include <chrono>
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

// Initialize a contiguous block of rows [rowStart, rowStart + numRows) of an NxN matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t numRows) {
    for (size_t li = 0; li < numRows; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Multiply numRows rows of A (starting at global row rowStart) by full B, producing local C rows
void matrixMultiply(const std::vector<double>& ALocal, const std::vector<double>& B,
                    std::vector<double>& CLocal, const size_t N, const size_t numRows) {
    for (size_t li = 0; li < numRows; ++li) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += ALocal[li * N + k] * B[k * N + j];
            }
            CLocal[li * N + j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare.
// A values are regenerated on the fly since getPseudoRndValue is a pure function of (N, i, j),
// avoiding the need to gather the distributed A matrix onto rank 0.
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
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical across all MPI ranks)
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
        printf("MPI processes: %d\n", numProcs);
    }

    // Row decomposition of A/C across ranks: as evenly as possible
    const size_t baseRows = N / static_cast<size_t>(numProcs);
    const size_t remainder = N % static_cast<size_t>(numProcs);
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowDispls(numProcs);
    {
        size_t offset = 0;
        for (int r = 0; r < numProcs; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            rowCounts[r] = static_cast<int>(rows);
            rowDispls[r] = static_cast<int>(offset);
            offset += rows;
        }
    }
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t localRowStart = static_cast<size_t>(rowDispls[rank]);

    // Allocate matrices: each rank holds only its own rows of A/C, but the full B
    // (needed by every rank to compute its local rows of C).
    std::vector<double> ALocal(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> CLocal(localRows * N);

    // Initialize matrices. Each rank initializes its own rows of A directly (no
    // communication needed since getPseudoRndValue is a pure function of indices),
    // and every rank independently initializes the full B matrix it needs.
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(ALocal, N, localRowStart, localRows);
    initMatrixRows(B, N, 0, N);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(ALocal, B, CLocal, N, localRows);

    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed C rows onto rank 0 for reporting/validation
    std::vector<double> C;
    std::vector<int> elemCounts(numProcs);
    std::vector<int> elemDispls(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        elemCounts[r] = rowCounts[r] * static_cast<int>(N);
        elemDispls[r] = rowDispls[r] * static_cast<int>(N);
    }
    if (rank == 0) C.resize(N * N);
    MPI_Gatherv(CLocal.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, elemCounts.data(), elemDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
