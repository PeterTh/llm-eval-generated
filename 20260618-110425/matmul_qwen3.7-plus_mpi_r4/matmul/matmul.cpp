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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize a contiguous block of rows [startRow, startRow+numRows)
void initMatrixRows(std::vector<double>& mat, const size_t N, size_t startRow, size_t numRows) {
    for (size_t i = 0; i < numRows; ++i) {
        size_t globalRow = startRow + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// Cache-optimized local matrix multiply: C_local += A_local * B
// Uses ikj loop order for sequential memory access in the inner loop.
void matrixMultiplyLocal(const double* __restrict__ A, const double* __restrict__ B,
                         double* __restrict__ C, const size_t N, const size_t localRows) {
    std::memset(C, 0, localRows * N * sizeof(double));

    for (size_t i = 0; i < localRows; ++i) {
        const double* __restrict__ Ai = A + i * N;
        double* __restrict__ Ci = C + i * N;
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = Ai[k];
            const double* __restrict__ Bk = B + k * N;
            for (size_t j = 0; j < N; ++j) {
                Ci[j] += a_ik * Bk[j];
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D block row decomposition: distribute rows of A and C across ranks.
    // Ranks [0, rem) each get (baseRows+1) rows; ranks [rem, nprocs) get baseRows rows.
    const size_t baseRows = N / nprocs;
    const size_t rem = N % static_cast<size_t>(nprocs);

    auto getLocalRows = [&](int r) -> size_t {
        return (static_cast<size_t>(r) < rem) ? baseRows + 1 : baseRows;
    };

    auto getStartRow = [&](int r) -> size_t {
        if (static_cast<size_t>(r) < rem) {
            return static_cast<size_t>(r) * (baseRows + 1);
        } else {
            return rem * (baseRows + 1) + (static_cast<size_t>(r) - rem) * baseRows;
        }
    };

    const size_t localRows = getLocalRows(rank);
    const size_t startRow = getStartRow(rank);

    // Allocate local matrices
    std::vector<double> A_local(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(localRows * N);

    // Each rank initializes its own rows of A deterministically (no communication)
    initMatrixRows(A_local, N, startRow, localRows);

    // Rank 0 initializes full B, then broadcast to all ranks
    if (rank == 0) {
        initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform local matrix multiplication (timed)
    double localStart = MPI_Wtime();

    matrixMultiplyLocal(A_local.data(), B.data(), C_local.data(), N, localRows);

    double localEnd = MPI_Wtime();
    double localTime = localEnd - localStart;

    // Reduce to get the maximum computation time across all ranks
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Build recvcounts and displacements for Gatherv
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        recvcounts[r] = static_cast<int>(getLocalRows(r) * N);
        displs[r] = static_cast<int>(getStartRow(r) * N);
    }

    // Gather C results to rank 0
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        long durationMs = static_cast<long>(maxTime * 1000.0);

        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxTime) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Re-initialize full A on rank 0 for validation
            std::vector<double> A_full(N * N);
            initMatrix(A_full, N);

            bool valid = validateResult(A_full, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Broadcast exit code so all ranks return consistently
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
