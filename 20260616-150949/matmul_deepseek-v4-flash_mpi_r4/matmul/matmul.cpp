#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Cache-optimized local matrix multiply using i-k-j loop ordering
// Computes C += A * B for a local row range, where A and C have 'localRows' rows
// and B has N rows. All matrices are in row-major order.
static void matrixMultiplyLocal(const double* local_A, const double* B,
                                 double* local_C, size_t N, size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double aik = local_A[i * N + k];
            const double* B_row = &B[k * N];
            double* C_row = &local_C[i * N];
            for (size_t j = 0; j < N; ++j) {
                C_row[j] += aik * B_row[j];
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

    int numProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0
    int shouldExit = 0; // 0 = run, 1 = help, 2 = error
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 2;
            }
        }
    }

    // Broadcast parameters and exit flag to all ranks
    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (shouldExit) {
        MPI_Finalize();
        return shouldExit == 2 ? 1 : 0;
    }

    // Compute row distribution across processes
    const size_t baseLocalRows = N / static_cast<size_t>(numProcs);
    const size_t remainder = N % static_cast<size_t>(numProcs);
    const size_t localRows = baseLocalRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Compute send counts and displacements for scatterv/gatherv (all ranks)
    std::vector<int> counts(numProcs);
    std::vector<int> displs(numProcs);
    {
        size_t offset = 0;
        for (int i = 0; i < numProcs; ++i) {
            const size_t rows = baseLocalRows + (static_cast<size_t>(i) < remainder ? 1 : 0);
            counts[i] = static_cast<int>(rows * N);
            displs[i] = static_cast<int>(offset);
            offset += rows * N;
        }
    }

    // Full matrices only on rank 0
    std::vector<double> A_full, B_full, C_full;
    std::vector<double> B_local;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        A_full.resize(N * N);
        B_full.resize(N * N);
        C_full.resize(N * N);

        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B_full, N);

        B_local = B_full; // rank 0 has full B
    } else {
        B_local.resize(N * N);
    }

    // Allocate local portions
    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N, 0.0);

    // Scatter rows of A to all processes
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast B to all processes
    MPI_Bcast(B_local.data(), static_cast<int>(N * N), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    // Synchronize before timing the computation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    // Each process computes its portion of C
    matrixMultiplyLocal(A_local.data(), B_local.data(), C_local.data(), N, localRows);

    const double endTime = MPI_Wtime();
    const double localDuration = endTime - startTime;

    // Gather C results back to rank 0
    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C_full.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Reduce to get maximum computation time across all processes
    double maxDuration;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxDuration * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / maxDuration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(A_full, B_full, C_full, N)) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
