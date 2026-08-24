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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Optimized matrix multiplication with (i,k,j) loop order for cache efficiency
// Each process computes its assigned row block of C
void matrixMultiplyBlock(const std::vector<double>& A,
                         const std::vector<double>& B, std::vector<double>& C,
                         const size_t N, const size_t localRows) {
    for (size_t ii = 0; ii < localRows; ++ii) {
        const double* A_row = A.data() + ii * N;
        double* C_row = C.data() + ii * N;

        // Zero out the result row
        std::memset(C_row, 0, N * sizeof(double));

        // (i,k,j) loop order: for each k, stream through B row k and accumulate into C row i
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A_row[k];
            const double* B_row = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                C_row[j] += a_ik * B_row[j];
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0, broadcast to all
    if (rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters from rank 0 to all processes
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution: each process gets a contiguous block of rows
    size_t totalRows = N;
    size_t baseRows = totalRows / static_cast<size_t>(numProcs);
    size_t remainder = totalRows % static_cast<size_t>(numProcs);
    size_t myRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Allocate matrices
    // Each process stores: its rows of A, full matrix B, its rows of C
    size_t totalElements = N * N;
    size_t myAElements = myRows * N;
    size_t myCElements = myRows * N;

    std::vector<double> A(myAElements);
    std::vector<double> B(totalElements);
    std::vector<double> C(myCElements);

    // Initialize matrices on rank 0, then distribute
    std::vector<double> A_full(totalElements);
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    }

    // Broadcast B to all processes (needed by all for multiplication)
    MPI_Bcast(B.data(), totalElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A: send each process its assigned rows
    // Build sendcounts and displacements for MPI_Scatterv
    std::vector<int> sendcounts(numProcs);
    std::vector<int> displs(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        size_t pBase = baseRows * static_cast<size_t>(p) + std::min(static_cast<size_t>(p), remainder);
        size_t pRows = baseRows + (static_cast<size_t>(p) < remainder ? 1 : 0);
        sendcounts[p] = static_cast<int>(pRows * N);
        displs[p] = static_cast<int>(pBase * N);
    }

    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), static_cast<int>(myAElements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyBlock(A, B, C, N, myRows);

    auto end = std::chrono::high_resolution_clock::now();

    // Synchronize all processes before timing to account for stragglers
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C back to rank 0 for validation and output
    std::vector<double> C_full(totalElements);
    MPI_Gatherv(C.data(), static_cast<int>(myCElements), MPI_DOUBLE,
                C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B, C_full, N);

            if (valid) {
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
