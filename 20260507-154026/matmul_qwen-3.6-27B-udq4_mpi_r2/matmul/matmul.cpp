#include <mpi.h>

#include <chrono>
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

// 1D block row distribution: each rank owns a contiguous block of rows of C.
// A's rows are distributed (rank r owns rows rowStart..rowEnd-1).
// B is broadcast in full to every rank.
// Each rank computes its local C block, then results are gathered to rank 0.
void matrixMultiplyMPI(const std::vector<double>& A, const std::vector<double>& B,
                       std::vector<double>& C, const size_t N,
                       const int rank, const int numRanks) {
    // Compute row distribution
    const size_t rowsPerRank = N / numRanks;
    const size_t remainder = N % numRanks;
    size_t rowStart = rowsPerRank * rank + (rank < static_cast<int>(remainder) ? rank : remainder);
    size_t localRows = rowsPerRank + (rank < static_cast<int>(remainder) ? 1 : 0);

    // Local buffers: A block (localRows x N), full B (N x N), local C (localRows x N)
    std::vector<double> localA(localRows * N);
    std::vector<double> localB(N * N);
    std::vector<double> localC(localRows * N, 0.0);

    // Extract my rows of A
    for (size_t i = 0; i < localRows; ++i) {
        std::memcpy(localA.data() + i * N,
                    A.data() + (rowStart + i) * N,
                    N * sizeof(double));
    }

    // Copy B to local buffer on rank 0, then broadcast to all ranks
    if (rank == 0) {
        std::memcpy(localB.data(), B.data(), N * N * sizeof(double));
    }
    MPI_Bcast(localB.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local matrix multiply: localC = localA * localB
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += localA[i * N + k] * localB[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }

    // Gather local C blocks back to rank 0
    // Build count/displacement arrays
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    size_t disp = 0;
    for (int r = 0; r < numRanks; ++r) {
        size_t rRows = rowsPerRank + (r < static_cast<int>(remainder) ? 1 : 0);
        counts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(disp);
        disp += rRows * N;
    }

    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, same args expected)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (full on all ranks for init, then distributed for compute)
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    // Initialize matrices on all ranks (deterministic, same result everywhere)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform distributed matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::high_resolution_clock::now();

    matrixMultiplyMPI(A, B, C, N, rank, numRanks);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

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
