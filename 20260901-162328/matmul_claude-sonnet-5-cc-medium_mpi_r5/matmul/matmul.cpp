#include <algorithm>
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

// Initialize only the rows [rowStart, rowStart + rowCount) of an NxN matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t rowCount) {
    for (size_t li = 0; li < rowCount; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Multiply a block of rows of A (rowCount x N) by the full B (N x N), writing rowCount x N rows of C
void matrixMultiplyRows(const std::vector<double>& Arows, const std::vector<double>& B,
                        std::vector<double>& Crows, const size_t N, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += Arows[i * N + k] * B[k * N + j];
            }
            Crows[i * N + j] = sum;
        }
    }
}

// Compute this rank's contiguous row range [rowStart, rowStart + rowCount) for an N-row matrix
// split as evenly as possible across numRanks ranks.
void computeRowRange(const size_t N, const int rank, const int numRanks,
                     size_t& rowStart, size_t& rowCount) {
    const size_t base = N / static_cast<size_t>(numRanks);
    const size_t rem = N % static_cast<size_t>(numRanks);
    const size_t r = static_cast<size_t>(rank);
    rowCount = base + (r < rem ? 1 : 0);
    rowStart = r * base + std::min(r, rem);
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Distribute rows of A (and C) across ranks; B is replicated on every rank.
    size_t rowStart = 0;
    size_t rowCount = 0;
    computeRowRange(N, rank, numRanks, rowStart, rowCount);

    std::vector<double> localA(rowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(rowCount * N);

    // Initialize matrices: each rank generates only its own rows of A, and the
    // full (deterministically derived) B locally, avoiding any communication.
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(localA, N, rowStart, rowCount);
    initMatrix(B, N);

    // Perform distributed matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(localA, B, localC, N, rowCount);

    auto end = std::chrono::high_resolution_clock::now();
    auto localDurationMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

    double maxDurationMs = 0.0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed rows of C onto rank 0
    std::vector<double> C;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        C.resize(N * N);
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            size_t rStart = 0;
            size_t rCount = 0;
            computeRowRange(N, r, numRanks, rStart, rCount);
            recvCounts[r] = static_cast<int>(rCount * N);
            displs[r] = static_cast<int>(rStart * N);
        }
    }
    MPI_Gatherv(localC.data(), static_cast<int>(rowCount * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxDurationMs));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDurationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            initMatrix(A, N);
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
