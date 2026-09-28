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

// Multiply only the local block of rows [rowStart, rowStart + localRows) of A by B,
// writing into the corresponding rows of localC (indexed from 0).
void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& localC, const size_t N,
                         const size_t rowStart, const size_t localRows) {
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            localC[li * N + j] = sum;
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

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank)
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

    // Row-block distribution of A/C across ranks
    const size_t baseRows = N / static_cast<size_t>(numProcs);
    const size_t extraRows = N % static_cast<size_t>(numProcs);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t rowStart = static_cast<size_t>(rank) * baseRows +
                             std::min(static_cast<size_t>(rank), extraRows);

    // A and B are cheap to (re)generate deterministically on every rank (O(N^2)),
    // so no communication is needed to distribute the inputs.
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication (distributed by row blocks)
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> localC(localRows * N);
    matrixMultiplyLocal(A, B, localC, N, rowStart, localRows);

    std::vector<double> C;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        C.resize(N * N);
        recvCounts.resize(numProcs);
        displs.resize(numProcs);
        for (int p = 0; p < numProcs; ++p) {
            const size_t rows = baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
            const size_t start_ = static_cast<size_t>(p) * baseRows +
                                   std::min(static_cast<size_t>(p), extraRows);
            recvCounts[p] = static_cast<int>(rows * N);
            displs[p] = static_cast<int>(start_ * N);
        }
    }

    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;

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
