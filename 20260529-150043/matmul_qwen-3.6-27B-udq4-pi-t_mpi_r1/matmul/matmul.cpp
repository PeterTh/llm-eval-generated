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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t localRows, const size_t N) {
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 handles args, broadcasts to others)
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

    // Print info from rank 0 only
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution: each rank gets N/numRanks rows, with remainder
    // distributed to the first (N % numRanks) ranks
    const size_t baseRows = N / numRanks;
    const size_t remainder = N % numRanks;

    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    size_t offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rrows = baseRows + (r < static_cast<int>(remainder) ? 1 : 0);
        counts[r] = static_cast<int>(rrows * N);
        displs[r] = static_cast<int>(offset * N);
        offset += rrows;
    }

    // Determine local row count for this rank
    const size_t localRows = baseRows + (rank < static_cast<int>(remainder) ? 1 : 0);

    // Allocate and initialize matrices
    // A: local rows only; B: full matrix on every rank; C: local rows only
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);

    // Initialize B on all ranks (deterministic, same values everywhere)
    initMatrix(B, N);

    // Initialize A on rank 0, then scatter to all ranks
    std::vector<double> A_full(N * N);
    if (rank == 0) {
        initMatrix(A_full, N);
    }

    // Scatter A rows: rank r gets rows [rowStart, rowStart + localRows)
    MPI_Scatterv(A_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, localRows, N);

    auto end = std::chrono::high_resolution_clock::now();

    // Synchronize after timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather C rows back to rank 0
    std::vector<double> C_full(N * N);
    MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Output results from rank 0 only
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = 0.0;
        if (duration.count() > 0) {
            gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        }
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
