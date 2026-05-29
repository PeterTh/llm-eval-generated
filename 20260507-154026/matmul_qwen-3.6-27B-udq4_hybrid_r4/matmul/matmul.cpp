#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "matmul_kernel.h"
#include "../common/results_output.hpp"

// Matrix initialization (OpenMP)
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Validation (sequential reference)
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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

    int rank = 0, numRanks = 1;
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

    // Print banner (rank 0 only)
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Parallelization: MPI (%d ranks) + CUDA + OpenMP (%d threads)\n",
               numRanks, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate & initialize matrices on rank 0
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Broadcast A and B to all ranks
    MPI_Bcast(A.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local row range (block distribution)
    size_t totalRows = N;
    size_t baseRows = totalRows / numRanks;
    size_t remainder = totalRows % numRanks;
    size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Allocate local C
    std::vector<double> C_local(localRows * N, 0.0);

    // Timed computation
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    // Each rank runs its GPU kernel on its assigned rows
    gpuMatmul(A.data(), B.data(), C_local.data(), static_cast<int>(localRows), static_cast<int>(N));

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results to rank 0
    std::vector<double> C(N * N);
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);

    for (int r = 0; r < numRanks; ++r) {
        size_t rBase = totalRows / numRanks;
        size_t rRem = totalRows % numRanks;
        size_t rLocalRows = rBase + (static_cast<size_t>(r) < rRem ? 1 : 0);
        recvCounts[r] = static_cast<int>(rLocalRows * N);

        size_t rStart = rBase * r + std::min(static_cast<size_t>(r), rRem);
        displs[r] = static_cast<int>(rStart * N);
    }

    MPI_Gatherv(C_local.data(),
                static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(),
                recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Report results (rank 0)
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
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
