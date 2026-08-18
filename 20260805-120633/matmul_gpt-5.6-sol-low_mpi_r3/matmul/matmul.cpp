#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N,
                const size_t firstRow = 0) {
    const size_t rows = mat.size() / N;
    for (size_t localI = 0; localI < rows; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void transposeMatrix(const std::vector<double>& B, std::vector<double>& BT,
                     const size_t N) {
    constexpr size_t block = 32;
    for (size_t ii = 0; ii < N; ii += block) {
        for (size_t jj = 0; jj < N; jj += block) {
            for (size_t i = ii; i < std::min(ii + block, N); ++i) {
                for (size_t j = jj; j < std::min(jj + block, N); ++j) {
                    BT[j * N + i] = B[i * N + j];
                }
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& BT,
                    std::vector<double>& C, const size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * BT[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N,
                    const size_t firstRow) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            if (i < firstRow || i >= firstRow + A.size() / N) continue;
            const size_t localI = i - firstRow;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[localI * N + k] * B[k * N + j];
            }
            
            const double actual = C[localI * N + j];
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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N > SIZE_MAX / N) {
        if (rank == 0) fprintf(stderr, "Matrix size must be a positive, representable value\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    
    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> BT(N * N);
    std::vector<double> C(localRows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, firstRow);
    initMatrix(B, N);
    transposeMatrix(B, BT, N);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, BT, C, N);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate GFLOPS
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(worldSize), displacements(worldSize);
        bool gatherable = true;
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows);
            const size_t offset = static_cast<size_t>(r) * baseRows +
                                  std::min(static_cast<size_t>(r), extraRows);
            if (rows * N > INT_MAX || offset * N > INT_MAX) gatherable = false;
            counts[r] = gatherable ? static_cast<int>(rows * N) : 0;
            displacements[r] = gatherable ? static_cast<int>(offset * N) : 0;
        }
        int allGatherable = gatherable;
        MPI_Allreduce(MPI_IN_PLACE, &allGatherable, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!allGatherable) {
            if (rank == 0) fprintf(stderr, "Result is too large for MPI_Gatherv\n");
            MPI_Finalize();
            return 1;
        }
        std::vector<double> fullC(rank == 0 ? N * N : 0);
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    fullC.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool localValid = validateResult(A, B, C, N, firstRow);
        int valid = localValid;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
