#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <mpi.h>
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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rows = N == 0 ? 0 : A.size() / N;
    std::fill(C.begin(), C.end(), 0.0);
    // The i-k-j ordering reuses each A value and streams across B and C rows.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
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
    int rank = 0, processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments on all ranks (MPI launches the same argv).
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

    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) printf("Invalid matrix size (MPI counts must fit in int).\n");
        MPI_Finalize();
        return 1;
    }
    const size_t total = N * N;
    std::vector<int> rowCounts(processCount), rowDisplacements(processCount);
    std::vector<int> elementCounts(processCount), elementDisplacements(processCount);
    for (int p = 0; p < processCount; ++p) {
        const size_t first = N * static_cast<size_t>(p) / processCount;
        const size_t last = N * static_cast<size_t>(p + 1) / processCount;
        rowCounts[p] = static_cast<int>(last - first);
        rowDisplacements[p] = static_cast<int>(first);
        elementCounts[p] = static_cast<int>((last - first) * N);
        elementDisplacements[p] = static_cast<int>(first * N);
    }
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    std::vector<double> A(rank == 0 ? total : 0);
    std::vector<double> localA(localRows * N);
    std::vector<double> B(total);
    std::vector<double> C(rank == 0 ? total : 0);
    std::vector<double> localC(localRows * N);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(total), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, elementCounts.data(), elementDisplacements.data(), MPI_DOUBLE,
                 localA.data(), elementCounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, B, localC, N);
    MPI_Gatherv(localC.data(), elementCounts[rank], MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, elementCounts.data(), elementDisplacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);
    
    // Calculate GFLOPS
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (rank == 0 && validate) {
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
    
    MPI_Finalize();
    return 0;
}
