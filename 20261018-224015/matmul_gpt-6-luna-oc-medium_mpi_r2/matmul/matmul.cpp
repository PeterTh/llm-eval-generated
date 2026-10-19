#include <chrono>
#include <algorithm>
#include <climits>
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

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow = 0) {
    for (size_t i = 0; i < mat.size() / N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, const size_t rows) {
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double aik = A[i * N + k];
            const size_t bRow = k * N;
            const size_t cRow = i * N;
            for (size_t j = 0; j < N; ++j) {
                C[cRow + j] += aik * B[bRow + j];
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

    if (N == 0 || N > static_cast<size_t>(INT_MAX) / N || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Invalid or unsupported matrix size\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        const size_t startRow = static_cast<size_t>(r) * baseRows +
                                (static_cast<size_t>(r) < extraRows ? static_cast<size_t>(r) : extraRows);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>(startRow * N);
    }
    
    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);
    std::vector<double> C(rank == 0 ? N * N : 0);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, firstRow);
    initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, localC, N, localRows);
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration);
    
    // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
    }
    
    // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA(N * N);
            initMatrix(fullA, N);
            bool valid = validateResult(fullA, B, C, N);
        
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
