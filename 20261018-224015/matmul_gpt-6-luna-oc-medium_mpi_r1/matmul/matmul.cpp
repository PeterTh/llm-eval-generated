#include <chrono>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const size_t firstRow, const size_t rowCount) {
    for (size_t i = firstRow; i < firstRow + rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[(i - firstRow) * N + j] = 0.0;
        }
    }
    // i-k-j ordering reuses each value from A across a contiguous row of B.
    for (size_t i = firstRow; i < firstRow + rowCount; ++i) {
        double* const out = C.data() + (i - firstRow) * N;
        for (size_t k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            const double* const brow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                out[j] += a * brow[j];
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
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
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
    
    if (N == 0 || N > static_cast<size_t>(INT_MAX) / sizeof(double) / N) {
        if (rank == 0) printf("Invalid matrix size\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rank == 0 ? N * N : 0);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    const int matrixElements = static_cast<int>(N * N);
    MPI_Bcast(A.data(), matrixElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), matrixElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    const size_t baseRows = N / static_cast<size_t>(processCount);
    const size_t extraRows = N % static_cast<size_t>(processCount);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);
    std::vector<double> localC(localRows * N);
    matrixMultiply(A, B, localC, N, firstRow, localRows);
    std::vector<int> recvCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        recvCounts.resize(processCount);
        displacements.resize(processCount);
        for (int r = 0; r < processCount; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            const size_t row = static_cast<size_t>(r) * baseRows +
                               (static_cast<size_t>(r) < extraRows ? static_cast<size_t>(r) : extraRows);
            recvCounts[r] = static_cast<int>(rows * N);
            displacements[r] = static_cast<int>(row * N);
        }
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = duration.count() ? (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9 : 0.0;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    int valid = 1;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
