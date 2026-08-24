#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow = 0) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; ++i) {
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
bool validateResult(const std::vector<double>& B, const std::vector<double>& C,
                   const size_t N, const size_t firstRow) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t globalI = firstRow + i;
            if (i >= C.size() / N) continue;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, globalI, k) * B[k * N + j];
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
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (N == 0 || N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Matrix size must be between 1 and INT_MAX\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    const size_t base = N / static_cast<size_t>(world);
    const size_t remainder = N % static_cast<size_t>(world);
    const size_t localRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = base * static_cast<size_t>(rank) +
                            std::min(static_cast<size_t>(rank), remainder);
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, firstRow);
    initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    long long localDuration = static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    long long durationUs = 0;
    MPI_Reduce(&localDuration, &durationUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    const double seconds = static_cast<double>(durationUs) / 1e6;
    if (rank == 0) printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) fullC.resize(N * N);
        std::vector<int> counts(world), displs(world);
        for (int p = 0; p < world; ++p) { size_t rows = base + (static_cast<size_t>(p) < remainder); counts[p] = static_cast<int>(rows * N); displs[p] = static_cast<int>((base * p + std::min(static_cast<size_t>(p), remainder)) * N); }
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE, rank == 0 ? fullC.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool localValid = validateResult(B, C, N, firstRow);
        int validInt = localValid ? 1 : 0, allValid = 0;
        MPI_Reduce(&validInt, &allValid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        bool valid = rank == 0 && allValid;
        
        if (rank == 0 && valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else if (rank == 0) {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
