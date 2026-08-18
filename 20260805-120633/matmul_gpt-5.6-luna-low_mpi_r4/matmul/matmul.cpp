#include <chrono>
#include <cmath>
#include <climits>
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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& BT,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    // BT is column-contiguous, which makes the innermost dot product cache-friendly.
    constexpr size_t block = 32;
    for (size_t ii = 0; ii < rows; ii += block) {
        for (size_t jj = 0; jj < N; jj += block) {
            for (size_t kk = 0; kk < N; kk += block) {
                const size_t iEnd = std::min(ii + block, rows);
                const size_t jEnd = std::min(jj + block, N);
                const size_t kEnd = std::min(kk + block, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    for (size_t j = jj; j < jEnd; ++j) {
                        double sum = (kk == 0) ? 0.0 : C[i * N + j];
                        for (size_t k = kk; k < kEnd; ++k)
                            sum += A[i * N + k] * BT[j * N + k];
                        C[i * N + j] = sum;
                    }
                }
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (N == 0) { if (rank == 0) printf("Matrix size must be positive\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
    }

    const size_t base = N / static_cast<size_t>(world);
    const size_t extra = N % static_cast<size_t>(world);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<int> counts(world), displacements(world);
    for (int p = 0; p < world; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        if (rows * N > static_cast<size_t>(INT_MAX)) MPI_Abort(MPI_COMM_WORLD, 2);
        counts[p] = static_cast<int>(rows * N);
        displacements[p] = (p == 0) ? 0 : displacements[p - 1] + counts[p - 1];
    }
    
    // Allocate matrices
    std::vector<double> A(rank == 0 ? N * N : 0);
    std::vector<double> B(N * N), BT(N * N);
    std::vector<double> C(rank == 0 ? N * N : 0);
    std::vector<double> localA(localRows * N), localC(localRows * N, 0.0);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    if (rank == 0) initMatrix(A, N);
    if (rank == 0) initMatrix(B, N);
    if (N * N > static_cast<size_t>(INT_MAX)) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) BT[j * N + i] = B[i * N + j];
    
    // Perform matrix multiplication
    MPI_Scatterv(A.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(localA, BT, localC, localRows, N);
    const double elapsed = MPI_Wtime() - start;
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, C.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank != 0) { MPI_Finalize(); return 0; }
    printf("Computation time: %.0f ms\n", maxElapsed * 1000.0);
    
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
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
