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

void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowOffset = 0) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowOffset + i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& BT,
                    std::vector<double>& C, const size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * BT[j * N + k];
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
        printf("MPI processes: %d\nInitializing matrices...\n", worldSize);
    }
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t rowOffset = static_cast<size_t>(rank) * baseRows +
                             (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);
    std::vector<double> A(localRows * N), BT(N * N), C(localRows * N);
    initMatrix(A, N, rowOffset);
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < N; ++k)
            BT[j * N + k] = getPseudoRndValue(N, k, j);

    // The replicated, transposed B avoids communication in the hot loop.
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, BT, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> fullA, fullB, fullC;
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        counts[r] = static_cast<int>(rows * N);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseRows +
                                             (static_cast<size_t>(r) < extraRows ? static_cast<size_t>(r) : extraRows)) * N);
    }
    if (rank == 0 && (validate || printResults)) {
        fullC.resize(N * N); fullA.resize(N * N); fullB.resize(N * N);
        initMatrix(fullA, N); initMatrix(fullB, N);
    }
    MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, rank == 0 ? fullC.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0,
               (seconds > 0.0) ? (2.0 * N * N * N / seconds / 1e9) : 0.0);
    }
    if (validate) {
        bool valid = true;
        if (rank == 0) { printf("Validating result...\n"); valid = validateResult(fullA, fullB, fullC, N); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Finalize(); return valid ? 0 : 1;
    }
    if (rank == 0 && printResults) print_results(fullC, "MatrixC");
    MPI_Finalize();
    return 0;
}
