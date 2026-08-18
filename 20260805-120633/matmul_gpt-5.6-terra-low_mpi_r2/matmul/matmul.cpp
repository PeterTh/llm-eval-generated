#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t firstRow, const size_t rowCount) {
    for (size_t localI = 0; localI < rowCount; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Store B transposed so each dot product reads contiguous memory.
void initTransposedMatrix(std::vector<double>& transposed, const size_t N) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
            transposed[j * N + k] = getPseudoRndValue(N, k, j);
        }
    }
}

void matrixMultiplyLocal(const std::vector<double>& A,
                         const std::vector<double>& transposedB,
                         std::vector<double>& C, const size_t N,
                         const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        const double* const aRow = A.data() + i * N;
        double* const cRow = C.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            const double* const bRow = transposedB.data() + j * N;
            for (size_t k = 0; k < N; ++k) {
                sum += aRow[k] * bRow[k];
            }
            cRow[j] = sum;
        }
    }
}

bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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

    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) printf("Matrix size must be positive and fit MPI count limits\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);
    const int localCount = static_cast<int>(localRows * N);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", ranks);
        printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localRows * N);
    std::vector<double> transposedB(N * N);
    std::vector<double> localC(localRows * N);
    initMatrixRows(localA, N, firstRow, localRows);
    initTransposedMatrix(transposedB, N);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiplyLocal(localA, transposedB, localC, N, localRows);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> C;
    if (rank == 0 && (printResults || validate)) C.resize(N * N);
    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
        size_t offset = 0;
        for (int process = 0; process < ranks; ++process) {
            const size_t rows = baseRows + (static_cast<size_t>(process) < extraRows ? 1 : 0);
            counts[process] = static_cast<int>(rows * N);
            displacements[process] = static_cast<int>(offset);
            offset += rows * N;
        }
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double gflops = elapsed > 0.0 ? (2.0 * N * N * N) / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults || validate) {
        MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    int success = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = validateResult(C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            success = 1;
        } else {
            printf("Validation: FAILED\n");
            success = 0;
        }
    }
    
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return success ? 0 : 1;
}
