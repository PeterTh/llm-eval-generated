#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t i = firstRow + localRow;
        for (size_t j = 0; j < N; ++j) {
            mat[localRow * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// B is shared read-only input.  Rows of A and C are owned by one MPI rank.
// Blocking keeps the B and C working sets cache-resident while retaining the
// increasing-k accumulation order used by the original implementation.
void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t localRows, const size_t N) {
    constexpr size_t blockSize = 64;
    std::fill(C.begin(), C.end(), 0.0);

    for (size_t i = 0; i < localRows; ++i) {
        const double* const aRow = A.data() + i * N;
        double* const cRow = C.data() + i * N;
        for (size_t kk = 0; kk < N; kk += blockSize) {
            const size_t kEnd = std::min(kk + blockSize, N);
            for (size_t jj = 0; jj < N; jj += blockSize) {
                const size_t jEnd = std::min(jj + blockSize, N);
                for (size_t k = kk; k < kEnd; ++k) {
                    const double a = aRow[k];
                    const double* const bRow = B.data() + k * N;
                    for (size_t j = jj; j < jEnd; ++j) {
                        cRow[j] += a * bRow[j];
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

    int rank = 0;
    int worldSize = 1;
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

    const size_t maxElements = static_cast<size_t>(std::numeric_limits<int>::max());
    if (N == 0 || N > maxElements / N) {
        if (rank == 0) {
            printf("Matrix order must be positive and its element count must not exceed %zu.\n", maxElements);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const size_t processRows = baseRows + (static_cast<size_t>(process) < extraRows ? 1 : 0);
        counts[process] = static_cast<int>(processRows * N);
        displacements[process] = static_cast<int>((static_cast<size_t>(process) * baseRows +
                                 std::min(static_cast<size_t>(process), extraRows)) * N);
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    initMatrixRows(A, N, firstRow, localRows);
    initMatrixRows(B, N, 0, N);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiplyLocal(A, B, C, localRows, N);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        const double gflops = (2.0 * static_cast<double>(N) * N * N) / elapsedSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalC;
    if (validate || printResults) {
        if (rank == 0) {
            globalC.resize(N * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalC, "MatrixC");
    }
    
    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        std::vector<double> globalA(N * N);
        initMatrixRows(globalA, N, 0, N);
        bool valid = validateResult(globalA, B, globalC, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
