#include <algorithm>
#include <cmath>
#include <climits>
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

constexpr size_t kBlockSize = 64;

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
    for (size_t localI = 0; localI < rowCount; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void transposeMatrix(const std::vector<double>& mat, std::vector<double>& transposed,
                     const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            transposed[j * N + i] = mat[i * N + j];
        }
    }
}

// Multiply local rows of A by B.  B is transposed so both inputs of the
// innermost dot product are contiguous.  Blocking keeps A, B, and C tiles in
// cache while preserving increasing-k accumulation for every C element.
void matrixMultiply(const std::vector<double>& localA, const std::vector<double>& transposedB,
                    std::vector<double>& localC, const size_t localRows, const size_t N) {
    std::fill(localC.begin(), localC.end(), 0.0);

    for (size_t ii = 0; ii < localRows; ii += kBlockSize) {
        const size_t iEnd = std::min(ii + kBlockSize, localRows);
        for (size_t jj = 0; jj < N; jj += kBlockSize) {
            const size_t jEnd = std::min(jj + kBlockSize, N);
            for (size_t kk = 0; kk < N; kk += kBlockSize) {
                const size_t kEnd = std::min(kk + kBlockSize, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* const a = localA.data() + i * N;
                    double* const c = localC.data() + i * N;
                    for (size_t j = jj; j < jEnd; ++j) {
                        const double* const b = transposedB.data() + j * N;
                        double sum = c[j];
                        for (size_t k = kk; k < kEnd; ++k) {
                            sum += a[k] * b[k];
                        }
                        c[j] = sum;
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare.  A and B have the
// same deterministic initializer, and B is stored transposed in this version.
bool validateResult(const std::vector<double>& transposedB, const std::vector<double>& C,
                    const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * transposedB[j * N + k];
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

bool validMatrixSize(const size_t N) {
    return N > 0 && N <= static_cast<size_t>(std::sqrt(static_cast<double>(INT_MAX)));
}

void makeRowPartition(const size_t N, const int rank, const int processCount,
                      size_t& firstRow, size_t& localRows) {
    const size_t ranks = static_cast<size_t>(processCount);
    const size_t baseRows = N / ranks;
    const size_t remainder = N % ranks;
    const size_t thisRank = static_cast<size_t>(rank);

    localRows = baseRows + (thisRank < remainder ? 1 : 0);
    firstRow = thisRank * baseRows + std::min(thisRank, remainder);
}

void makeGatherLayout(const size_t N, const int processCount,
                      std::vector<int>& counts, std::vector<int>& displacements) {
    counts.resize(processCount);
    displacements.resize(processCount);

    size_t firstRow = 0;
    for (int rank = 0; rank < processCount; ++rank) {
        const size_t ranks = static_cast<size_t>(processCount);
        const size_t thisRank = static_cast<size_t>(rank);
        const size_t rows = N / ranks + (thisRank < N % ranks ? 1 : 0);
        counts[rank] = static_cast<int>(rows * N);
        displacements[rank] = static_cast<int>(firstRow * N);
        firstRow += rows;
    }
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
    int exitCode = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
                exitCode = 1;
                if (rank == 0) {
                    printf("Invalid matrix size: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                break;
            }
            N = static_cast<size_t>(parsed);
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
            exitCode = 1;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }

    if (exitCode == 0 && !validMatrixSize(N)) {
        exitCode = 1;
        if (rank == 0) {
            printf("Matrix size must be between 1 and %d for this MPI implementation\n",
                   static_cast<int>(std::sqrt(static_cast<double>(INT_MAX))));
        }
    }

    int globalExitCode = 0;
    MPI_Allreduce(&exitCode, &globalExitCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalExitCode != 0) {
        MPI_Finalize();
        return globalExitCode;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    
    size_t firstRow = 0;
    size_t localRows = 0;
    makeRowPartition(N, rank, processCount, firstRow, localRows);

    // A and C stay distributed by contiguous row blocks.  B is replicated,
    // which avoids communication in the O(N^3) kernel.
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    
    initMatrixRows(A, N, firstRow, localRows);
    if (rank == 0) {
        initMatrixRows(B, N, 0, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> transposedB(N * N);
    transposeMatrix(B, transposedB, N);
    B.clear();
    B.shrink_to_fit();
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiply(A, transposedB, C, localRows, N);
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
    
        // Use the slowest rank's elapsed time, since it determines the
        // completion time of a distributed multiplication.
        const double gflops = duration > 0.0
            ? (2.0 * static_cast<double>(N) * N * N) / duration / 1e9
            : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // The full result is needed only by root for output or validation.
    std::vector<double> globalC;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displacements;
        makeGatherLayout(N, processCount, counts, displacements);
        if (rank == 0) {
            globalC.resize(N * N);
        }

        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if (rank == 0 && printResults) {
        print_results(globalC, "MatrixC");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(transposedB, globalC, N);
        
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
