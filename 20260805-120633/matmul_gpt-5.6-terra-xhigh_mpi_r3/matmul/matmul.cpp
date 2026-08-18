#include <algorithm>
#include <cerrno>
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

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t firstRow) {
    const size_t rowCount = mat.size() / N;
    for (size_t localI = 0; localI < rowCount; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Rows are assigned contiguously to ranks.  B is read-only and replicated,
// which keeps the compute phase communication-free and lets each rank use a
// cache-blocked, vectorizable i-k-j kernel on its local rows.
void matrixMultiplyLocalRows(const std::vector<double>& A, const std::vector<double>& B,
                             std::vector<double>& C, const size_t localRows,
                             const size_t N) {
    constexpr size_t kRowBlock = 32;
    constexpr size_t kKBlock = 128;
    constexpr size_t kColumnBlock = 256;

    std::fill(C.begin(), C.end(), 0.0);

    const double* const aData = A.data();
    const double* const bData = B.data();
    double* const cData = C.data();

    for (size_t ii = 0; ii < localRows; ii += kRowBlock) {
        const size_t iEnd = std::min(ii + kRowBlock, localRows);
        for (size_t kk = 0; kk < N; kk += kKBlock) {
            const size_t kEnd = std::min(kk + kKBlock, N);
            for (size_t jj = 0; jj < N; jj += kColumnBlock) {
                const size_t jEnd = std::min(jj + kColumnBlock, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* const aRow = aData + i * N;
                    double* const cRow = cData + i * N;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aValue = aRow[k];
                        const double* const bRow = bData + k * N;
#pragma GCC ivdep
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aValue * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a few local elements and compare.  Each global
// checkpoint is owned by exactly one rank, then the result is combined below.
bool validateLocalResult(const std::vector<double>& A, const std::vector<double>& B,
                         const std::vector<double>& C, const size_t N,
                         const size_t firstRow, const size_t localRows) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            if (i < firstRow || i >= firstRow + localRows) {
                continue;
            }

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

size_t rowsForRank(const size_t N, const int rank, const int rankCount) {
    const size_t baseRows = N / static_cast<size_t>(rankCount);
    const size_t remainder = N % static_cast<size_t>(rankCount);
    return baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t firstRowForRank(const size_t N, const int rank, const int rankCount) {
    const size_t baseRows = N / static_cast<size_t>(rankCount);
    const size_t remainder = N % static_cast<size_t>(rankCount);
    const size_t rankIndex = static_cast<size_t>(rank);
    return rankIndex * baseRows + std::min(rankIndex, remainder);
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
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    int exitCode = 0;
    
    if (rank == 0) {
        // Parse command line arguments on the root, then broadcast them so
        // every rank follows precisely the same execution path.
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                char* end = nullptr;
                errno = 0;
                const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
                if (*argv[i] == '\0' || *argv[i] == '-' || *end != '\0' || errno == ERANGE ||
                    parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                    printf("Invalid matrix size: %s\n", argv[i]);
                    exitCode = 1;
                    break;
                }
                N = static_cast<size_t>(parsed);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                showHelp = true;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
    }

    unsigned long long broadcastN = static_cast<unsigned long long>(N);
    int options[4] = {validate ? 1 : 0, printResults ? 1 : 0, showHelp ? 1 : 0, exitCode};
    MPI_Bcast(&broadcastN, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(broadcastN);
    validate = options[0] != 0;
    printResults = options[1] != 0;
    showHelp = options[2] != 0;
    exitCode = options[3];

    if (exitCode != 0 || showHelp) {
        MPI_Finalize();
        return exitCode;
    }

    if (N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) {
            printf("Matrix size is too large for this MPI implementation: %zu\n", N);
        }
        MPI_Finalize();
        return 1;
    }
    const size_t matrixElementCount = N * N;

    if (printResults && matrixElementCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Matrix size is too large to gather results with MPI_Gatherv: %zu\n", N);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const size_t localRows = rowsForRank(N, rank, rankCount);
    const size_t firstRow = firstRowForRank(N, rank, rankCount);
    const size_t localElementCount = localRows * N;

    // Each rank owns a contiguous row range of A and C.  B is replicated so
    // no communication occurs in the timed matrix multiplication.
    std::vector<double> A(localElementCount);
    std::vector<double> B(matrixElementCount);
    std::vector<double> C(localElementCount);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A, N, firstRow);
    initMatrixRows(B, N, 0);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyLocalRows(A, B, C, localRows, N);
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long long durationMs = static_cast<long long>(duration * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
        
        // Calculate aggregate GFLOPS from the slowest rank's elapsed time.
        const double gflops = (2.0 * static_cast<double>(N) * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> fullC;
        if (rank == 0) {
            receiveCounts.resize(rankCount);
            displacements.resize(rankCount);
            size_t displacement = 0;
            for (int process = 0; process < rankCount; ++process) {
                const size_t count = rowsForRank(N, process, rankCount) * N;
                receiveCounts[process] = static_cast<int>(count);
                displacements[process] = static_cast<int>(displacement);
                displacement += count;
            }
            fullC.resize(matrixElementCount);
        }

        MPI_Gatherv(C.data(), static_cast<int>(localElementCount), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(fullC, "MatrixC");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool localValid = validateLocalResult(A, B, C, N, firstRow, localRows);
        int validValue = localValid ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&validValue, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return exitCode;
}
