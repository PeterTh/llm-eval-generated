#include <algorithm>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t firstRow, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

// Multiply a contiguous block of rows of A by the full matrix B.  The k
// iteration remains in increasing order for every C element, matching the
// original scalar operation while using a cache-friendly i-k-j kernel.
void matrixMultiplyRows(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t rowCount,
                        const size_t N) {
    constexpr size_t tileSize = 64;
    std::fill(C.begin(), C.end(), 0.0);

    for (size_t i0 = 0; i0 < rowCount; i0 += tileSize) {
        const size_t iEnd = std::min(i0 + tileSize, rowCount);
        for (size_t k0 = 0; k0 < N; k0 += tileSize) {
            const size_t kEnd = std::min(k0 + tileSize, N);
            for (size_t j0 = 0; j0 < N; j0 += tileSize) {
                const size_t jEnd = std::min(j0 + tileSize, N);
                for (size_t i = i0; i < iEnd; ++i) {
                    double* cRow = C.data() + i * N;
                    const double* aRow = A.data() + i * N;
                    for (size_t k = k0; k < kEnd; ++k) {
                        const double aik = aRow[k];
                        const double* bRow = B.data() + k * N + j0;
                        for (size_t j = j0; j < jEnd; ++j) {
                            cRow[j] += aik * bRow[j - j0];
                        }
                    }
                }
            }
        }
    }
}

size_t rowsForRank(const size_t N, const int rank, const int processCount) {
    const size_t baseRows = N / static_cast<size_t>(processCount);
    const size_t remainder = N % static_cast<size_t>(processCount);
    return baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t firstRowForRank(const size_t N, const int rank, const int processCount) {
    const size_t baseRows = N / static_cast<size_t>(processCount);
    const size_t remainder = N % static_cast<size_t>(processCount);
    return static_cast<size_t>(rank) * baseRows +
           std::min(static_cast<size_t>(rank), remainder);
}

bool checkedMatrixElementCount(const size_t N, size_t& elementCount) {
    if (N != 0 && N > std::numeric_limits<size_t>::max() / N) {
        return false;
    }
    elementCount = N * N;
    return true;
}

// MPI collectives traditionally take an int count.  Chunking the broadcast
// keeps the implementation correct for matrices whose element count exceeds
// that limit, without requiring MPI-4 large-count extensions.
bool broadcastDoubles(std::vector<double>& values, const int root, MPI_Comm comm) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < values.size();) {
        const size_t remaining = values.size() - offset;
        const int chunk = static_cast<int>(std::min(remaining, maxChunk));
        if (MPI_Bcast(values.data() + offset, chunk, MPI_DOUBLE, root, comm) != MPI_SUCCESS) {
            return false;
        }
        offset += static_cast<size_t>(chunk);
    }
    return true;
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
    bool showHelp = false;
    bool parseError = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or incomplete option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    size_t matrixElements = 0;
    if (!checkedMatrixElementCount(N, matrixElements) || N == 0) {
        if (rank == 0) {
            printf("Matrix size must be a positive value that fits in memory\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t localRows = rowsForRank(N, rank, processCount);
    const size_t firstRow = firstRowForRank(N, rank, processCount);
    const size_t localElements = localRows * N;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Only rank 0 needs the complete input/output matrices for validation or
    // result reporting. Every rank owns and initializes its local A rows.
    std::vector<double> A;
    std::vector<double> B(matrixElements);
    std::vector<double> localA(localElements);
    std::vector<double> localC(localElements);
    std::vector<double> C;

    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(B, N);
        if (validate) {
            A.resize(matrixElements);
            initMatrix(A, N);
        }
    }
    initMatrixRows(localA, N, firstRow, localRows);

    // B is read-only during multiplication and is replicated once on each
    // rank. This avoids repeatedly communicating panels during the hot loop.
    if (!broadcastDoubles(B, 0, MPI_COMM_WORLD)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (printResults || validate) {
        if (rank == 0) {
            C.resize(matrixElements);
        }
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiplyRows(localA, B, localC, localRows, N);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (printResults || validate) {
        // A row block is contiguous, so one Gatherv reconstructs C in the
        // original global row-major layout on rank 0.
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        bool countsFit = true;
        if (rank == 0) {
            receiveCounts.resize(static_cast<size_t>(processCount));
            displacements.resize(static_cast<size_t>(processCount));
            for (int r = 0; r < processCount; ++r) {
                const size_t rows = rowsForRank(N, r, processCount);
                const size_t offset = firstRowForRank(N, r, processCount) * N;
                if (rows * N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    countsFit = false;
                    break;
                }
                receiveCounts[static_cast<size_t>(r)] = static_cast<int>(rows * N);
                displacements[static_cast<size_t>(r)] = static_cast<int>(offset);
            }
        }
        int countsFitInt = countsFit ? 1 : 0;
        MPI_Bcast(&countsFitInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!countsFitInt) {
            if (rank == 0) {
                printf("Result collection is too large for this MPI implementation\n");
            }
            MPI_Finalize();
            return 1;
        }

        const int localCount = static_cast<int>(localElements);
        MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long durationMs = std::max(1L, static_cast<long>(elapsed * 1000.0));
        printf("Computation time: %ld ms\n", durationMs);
        const double nAsDouble = static_cast<double>(N);
        const double gflops = (2.0 * nAsDouble * nAsDouble * nAsDouble) /
                              (static_cast<double>(durationMs) / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(A, B, C, N) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
