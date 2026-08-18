#include <algorithm>
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

struct Block {
    size_t offset;
    size_t size;
};

Block blockForRank(const size_t N, const int coordinate, const int dimensions) {
    const size_t base = N / static_cast<size_t>(dimensions);
    const size_t remainder = N % static_cast<size_t>(dimensions);
    const size_t coordinateSize = static_cast<size_t>(coordinate);

    return {
        coordinateSize * base + std::min(coordinateSize, remainder),
        base + (coordinateSize < remainder ? 1 : 0),
    };
}

void initLocalA(std::vector<double>& mat, const size_t N, const size_t firstRow) {
    const size_t localRows = mat.empty() ? 0 : mat.size() / N;
    for (size_t i = 0; i < localRows; ++i) {
        const size_t globalRow = firstRow + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initLocalB(std::vector<double>& mat, const size_t N, const size_t firstColumn) {
    const size_t localColumns = mat.empty() ? 0 : mat.size() / N;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < localColumns; ++j) {
            mat[i * localColumns + j] = getPseudoRndValue(N, i, firstColumn + j);
        }
    }
}

// Computes one two-dimensional block of C.  The k loop remains increasing for
// every C element, preserving the original accumulation order while the i-k-j
// order gives contiguous B and C accesses and enables SIMD vectorization.
void matrixMultiplyBlock(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N,
                         const size_t localRows, const size_t localColumns) {
    constexpr size_t I_BLOCK = 32;
    constexpr size_t K_BLOCK = 128;
    constexpr size_t J_BLOCK = 128;

    for (size_t ii = 0; ii < localRows; ii += I_BLOCK) {
        const size_t iEnd = std::min(ii + I_BLOCK, localRows);
        for (size_t kk = 0; kk < N; kk += K_BLOCK) {
            const size_t kEnd = std::min(kk + K_BLOCK, N);
            for (size_t jj = 0; jj < localColumns; jj += J_BLOCK) {
                const size_t jEnd = std::min(jj + J_BLOCK, localColumns);
                for (size_t i = ii; i < iEnd; ++i) {
                    const double* const aRow = A.data() + i * N;
                    double* const cRow = C.data() + i * localColumns;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double aValue = aRow[k];
                        const double* const bRow = B.data() + k * localColumns + jj;
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += aValue * bRow[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a few elements directly from the deterministic
// input definition and compare them with the gathered distributed result.
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
    int processCount = 0;
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
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int gridDimensions[2] = {0, 0};
    MPI_Dims_create(processCount, 2, gridDimensions);
    const int gridRow = rank / gridDimensions[1];
    const int gridColumn = rank % gridDimensions[1];
    const Block rowBlock = blockForRank(N, gridRow, gridDimensions[0]);
    const Block columnBlock = blockForRank(N, gridColumn, gridDimensions[1]);
    
    // Each rank owns one C block.  A is partitioned by rows and B by columns,
    // so no root-side distribution or per-iteration communication is needed.
    std::vector<double> A(rowBlock.size * N);
    std::vector<double> B(N * columnBlock.size);
    std::vector<double> C(rowBlock.size * columnBlock.size, 0.0);
    
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initLocalA(A, N, rowBlock.offset);
    initLocalB(B, N, columnBlock.offset);
    
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyBlock(A, B, C, N, rowBlock.size, columnBlock.size);
    
    const double localDuration = MPI_Wtime() - start;
    double durationSeconds = 0.0;
    MPI_Reduce(&localDuration, &durationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(durationSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        // Calculate aggregate GFLOPS using the slowest rank's elapsed time.
        const double gflops = (2.0 * N * N * N) / durationSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalC;
    if (printResults || validate) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> gatheredBlocks;

        if (rank == 0) {
            receiveCounts.resize(processCount);
            displacements.resize(processCount);
            size_t displacement = 0;
            for (int sourceRank = 0; sourceRank < processCount; ++sourceRank) {
                const int sourceRow = sourceRank / gridDimensions[1];
                const int sourceColumn = sourceRank % gridDimensions[1];
                const Block sourceRows = blockForRank(N, sourceRow, gridDimensions[0]);
                const Block sourceColumns = blockForRank(N, sourceColumn, gridDimensions[1]);
                receiveCounts[sourceRank] = static_cast<int>(sourceRows.size * sourceColumns.size);
                displacements[sourceRank] = static_cast<int>(displacement);
                displacement += sourceRows.size * sourceColumns.size;
            }
            gatheredBlocks.resize(displacement);
            globalC.resize(N * N);
        }

        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    gatheredBlocks.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int sourceRank = 0; sourceRank < processCount; ++sourceRank) {
                const int sourceRow = sourceRank / gridDimensions[1];
                const int sourceColumn = sourceRank % gridDimensions[1];
                const Block sourceRows = blockForRank(N, sourceRow, gridDimensions[0]);
                const Block sourceColumns = blockForRank(N, sourceColumn, gridDimensions[1]);
                const size_t sourceOffset = static_cast<size_t>(displacements[sourceRank]);

                for (size_t i = 0; i < sourceRows.size; ++i) {
                    std::copy_n(gatheredBlocks.data() + sourceOffset + i * sourceColumns.size,
                                sourceColumns.size,
                                globalC.data() + (sourceRows.offset + i) * N + sourceColumns.offset);
                }
            }
        }
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(globalC, "MatrixC");
    }
    
    // Validation
    int exitCode = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(globalC, N);
        
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
