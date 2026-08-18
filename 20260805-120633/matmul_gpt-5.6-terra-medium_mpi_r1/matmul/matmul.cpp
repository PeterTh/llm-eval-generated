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

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow,
                const size_t rows) {
    for (size_t localI = 0; localI < rows; ++localI) {
        const size_t i = firstRow + localI;
        for (size_t j = 0; j < N; ++j) {
            mat[localI * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// A and C contain only this rank's contiguous row block; B is replicated.
// Cache blocking reuses B panels between local A rows, while the i-k-j inner
// ordering retains increasing-k accumulation for every C element.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t blockSize = 48;
    for (size_t ii = 0; ii < localRows; ii += blockSize) {
        const size_t iEnd = std::min(ii + blockSize, localRows);
        for (size_t kk = 0; kk < N; kk += blockSize) {
            const size_t kEnd = std::min(kk + blockSize, N);
            for (size_t jj = 0; jj < N; jj += blockSize) {
                const size_t jEnd = std::min(jj + blockSize, N);
                for (size_t i = ii; i < iEnd; ++i) {
                    double* const cRow = C.data() + i * N;
                    const double* const aRow = A.data() + i * N;
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
}

// Validate the rows owned by this rank, avoiding a full result gather.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
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
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[(i - firstRow) * N + k] * B[k * N + j];
            }
            
            const double actual = C[(i - firstRow) * N + j];
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
    
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) / N) {
        if (rank == 0) {
            printf("Matrix size is too large for MPI counts: %zu\n", N);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(numRanks);
    const size_t extraRows = N % static_cast<size_t>(numRanks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N, firstRow, localRows);
    initMatrix(B, N, 0, N);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiply(A, B, C, N, localRows);
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long long durationMs = static_cast<long long>(duration * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
    
        // Calculate GFLOPS from the globally synchronized critical-path time.
        const double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> globalC;
        if (rank == 0) {
            counts.resize(numRanks);
            displacements.resize(numRanks);
            for (int process = 0; process < numRanks; ++process) {
                const size_t processRows = baseRows + (static_cast<size_t>(process) < extraRows);
                const size_t processFirstRow = static_cast<size_t>(process) * baseRows +
                                               (static_cast<size_t>(process) < extraRows ? static_cast<size_t>(process) : extraRows);
                counts[process] = static_cast<int>(processRows * N);
                displacements[process] = static_cast<int>(processFirstRow * N);
            }
            globalC.resize(N * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalC, "MatrixC");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool localValid = validateResult(A, B, C, N, firstRow, localRows);
        int validValue = localValid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &validValue, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Validation: %s\n", validValue ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return validValue ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
