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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = rowStart + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrixTranspose(std::vector<double>& matT, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            matT[j * N + i] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B_T,
                         std::vector<double>& C, const size_t N) {
    const size_t rows = A.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        const double* aRow = &A[i * N];
        for (size_t j = 0; j < N; ++j) {
            const double* bRow = &B_T[j * N];
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += aRow[k] * bRow[k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool invalidArgs = false;
    const char* invalidArg = nullptr;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            invalidArgs = true;
            invalidArg = argv[i];
        }
    }

    if (showHelp || invalidArgs) {
        if (rank == 0) {
            if (invalidArgs) {
                printf("Unknown option: %s\n", invalidArg);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return invalidArgs ? 1 : 0;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t sizeT = static_cast<size_t>(size);
    const size_t rankT = static_cast<size_t>(rank);
    const size_t baseRows = N / sizeT;
    const size_t remainder = N % sizeT;
    const size_t localRows = baseRows + (rankT < remainder ? 1 : 0);
    const size_t rowStart = baseRows * rankT + (rankT < remainder ? rankT : remainder);
    
    // Allocate matrices
    std::vector<double> A_local(localRows * N);
    std::vector<double> B_T(N * N);
    std::vector<double> C_local(localRows * N);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A_local, N, rowStart);
    initMatrixTranspose(B_T, N);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyLocal(A_local, B_T, C_local, N);
    
    const double end = MPI_Wtime();
    const double localDuration = end - start;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxDuration * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        
        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / maxDuration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> C;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (printResults || validate) {
        if (rank == 0) {
            C.resize(N * N);
            recvCounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t rT = static_cast<size_t>(r);
                const size_t rows = baseRows + (rT < remainder ? 1 : 0);
                recvCounts[r] = static_cast<int>(rows * N);
                displs[r] = static_cast<int>((baseRows * rT + (rT < remainder ? rT : remainder)) * N);
            }
        }
        const int localCount = static_cast<int>(localRows * N);
        MPI_Gatherv(C_local.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        int validInt = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            validInt = validateResult(C, N) ? 1 : 0;
            if (validInt == 1) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt == 1 ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
