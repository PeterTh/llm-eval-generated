#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N, const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        double* cRow = &C[i * N];
        const double* aRow = &A[i * N];
        for (size_t k = 0; k < N; ++k) {
            const double aVal = aRow[k];
            const double* bRow = &B[k * N];
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += aVal * bRow[j];
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
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int shouldExit = 0;
    int exitCode = 0;
    
    // Parse command line arguments
    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    uint64_t n64 = static_cast<uint64_t>(N);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(n64);
    validate = validateFlag != 0;
    printResults = printFlag != 0;
    
    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(worldRank) < extraRows ? 1 : 0);
    const size_t localElems = localRows * N;
    
    // Allocate matrices
    std::vector<double> B(N * N);
    std::vector<double> localA(localElems);
    std::vector<double> localC(localElems, 0.0);
    std::vector<double> A;
    std::vector<double> C;
    
    if (worldRank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> counts;
    std::vector<int> displs;
    if (worldRank == 0) {
        counts.resize(worldSize);
        displs.resize(worldSize);
        size_t offset = 0;
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            const size_t elems = rows * N;
            counts[r] = static_cast<int>(elems);
            displs[r] = static_cast<int>(offset);
            offset += elems;
        }
    }

    const int* sendcounts = worldRank == 0 ? counts.data() : nullptr;
    const int* senddispls = worldRank == 0 ? displs.data() : nullptr;
    MPI_Scatterv(A.empty() ? nullptr : A.data(), sendcounts, senddispls, MPI_DOUBLE,
                 localA.data(), static_cast<int>(localElems), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiplyLocal(localA, B, localC, N, localRows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    const int* recvcounts = worldRank == 0 ? counts.data() : nullptr;
    const int* recvdispls = worldRank == 0 ? displs.data() : nullptr;
    MPI_Gatherv(localC.data(), static_cast<int>(localElems), MPI_DOUBLE,
                C.empty() ? nullptr : C.data(), recvcounts, recvdispls,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (worldRank == 0) {
        const long durationMs = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        
        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / maxTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
        print_results(C, "MatrixC");
    }
    
    int resultCode = 0;
    if (validate && worldRank == 0) {
        printf("Validating result...\n");
        const bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            resultCode = 0;
        } else {
            printf("Validation: FAILED\n");
            resultCode = 1;
        }
    }
    MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Finalize();
    return resultCode;
}
