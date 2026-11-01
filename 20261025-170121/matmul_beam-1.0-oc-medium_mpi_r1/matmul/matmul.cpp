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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, size_t rowStart, size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        const size_t gi = rowStart + i;
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[gi * N + j] = sum;
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks)
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A, B, C;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
    } else {
        B.resize(N * N);
    }
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    
    // Broadcast B to all ranks
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Broadcast A? Each rank needs its rows. Scatter rows of A.
    
    // Determine row distribution
    size_t rowsPerRank = N / size;
    size_t remainder = N % size;
    size_t rowCount = rowsPerRank + (rank < (int)remainder ? 1 : 0);
    
    // Scatter A rows
    std::vector<double> localA(rowCount * N);
    // Build counts/displs
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t rc = rowsPerRank + (r < (int)remainder ? 1 : 0);
        size_t rs = r * rowsPerRank + std::min((size_t)r, remainder);
        counts[r] = rc * N;
        displs[r] = rs * N;
    }
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), rowCount * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    std::vector<double> localC(rowCount * N, 0.0);
    
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // compute using localA as A-subset (rows), B full, store into localC
    std::vector<double> fullLocalA = localA;
    // matrixMultiply expects A with rows starting at 0 in local buffer; C indices global. Adjust: pass localC local and map.
    // We'll compute into localC (local rows) then gather.
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += fullLocalA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }
    
    // Gather localC into C on rank 0
    std::vector<int> rcounts(size), rdispls(size);
    for (int r = 0; r < size; ++r) {
        size_t rc = rowsPerRank + (r < (int)remainder ? 1 : 0);
        size_t rs = r * rowsPerRank + std::min((size_t)r, remainder);
        rcounts[r] = rc * N;
        rdispls[r] = rs * N;
    }
    MPI_Gatherv(localC.data(), rowCount * N, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, rcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);
        double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
