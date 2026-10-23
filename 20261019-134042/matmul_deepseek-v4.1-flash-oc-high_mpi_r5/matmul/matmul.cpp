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

// Initialize only rows [rowBegin, rowEnd) of the global matrix. The values
// generated for a given global (i, j) are identical to initMatrix().
void initMatrixRange(std::vector<double>& mat, const size_t N,
                     const size_t rowBegin, const size_t rowEnd) {
    const size_t localRows = rowEnd - rowBegin;
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Each rank owns a contiguous block of rows of A and C. B is replicated so
// that the arithmetic is purely local to each rank. The loops are tiled for
// cache locality and vectorization while keeping the accumulation order over
// k ascending. k is consumed in pairs followed by an explicit FMA for a
// possible odd tail, which reproduces the rounding of the original optimized
// serial loop exactly (the build disables implicit FP contraction).
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t IB = 8;
    constexpr size_t JB = 256;

    for (size_t i0 = 0; i0 < localRows; i0 += IB) {
        const size_t iEnd = (i0 + IB < localRows) ? i0 + IB : localRows;

        for (size_t j0 = 0; j0 < N; j0 += JB) {
            const size_t jEnd = (j0 + JB < N) ? j0 + JB : N;
            const size_t jLen = jEnd - j0;

            size_t k = 0;
            for (; k + 1 < N; k += 2) {
                const double* bRow0 = &B[k * N + j0];
                const double* bRow1 = &B[(k + 1) * N + j0];

                for (size_t i = i0; i < iEnd; ++i) {
                    const double a0 = A[i * N + k];
                    const double a1 = A[i * N + k + 1];
                    double* cRow = &C[i * N + j0];

                    for (size_t j = 0; j < jLen; ++j) {
                        cRow[j] += a0 * bRow0[j];
                        cRow[j] += a1 * bRow1[j];
                    }
                }
            }

            if (k < N) {
                const double* bRow = &B[k * N + j0];

                for (size_t i = i0; i < iEnd; ++i) {
                    const double a = A[i * N + k];
                    double* cRow = &C[i * N + j0];

                    for (size_t j = 0; j < jLen; ++j) {
                        cRow[j] = std::fma(a, bRow[j], cRow[j]);
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

    // Contiguous, balanced row distribution across all ranks.
    const size_t rowBegin = (N * static_cast<size_t>(rank)) / static_cast<size_t>(numRanks);
    const size_t rowEnd = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(numRanks);
    const size_t localRows = rowEnd - rowBegin;
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices. A and C are distributed row-wise; B is replicated.
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N, 0.0);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRange(A, N, rowBegin, rowEnd);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, localRows);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        
        // Calculate GFLOPS for the whole distributed computation
        const double gflops = (2.0 * N * N * N) / maxSeconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather the distributed C on rank 0 only when it is actually needed.
    std::vector<double> Cfull;
    if (printResults || validate) {
        if (rank == 0) {
            Cfull.resize(N * N);
        }
        
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(numRanks));
            displs.resize(static_cast<size_t>(numRanks));
            for (int r = 0; r < numRanks; ++r) {
                const size_t begin = (N * static_cast<size_t>(r)) / static_cast<size_t>(numRanks);
                const size_t finish = (N * static_cast<size_t>(r + 1)) / static_cast<size_t>(numRanks);
                counts[static_cast<size_t>(r)] = static_cast<int>((finish - begin) * N);
                displs[static_cast<size_t>(r)] = static_cast<int>(begin * N);
            }
        }
        
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    (rank == 0) ? Cfull.data() : nullptr,
                    (rank == 0) ? counts.data() : nullptr,
                    (rank == 0) ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(Cfull, "MatrixC");
    }
    
    // Validation
    int returnCode = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        
        std::vector<double> Afull(N * N);
        initMatrix(Afull, N);
        
        bool valid = validateResult(Afull, B, Cfull, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            returnCode = 1;
        }
    }
    
    MPI_Finalize();
    return returnCode;
}
