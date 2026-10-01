#include <algorithm>
#include <climits>
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

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow = 0) {
    for (size_t i = 0; i < mat.size() / N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

size_t firstRow(const size_t N, const int rank, const int ranks) {
    return (N / ranks) * rank + std::min<size_t>(rank, N % ranks);
}

size_t rowCount(const size_t N, const int rank, const int ranks) {
    return N / ranks + (static_cast<size_t>(rank) < N % ranks);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    constexpr size_t rowTile = 32, colTile = 128, depthTile = 64;
    const size_t rows = A.size() / N;
    for (size_t ib = 0; ib < rows; ib += rowTile) {
        for (size_t jb = 0; jb < N; jb += colTile) {
            for (size_t kb = 0; kb < N; kb += depthTile) {
                for (size_t i = ib; i < std::min(ib + rowTile, rows); ++i) {
                    double* cRow = C.data() + i * N;
                    const double* aRow = A.data() + i * N;
                    for (size_t k = kb; k < std::min(kb + depthTile, N); ++k) {
                        const double aik = aRow[k];
                        const double* bRow = B.data() + k * N;
                        for (size_t j = jb; j < std::min(jb + colTile, N); ++j) {
                            cRow[j] += aik * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N,
                   const size_t first, const size_t rows) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            if (i < first || i >= first + rows) continue;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[(i - first) * N + k] * B[k * N + j];
            }
            
            const double actual = C[(i - first) * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (!(relError <= 1e-6)) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void gatherResults(const std::vector<double>& localC, std::vector<double>& C,
                   const size_t N, const int rank, const int ranks) {
    if (N * N <= INT_MAX) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(rowCount(N, r, ranks) * N);
                displacements[r] = static_cast<int>(firstRow(N, r, ranks) * N);
            }
        }
        MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        // Counts and displacements in MPI_Gatherv are int. Use chunks for larger matrices.
        std::copy(localC.begin(), localC.end(), C.begin());
        for (int r = 1; r < ranks; ++r) {
            const size_t offset = firstRow(N, r, ranks) * N;
            const size_t count = rowCount(N, r, ranks) * N;
            for (size_t pos = 0; pos < count; pos += INT_MAX) {
                MPI_Recv(C.data() + offset + pos,
                         static_cast<int>(std::min<size_t>(INT_MAX, count - pos)),
                         MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t pos = 0; pos < localC.size(); pos += INT_MAX) {
            MPI_Send(localC.data() + pos,
                     static_cast<int>(std::min<size_t>(INT_MAX, localC.size() - pos)),
                     MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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

    if (N == 0 || N > static_cast<size_t>(-1) / N) {
        if (rank == 0) fprintf(stderr, "Matrix size must be positive and fit in memory.\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    
    const size_t first = firstRow(N, rank, ranks);
    const size_t rows = rowCount(N, rank, ranks);
    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(rows * N, 0.0);
    
    initMatrix(A, N, first);
    initMatrix(B, N);
    
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiply(A, B, localC, N);
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = duration > 0.0 ? (2.0 * N * N * N) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    if (printResults) {
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        gatherResults(localC, C, N, rank, ranks);
        if (rank == 0) print_results(C, "MatrixC");
    }
    
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(A, B, localC, N, first, rows) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    
    MPI_Finalize();
    return valid ? 0 : 1;
}
