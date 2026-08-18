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

void initMatrix(std::vector<double>& mat, const size_t N, const size_t firstRow = 0) {
    const size_t rows = mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t localRows = A.size() / N;
    std::fill(C.begin(), C.end(), 0.0);

    // This loop order gives contiguous access to both B and C.  Each MPI rank
    // works on an independent block of C rows, so no communication is needed
    // during the compute phase.
    for (size_t i = 0; i < localRows; ++i) {
        double* const cRow = C.data() + i * N;
        const double* const aRow = A.data() + i * N;
        for (size_t k = 0; k < N; ++k) {
            const double a = aRow[k];
            const double* const bRow = B.data() + k * N;
            for (size_t j = 0; j < N; ++j) {
                cRow[j] += a * bRow[j];
            }
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
    int ranks = 1;
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

    const size_t matrixElements = N * N;
    if (N == 0 || matrixElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Matrix size is unsupported by this MPI implementation.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = N / static_cast<size_t>(ranks);
    const size_t extraRows = N % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            (static_cast<size_t>(rank) < extraRows ? static_cast<size_t>(rank) : extraRows);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(localRows * N);
    std::vector<double> B(matrixElements);
    std::vector<double> C(localRows * N);
    initMatrix(A, N, firstRow);
    if (rank == 0) {
        initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(matrixElements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        const double gflops = elapsedSeconds > 0.0 ?
            (2.0 * static_cast<double>(N) * N * N) / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    std::vector<double> globalC;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            globalC.resize(matrixElements);
            counts.resize(ranks);
            displacements.resize(ranks);
            size_t offset = 0;
            for (int process = 0; process < ranks; ++process) {
                const size_t rows = baseRows + (static_cast<size_t>(process) < extraRows ? 1 : 0);
                counts[process] = static_cast<int>(rows * N);
                displacements[process] = static_cast<int>(offset);
                offset += rows * N;
            }
        }
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0 && printResults) {
            print_results(globalC, "MatrixC");
        }
    }
    
    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalC, N) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
