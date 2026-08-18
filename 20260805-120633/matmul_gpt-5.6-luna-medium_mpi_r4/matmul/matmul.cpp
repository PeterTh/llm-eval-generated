#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N,
                const size_t firstRow = 0) {
    const size_t rows = N == 0 ? 0 : mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        const size_t globalRow = firstRow + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t rows, const size_t N) {
    // The k loop remains the innermost accumulation order for every element,
    // matching the serial implementation while improving cache locality.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
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

static bool checkedSquareSize(const size_t N) {
    return N != 0 && N <= std::numeric_limits<size_t>::max() / N;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments identically on every rank.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || end == argv[i] || *end != '\0') {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
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

    if (!checkedSquareSize(N)) {
        if (rank == 0) printf("Matrix size is too large or zero: %zu\n", N);
        MPI_Finalize();
        return 1;
    }

    const size_t rows = N / static_cast<size_t>(worldSize) +
                        (static_cast<size_t>(rank) < N % static_cast<size_t>(worldSize) ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * (N / static_cast<size_t>(worldSize)) +
                            std::min(static_cast<size_t>(rank), N % static_cast<size_t>(worldSize));
    const size_t localElements = rows * N;
    if (localElements > static_cast<size_t>(INT_MAX) || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Matrix is too large for this MPI implementation's count limits\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // A and C are distributed by rows; B is replicated because every rank
    // needs all columns for its local output rows.
    std::vector<double> A(localElements);
    std::vector<double> B(N * N);
    std::vector<double> C(localElements, 0.0);
    
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, firstRow);
    if (rank == 0) initMatrix(B, N);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    matrixMultiply(A, B, C, rows, N);

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(seconds * 1000.0);
    
    if (rank == 0) printf("Computation time: %ld ms\n", durationMs);
    
    if (rank == 0) {
        const double gflops = seconds > 0.0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullC;
    const bool needFullResult = printResults || validate;
    if (rank == 0 && needFullResult) fullC.resize(N * N);
    if (needFullResult) {
        std::vector<int> counts(worldSize), displacements(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rRows = N / static_cast<size_t>(worldSize) +
                (static_cast<size_t>(r) < N % static_cast<size_t>(worldSize) ? 1 : 0);
            const size_t rFirst = static_cast<size_t>(r) * (N / static_cast<size_t>(worldSize)) +
                std::min(static_cast<size_t>(r), N % static_cast<size_t>(worldSize));
            counts[r] = static_cast<int>(rRows * N);
            displacements[r] = static_cast<int>(rFirst * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if (rank == 0 && printResults) {
        print_results(fullC, "MatrixC");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = false;
        if (rank == 0) {
            std::vector<double> fullA(N * N);
            initMatrix(fullA, N);
            valid = validateResult(fullA, B, fullC, N);
        }
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else if (rank == 0) {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
