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

// Initialize a block of rows [rowBegin, rowEnd) of an NxN matrix
void initMatrixRows(std::vector<double>& mat, const size_t N,
                    const size_t rowBegin, const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Multiply the local row block of A (rows [rowBegin, rowEnd)) with full B.
// The i-k-j loop order accumulates over k in the same order as the naive
// triple loop (bitwise-identical results) while enabling vectorization and
// cache-friendly access to B.
void matrixMultiplyRows(const std::vector<double>& Alocal, const std::vector<double>& B,
                        std::vector<double>& Clocal, const size_t N,
                        const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        const double* Ai = &Alocal[i * N];
        double* Ci = &Clocal[i * N];
        for (size_t k = 0; k < N; ++k) {
            const double a = Ai[k];
            const double* Bk = &B[k * N];
            for (size_t j = 0; j < N; ++j) {
                Ci[j] += a * Bk[j];
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI processes: %d\n", nprocs);
    }

    // Row-block decomposition: rank r owns rows [rowBegin, rowEnd)
    const size_t rowsPerRank = N / nprocs;
    const size_t remainder = N % nprocs;
    const size_t rowBegin = rank * rowsPerRank + std::min<size_t>(rank, remainder);
    const size_t localRows = rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t rowEnd = rowBegin + localRows;

    // Allocate matrices: each rank holds its row block of A and C, plus full B
    std::vector<double> Alocal(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Clocal(localRows * N, 0.0);

    // Initialize matrices: initialization is deterministic, so every rank
    // generates its own data locally without communication
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(Alocal, N, rowBegin, rowEnd);
    initMatrixRows(B, N, 0, N);

    // Gather layout for assembling full C on rank 0
    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t rBegin = r * rowsPerRank + std::min<size_t>(r, remainder);
        const size_t rRows = rowsPerRank + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rBegin * N);
    }
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(Alocal, B, Clocal, N, localRows);

    MPI_Gatherv(Clocal.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Rank 0 regenerates full A for validation (deterministic init)
            std::vector<double> A(N * N);
            initMatrixRows(A, N, 0, N);
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
