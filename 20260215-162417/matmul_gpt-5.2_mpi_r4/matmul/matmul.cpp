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

static inline void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void initMatrixRows(std::vector<double>& matRows, const size_t N, const size_t rowStart,
                                 const size_t numRows) {
    for (size_t ii = 0; ii < numRows; ++ii) {
        const size_t i = rowStart + ii;
        for (size_t j = 0; j < N; ++j) {
            matRows[ii * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void computeRowDecomposition(const size_t N, const int commSize, const int rank,
                                          size_t& rowStart, size_t& numRows) {
    const size_t p = static_cast<size_t>(commSize);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = (p == 0) ? 0 : (N / p);
    const size_t rem = (p == 0) ? 0 : (N % p);
    numRows = base + ((r < rem) ? 1 : 0);
    rowStart = (r < rem) ? (r * (base + 1)) : (rem * (base + 1) + (r - rem) * base);
}

static inline void matrixMultiplyLocal(const double* __restrict__ A, const double* __restrict__ B,
                                      double* __restrict__ C, const size_t N, const size_t localRows) {
    // Tile over columns to keep B streaming-friendly and enable vectorization.
    constexpr size_t JB = 64; // columns per tile

    for (size_t ii = 0; ii < localRows; ++ii) {
        const double* __restrict__ aRow = A + ii * N;
        double* __restrict__ cRow = C + ii * N;

        for (size_t j0 = 0; j0 < N; j0 += JB) {
            const size_t j1 = (j0 + JB < N) ? (j0 + JB) : N;
            // keep partial sums in registers/L1 and write out once
            double sum[JB];
            const size_t width = j1 - j0;
            for (size_t jj = 0; jj < width; ++jj) sum[jj] = 0.0;

            for (size_t k = 0; k < N; ++k) {
                const double a = aRow[k];
                const double* __restrict__ bRow = B + k * N + j0;
                for (size_t jj = 0; jj < width; ++jj) {
                    sum[jj] += a * bRow[jj];
                }
            }

            for (size_t jj = 0; jj < width; ++jj) {
                cRow[j0 + jj] = sum[jj];
            }
        }
    }
}

static inline bool validateResultMPI(const std::vector<double>& localC, const size_t N,
                                     const size_t rowStart, const size_t localRows, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Check a few fixed positions (same as original, but computed from the closed-form initializer)
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    int ok = 1;
    if (rank == 0) {
        printf("Validating result...\n");
    }

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double contrib = 0.0;
            if (i >= rowStart && i < rowStart + localRows) {
                const size_t li = i - rowStart;
                contrib = localC[li * N + j];
            }

            double actual = 0.0;
            MPI_Reduce(&contrib, &actual, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

            if (rank == 0) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                }

                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf(
                        "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, relError);
                    ok = 0;
                }
            }
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    return ok != 0;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation (gathers MatrixC to rank 0)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
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

    if (N == 0) {
        if (rank == 0) printf("Matrix size N must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", commSize);
        printf("Initializing matrices...\n");
    }

    size_t rowStart = 0, localRows = 0;
    computeRowDecomposition(N, commSize, rank, rowStart, localRows);

    // Local A rows and local C rows; B is replicated on all ranks.
    std::vector<double> localA(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);

    initMatrixRows(localA, N, rowStart, localRows);
    initMatrix(B, N);

    // Perform matrix multiplication (distributed by rows)
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyLocal(localA.data(), B.data(), localC.data(), N, localRows);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double elapsed = t1 - t0;

    double elapsedMax = 0.0;
    MPI_Reduce(&elapsed, &elapsedMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(elapsedMax * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              elapsedMax / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> recvCounts;
        std::vector<int> displs;
        std::vector<double> C;
        if (rank == 0) {
            recvCounts.resize(commSize);
            displs.resize(commSize);
            for (int r = 0; r < commSize; ++r) {
                size_t rs = 0, nr = 0;
                computeRowDecomposition(N, commSize, r, rs, nr);
                recvCounts[r] = static_cast<int>(nr * N);
                displs[r] = static_cast<int>(rs * N);
            }
            C.resize(N * N);
        }

        const int sendCount = static_cast<int>(localRows * N);
        MPI_Gatherv(localC.data(), sendCount, MPI_DOUBLE, rank == 0 ? C.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }

    // Validation
    if (validate) {
        const bool valid = validateResultMPI(localC, N, rowStart, localRows, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
