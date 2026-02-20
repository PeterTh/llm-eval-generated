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

static inline void initMatrixRows(std::vector<double>& matRows, const size_t N, const size_t rowOffset) {
    const size_t localRows = matRows.size() / N;
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = rowOffset + li;
        for (size_t j = 0; j < N; ++j) {
            matRows[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void initMatrixFull(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void matrixMultiplyLocal(const std::vector<double>& A_local,
                                      const std::vector<double>& B,
                                      std::vector<double>& C_local,
                                      const size_t localRows,
                                      const size_t N) {
    std::fill(C_local.begin(), C_local.end(), 0.0);

    // Cache-friendly loop order: i-k-j (B rows contiguous, C row updated in-place)
    for (size_t i = 0; i < localRows; ++i) {
        const double* __restrict aRow = A_local.data() + i * N;
        double* __restrict cRow = C_local.data() + i * N;

        // Optional j-blocking to improve cache locality on large N
        constexpr size_t JB = 256;
        for (size_t k = 0; k < N; ++k) {
            const double a = aRow[k];
            const double* __restrict bRow = B.data() + k * N;

            for (size_t jb = 0; jb < N; jb += JB) {
                const size_t jEnd = (jb + JB < N) ? (jb + JB) : N;
                for (size_t j = jb; j < jEnd; ++j) {
                    cRow[j] += a * bRow[j];
                }
            }
        }
    }
}

static bool validateResultDeterministic(const std::vector<double>& C, const size_t N) {
    // Check a few fixed positions (same as original)
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static inline void bcast_doubles(std::vector<double>& v, const int root) {
    const size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t off = 0;
    while (off < v.size()) {
        const size_t remaining = v.size() - off;
        const int chunk = static_cast<int>(remaining > maxChunk ? maxChunk : remaining);
        MPI_Bcast(v.data() + off, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        off += static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    int validate = 0;
    int printResults = 0;
    int doExit = 0;
    int exitCode = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const long long nIn = atoll(argv[++i]);
                if (nIn <= 0) {
                    printf("Matrix size N must be > 0\n");
                    doExit = 1;
                    exitCode = 1;
                } else {
                    N = static_cast<size_t>(nIn);
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                doExit = 1;
                exitCode = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                doExit = 1;
                exitCode = 1;
            }
        }

        if (N == 0) {
            printf("Matrix size N must be > 0\n");
            doExit = 1;
            exitCode = 1;
        }
    }

    // Broadcast configuration
    uint64_t N_u64 = static_cast<uint64_t>(N);
    MPI_Bcast(&N_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&doExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_u64);

    if (doExit) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t rem = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowOffset = static_cast<size_t>(rank) * baseRows +
                             (static_cast<size_t>(rank) < rem ? static_cast<size_t>(rank) : rem);

    // Allocate local A and C; full B replicated
    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);
    std::vector<double> B;
    B.resize(N * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    initMatrixRows(A_local, N, rowOffset);

    if (rank == 0) {
        initMatrixFull(B, N);
    }
    bcast_doubles(B, 0);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyLocal(A_local, B, C_local, localRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localSeconds = t1 - t0;
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxSeconds * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double denom = (maxSeconds > 0.0) ? maxSeconds : 1e-12;
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) /
                              denom / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather C for printing/validation if requested
    std::vector<double> C_full;
    if (printResults || validate) {
        if (rank == 0) {
            C_full.resize(N * N);
        }

        std::vector<int> recvcounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvcounts.resize(worldSize);
            displs.resize(worldSize);
            size_t disp = 0;
            for (int r = 0; r < worldSize; ++r) {
                const size_t rRows = baseRows + (static_cast<size_t>(r) < rem ? 1 : 0);
                const size_t cnt = rRows * N;
                if (cnt > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    disp > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    printf("Matrix too large for MPI_Gatherv counts (int overflow)\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                recvcounts[r] = static_cast<int>(cnt);
                displs[r] = static_cast<int>(disp);
                disp += cnt;
            }
        }

        const size_t sendCountSz = localRows * N;
        if (sendCountSz > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                printf("Matrix too large for MPI_Gatherv sendcount (int overflow)\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        const int sendCount = static_cast<int>(sendCountSz);

        MPI_Gatherv(C_local.data(), sendCount, MPI_DOUBLE,
                    rank == 0 ? C_full.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int finalRc = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResultDeterministic(C_full, N);
            if (ok) {
                printf("Validation: PASSED\n");
                finalRc = 0;
            } else {
                printf("Validation: FAILED\n");
                finalRc = 1;
            }
        }
    }

    MPI_Bcast(&finalRc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return finalRc;
}
