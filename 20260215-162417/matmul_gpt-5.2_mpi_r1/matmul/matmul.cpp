#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <algorithm>
#include <cstdint>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization (deterministic across ranks)
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static inline void initMatrixRange(double* mat, const size_t N, const size_t row0, const size_t rows) {
    for (size_t ii = 0; ii < rows; ++ii) {
        const size_t i = row0 + ii;
        double* row = mat + ii * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void initMatrixFull(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        double* row = mat.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void transposeBlocked(const std::vector<double>& B, std::vector<double>& BT, const size_t N) {
    constexpr size_t TB = 32;
    for (size_t i0 = 0; i0 < N; i0 += TB) {
        const size_t iMax = std::min(N, i0 + TB);
        for (size_t j0 = 0; j0 < N; j0 += TB) {
            const size_t jMax = std::min(N, j0 + TB);
            for (size_t i = i0; i < iMax; ++i) {
                const double* __restrict__ brow = B.data() + i * N;
                for (size_t j = j0; j < jMax; ++j) {
                    BT[j * N + i] = brow[j];
                }
            }
        }
    }
}

static inline void matrixMultiplyBlockedLocal(const double* __restrict__ A,
                                             const double* __restrict__ BT,
                                             double* __restrict__ C,
                                             const size_t N,
                                             const size_t rows) {
    std::memset(C, 0, rows * N * sizeof(double));

    constexpr size_t JB = 64;
    constexpr size_t KB = 256;

    for (size_t ii = 0; ii < rows; ++ii) {
        const double* __restrict__ Arow = A + ii * N;
        double* __restrict__ Crow = C + ii * N;

        for (size_t jb = 0; jb < N; jb += JB) {
            const size_t jEnd = std::min(N, jb + JB);
            for (size_t kb = 0; kb < N; kb += KB) {
                const size_t kEnd = std::min(N, kb + KB);
                const size_t len = kEnd - kb;
                const double* __restrict__ Ablk = Arow + kb;

                for (size_t j = jb; j < jEnd; ++j) {
                    const double* __restrict__ BTblk = BT + j * N + kb;
                    double sum = Crow[j];

                    size_t t = 0;
                    for (; t + 3 < len; t += 4) {
                        sum += Ablk[t + 0] * BTblk[t + 0] + Ablk[t + 1] * BTblk[t + 1] +
                               Ablk[t + 2] * BTblk[t + 2] + Ablk[t + 3] * BTblk[t + 3];
                    }
                    for (; t < len; ++t) {
                        sum += Ablk[t] * BTblk[t];
                    }

                    Crow[j] = sum;
                }
            }
        }
    }
}

static inline void computeRowPartition(const size_t N, const int worldSize, const int rank,
                                      size_t& row0, size_t& rows) {
    const size_t base = N / static_cast<size_t>(worldSize);
    const size_t rem = N % static_cast<size_t>(worldSize);
    rows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    row0 = base * static_cast<size_t>(rank) + std::min(rem, static_cast<size_t>(rank));
}

static inline int ownerOfRow(const size_t N, const int worldSize, const size_t i) {
    const size_t base = N / static_cast<size_t>(worldSize);
    const size_t rem = N % static_cast<size_t>(worldSize);
    const size_t cut = (base + 1) * rem;
    if (i < cut) return static_cast<int>(i / (base + 1));
    return static_cast<int>(rem + (i - cut) / base);
}

static inline size_t rowStartOfRank(const size_t N, const int worldSize, const int r) {
    const size_t base = N / static_cast<size_t>(worldSize);
    const size_t rem = N % static_cast<size_t>(worldSize);
    if (static_cast<size_t>(r) < rem) return (base + 1) * static_cast<size_t>(r);
    return (base + 1) * rem + base * (static_cast<size_t>(r) - rem);
}

static inline bool validateDistributed(const double* __restrict__ localC,
                                      const size_t N,
                                      const size_t row0,
                                      const size_t rows,
                                      const int worldSize,
                                      const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    // Non-blocking send of owned checkpoints to rank 0, point-to-point recv on rank 0.
    std::vector<MPI_Request> reqs;
    reqs.reserve(25);

    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            const int tag = pi * 8 + pj;
            const int owner = ownerOfRow(N, worldSize, i);

            if (rank == owner && rank != 0) {
                const size_t li = i - row0;
                const double val = localC[li * N + j];
                MPI_Request req{};
                MPI_Isend(&val, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &req);
                reqs.push_back(req);
            }
        }
    }

    bool ok = true;

    if (rank == 0) {
        for (int pi = 0; pi < 5; ++pi) {
            for (int pj = 0; pj < 5; ++pj) {
                const size_t i = checkPoints[pi] % N;
                const size_t j = checkPoints[pj] % N;
                const int tag = pi * 8 + pj;
                const int owner = ownerOfRow(N, worldSize, i);

                double actual = 0.0;
                if (owner == 0) {
                    const size_t li = i - row0;
                    actual = localC[li * N + j];
                } else {
                    MPI_Recv(&actual, 1, MPI_DOUBLE, owner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                }

                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    std::printf(
                        "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, relError);
                    ok = false;
                }
            }
        }
    }

    if (!reqs.empty()) {
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

    int allOk = ok ? 1 : 0;
    MPI_Bcast(&allOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return allOk != 0;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation (gathers full C to rank 0)\n");
    std::printf("  -h           Show this help message\n");
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
    int exitNow = 0;
    int exitStatus = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitNow = 1;
                exitStatus = 0;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitNow = 1;
                exitStatus = 1;
            }
        }
    }

    MPI_Bcast(&exitNow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitNow) {
        MPI_Finalize();
        return exitStatus;
    }

    // Broadcast runtime configuration
    uint64_t N64 = static_cast<uint64_t>(N);
    MPI_Bcast(&N64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N64);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N == 0) {
        if (rank == 0) {
            std::printf("Matrix size must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark (MPI)\n");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t row0 = 0, rows = 0;
    computeRowPartition(N, worldSize, rank, row0, rows);

    // Local data
    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> BT(N * N);
    std::vector<double> C(rows * N);

    if (rank == 0) {
        std::printf("Initializing matrices...\n");
    }

    initMatrixRange(A.data(), N, row0, rows);
    initMatrixFull(B, N);
    transposeBlocked(B, BT, N);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyBlockedLocal(A.data(), BT.data(), C.data(), N, rows);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        std::printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) /
                              (maxTime) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Optional gather for result printing
    std::vector<double> Cglobal;
    if (printResults) {
        if (rank == 0) {
            Cglobal.resize(N * N);
        }

        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(worldSize);
            displs.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                size_t r0 = 0, rr = 0;
                computeRowPartition(N, worldSize, r, r0, rr);
                counts[r] = static_cast<int>(rr * N);
                displs[r] = static_cast<int>(r0 * N);
            }
        }

        MPI_Gatherv(C.data(), static_cast<int>(rows * N), MPI_DOUBLE,
                    rank == 0 ? Cglobal.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(Cglobal, "MatrixC");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }

        const bool valid = validateDistributed(C.data(), N, row0, rows, worldSize, rank);

        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
