#include <algorithm>
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

// Initialize the [rowBegin, rowEnd) x [colBegin, colEnd) block of a matrix into a
// dense local buffer with leading dimension ld. Generation is a pure function of the
// global indices, so every rank can produce its own block without communication.
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowEnd,
                     const size_t colBegin, const size_t colEnd, const size_t ld) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const row = mat.data() + (i - rowBegin) * ld;
        for (size_t j = colBegin; j < colEnd; ++j) {
            row[j - colBegin] = getPseudoRndValue(N, i, j);
        }
    }
}

// Register-tile size of the micro-kernel; the compiler vectorizes the NR dimension
// and keeps the MR x NR accumulator tile hot in registers.
constexpr size_t MR = 8;
constexpr size_t NR = 12;
// Cache blocking parameters: A panel (MC x KC) is kept in L2, B panel (KC x NC) in L3.
constexpr size_t MC = 128;
constexpr size_t KC = 256;
constexpr size_t NC = 384;

// Micro-kernel: C tile (mValid x nValid, leading dimension ldC) += Ap (kLen x MR) * Bp (kLen x NR).
// Both packed operands are traversed contiguously and in ascending k order.
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               double* __restrict__ C, const size_t ldC, const size_t kLen,
                               const size_t mValid, const size_t nValid) {
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t c = 0; c < NR; ++c) {
            acc[r][c] = (r < mValid && c < nValid) ? C[r * ldC + c] : 0.0;
        }
    }
    for (size_t k = 0; k < kLen; ++k) {
        const double* const __restrict__ a = Ap + k * MR;
        const double* const __restrict__ b = Bp + k * NR;
        for (size_t r = 0; r < MR; ++r) {
            for (size_t c = 0; c < NR; ++c) {
                acc[r][c] += a[r] * b[c];
            }
        }
    }
    for (size_t r = 0; r < mValid; ++r) {
        for (size_t c = 0; c < nValid; ++c) {
            C[r * ldC + c] = acc[r][c];
        }
    }
}

// Cache-blocked local multiplication: C(m x n) = A(m x K) * B(K x n), with the given
// leading dimensions. The k loop runs in ascending order for every output element, so
// the accumulation order (and hence the floating-point result) matches the reference.
void matrixMultiplyLocal(const double* __restrict__ A, const size_t ldA,
                         const double* __restrict__ B, const size_t ldB,
                         double* __restrict__ C, const size_t ldC,
                         const size_t m, const size_t n, const size_t K) {
    for (size_t i = 0; i < m; ++i) {
        std::fill(C + i * ldC, C + i * ldC + n, 0.0);
    }
    if (m == 0 || n == 0 || K == 0) return;

    std::vector<double> Bpack(KC * NC);
    std::vector<double> Apack(MC * KC);

    for (size_t jj = 0; jj < n; jj += NC) {
        const size_t nLen = std::min(NC, n - jj);
        for (size_t kk = 0; kk < K; kk += KC) {
            const size_t kLen = std::min(KC, K - kk);

            // Pack B block into NR-wide panels, zero padded at the edges.
            for (size_t jp = 0; jp < nLen; jp += NR) {
                const size_t nValid = std::min(NR, nLen - jp);
                double* const __restrict__ dst = Bpack.data() + jp * kLen;
                for (size_t k = 0; k < kLen; ++k) {
                    const double* const __restrict__ src = B + (kk + k) * ldB + jj + jp;
                    for (size_t c = 0; c < NR; ++c) {
                        dst[k * NR + c] = (c < nValid) ? src[c] : 0.0;
                    }
                }
            }

            for (size_t ii = 0; ii < m; ii += MC) {
                const size_t mLen = std::min(MC, m - ii);

                // Pack A block into MR-tall panels, zero padded at the edges.
                for (size_t ip = 0; ip < mLen; ip += MR) {
                    const size_t mValid = std::min(MR, mLen - ip);
                    double* const __restrict__ dst = Apack.data() + ip * kLen;
                    for (size_t k = 0; k < kLen; ++k) {
                        for (size_t r = 0; r < MR; ++r) {
                            dst[k * MR + r] = (r < mValid) ? A[(ii + ip + r) * ldA + kk + k] : 0.0;
                        }
                    }
                }

                for (size_t jp = 0; jp < nLen; jp += NR) {
                    const size_t nValid = std::min(NR, nLen - jp);
                    for (size_t ip = 0; ip < mLen; ip += MR) {
                        const size_t mValid = std::min(MR, mLen - ip);
                        microKernel(Apack.data() + ip * kLen, Bpack.data() + jp * kLen,
                                    C + (ii + ip) * ldC + jj + jp, ldC, kLen, mValid, nValid);
                    }
                }
            }
        }
    }
}

// Simple validation: compute a few elements and compare. The operand values are
// regenerated on the fly (they are a pure function of the global indices).
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
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 2D block decomposition: the ranks form a pr x pc process grid, rank (gr, gc)
    // owns the C block [rowBegin, rowEnd) x [colBegin, colEnd), and therefore needs
    // only the corresponding row panel of A and column panel of B. Both operands are
    // generated locally, so no data has to be distributed.
    int pr = static_cast<int>(std::sqrt(static_cast<double>(nranks)));
    while (pr > 1 && nranks % pr != 0) --pr;
    if (pr < 1) pr = 1;
    const int pc = nranks / pr;
    const int gridRow = rank / pc;
    const int gridCol = rank % pc;

    // Split [0, N) into cnt balanced chunks and return the begin index / size of chunk idx.
    const auto chunkBegin = [N](const size_t idx, const size_t cnt) {
        return idx * (N / cnt) + std::min(idx, N % cnt);
    };
    const auto chunkSize = [N](const size_t idx, const size_t cnt) {
        return N / cnt + (idx < N % cnt ? 1 : 0);
    };

    const size_t rowBegin = chunkBegin(static_cast<size_t>(gridRow), static_cast<size_t>(pr));
    const size_t localRows = chunkSize(static_cast<size_t>(gridRow), static_cast<size_t>(pr));
    const size_t colBegin = chunkBegin(static_cast<size_t>(gridCol), static_cast<size_t>(pc));
    const size_t localCols = chunkSize(static_cast<size_t>(gridCol), static_cast<size_t>(pc));

    if (rank == 0) {
        printf("MPI ranks: %d (%d x %d process grid)\n", nranks, pr, pc);
    }

    // Allocate the local blocks: A row panel (localRows x N), B column panel
    // (N x localCols) and the local C block (localRows x localCols).
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * localCols);
    std::vector<double> C(localRows * localCols);

    // Initialize matrices (fully deterministic, so every rank generates its own part)
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixBlock(A, N, rowBegin, rowBegin + localRows, 0, N, N);
    initMatrixBlock(B, N, 0, N, colBegin, colBegin + localCols, localCols);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A.data(), N, B.data(), localCols, C.data(), localCols,
                        localRows, localCols, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = static_cast<long>(duration.count());
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the full result on rank 0 only if it is needed
    if (printResults || validate) {
        std::vector<double> fullC;
        std::vector<MPI_Request> recvReqs;

        if (rank == 0) {
            fullC.resize(N * N);
            recvReqs.reserve(nranks);
            for (int r = 0; r < nranks; ++r) {
                const size_t rBegin = chunkBegin(static_cast<size_t>(r / pc), static_cast<size_t>(pr));
                const size_t rRows = chunkSize(static_cast<size_t>(r / pc), static_cast<size_t>(pr));
                const size_t cBegin = chunkBegin(static_cast<size_t>(r % pc), static_cast<size_t>(pc));
                const size_t cCols = chunkSize(static_cast<size_t>(r % pc), static_cast<size_t>(pc));
                if (rRows == 0 || cCols == 0) continue;

                // Receive the remote block directly into its place in the full matrix.
                MPI_Datatype blockType;
                MPI_Type_vector(static_cast<int>(rRows), static_cast<int>(cCols),
                                static_cast<int>(N), MPI_DOUBLE, &blockType);
                MPI_Type_commit(&blockType);
                recvReqs.emplace_back();
                MPI_Irecv(fullC.data() + rBegin * N + cBegin, 1, blockType, r, 0, MPI_COMM_WORLD,
                          &recvReqs.back());
                MPI_Type_free(&blockType);
            }
        }

        MPI_Request sendReq;
        MPI_Isend(C.data(), static_cast<int>(localRows * localCols), MPI_DOUBLE, 0, 0,
                  MPI_COMM_WORLD, &sendReq);
        if (rank == 0 && !recvReqs.empty()) {
            MPI_Waitall(static_cast<int>(recvReqs.size()), recvReqs.data(), MPI_STATUSES_IGNORE);
        }
        MPI_Wait(&sendReq, MPI_STATUS_IGNORE);

        if (rank == 0) {
            if (printResults) {
                print_results(fullC, "MatrixC");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullC, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
