#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowBegin + rows) of an NxN matrix; the block is stored
// densely in "mat" (rows x N), i.e. local row 0 is global row rowBegin.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        const size_t gi = rowBegin + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, gi, j);
        }
    }
}

namespace {
constexpr size_t JB = 512; // column block of B/C
constexpr size_t KB = 256; // k block (packed B panel stays L2-resident)
constexpr size_t MR = 6;   // rows per register tile
constexpr size_t NR = 8;   // columns per register tile (two AVX2 vectors)

#if defined(__AVX2__)
// C tile (MR x NR) is held in vector registers across the whole k block; the
// multiply and add stay separate (see -ffp-contract=off) so that every C element is
// rounded exactly like "sum += A[i][k] * B[k][j]" in the original scalar kernel.
inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                        double* __restrict__ C, const size_t kb, const size_t N) {
    __m256d c[MR][2];
    for (size_t u = 0; u < MR; ++u) {
        c[u][0] = _mm256_loadu_pd(C + u * N);
        c[u][1] = _mm256_loadu_pd(C + u * N + 4);
    }
    for (size_t k = 0; k < kb; ++k) {
        const __m256d b0 = _mm256_loadu_pd(Bp + k * NR);
        const __m256d b1 = _mm256_loadu_pd(Bp + k * NR + 4);
        for (size_t u = 0; u < MR; ++u) {
            const __m256d av = _mm256_broadcast_sd(Ap + u * kb + k);
            c[u][0] = _mm256_add_pd(c[u][0], _mm256_mul_pd(av, b0));
            c[u][1] = _mm256_add_pd(c[u][1], _mm256_mul_pd(av, b1));
        }
    }
    for (size_t u = 0; u < MR; ++u) {
        _mm256_storeu_pd(C + u * N, c[u][0]);
        _mm256_storeu_pd(C + u * N + 4, c[u][1]);
    }
}
#else
// Portable fallback with the same accumulation order.
inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                        double* __restrict__ C, const size_t kb, const size_t N) {
    double c[MR][NR];
    for (size_t u = 0; u < MR; ++u) {
        for (size_t v = 0; v < NR; ++v) {
            c[u][v] = C[u * N + v];
        }
    }
    for (size_t k = 0; k < kb; ++k) {
        for (size_t u = 0; u < MR; ++u) {
            const double av = Ap[u * kb + k];
            for (size_t v = 0; v < NR; ++v) {
                c[u][v] += av * Bp[k * NR + v];
            }
        }
    }
    for (size_t u = 0; u < MR; ++u) {
        for (size_t v = 0; v < NR; ++v) {
            C[u * N + v] = c[u][v];
        }
    }
}
#endif

// Scratch buffers for the packed A/B panels, allocated once per rank.
struct Scratch {
    std::vector<double> Bp; // KB x JB, split into panels of NR columns
    std::vector<double> Ap; // MR x KB
    Scratch() : Bp(KB * JB + NR), Ap(MR * KB) {}
};

// C(rows x N) += A(rows x N)[:, kOff : kOff+kCount] * Bblk(kCount x N).
// Each C element is accumulated over k in strictly ascending order, so the result is
// bit-identical to the original scalar kernel (the caller feeds ascending k blocks).
void multiplyAccumulate(const double* __restrict__ A, const size_t kOff, const double* __restrict__ Bblk,
                        const size_t kCount, double* __restrict__ C, const size_t rows, const size_t N,
                        Scratch& scratch) {
    double* const Bp = scratch.Bp.data();
    double* const Ap = scratch.Ap.data();

    for (size_t kk = 0; kk < kCount; kk += KB) {
        const size_t kb = (kk + KB < kCount) ? KB : (kCount - kk);

        for (size_t jj = 0; jj < N; jj += JB) {
            const size_t jb = (jj + JB < N) ? JB : (N - jj);
            const size_t jFull = (jb / NR) * NR;

            // Pack the B block into NR-wide, k-major panels for unit-stride access.
            for (size_t jt = 0; jt < jFull; jt += NR) {
                double* dst = Bp + jt * kb;
                for (size_t k = 0; k < kb; ++k) {
                    std::memcpy(dst + k * NR, Bblk + (kk + k) * N + jj + jt, NR * sizeof(double));
                }
            }

            const size_t iFull = (rows / MR) * MR;
            for (size_t i = 0; i < iFull; i += MR) {
                for (size_t u = 0; u < MR; ++u) {
                    std::memcpy(Ap + u * kb, A + (i + u) * N + kOff + kk, kb * sizeof(double));
                }
                for (size_t jt = 0; jt < jFull; jt += NR) {
                    microKernel(Ap, Bp + jt * kb, C + i * N + jj + jt, kb, N);
                }
                // Leftover columns (N not a multiple of NR)
                for (size_t j = jFull; j < jb; ++j) {
                    for (size_t u = 0; u < MR; ++u) {
                        double sum = C[(i + u) * N + jj + j];
                        for (size_t k = 0; k < kb; ++k) {
                            sum += A[(i + u) * N + kOff + kk + k] * Bblk[(kk + k) * N + jj + j];
                        }
                        C[(i + u) * N + jj + j] = sum;
                    }
                }
            }

            // Leftover rows (local row count not a multiple of MR)
            for (size_t i = iFull; i < rows; ++i) {
                double* __restrict__ c0 = C + i * N + jj;
                const double* __restrict__ a0 = A + i * N + kOff + kk;
                for (size_t k = 0; k < kb; ++k) {
                    const double* __restrict__ b = Bblk + (kk + k) * N + jj;
                    const double v0 = a0[k];
                    for (size_t j = 0; j < jb; ++j) {
                        c0[j] += v0 * b[j];
                    }
                }
            }
        }
    }
}
} // namespace

// Simple validation: recompute a few elements and compare.
// Each rank checks the check points whose row it owns; "Bcols" holds the full columns
// of B needed by the check points (N x numCols, column c corresponds to checkCols[c]).
bool validateLocal(const std::vector<double>& A, const std::vector<double>& Bcols, const size_t numCols,
                   const std::vector<double>& C, const size_t N, const size_t rowBegin, const size_t rows) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            if (i < rowBegin || i >= rowBegin + rows) {
                continue; // row not owned by this rank
            }
            const size_t li = i - rowBegin;
            const size_t col = j % numCols; // check columns are 0..numCols-1 (numCols = min(5, N))

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[li * N + k] * Bcols[k * numCols + col];
            }

            const double actual = C[li * N + j];
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
        printf("MPI ranks: %d\n", nranks);
    }

    // 1D row decomposition: rank r owns rows [rowOffsets[r], rowOffsets[r] + rowCounts[r])
    // of A and C, and the same row block of B (i.e. a k slab).
    std::vector<int> rowCounts(nranks);
    std::vector<int> rowOffsets(nranks);
    size_t maxRows = 0;
    {
        const size_t base = N / static_cast<size_t>(nranks);
        const size_t rem = N % static_cast<size_t>(nranks);
        size_t off = 0;
        for (int r = 0; r < nranks; ++r) {
            const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            rowCounts[r] = static_cast<int>(cnt);
            rowOffsets[r] = static_cast<int>(off);
            off += cnt;
            maxRows = (cnt > maxRows) ? cnt : maxRows;
        }
    }
    const size_t myRows = static_cast<size_t>(rowCounts[rank]);
    const size_t myFirstRow = static_cast<size_t>(rowOffsets[rank]);

    // B is needed in full by every rank. If the ranks sharing a node can afford a private
    // copy, each one materializes B locally (the generator is a pure function of (N, i, j),
    // so this needs no communication at all). Otherwise B stays distributed as k slabs that
    // are streamed through a pipelined broadcast during the multiplication.
    bool replicateB = false;
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int ranksPerNode = 1;
        MPI_Comm_size(nodeComm, &ranksPerNode);
        MPI_Comm_free(&nodeComm);

        const double pages = static_cast<double>(sysconf(_SC_PHYS_PAGES));
        const double pageSize = static_cast<double>(sysconf(_SC_PAGESIZE));
        const double nodeMemory = (pages > 0 && pageSize > 0) ? pages * pageSize : 0.0;
        const double needed = static_cast<double>(ranksPerNode) * static_cast<double>(N) *
                              static_cast<double>(N) * static_cast<double>(sizeof(double));
        int local = (nodeMemory > 0.0 && needed <= 0.25 * nodeMemory) ? 1 : 0;
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        replicateB = (global != 0);
    }

    // Local storage: row block of A and C, plus either all of B or the own k slab of B
    // together with two receive buffers for the pipelined broadcast.
    std::vector<double> A(myRows * N);
    std::vector<double> C(myRows * N);
    std::vector<double> B(replicateB ? N * N : myRows * N);
    std::vector<double> Bbuf[2];
    if (!replicateB) {
        Bbuf[0].resize(maxRows * N);
        Bbuf[1].resize(maxRows * N);
    }

    // Initialize matrices; every rank produces exactly the data it holds.
    if (rank == 0) {
        printf("Initializing matrices...\n");
        printf("B distribution: %s\n", replicateB ? "replicated" : "pipelined broadcast");
    }
    initMatrixRows(A, N, myFirstRow, myRows);
    if (replicateB) {
        initMatrixRows(B, N, 0, N);
    } else {
        initMatrixRows(B, N, myFirstRow, myRows);
    }

    // A row of the matrix is the unit of communication (avoids int element counts).
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Packing buffers for the local kernel (allocated before the timed region)
    Scratch scratch;

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::memset(C.data(), 0, myRows * N * sizeof(double));

    if (replicateB) {
        // No communication needed: the whole k range is available locally.
        if (myRows > 0) {
            multiplyAccumulate(A.data(), 0, B.data(), N, C.data(), myRows, N, scratch);
        }
    } else {
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        // Broadcast the k slab owned by rank s to everyone; the owner stages it in the buffer
        // so that all ranks run the identical compute path.
        auto postBcast = [&](const int s) {
            if (rowCounts[s] == 0) {
                return;
            }
            double* buf = Bbuf[s & 1].data();
            if (s == rank) {
                std::memcpy(buf, B.data(), myRows * N * sizeof(double));
            }
            MPI_Ibcast(buf, rowCounts[s], rowType, s, MPI_COMM_WORLD, &requests[s & 1]);
        };

        postBcast(0);
        if (nranks > 1) {
            postBcast(1);
        }

        // Steps run in ascending rank order, so every C element accumulates its k indices
        // in ascending order - matching the original kernel bit-for-bit.
        for (int s = 0; s < nranks; ++s) {
            const size_t kCount = static_cast<size_t>(rowCounts[s]);
            if (kCount == 0) {
                continue;
            }
            MPI_Wait(&requests[s & 1], MPI_STATUS_IGNORE);

            if (myRows > 0) {
                multiplyAccumulate(A.data(), static_cast<size_t>(rowOffsets[s]), Bbuf[s & 1].data(), kCount,
                                   C.data(), myRows, N, scratch);
            }

            if (s + 2 < nranks) {
                postBcast(s + 2); // buffer is free again, keep the pipeline filled
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

    }

    // Print results for external validation; C is assembled on rank 0 only when asked for,
    // outside of the timed region (this is output, not part of the multiplication).
    if (printResults) {
        std::vector<double> Cfull;
        if (rank == 0) {
            Cfull.resize(N * N);
        }
        MPI_Gatherv(C.data(), rowCounts[rank], rowType, Cfull.data(), rowCounts.data(), rowOffsets.data(),
                    rowType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(Cfull, "MatrixC");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        // Gather the (at most 5) B columns touched by the check points on all ranks.
        const size_t numCols = (N < 5) ? N : 5;
        const size_t localBRows = replicateB ? N : myRows;
        std::vector<double> myBcols(localBRows * numCols);
        for (size_t i = 0; i < localBRows; ++i) {
            for (size_t c = 0; c < numCols; ++c) {
                myBcols[i * numCols + c] = B[i * N + c];
            }
        }
        std::vector<double> Bcols;
        if (replicateB) {
            Bcols = std::move(myBcols);
        } else {
            Bcols.resize(N * numCols);
            std::vector<int> colCounts(nranks), colOffsets(nranks);
            for (int r = 0; r < nranks; ++r) {
                colCounts[r] = static_cast<int>(rowCounts[r] * static_cast<int>(numCols));
                colOffsets[r] = static_cast<int>(rowOffsets[r] * static_cast<int>(numCols));
            }
            MPI_Allgatherv(myBcols.data(), colCounts[rank], MPI_DOUBLE, Bcols.data(), colCounts.data(),
                           colOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }

        const int localValid = validateLocal(A, Bcols, numCols, C, N, myFirstRow, myRows) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Type_free(&rowType);
    MPI_Finalize();
    return exitCode;
}
