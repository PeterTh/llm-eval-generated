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

// Initialize the block [rowBegin, rowEnd) x [colBegin, colEnd) of an NxN matrix,
// stored locally as a dense row-major block with leading dimension (colEnd - colBegin).
// The generator is a pure function of (N, i, j), so every rank can produce exactly
// the part of the input it owns without any communication.
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                     const size_t rowEnd, const size_t colBegin, const size_t colEnd) {
    const size_t ld = colEnd - colBegin;
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat.data() + (i - rowBegin) * ld;
        for (size_t j = colBegin; j < colEnd; ++j) {
            row[j - colBegin] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Rank-local GEMM: cache-blocked, packed, register-blocked kernel computing the
// C block owned by this rank.
// ---------------------------------------------------------------------------

namespace {

// SIMD width (in doubles) of the target ISA, using portable GCC/Clang vector
// extensions so that no external kernel library is required.
#if defined(__AVX512F__)
constexpr size_t VW = 8;
#elif defined(__AVX__)
constexpr size_t VW = 4;
#elif defined(__SSE2__) || defined(__aarch64__)
constexpr size_t VW = 2;
#else
constexpr size_t VW = 1;
#endif

typedef double vec __attribute__((vector_size(VW * sizeof(double))));
// Same type, but usable on unaligned addresses (the packed buffers and C).
typedef double vecu __attribute__((vector_size(VW * sizeof(double)), aligned(8)));

constexpr size_t MR = 6;       // micro-kernel rows
constexpr size_t NR = 2 * VW;  // micro-kernel columns (two vector registers wide)
constexpr size_t MC = 120;     // rows of A per L2 block (multiple of MR)
constexpr size_t KC = 256;     // depth per block
constexpr size_t NC = 512;     // columns of B per L3 block

// Pack a kc x nc block of B (row-major, leading dimension ldb) into column
// panels of width NR: panel p holds rows 0..kc-1, columns p*NR..p*NR+NR-1.
void packB(const double* __restrict B, const size_t ldb, const size_t kc, const size_t nc,
           double* __restrict Bp) {
    const size_t panels = (nc + NR - 1) / NR;
    for (size_t p = 0; p < panels; ++p) {
        const size_t j0 = p * NR;
        const size_t jn = (j0 + NR <= nc) ? NR : nc - j0;
        double* __restrict dst = Bp + p * kc * NR;
        for (size_t k = 0; k < kc; ++k) {
            const double* __restrict src = B + k * ldb + j0;
            for (size_t j = 0; j < jn; ++j) {
                dst[k * NR + j] = src[j];
            }
            for (size_t j = jn; j < NR; ++j) {
                dst[k * NR + j] = 0.0;
            }
        }
    }
}

// Pack an mc x kc block of A (row-major, leading dimension lda) into row panels
// of height MR, stored so that the micro-kernel reads MR contiguous values per k.
void packA(const double* __restrict A, const size_t lda, const size_t mc, const size_t kc,
           double* __restrict Ap) {
    const size_t panels = (mc + MR - 1) / MR;
    for (size_t p = 0; p < panels; ++p) {
        const size_t i0 = p * MR;
        const size_t in = (i0 + MR <= mc) ? MR : mc - i0;
        double* __restrict dst = Ap + p * kc * MR;
        for (size_t k = 0; k < kc; ++k) {
            for (size_t i = 0; i < in; ++i) {
                dst[k * MR + i] = A[(i0 + i) * lda + k];
            }
            for (size_t i = in; i < MR; ++i) {
                dst[k * MR + i] = 0.0;
            }
        }
    }
}

// MR x NR micro-kernel: acc = Ap_panel * Bp_panel accumulated over kc steps.
// Written with explicit vector types so that the 2*MR accumulators stay in
// vector registers for the whole k loop (2 loads + MR broadcasts + 2*MR FMAs
// per step).
inline void microKernelAcc(const double* __restrict Ap, const double* __restrict Bp,
                           const size_t kc, vec (&acc)[MR][2]) {
    for (size_t i = 0; i < MR; ++i) {
        acc[i][0] = vec{};
        acc[i][1] = vec{};
    }

    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict a = Ap + k * MR;
        const vecu* __restrict b = reinterpret_cast<const vecu*>(Bp + k * NR);
        const vec b0 = b[0];
        const vec b1 = b[1];
        for (size_t i = 0; i < MR; ++i) {
            vec av;
            for (size_t t = 0; t < VW; ++t) av[t] = a[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }
}

// Full tile: accumulate straight into C.
void microKernelFull(const double* __restrict Ap, const double* __restrict Bp, const size_t kc,
                     double* __restrict C, const size_t ldc) {
    vec acc[MR][2];
    microKernelAcc(Ap, Bp, kc, acc);

    for (size_t i = 0; i < MR; ++i) {
        vecu* __restrict c = reinterpret_cast<vecu*>(C + i * ldc);
        c[0] += acc[i][0];
        c[1] += acc[i][1];
    }
}

// Edge tile: write back through a padded scratch tile.
void microKernelEdge(const double* __restrict Ap, const double* __restrict Bp, const size_t kc,
                     double* __restrict C, const size_t ldc, const size_t mn, const size_t nn) {
    vec acc[MR][2];
    microKernelAcc(Ap, Bp, kc, acc);

    alignas(64) double tile[MR][NR];
    for (size_t i = 0; i < MR; ++i) {
        reinterpret_cast<vecu*>(tile[i])[0] = acc[i][0];
        reinterpret_cast<vecu*>(tile[i])[1] = acc[i][1];
    }

    for (size_t i = 0; i < mn; ++i) {
        double* __restrict c = C + i * ldc;
        for (size_t j = 0; j < nn; ++j) c[j] += tile[i][j];
    }
}

// Scratch buffers for the packed panels, allocated once per rank.
std::vector<double> Bpack;
std::vector<double> Apack;

// C(m x n) += A(m x kTotal) * B(kTotal x n), all row-major with the given
// leading dimensions. C is accumulated into, never cleared.
void localGemm(const double* __restrict A, const size_t lda, const double* __restrict B,
               const size_t ldb, double* __restrict C, const size_t ldc, const size_t m,
               const size_t n, const size_t kTotal) {
    if (m == 0 || n == 0 || kTotal == 0) return;

    if (Bpack.empty()) {
        Bpack.resize(KC * ((NC + NR - 1) / NR) * NR);
        Apack.resize(((MC + MR - 1) / MR) * MR * KC);
    }
    std::vector<double>& Bp = Bpack;
    std::vector<double>& Ap = Apack;

    for (size_t jc = 0; jc < n; jc += NC) {
        const size_t nc = std::min(NC, n - jc);
        for (size_t kc0 = 0; kc0 < kTotal; kc0 += KC) {
            const size_t kc = std::min(KC, kTotal - kc0);
            packB(B + kc0 * ldb + jc, ldb, kc, nc, Bp.data());

            for (size_t ic = 0; ic < m; ic += MC) {
                const size_t mc = std::min(MC, m - ic);
                packA(A + ic * lda + kc0, lda, mc, kc, Ap.data());

                for (size_t ir = 0; ir < mc; ir += MR) {
                    const size_t mn = std::min(MR, mc - ir);
                    const double* aPanel = Ap.data() + (ir / MR) * kc * MR;
                    for (size_t jr = 0; jr < nc; jr += NR) {
                        const size_t nn = std::min(NR, nc - jr);
                        const double* bPanel = Bp.data() + (jr / NR) * kc * NR;
                        double* cTile = C + (ic + ir) * ldc + jc + jr;
                        if (mn == MR && nn == NR) {
                            microKernelFull(aPanel, bPanel, kc, cTile, ldc);
                        } else {
                            microKernelEdge(aPanel, bPanel, kc, cTile, ldc, mn, nn);
                        }
                    }
                }
            }
        }
    }
}

} // namespace

// Simple validation: compute a few elements and compare.
// Each rank checks the points that fall into the C block it owns; it holds the
// full A rows and full B columns needed for those, so no communication is required.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N,
                    const size_t rowBegin, const size_t rowEnd,
                    const size_t colBegin, const size_t colEnd) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    const size_t localCols = colEnd - colBegin;

    bool valid = true;
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            if (i < rowBegin || i >= rowEnd || j < colBegin || j >= colEnd) continue;
            const size_t li = i - rowBegin;
            const size_t lj = j - colBegin;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[li * N + k] * B[k * localCols + lj];
            }

            const double actual = C[li * localCols + lj];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = false;
            }
        }
    }

    return valid;
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

    int rank = 0, numRanks = 1;
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

    // 2D block decomposition of C over a gridRows x gridCols process grid, chosen
    // as square as possible. Rank r owns the C block of process row r / gridCols and
    // process column r % gridCols, and correspondingly the A rows and B columns that
    // block needs. This keeps the per-rank data volume at O(N^2 / sqrt(P)) instead of
    // the O(N^2) that a pure row decomposition would stream through memory.
    int gridCols = 1;
    for (int c = 1; c * c <= numRanks; ++c) {
        if (numRanks % c == 0) gridCols = c;
    }
    const int gridRows = numRanks / gridCols;
    const int procRow = rank / gridCols;
    const int procCol = rank % gridCols;

    const auto blockBegin = [](const size_t total, const int parts, const int idx) {
        const size_t base = total / static_cast<size_t>(parts);
        const size_t rem = total % static_cast<size_t>(parts);
        return base * static_cast<size_t>(idx) + std::min(static_cast<size_t>(idx), rem);
    };

    const size_t rowBegin = blockBegin(N, gridRows, procRow);
    const size_t rowEnd = blockBegin(N, gridRows, procRow + 1);
    const size_t colBegin = blockBegin(N, gridCols, procCol);
    const size_t colEnd = blockBegin(N, gridCols, procCol + 1);
    const size_t localRows = rowEnd - rowBegin;
    const size_t localCols = colEnd - colBegin;

    if (rank == 0) {
        printf("MPI ranks: %d (process grid %d x %d)\n", numRanks, gridRows, gridCols);
    }

    // Allocate matrices: the A row panel, the B column panel and the C block owned
    // by this rank.
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * localCols);
    std::vector<double> C(localRows * localCols, 0.0);

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixBlock(A, N, rowBegin, rowEnd, 0, N);
    initMatrixBlock(B, N, 0, N, colBegin, colEnd);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    localGemm(A.data(), N, B.data(), localCols, C.data(), localCols, localRows, localCols, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long elapsedMs = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", elapsedMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation: assemble the distributed C blocks
    // on rank 0 first.
    if (printResults) {
        std::vector<double> Cfull;
        if (rank == 0) {
            Cfull.resize(N * N);
            for (size_t i = 0; i < localRows; ++i) {
                std::copy(C.begin() + i * localCols, C.begin() + (i + 1) * localCols,
                          Cfull.begin() + (rowBegin + i) * N + colBegin);
            }
            for (int r = 1; r < numRanks; ++r) {
                const size_t rb = blockBegin(N, gridRows, r / gridCols);
                const size_t re = blockBegin(N, gridRows, r / gridCols + 1);
                const size_t cb = blockBegin(N, gridCols, r % gridCols);
                const size_t ce = blockBegin(N, gridCols, r % gridCols + 1);
                if (re == rb || ce == cb) continue;

                MPI_Datatype blockType;
                MPI_Type_vector(static_cast<int>(re - rb), static_cast<int>(ce - cb),
                                static_cast<int>(N), MPI_DOUBLE, &blockType);
                MPI_Type_commit(&blockType);
                MPI_Recv(Cfull.data() + rb * N + cb, 1, blockType, r, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Type_free(&blockType);
            }
        } else if (localRows > 0 && localCols > 0) {
            MPI_Send(C.data(), static_cast<int>(localRows * localCols), MPI_DOUBLE, 0, 1,
                     MPI_COMM_WORLD);
        }

        if (rank == 0) print_results(Cfull, "MatrixC");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = validateResult(A, B, C, N, rowBegin, rowEnd, colBegin, colEnd) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            rc = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
