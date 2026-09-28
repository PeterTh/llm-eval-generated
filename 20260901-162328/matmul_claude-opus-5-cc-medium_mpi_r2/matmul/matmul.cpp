#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <mpi.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// ---------------------------------------------------------------------------
// Distribution helpers
//
// The matrices are distributed over a 2D process grid of pr x pc ranks.
// Rank (pi, pj) owns the block C[rowOff .. rowOff+rows) x [colOff .. colOff+cols).
// To compute it, the rank needs the corresponding block-row of A (rows x N) and
// block-column of B (N x cols).  Since the matrix entries are a pure function of
// (i, j, N), every rank generates exactly the parts it needs locally -- this is
// the distributed-memory equivalent of the original initMatrix() and keeps the
// multiplication itself completely communication free.
// ---------------------------------------------------------------------------

static inline size_t blockBegin(const size_t N, const size_t parts, const size_t idx) noexcept {
    const size_t q = N / parts;
    const size_t r = N % parts;
    return idx * q + std::min(idx, r);
}

// Choose the most square process grid (pr <= pc) for the given number of ranks.
static void makeGrid(const int size, int& pr, int& pc) {
    pr = 1;
    for (int d = 1; d * d <= size; ++d) {
        if (size % d == 0) pr = d;
    }
    pc = size / pr;
}

// ---------------------------------------------------------------------------
// Cache blocked local GEMM (C += A * B), single threaded, one instance per rank.
// ---------------------------------------------------------------------------

namespace gemm {

constexpr size_t MR = 6;   // micro-kernel rows
constexpr size_t NR = 8;   // micro-kernel columns (2 AVX registers)
constexpr size_t MC = 144; // A panel rows   (multiple of MR)
constexpr size_t KC = 256; // panel depth
constexpr size_t NC = 512; // B panel columns (multiple of NR)

struct AlignedFree {
    void operator()(double* p) const noexcept { std::free(p); }
};
using Buffer = std::unique_ptr<double[], AlignedFree>;

static Buffer allocate(const size_t n) {
    void* p = std::aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64);
    if (p == nullptr) {
        fprintf(stderr, "Out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return Buffer(static_cast<double*>(p));
}

// Pack an MC x KC block of A into MR-row panels, zero padded at the edges.
static void packA(const double* __restrict A, const size_t lda, const size_t mc, const size_t kc,
                  double* __restrict Ap) {
    for (size_t i0 = 0; i0 < mc; i0 += MR) {
        const size_t m = std::min(MR, mc - i0);
        for (size_t k = 0; k < kc; ++k) {
            for (size_t i = 0; i < m; ++i) Ap[i] = A[(i0 + i) * lda + k];
            for (size_t i = m; i < MR; ++i) Ap[i] = 0.0;
            Ap += MR;
        }
    }
}

// Pack a KC x NC block of B into NR-column panels, zero padded at the edges.
static void packB(const double* __restrict B, const size_t ldb, const size_t kc, const size_t nc,
                  double* __restrict Bp) {
    for (size_t j0 = 0; j0 < nc; j0 += NR) {
        const size_t n = std::min(NR, nc - j0);
        for (size_t k = 0; k < kc; ++k) {
            const double* __restrict src = B + k * ldb + j0;
            for (size_t j = 0; j < n; ++j) Bp[j] = src[j];
            for (size_t j = n; j < NR; ++j) Bp[j] = 0.0;
            Bp += NR;
        }
    }
}

// MR x NR micro-kernel: the 6x8 accumulator block occupies 12 of the 16 AVX registers
// and stays resident for the whole k loop, leaving 4 registers for the operands.
static void microKernel(const size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, const size_t ldc) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c0l = _mm256_setzero_pd(), c0r = _mm256_setzero_pd();
    __m256d c1l = _mm256_setzero_pd(), c1r = _mm256_setzero_pd();
    __m256d c2l = _mm256_setzero_pd(), c2r = _mm256_setzero_pd();
    __m256d c3l = _mm256_setzero_pd(), c3r = _mm256_setzero_pd();
    __m256d c4l = _mm256_setzero_pd(), c4r = _mm256_setzero_pd();
    __m256d c5l = _mm256_setzero_pd(), c5r = _mm256_setzero_pd();

    for (size_t k = 0; k < kc; ++k, Ap += MR, Bp += NR) {
        _mm_prefetch(reinterpret_cast<const char*>(Bp + 16 * NR), _MM_HINT_T0);
        const __m256d bl = _mm256_load_pd(Bp);
        const __m256d br = _mm256_load_pd(Bp + 4);

        __m256d a = _mm256_broadcast_sd(Ap + 0);
        c0l = _mm256_fmadd_pd(a, bl, c0l);
        c0r = _mm256_fmadd_pd(a, br, c0r);
        a = _mm256_broadcast_sd(Ap + 1);
        c1l = _mm256_fmadd_pd(a, bl, c1l);
        c1r = _mm256_fmadd_pd(a, br, c1r);
        a = _mm256_broadcast_sd(Ap + 2);
        c2l = _mm256_fmadd_pd(a, bl, c2l);
        c2r = _mm256_fmadd_pd(a, br, c2r);
        a = _mm256_broadcast_sd(Ap + 3);
        c3l = _mm256_fmadd_pd(a, bl, c3l);
        c3r = _mm256_fmadd_pd(a, br, c3r);
        a = _mm256_broadcast_sd(Ap + 4);
        c4l = _mm256_fmadd_pd(a, bl, c4l);
        c4r = _mm256_fmadd_pd(a, br, c4r);
        a = _mm256_broadcast_sd(Ap + 5);
        c5l = _mm256_fmadd_pd(a, bl, c5l);
        c5r = _mm256_fmadd_pd(a, br, c5r);
    }

    const __m256d acc[MR][2] = {{c0l, c0r}, {c1l, c1r}, {c2l, c2r}, {c3l, c3r}, {c4l, c4r}, {c5l, c5r}};
    for (size_t i = 0; i < MR; ++i) {
        double* row = C + i * ldc;
        _mm256_storeu_pd(row, _mm256_add_pd(_mm256_loadu_pd(row), acc[i][0]));
        _mm256_storeu_pd(row + 4, _mm256_add_pd(_mm256_loadu_pd(row + 4), acc[i][1]));
    }
#else
    double acc[MR][NR];
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) acc[i][j] = 0.0;
    }

    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict a = Ap + k * MR;
        const double* __restrict b = Bp + k * NR;
        for (size_t i = 0; i < MR; ++i) {
            for (size_t j = 0; j < NR; ++j) acc[i][j] += a[i] * b[j];
        }
    }

    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) C[i * ldc + j] += acc[i][j];
    }
#endif
}

// Slow path for partial tiles at the right/bottom edge of the block.  Uses fused
// multiply-add just like the vectorized kernel so that the result of a tile does not
// depend on which of the two paths computed it.
static void microKernelEdge(const size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                            double* __restrict C, const size_t ldc, const size_t mr, const size_t nr) {
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < kc; ++k) sum = std::fma(Ap[k * MR + i], Bp[k * NR + j], sum);
            C[i * ldc + j] += sum;
        }
    }
}

// C (M x N, ld ldc) += A (M x K, ld lda) * B (K x N, ld ldb)
static void multiply(const size_t M, const size_t N, const size_t K, const double* A, const size_t lda,
                     const double* B, const size_t ldb, double* C, const size_t ldc) {
    if (M == 0 || N == 0 || K == 0) return;

    Buffer Ap = allocate(MC * KC);
    Buffer Bp = allocate(NC * KC);

    for (size_t jc = 0; jc < N; jc += NC) {
        const size_t nc = std::min(NC, N - jc);
        for (size_t pc = 0; pc < K; pc += KC) {
            const size_t kc = std::min(KC, K - pc);
            packB(B + pc * ldb + jc, ldb, kc, nc, Bp.get());

            for (size_t ic = 0; ic < M; ic += MC) {
                const size_t mc = std::min(MC, M - ic);
                packA(A + ic * lda + pc, lda, mc, kc, Ap.get());

                for (size_t jr = 0; jr < nc; jr += NR) {
                    const double* bPanel = Bp.get() + (jr / NR) * NR * kc;
                    const size_t nr = std::min(NR, nc - jr);
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        const double* aPanel = Ap.get() + (ir / MR) * MR * kc;
                        double* cTile = C + (ic + ir) * ldc + jc + jr;
                        const size_t mr = std::min(MR, mc - ir);
                        if (mr == MR && nr == NR) {
                            microKernel(kc, aPanel, bPanel, cTile, ldc);
                        } else {
                            microKernelEdge(kc, aPanel, bPanel, cTile, ldc, mr, nr);
                        }
                    }
                }
            }
        }
    }
}

} // namespace gemm

// Simple validation: compute a single element and compare.
// Each rank validates the check points that fall into its own block of C.
bool validateResult(const std::vector<double>& C, const size_t N, const size_t rowOff,
                    const size_t rows, const size_t colOff, const size_t cols) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    bool valid = true;
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            if (i < rowOff || i >= rowOff + rows || j < colOff || j >= colOff + cols) continue;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[(i - rowOff) * cols + (j - colOff)];
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
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    if (N == 0) {
        MPI_Finalize();
        return 0;
    }

    // 2D process grid; rank r owns block (r / pc, r % pc) of C.
    int pr = 1, pc = 1;
    makeGrid(numRanks, pr, pc);
    const size_t pi = static_cast<size_t>(rank / pc);
    const size_t pj = static_cast<size_t>(rank % pc);

    const size_t rowOff = blockBegin(N, static_cast<size_t>(pr), pi);
    const size_t rows = blockBegin(N, static_cast<size_t>(pr), pi + 1) - rowOff;
    const size_t colOff = blockBegin(N, static_cast<size_t>(pc), pj);
    const size_t cols = blockBegin(N, static_cast<size_t>(pc), pj + 1) - colOff;

    // Allocate the local parts: block-row of A, block-column of B, block of C
    std::vector<double> A(rows * N);
    std::vector<double> B(N * cols);
    std::vector<double> C(rows * cols, 0.0);

    // Initialize matrices (every rank generates the portion it needs)
    if (isRoot) printf("Initializing matrices...\n");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = getPseudoRndValue(N, rowOff + i, j);
        }
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            B[i * cols + j] = getPseudoRndValue(N, i, colOff + j);
        }
    }

    // Perform matrix multiplication
    if (isRoot) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    gemm::multiply(rows, cols, N, A.data(), N, B.data(), cols, C.data(), cols);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = static_cast<long>(duration.count());
    long elapsedMs = localMs;
    MPI_Allreduce(&localMs, &elapsedMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", elapsedMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather the full matrix on the root)
    if (printResults) {
        if (isRoot) {
            std::vector<double> full(N * N);
            for (size_t i = 0; i < rows; ++i) {
                std::copy_n(&C[i * cols], cols, &full[(rowOff + i) * N + colOff]);
            }
            std::vector<double> recv;
            for (int r = 1; r < numRanks; ++r) {
                const size_t ri = static_cast<size_t>(r / pc);
                const size_t rj = static_cast<size_t>(r % pc);
                const size_t rOff = blockBegin(N, static_cast<size_t>(pr), ri);
                const size_t rCnt = blockBegin(N, static_cast<size_t>(pr), ri + 1) - rOff;
                const size_t cOff = blockBegin(N, static_cast<size_t>(pc), rj);
                const size_t cCnt = blockBegin(N, static_cast<size_t>(pc), rj + 1) - cOff;
                if (rCnt == 0 || cCnt == 0) continue;
                recv.resize(rCnt * cCnt);
                MPI_Recv(recv.data(), static_cast<int>(rCnt * cCnt), MPI_DOUBLE, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (size_t i = 0; i < rCnt; ++i) {
                    std::copy_n(&recv[i * cCnt], cCnt, &full[(rOff + i) * N + cOff]);
                }
            }
            print_results(full, "MatrixC");
        } else if (rows > 0 && cols > 0) {
            MPI_Send(C.data(), static_cast<int>(rows * cols), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Validation
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        int localValid = validateResult(C, N, rowOff, rows, colOff, cols) ? 1 : 0;
        int valid = localValid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (isRoot) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
