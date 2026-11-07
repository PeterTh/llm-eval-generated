#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Cholesky decomposition, parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is processed in square tiles with a right-looking algorithm:
// step k factors diagonal tile (k,k), solves the tiles (i,k) below it and
// applies the rank-nb update to the trailing tiles. All of these operations,
// for all steps, form one global work list that the OpenMP threads consume
// with a dynamic schedule; each item waits on small per-tile dependency
// stamps, so there are no per-step barriers and later steps overlap the
// trailing update of earlier ones.
//
// Numerical semantics follow the original unblocked algorithm: for every
// element the products A[i][k]*A[j][k] are accumulated in increasing k order
// into a separate accumulator (one fused multiply-add chain starting at zero),
// and the final value is (A[i][j] - sum) / L[j][j] (or sqrt(A[j][j] - sum) on
// the diagonal). The result is deterministic and independent of the number of
// threads or the tile size; it agrees with the sequential code to rounding.

namespace {

// Micro-kernel block: kCB tile columns x kRB tile rows (two AVX2 vectors).
constexpr size_t kRB = 8;
constexpr size_t kCB = 4;
constexpr size_t kAlign = 64;

// Tile buffer layouts (nb x nb doubles per tile, nb a multiple of 16):
//  * "transposed" tile T (factored tile, T(kk, r) = L[row r][col kk] of the tile):
//    panel layout, panels of kRB rows: T[(r/8)*nb*8 + kk*8 + r%8].
//    For fixed r-panel the data for consecutive kk is contiguous (sequential stream).
//  * accumulator tile ACC (ACC(c, r) = running sum for element (row r, col c)):
//    blocks of kCB columns x kRB rows: ACC[((c/4)*(nb/8) + r/8)*32 + (c%4)*8 + r%8].
inline size_t trIdx(size_t kk, size_t r, size_t nb) {
    return ((r >> 3) * nb + kk) * kRB + (r & 7);
}
inline size_t accIdx(size_t c, size_t r, size_t nb) {
    return (((c >> 2) * (nb >> 3) + (r >> 3)) * kCB + (c & 3)) * kRB + (r & 7);
}

inline size_t tileIndex(size_t i, size_t j) {
    // Index of lower-triangular tile (i, j), j <= i.
    return i * (i + 1) / 2 + j;
}

// Pick the tile size: large tiles for compute efficiency, small enough to
// expose parallelism for small matrices.
size_t chooseTileSize(size_t n) {
    size_t nb = n / 16;
    nb = (nb / 16) * 16;  // multiple of the TRSM chunk and micro-kernel widths
    if (nb < 32) nb = 32;
    if (nb > 128) nb = 128;
    return nb;
}

struct AlignedDeleter {
    void operator()(double* p) const { std::free(p); }
};
using AlignedBuf = std::unique_ptr<double[], AlignedDeleter>;

AlignedBuf allocAligned(size_t count) {
    size_t bytes = count * sizeof(double);
    bytes = (bytes + kAlign - 1) / kAlign * kAlign;
    void* p = std::aligned_alloc(kAlign, bytes);
    if (!p) {
        fprintf(stderr, "Error: out of memory\n");
        std::abort();
    }
    return AlignedBuf(static_cast<double*>(p));
}

#if defined(__GNUC__)
typedef double v4d __attribute__((vector_size(32)));

inline v4d loadv(const double* p) {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, v4d v) { __builtin_memcpy(p, &v, sizeof(v)); }
inline v4d fmav(v4d a, v4d b, v4d c) {
    return v4d{std::fma(a[0], b[0], c[0]), std::fma(a[1], b[1], c[1]),
               std::fma(a[2], b[2], c[2]), std::fma(a[3], b[3], c[3])};
}

// ACC block (c0.., r0..) += sum_{kk < kb} T_j(kk, c) * T_i(kk, r)   (FMA chain, kk ascending)
inline void gemmBlock(double* __restrict ap, const double* __restrict tip,
                      const double* __restrict tjp, const size_t kb) {
    v4d a00 = loadv(ap + 0), a01 = loadv(ap + 4);
    v4d a10 = loadv(ap + 8), a11 = loadv(ap + 12);
    v4d a20 = loadv(ap + 16), a21 = loadv(ap + 20);
    v4d a30 = loadv(ap + 24), a31 = loadv(ap + 28);
    for (size_t kk = 0; kk < kb; ++kk) {
        v4d t0 = loadv(tip + kk * kRB);
        v4d t1 = loadv(tip + kk * kRB + 4);
        // Keep the row vectors in registers instead of re-reading them per FMA.
        __asm__("" : "+x"(t0), "+x"(t1));
        const double* tj = tjp + kk * kRB;
        v4d b;
        b = v4d{tj[0], tj[0], tj[0], tj[0]};
        a00 = fmav(t0, b, a00); a01 = fmav(t1, b, a01);
        b = v4d{tj[1], tj[1], tj[1], tj[1]};
        a10 = fmav(t0, b, a10); a11 = fmav(t1, b, a11);
        b = v4d{tj[2], tj[2], tj[2], tj[2]};
        a20 = fmav(t0, b, a20); a21 = fmav(t1, b, a21);
        b = v4d{tj[3], tj[3], tj[3], tj[3]};
        a30 = fmav(t0, b, a30); a31 = fmav(t1, b, a31);
    }
    storev(ap + 0, a00); storev(ap + 4, a01);
    storev(ap + 8, a10); storev(ap + 12, a11);
    storev(ap + 16, a20); storev(ap + 20, a21);
    storev(ap + 24, a30); storev(ap + 28, a31);
}
#else
inline void gemmBlock(double* __restrict ap, const double* __restrict tip,
                      const double* __restrict tjp, const size_t kb) {
    double a[kCB][kRB];
    for (size_t cc = 0; cc < kCB; ++cc)
        for (size_t rr = 0; rr < kRB; ++rr) a[cc][rr] = ap[cc * kRB + rr];
    for (size_t kk = 0; kk < kb; ++kk) {
        const double* ti = tip + kk * kRB;
        const double* tj = tjp + kk * kRB;
        for (size_t cc = 0; cc < kCB; ++cc) {
            const double b = tj[cc];
#pragma omp simd
            for (size_t rr = 0; rr < kRB; ++rr) a[cc][rr] = std::fma(ti[rr], b, a[cc][rr]);
        }
    }
    for (size_t cc = 0; cc < kCB; ++cc)
        for (size_t rr = 0; rr < kRB; ++rr) ap[cc * kRB + rr] = a[cc][rr];
}
#endif

// ACC(c, r) += sum_{kk < kb} T_j(kk, c) * T_i(kk, r) for the whole tile.
void gemmTile(double* __restrict acc, const double* __restrict Ti,
              const double* __restrict Tj, const size_t kb, const size_t nb) {
    const size_t npan = nb / kRB;
    for (size_t c0 = 0; c0 < nb; c0 += kCB) {
        const double* tjp = Tj + (c0 >> 3) * nb * kRB + (c0 & 7);
        double* ap = acc + (c0 >> 2) * npan * (kCB * kRB);
        for (size_t p = 0; p < npan; ++p) {
            gemmBlock(ap + p * (kCB * kRB), Ti + p * nb * kRB, tjp, kb);
        }
    }
}

// Factor diagonal tile k. Returns false if the matrix is not positive definite.
// Columns are processed in blocks of kCB: the contribution of all previous
// columns (kk < c0) is applied with the micro-kernel, then the in-block
// triangular part (kk in [c0, c)) is applied column by column, so each element
// still accumulates its products in increasing kk order.
bool potrfTile(double* __restrict A, const size_t n, const size_t nb, const size_t k,
               double* __restrict acc, double* __restrict Tkk) {
    const size_t base = k * nb;
    const size_t cols = std::min(nb, n - base);
    const size_t npan = nb / kRB;
    for (size_t c0 = 0; c0 < cols; c0 += kCB) {
        const size_t p0 = c0 / kRB;  // first row panel that holds rows >= c0
        const double* tjp = Tkk + (c0 >> 3) * nb * kRB + (c0 & 7);
        for (size_t p = p0; p < npan; ++p) {
            gemmBlock(acc + accIdx(c0, p * kRB, nb), Tkk + p * nb * kRB, tjp, c0);
        }
        for (size_t cc = 0; cc < kCB && c0 + cc < cols; ++cc) {
            const size_t c = c0 + cc;
            const size_t j = base + c;
            const double val = A[j * n + j] - acc[accIdx(c, c, nb)];
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            const double d = sqrt(val);
            A[j * n + j] = d;
            for (size_t p = p0; p < npan; ++p) {
                double* __restrict tcol = Tkk + trIdx(c, p * kRB, nb);
                const double* __restrict ab = acc + accIdx(c, p * kRB, nb);
                for (size_t rr = 0; rr < kRB; ++rr) {
                    const size_t r = p * kRB + rr;
                    if (r < c) {
                        tcol[rr] = 0.0;
                    } else if (r == c) {
                        tcol[rr] = d;
                    } else if (r < cols) {
                        const double v = (A[(base + r) * n + j] - ab[rr]) / d;
                        A[(base + r) * n + j] = v;
                        tcol[rr] = v;
                    } else {
                        tcol[rr] = 0.0;
                    }
                }
            }
            // In-block rank-1 update of the following columns (term kk = j).
            for (size_t cc2 = cc + 1; cc2 < kCB; ++cc2) {
                const double b = Tkk[trIdx(c, c0 + cc2, nb)];
                for (size_t p = p0; p < npan; ++p) {
                    const double* __restrict tcol = Tkk + trIdx(c, p * kRB, nb);
                    double* __restrict ab = acc + accIdx(c0 + cc2, p * kRB, nb);
#pragma omp simd
                    for (size_t rr = 0; rr < kRB; ++rr) {
                        ab[rr] = std::fma(tcol[rr], b, ab[rr]);
                    }
                }
            }
        }
    }
    // Zero the (never used) padding columns of the last tile so later reads
    // of the transposed tile are well defined.
    for (size_t c = cols; c < nb; ++c) {
        for (size_t r = 0; r < nb; ++r) Tkk[trIdx(c, r, nb)] = 0.0;
    }
    return true;
}

// Solve row panels [p0, p1) (kRB rows each) of tile (i, k), i > k, against the
// factored diagonal tile k. Different row panels of a tile are independent.
void trsmRows(double* __restrict A, const size_t n, const size_t nb, const size_t i,
              const size_t k, const size_t p0, const size_t p1, double* __restrict acc,
              double* __restrict Tik, const double* __restrict Tkk) {
    const size_t rbase = i * nb;
    const size_t cbase = k * nb;
    const size_t rows = std::min(nb, n - rbase);  // valid rows of the tile
    const size_t cols = std::min(nb, n - cbase);
    for (size_t p = p0; p < p1; ++p) {
        const double* __restrict tip = Tik + p * nb * kRB;
        for (size_t c0 = 0; c0 < cols; c0 += kCB) {
            double* __restrict ab = acc + accIdx(c0, p * kRB, nb);  // kCB x kRB block
            // Contribution of all columns kk < c0.
            gemmBlock(ab, tip, Tkk + (c0 >> 3) * nb * kRB + (c0 & 7), c0);
            for (size_t cc = 0; cc < kCB && c0 + cc < cols; ++cc) {
                const size_t c = c0 + cc;
                const size_t j = cbase + c;
                const double d = Tkk[trIdx(c, c, nb)];  // L[j][j]
                double* __restrict tcol = Tik + trIdx(c, p * kRB, nb);
                for (size_t rr = 0; rr < kRB; ++rr) {
                    const size_t r = p * kRB + rr;
                    if (r < rows) {
                        const double v = (A[(rbase + r) * n + j] - ab[cc * kRB + rr]) / d;
                        A[(rbase + r) * n + j] = v;
                        tcol[rr] = v;
                    } else {
                        tcol[rr] = 0.0;
                    }
                }
                // In-block rank-1 update of the following columns (term kk = j).
                for (size_t cc2 = cc + 1; cc2 < kCB; ++cc2) {
                    const double b = Tkk[trIdx(c, c0 + cc2, nb)];  // L[cbase + c0 + cc2][j]
                    double* __restrict ab2 = ab + cc2 * kRB;
#pragma omp simd
                    for (size_t rr = 0; rr < kRB; ++rr) {
                        ab2[rr] = std::fma(tcol[rr], b, ab2[rr]);
                    }
                }
            }
        }
    }
}

// Row panels (of kRB rows) per TRSM work item.
constexpr size_t kTrsmPanels = 2;

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Cache-line padded counter so that spinning readers do not disturb each other.
struct alignas(64) Stamp {
    std::atomic<size_t> v{0};
};

// Spin until `stamp` reaches `target` (or the computation has failed).
// After a while the CPU is yielded so that a preempted producer can finish.
inline void waitFor(const Stamp& stamp, const size_t target, const std::atomic<bool>& failed) {
    size_t spins = 0;
    while (stamp.v.load(std::memory_order_acquire) < target) {
        if (failed.load(std::memory_order_relaxed)) return;
        if (++spins < 4096) {
            cpuRelax();
        } else {
            std::this_thread::yield();
        }
    }
}

// Number of threads to use: all of them, unless the problem is so small that
// per-item overhead would dominate (about 2M fused multiply-adds per thread).
int chooseThreads(size_t n) {
    const int maxThreads = omp_get_max_threads();
    const double work = static_cast<double>(n) * n * n / 3.0;
    const double perThread = 2.0e6;
    int t = static_cast<int>(work / perThread);
    if (t < 1) t = 1;
    if (t > maxThreads) t = maxThreads;
    return t;
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const size_t nb = chooseTileSize(n);
    const size_t nt = (n + nb - 1) / nb;          // number of tile rows/cols
    const size_t ntiles = nt * (nt + 1) / 2;      // lower-triangular tiles
    const size_t tsz = nb * nb;
    const size_t nchunks = nb / (kRB * kTrsmPanels);  // TRSM work items per tile

    // Accumulators (transposed tiles) and transposed copies of the factored tiles.
    AlignedBuf accBuf = allocAligned(ntiles * tsz);
    AlignedBuf trBuf = allocAligned(ntiles * tsz);
    double* __restrict acc = accBuf.get();
    double* __restrict tr = trBuf.get();
    double* __restrict a = A.data();

    // Dependency tracking (all stamps are monotonic, so no resets are needed):
    //   potrfDone            = number of factored diagonal tiles
    //   trsmCount[i]         = TRSM row chunks completed in tile row i (nchunks per step)
    //   gemmStamp[tile(i,j)] = 1 + last step k whose update was applied to tile (i, j)
    std::atomic<bool> failed(false);
    Stamp potrfDone;
    std::unique_ptr<Stamp[]> trsmCount(new Stamp[nt]);
    std::unique_ptr<Stamp[]> gemmStamp(new Stamp[ntiles]);

    // Global work list. Step k consists of (in this order):
    //   [k == 0]  potrf(0)
    //   TRSM row chunks of tiles (i, k), i > k
    //   GEMM updates of tiles (i, j), k < j <= i, ordered by j (column k+1 first)
    //   potrf(k+1) (placed right after the column k+1 updates)
    // Every item only depends on items that precede it in the list, and items
    // are handed out in list order, so spinning on the stamps cannot deadlock.
    std::vector<size_t> stepBegin(nt + 1);
    stepBegin[0] = 1;  // potrf(0)
    for (size_t k = 0; k < nt; ++k) {
        const size_t m = nt - k - 1;
        stepBegin[k + 1] = stepBegin[k] + m * nchunks + m * (m + 1) / 2 + (m > 0 ? 1 : 0);
    }
    const size_t nwork = stepBegin[nt];

    auto potrf = [&](size_t k) {
        if (!failed.load(std::memory_order_relaxed)) {
            if (!potrfTile(a, n, nb, k, acc + tileIndex(k, k) * tsz, tr + tileIndex(k, k) * tsz)) {
                failed.store(true, std::memory_order_relaxed);
            }
        }
        potrfDone.v.store(k + 1, std::memory_order_release);
    };

    const int nthreads = chooseThreads(n);

#pragma omp parallel num_threads(nthreads)
    {
        // Zero the accumulators (parallel first touch).
#pragma omp for schedule(static)
        for (size_t t = 0; t < ntiles; ++t) {
            std::memset(acc + t * tsz, 0, tsz * sizeof(double));
        }

        // Zero out upper triangular part (never read by the factorization).
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                a[i * n + j] = 0.0;
            }
        }

#pragma omp for schedule(dynamic, 1) nowait
        for (size_t w = 0; w < nwork; ++w) {
            if (failed.load(std::memory_order_relaxed)) continue;
            if (w == 0) {
                potrf(0);
                continue;
            }
            // Locate the step this item belongs to.
            const size_t k = static_cast<size_t>(
                std::upper_bound(stepBegin.begin(), stepBegin.end(), w) - stepBegin.begin()) - 1;
            const size_t m = nt - k - 1;
            size_t idx = w - stepBegin[k];

            if (idx < m * nchunks) {
                // TRSM row chunk of tile (i, k).
                const size_t i = k + 1 + idx / nchunks;
                const size_t p0 = (idx % nchunks) * kTrsmPanels;
                waitFor(potrfDone, k + 1, failed);
                if (k > 0) waitFor(gemmStamp[tileIndex(i, k)], k, failed);
                if (failed.load(std::memory_order_relaxed)) continue;
                trsmRows(a, n, nb, i, k, p0, p0 + kTrsmPanels, acc + tileIndex(i, k) * tsz,
                         tr + tileIndex(i, k) * tsz, tr + tileIndex(k, k) * tsz);
                trsmCount[i].v.fetch_add(1, std::memory_order_release);
                continue;
            }
            idx -= m * nchunks;

            // GEMM items: column j = k+1 first (m items), then potrf(k+1), then the rest.
            if (idx == m) {
                waitFor(gemmStamp[tileIndex(k + 1, k + 1)], k + 1, failed);
                potrf(k + 1);
                continue;
            }
            if (idx > m) --idx;
            // Decode idx into (jp, ip) with j = k+1+jp outer ascending, i = j..nt-1.
            size_t jp = 0;
            {
                // Number of items in columns < jp is jp*m - jp*(jp-1)/2.
                // Solve via the complementary triangular index.
                const size_t rem = m * (m + 1) / 2 - 1 - idx;  // reversed index
                size_t q = static_cast<size_t>((std::sqrt(8.0 * static_cast<double>(rem) + 1.0) - 1.0) / 2.0);
                while (q * (q + 1) / 2 > rem) --q;
                while ((q + 1) * (q + 2) / 2 <= rem) ++q;
                jp = m - 1 - q;
            }
            const size_t colStart = jp * m - jp * (jp - 1) / 2;
            const size_t j = k + 1 + jp;
            const size_t i = j + (idx - colStart);
            const size_t tij = tileIndex(i, j);
            const size_t kb = std::min(nb, n - k * nb);
            const size_t target = (k + 1) * nchunks;
            waitFor(trsmCount[i], target, failed);
            if (i != j) waitFor(trsmCount[j], target, failed);
            if (k > 0) waitFor(gemmStamp[tij], k, failed);
            if (failed.load(std::memory_order_relaxed)) continue;
            gemmTile(acc + tij * tsz, tr + tileIndex(i, k) * tsz, tr + tileIndex(j, k) * tsz, kb, nb);
            gemmStamp[tij].v.store(k + 1, std::memory_order_release);
        }
    }

    return !failed.load();
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (sequential: the RNG stream must be preserved)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (each element keeps its sequential summation order)
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;
    
    // Compute L * L^T and compare with original
#pragma omp parallel for schedule(dynamic, 8) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            
            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
