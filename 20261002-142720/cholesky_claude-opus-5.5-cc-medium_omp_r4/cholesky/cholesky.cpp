#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (OpenMP, shared memory)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Recursive blocked algorithm: the diagonal block is split in two halves,
//   L11 = chol(A11), L21 = A21 * L11^-T, A22 -= L21 * L21^T, L22 = chol(A22),
// so that most of the work are large matrix multiplications (high arithmetic intensity).
// All threads of one parallel region execute the recursion together; the triangular
// solves and multiplications are distributed with OpenMP worksharing loops, small
// diagonal blocks are factored by a single thread with cache-blocked serial kernels.

namespace {

typedef double v4d __attribute__((vector_size(32)));

constexpr size_t MR = 6;          // micro-kernel rows
constexpr size_t NR = 8;          // micro-kernel columns (2 x 4 doubles)
constexpr size_t MAX_NB = 256;    // maximum block size for the single-threaded kernels

inline v4d loadu(const double* p) {
    v4d v;
    memcpy(&v, p, sizeof(v));
    return v;
}

// Pack rows [0, nc) x cols [0, kb) of B (row stride ldb) transposed into micro-panels of NR
// columns: Bp[panel][p][j] = B[c0 + j][p]
void packBT(const double* B, size_t ldb, size_t nc, size_t kb, double* Bp) {
    for (size_t c0 = 0; c0 < nc; c0 += NR) {
        const size_t nr = std::min(NR, nc - c0);
        double* dst = Bp + c0 * kb;
        if (nr == NR) {
            const double* src = B + c0 * ldb;
            for (size_t p = 0; p < kb; ++p)
                for (size_t j = 0; j < NR; ++j) dst[p * NR + j] = src[j * ldb + p];
        } else {
            for (size_t p = 0; p < kb; ++p)
                for (size_t j = 0; j < NR; ++j) dst[p * NR + j] = j < nr ? B[(c0 + j) * ldb + p] : 0.0;
        }
    }
}

constexpr size_t KC = 256;        // k-blocking
constexpr size_t MC = 96;         // row blocking: packed A block (MC x KC) stays in L2

inline size_t packedSizeB(size_t n, size_t kb) { return (n + NR - 1) / NR * NR * kb; }

// C[0:mr, 0:nr] -= sum_p ap[p][0:mr] * bp[p][0:nr]   (MR x NR register-blocked micro-kernel)
inline void microKernel(double* C, size_t ld, const double* ap, const double* bp, size_t kc,
                        size_t mr, size_t nr) {
    v4d c00 = {0, 0, 0, 0}, c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
    v4d c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    for (size_t p = 0; p < kc; ++p) {
        const v4d b0 = loadu(bp + p * NR);
        const v4d b1 = loadu(bp + p * NR + 4);
        const double* a = ap + p * MR;
        v4d x = {a[0], a[0], a[0], a[0]};
        c00 += x * b0; c01 += x * b1;
        x = (v4d){a[1], a[1], a[1], a[1]};
        c10 += x * b0; c11 += x * b1;
        x = (v4d){a[2], a[2], a[2], a[2]};
        c20 += x * b0; c21 += x * b1;
        x = (v4d){a[3], a[3], a[3], a[3]};
        c30 += x * b0; c31 += x * b1;
        x = (v4d){a[4], a[4], a[4], a[4]};
        c40 += x * b0; c41 += x * b1;
        x = (v4d){a[5], a[5], a[5], a[5]};
        c50 += x * b0; c51 += x * b1;
    }
    if (mr == MR && nr == NR) {
        const v4d acc[MR][2] = {{c00, c01}, {c10, c11}, {c20, c21},
                                {c30, c31}, {c40, c41}, {c50, c51}};
        for (size_t i = 0; i < MR; ++i) {
            double* crow = C + i * ld;
            v4d x0 = loadu(crow) - acc[i][0];
            v4d x1 = loadu(crow + 4) - acc[i][1];
            memcpy(crow, &x0, 32);
            memcpy(crow + 4, &x1, 32);
        }
        return;
    }
    alignas(32) double acc[MR][NR];
    memcpy(&acc[0][0], &c00, 32); memcpy(&acc[0][4], &c01, 32);
    memcpy(&acc[1][0], &c10, 32); memcpy(&acc[1][4], &c11, 32);
    memcpy(&acc[2][0], &c20, 32); memcpy(&acc[2][4], &c21, 32);
    memcpy(&acc[3][0], &c30, 32); memcpy(&acc[3][4], &c31, 32);
    memcpy(&acc[4][0], &c40, 32); memcpy(&acc[4][4], &c41, 32);
    memcpy(&acc[5][0], &c50, 32); memcpy(&acc[5][4], &c51, 32);
    for (size_t i = 0; i < mr; ++i) {
        double* crow = C + i * ld;
        for (size_t j = 0; j < nr; ++j) crow[j] -= acc[i][j];
    }
}

// Pack the MC x kc block A[0:m, 0:kc] (row stride lda) into micro-panels of MR rows:
// Ap[panel][p][i]; rows beyond m are zero-filled.
void packA(const double* A, size_t lda, size_t m, size_t kc, double* Ap) {
    for (size_t r0 = 0; r0 < m; r0 += MR) {
        const size_t mr = std::min(MR, m - r0);
        double* dst = Ap + r0 * kc;
        for (size_t p = 0; p < kc; ++p) {
            size_t i = 0;
            for (; i < mr; ++i) dst[p * MR + i] = A[(r0 + i) * lda + p];
            for (; i < MR; ++i) dst[p * MR + i] = 0.0;
        }
    }
}

// C[0:m, 0:nc] -= A * B^T with A (m x kb, row-major, stride ld) and B (nc x kb) in packed form.
// If lowerOnly, only micro-tiles touching the lower triangle (diagonal tile) are computed.
void gemmPacked(double* C, const double* A, size_t ld, const double* Bp, size_t m, size_t nc,
                size_t kb, bool lowerOnly) {
    alignas(64) static thread_local double Ablk[MC * KC];
    for (size_t p0 = 0; p0 < kb; p0 += KC) {
        const size_t kc = std::min(KC, kb - p0);
        for (size_t m0 = 0; m0 < m; m0 += MC) {          // packed A block (MC x kc) stays in L2
            const size_t m1 = std::min(m, m0 + MC);
            packA(A + m0 * ld + p0, ld, m1 - m0, kc, Ablk);
            for (size_t c0 = 0; c0 < nc; c0 += NR) {      // B micro-panel (kc x NR) in L1
                if (lowerOnly && c0 >= m1) break;
                const size_t nr = std::min(NR, nc - c0);
                const double* bp = Bp + c0 * kb + p0 * NR;
                const size_t rStart = lowerOnly ? std::max(m0, c0 / MR * MR) : m0;
                for (size_t r0 = rStart; r0 < m1; r0 += MR) {
                    const size_t mr = std::min(MR, m1 - r0);
                    microKernel(C + r0 * ld + c0, ld, Ablk + (r0 - m0) * kc, bp, kc, mr, nr);
                }
            }
        }
    }
}

// C[0:m, 0:nc] -= A[0:m, 0:kb] * B[0:nc, 0:kb]^T   (all row-major with stride ld)
void gemmNT(double* C, const double* A, const double* B, size_t ld, size_t m, size_t nc,
            size_t kb, bool lowerOnly) {
    alignas(64) static thread_local double Bp[MAX_NB * MAX_NB + NR * MAX_NB];
    packBT(B, ld, nc, kb, Bp);
    gemmPacked(C, A, ld, Bp, m, nc, kb, lowerOnly);
}

constexpr size_t LEAF = 16;       // switch to unblocked kernels below this size

// Unblocked Cholesky of a diagonal block (bs x bs at D, stride ld), lower part only.
// Returns the local index of the failing diagonal element, or -1 on success.
long potrfLeaf(double* D, size_t ld, size_t bs) {
    for (size_t i = 0; i < bs; ++i) {
        double* Ri = D + i * ld;
        for (size_t j = 0; j < i; ++j) {
            const double* Rj = D + j * ld;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += Ri[k] * Rj[k];
            Ri[j] = (Ri[j] - sum) / Rj[j];
        }
        double sum = 0.0;
        for (size_t k = 0; k < i; ++k) sum += Ri[k] * Ri[k];
        const double val = Ri[i] - sum;
        if (val <= 0.0) return (long)i;
        Ri[i] = std::sqrt(val);
    }
    return -1;
}

// Unblocked triangular solve X * L^T = B (L: bs x bs lower, B: m x bs, bs <= LEAF),
// X overwrites B. Works on a transposed copy so the inner loops vectorize over rows.
void trsmLeaf(double* B, const double* L, size_t ld, size_t m, size_t bs) {
    alignas(64) static thread_local double Xt[LEAF * MAX_NB];
    for (size_t r0 = 0; r0 < m; r0 += MAX_NB) {
        const size_t mb = std::min(MAX_NB, m - r0);
        double* Bb = B + r0 * ld;
        for (size_t r = 0; r < mb; ++r)
            for (size_t c = 0; c < bs; ++c) Xt[c * MAX_NB + r] = Bb[r * ld + c];
        for (size_t c = 0; c < bs; ++c) {
            double* xc = Xt + c * MAX_NB;
            const double d = L[c * ld + c];
            #pragma omp simd
            for (size_t r = 0; r < mb; ++r) xc[r] /= d;
            for (size_t q = c + 1; q < bs; ++q) {
                double* xq = Xt + q * MAX_NB;
                const double l = L[q * ld + c];
                #pragma omp simd
                for (size_t r = 0; r < mb; ++r) xq[r] -= l * xc[r];
            }
        }
        for (size_t r = 0; r < mb; ++r)
            for (size_t c = 0; c < bs; ++c) Bb[r * ld + c] = Xt[c * MAX_NB + r];
    }
}

// Recursive triangular solve X * L^T = B; bulk of the work is done by gemmNT.
void trsmTile(double* B, const double* L, size_t ld, size_t m, size_t bs) {
    if (bs <= LEAF) {
        trsmLeaf(B, L, ld, m, bs);
        return;
    }
    const size_t h = (bs / 2 + 3) / 4 * 4;
    trsmTile(B, L, ld, m, h);                                     // X1 = B1 * L11^-T
    gemmNT(B + h, B, L + h * ld, ld, m, bs - h, h, false);       // B2 -= X1 * L21^T
    trsmTile(B + h, L + h * ld + h, ld, m, bs - h);               // X2 = B2 * L22^-T
}

// Recursive Cholesky of a diagonal tile, lower part only.
// Returns the local index of the failing diagonal element, or -1 on success.
long potrfTile(double* D, size_t ld, size_t bs) {
    if (bs <= LEAF) return potrfLeaf(D, ld, bs);
    const size_t h = (bs / 2 + 3) / 4 * 4;
    long f = potrfTile(D, ld, h);                                 // L11
    if (f >= 0) return f;
    double* D21 = D + h * ld;
    double* D22 = D21 + h;
    trsmTile(D21, D, ld, bs - h, h);                              // L21 = A21 * L11^-T
    gemmNT(D22, D21, D21, ld, bs - h, bs - h, h, true);           // A22 -= L21 * L21^T
    f = potrfTile(D22, ld, bs - h);                               // L22
    return f >= 0 ? f + (long)h : f;
}

constexpr size_t ALIGN = 24;      // block granularity: multiple of MR and NR
constexpr size_t LEAF_N = 192;    // blocks up to this size are handled by a single thread
static_assert(LEAF_N <= MAX_NB, "single-threaded kernels use buffers of MAX_NB x MAX_NB");

inline size_t splitPoint(size_t s) { return std::max(ALIGN, (s / 2 + ALIGN / 2) / ALIGN * ALIGN); }

// Number of hardware threads per physical core (1 if unknown)
size_t threadsPerCore() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!f) return 1;
    char buf[256] = {0};
    const size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    size_t count = 0;
    for (char* p = buf; p < buf + len && *p && *p != '\n';) {
        char* end;
        const long lo = strtol(p, &end, 10);
        if (end == p) break;
        long hi = lo;
        if (*end == '-') hi = strtol(end + 1, &end, 10);
        count += (size_t)(hi - lo + 1);
        p = (*end == ',') ? end + 1 : end;
        if (*end != ',') break;
    }
    return std::max<size_t>(count, 1);
}

// Number of threads used by all parallel regions. Kept identical across regions so that
// the OpenMP runtime never has to shrink (destroy) and regrow its thread pool.
// Small problems do not have enough parallel work for large thread counts, and the
// compute-bound kernels do not profit from running two threads per physical core.
int numThreads(size_t n) {
    static const size_t maxThreads = [] {
        size_t t = (size_t)omp_get_max_threads();
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online > 0) {
            const size_t cores = std::max<size_t>((size_t)online / threadsPerCore(), 1);
            t = std::min(t, cores);
        }
        return t;
    }();
    return (int)std::clamp<size_t>(n / 32, 1, maxThreads);
}

// State shared by all threads of the team while executing the recursive factorization.
// All functions below are called by every thread of the team (orphaned worksharing).
struct Shared {
    double* packB;       // packed right operand of the current parallel GEMM
    long failIndex;      // first non positive definite diagonal element, or -1
    double* base;        // matrix storage (to compute global diagonal indices)
    size_t n;
};

// C[0:m, 0:nc] -= A[0:m, 0:K] * B[0:nc, 0:K]^T, executed cooperatively by the team.
// If lower, C is a diagonal block and only its lower triangle is needed.
void parGemm(Shared& sh, double* C, const double* A, const double* B, size_t ld, size_t m,
             size_t nc, size_t K, bool lower) {
    const size_t nth = (size_t)omp_get_num_threads();

    // Pack the right operand once (shared, read-only during the multiplication)
    #pragma omp for schedule(dynamic, 1)
    for (size_t r = 0; r < nc; r += ALIGN)
        packBT(B + r * ld, ld, std::min(ALIGN, nc - r), K, sh.packB + r * K);

    // Choose the output tile size minimizing the estimated time: number of rounds of
    // tiles over the threads times the cost of one tile (small tiles are less efficient)
    size_t tb = 144;
    double best = 0.0;
    for (size_t cand = 144; cand >= ALIGN; cand -= ALIGN) {
        const size_t tm = (m + cand - 1) / cand, tn = (nc + cand - 1) / cand;
        const size_t tiles = lower ? tm * (tm + 1) / 2 : tm * tn;
        const size_t rounds = (tiles + nth - 1) / nth;
        const double cost = (double)rounds * (double)(cand * cand) * (1.0 + 48.0 / (double)cand);
        if (cand == 144 || cost < best) {
            best = cost;
            tb = cand;
        }
    }
    const size_t tm = (m + tb - 1) / tb, tn = (nc + tb - 1) / tb;
    const size_t total = lower ? tm * (tm + 1) / 2 : tm * tn;

    #pragma omp for schedule(dynamic, 1)
    for (size_t t = 0; t < total; ++t) {
        size_t i, j;
        if (lower) {  // map linear index to (i, j) with j <= i
            i = (size_t)((std::sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
            while (i * (i + 1) / 2 > t) --i;
            while ((i + 1) * (i + 2) / 2 <= t) ++i;
            j = t - i * (i + 1) / 2;
        } else {
            i = t / tn;
            j = t % tn;
        }
        const size_t r = i * tb, c = j * tb;
        gemmPacked(C + r * ld + c, A + r * ld, ld, sh.packB + c * K,
                   std::min(tb, m - r), std::min(tb, nc - c), K, lower && i == j);
    }
}

// Solve X * L^T = B in place (B: m x c, L: c x c lower), executed cooperatively by the team.
void parTrsm(Shared& sh, double* B, const double* L, size_t ld, size_t m, size_t c) {
    if (c <= LEAF_N) {
        #pragma omp for schedule(dynamic, 1)
        for (size_t r = 0; r < m; r += ALIGN)
            trsmTile(B + r * ld, L, ld, std::min(ALIGN, m - r), c);
        return;
    }
    const size_t h = splitPoint(c);
    parTrsm(sh, B, L, ld, m, h);                                   // X1 = B1 * L11^-T
    parGemm(sh, B + h, B, L + h * ld, ld, m, c - h, h, false);     // B2 -= X1 * L21^T
    parTrsm(sh, B + h, L + h * ld + h, ld, m, c - h);              // X2 = B2 * L22^-T
}

// Recursive Cholesky of the s x s diagonal block D, executed cooperatively by the team.
// Returns false (after recording the failing index) if the matrix is not positive definite.
bool parPotrf(Shared& sh, double* D, size_t s) {
    if (s <= LEAF_N) {
        #pragma omp single
        {
            const long f = potrfTile(D, sh.n, s);
            if (f >= 0) sh.failIndex = (long)((size_t)(D - sh.base) / (sh.n + 1)) + f;
        }
        return sh.failIndex < 0;  // read after the implicit barrier: same on all threads
    }
    const size_t ld = sh.n, h = splitPoint(s);
    double* D21 = D + h * ld;
    double* D22 = D21 + h;
    if (!parPotrf(sh, D, h)) return false;                         // L11
    parTrsm(sh, D21, D, ld, s - h, h);                             // L21 = A21 * L11^-T
    parGemm(sh, D22, D21, D21, ld, s - h, s - h, h, true);         // A22 -= L21 * L21^T
    return parPotrf(sh, D22, s - h);                               // L22
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const size_t half = splitPoint(n);
    const size_t packSize = packedSizeB(n, half) + 64;  // bound for any nc x K operand used
    std::unique_ptr<double[]> packB(new double[packSize]);
    Shared sh{packB.get(), -1, A.data(), n};
    double* a = A.data();

    #pragma omp parallel num_threads(numThreads(n))
    {
        if (parPotrf(sh, a, n)) {
            // Zero out upper triangular part
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
            }
        }
    }

    if (sh.failIndex >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)sh.failIndex);
        return false;
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (symmetric: compute lower part, mirror to upper)
    #pragma omp parallel for schedule(dynamic, 4) num_threads(numThreads(n))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    #pragma omp parallel for schedule(dynamic, 16) num_threads(numThreads(n))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = A[j * n + i];
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
    
    #pragma omp parallel for schedule(dynamic, 4) num_threads(numThreads(n)) reduction(max : maxError, relError)
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
