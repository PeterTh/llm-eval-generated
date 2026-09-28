#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

// Tiled Cholesky decomposition, parallelized with OpenMP for shared memory multicore CPUs.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The matrix is factorized tile by tile with a right-looking blocked algorithm: the trailing
// submatrix update, which holds nearly all of the work, is spread over the threads one tile at a
// time, and the sequential factorization of the next diagonal tile is overlapped with it.

#if defined(__AVX2__) && defined(__FMA__)
#define CHOLESKY_AVX2 1
#include <immintrin.h>
#endif

namespace {

// Register blocking factors of the GEMM micro kernel: MR x NR accumulators (NR / 4 vectors per
// row) plus the two operand registers have to fit into the 16 architectural AVX registers.
#ifdef CHOLESKY_AVX2
constexpr size_t MR = 6;
#else
constexpr size_t MR = 4;
#endif
constexpr size_t NR = 8;

// Rows processed simultaneously by the triangular solve.
constexpr size_t TR = 4;

// Scratch buffer used to pack tiles (one per thread, reused across the whole factorization).
double* tileScratch(const size_t elems) {
    static thread_local std::vector<double> buf;
    if (buf.size() < elems) {
        buf.resize(elems);
    }
    return buf.data();
}

// Generic (unblocked) variant of the update below, used for the ragged tile borders.
void gemmGeneric(double* __restrict C, const size_t ldc, const double* __restrict A, const size_t lda,
                 const double* __restrict Bt, const size_t ldbt, const size_t mb, const size_t nb, const size_t kb) {
    for (size_t i = 0; i < mb; ++i) {
        double* __restrict c = C + i * ldc;
        for (size_t k = 0; k < kb; ++k) {
            const double a = A[i * lda + k];
            const double* __restrict b = Bt + k * ldbt;
            for (size_t j = 0; j < nb; ++j) {
                c[j] -= a * b[j];
            }
        }
    }
}

// Micro kernel: updates MR rows and nb (multiple of NR) columns of C, keeping the whole
// MR x NR accumulator block in vector registers over the entire k loop.
// Ap holds the MR rows of A packed as Ap[k * MR + r].
void microKernel(double* __restrict C, const size_t ldc, const double* __restrict Ap,
                 const double* __restrict Bt, const size_t ldbt, const size_t nb, const size_t kb) {
#ifdef CHOLESKY_AVX2
    for (size_t j = 0; j < nb; j += NR) {
        __m256d acc[MR][NR / 4];
        for (size_t r = 0; r < MR; ++r) {
            acc[r][0] = _mm256_loadu_pd(C + r * ldc + j);
            acc[r][1] = _mm256_loadu_pd(C + r * ldc + j + 4);
        }
        const double* __restrict b = Bt + j;
        const double* __restrict a = Ap;
        for (size_t k = 0; k < kb; ++k) {
            const __m256d b0 = _mm256_loadu_pd(b);
            const __m256d b1 = _mm256_loadu_pd(b + 4);
            for (size_t r = 0; r < MR; ++r) {
                const __m256d av = _mm256_broadcast_sd(a + r);
                acc[r][0] = _mm256_fnmadd_pd(av, b0, acc[r][0]);
                acc[r][1] = _mm256_fnmadd_pd(av, b1, acc[r][1]);
            }
            b += ldbt;
            a += MR;
        }
        for (size_t r = 0; r < MR; ++r) {
            _mm256_storeu_pd(C + r * ldc + j, acc[r][0]);
            _mm256_storeu_pd(C + r * ldc + j + 4, acc[r][1]);
        }
    }
#else
    for (size_t j = 0; j < nb; j += NR) {
        double acc[MR][NR];
        for (size_t r = 0; r < MR; ++r) {
            for (size_t c = 0; c < NR; ++c) {
                acc[r][c] = C[r * ldc + j + c];
            }
        }
        for (size_t k = 0; k < kb; ++k) {
            const double* __restrict b = Bt + k * ldbt + j;
            for (size_t r = 0; r < MR; ++r) {
                const double a = Ap[k * MR + r];
                for (size_t c = 0; c < NR; ++c) {
                    acc[r][c] -= a * b[c];
                }
            }
        }
        for (size_t r = 0; r < MR; ++r) {
            for (size_t c = 0; c < NR; ++c) {
                C[r * ldc + j + c] = acc[r][c];
            }
        }
    }
#endif
}

// C[mb x nb] -= A[mb x kb] * B[nb x kb]^T, with B given transposed (Bt[kb x nb], row-major).
// Ap is scratch space for MR * kb doubles.
void gemmUpdate(double* __restrict C, const size_t ldc, const double* __restrict A, const size_t lda,
                const double* __restrict Bt, const size_t ldbt, double* __restrict Ap,
                const size_t mb, const size_t nb, const size_t kb) {
    const size_t mFull = mb - mb % MR;
    const size_t nFull = nb - nb % NR;

    for (size_t i = 0; i < mFull; i += MR) {
        // Pack MR rows of A so that the micro kernel reads them contiguously.
        for (size_t r = 0; r < MR; ++r) {
            const double* __restrict src = A + (i + r) * lda;
            for (size_t k = 0; k < kb; ++k) {
                Ap[k * MR + r] = src[k];
            }
        }
        microKernel(C + i * ldc, ldc, Ap, Bt, ldbt, nFull, kb);
        // Remaining columns of this row block
        if (nFull < nb) {
            gemmGeneric(C + i * ldc + nFull, ldc, A + i * lda, lda, Bt + nFull, ldbt, MR, nb - nFull, kb);
        }
    }
    // Remaining rows
    if (mFull < mb) {
        gemmGeneric(C + mFull * ldc, ldc, A + mFull * lda, lda, Bt, ldbt, mb - mFull, nb, kb);
    }
}

// Bt[kb x nb] <- transpose of B[nb x kb]
void packTranspose(double* __restrict Bt, const size_t ldbt, const double* __restrict B, const size_t ldb,
                   const size_t nb, const size_t kb) {
    constexpr size_t TB = 16;
    for (size_t jj = 0; jj < nb; jj += TB) {
        const size_t jEnd = std::min(jj + TB, nb);
        for (size_t kk = 0; kk < kb; kk += TB) {
            const size_t kEnd = std::min(kk + TB, kb);
            for (size_t j = jj; j < jEnd; ++j) {
                const double* __restrict src = B + j * ldb;
                for (size_t k = kk; k < kEnd; ++k) {
                    Bt[k * ldbt + j] = src[k];
                }
            }
        }
    }
}

// Unblocked Cholesky factorization of the diagonal tile D[mb x mb] (lower triangle).
// Right-looking formulation: the trailing part of the tile is updated by a rank-1 update after
// every column, which keeps the inner loop contiguous and vectorizable.
bool potrfDiag(double* __restrict D, const size_t ld, const size_t mb, const size_t base) {
    double* const col = tileScratch(mb);
    for (size_t j = 0; j < mb; ++j) {
        const double val = D[j * ld + j];
        if (val <= 0.0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", base + j);
            return false;
        }
        const double d = sqrt(val);
        D[j * ld + j] = d;
        for (size_t i = j + 1; i < mb; ++i) {
            col[i] = D[i * ld + j] / d;
            D[i * ld + j] = col[i];
        }
        // Trailing lower triangle: D(i,t) -= col[i] * col[t] for j < t <= i
        for (size_t i = j + 1; i < mb; ++i) {
            const double f = col[i];
            double* __restrict row = D + i * ld;
            for (size_t t = j + 1; t <= i; ++t) {
                row[t] -= f * col[t];
            }
        }
    }
    return true;
}

// Triangular solve X * D^T = X for a tile X[mb x kb] against the factorized diagonal tile D[kb x kb].
void trsmTile(double* __restrict X, const size_t ldx, const double* __restrict D, const size_t ldd,
              const size_t mb, const size_t kb) {
    // Column-major copy of the diagonal tile so that the inner update runs over contiguous memory.
    double* __restrict Dt = tileScratch(kb * kb);
    packTranspose(Dt, kb, D, ldd, kb, kb);

    size_t i = 0;
    for (; i + TR <= mb; i += TR) {
        double* __restrict x0 = X + (i + 0) * ldx;
        double* __restrict x1 = X + (i + 1) * ldx;
        double* __restrict x2 = X + (i + 2) * ldx;
        double* __restrict x3 = X + (i + 3) * ldx;
        for (size_t j = 0; j < kb; ++j) {
            const double* __restrict col = Dt + j * kb;
            const double v0 = x0[j] / col[j];
            const double v1 = x1[j] / col[j];
            const double v2 = x2[j] / col[j];
            const double v3 = x3[j] / col[j];
            x0[j] = v0;
            x1[j] = v1;
            x2[j] = v2;
            x3[j] = v3;
            for (size_t t = j + 1; t < kb; ++t) {
                const double l = col[t];
                x0[t] -= v0 * l;
                x1[t] -= v1 * l;
                x2[t] -= v2 * l;
                x3[t] -= v3 * l;
            }
        }
    }
    for (; i < mb; ++i) {
        double* __restrict x = X + i * ldx;
        for (size_t j = 0; j < kb; ++j) {
            const double* __restrict col = Dt + j * kb;
            const double v = x[j] / col[j];
            x[j] = v;
            for (size_t t = j + 1; t < kb; ++t) {
                x[t] -= v * col[t];
            }
        }
    }
}

// Tile size: large enough to keep the micro kernel efficient and the tiles cache resident, small
// enough to expose enough independent tiles for the available threads. A multiple of MR * 4 and
// of NR keeps the kernel edges out of the hot path.
size_t chooseTileSize(const size_t n) {
    size_t bs = (n / 34 / 24) * 24;
    if (bs > 216) bs = 216;
    if (bs < 72) bs = 72;
    return bs;
}

// Number of threads that the factorization of a matrix with nt x nt tiles can keep busy: during
// step k there are (nt - k)^2 / 2 independent tiles, which averages to nt^2 / 6 over the run, and
// giving every thread a couple of tiles per phase amortizes the synchronization between the
// phases. Using more threads than that only adds overhead.
int usefulThreads(const size_t nt) {
    const size_t useful = std::max<size_t>(8, nt * nt / 12);
    const int maxThreads = omp_get_max_threads();
    return useful < (size_t)maxThreads ? (int)useful : maxThreads;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) {
        return true;
    }

    double* const M = A.data();
    const size_t bs = chooseTileSize(n);
    const size_t nt = (n + bs - 1) / bs;

    bool failed = false;

    // Applies A(i,j) -= L(i,k) * L(j,k)^T to the rows [r0, r0 + rows) of one tile.
    const auto updateTile = [&](const size_t i, const size_t j, const size_t k,
                                const size_t r0, const size_t rows) {
        const size_t i0 = i * bs + r0;
        const size_t j0 = j * bs;
        const size_t k0 = k * bs;
        const size_t nb = std::min(bs, n - j0);
        const size_t kb = std::min(bs, n - k0);

        double* const scratch = tileScratch(kb * (nb + MR));
        double* const Bt = scratch;
        double* const Ap = scratch + kb * nb;
        packTranspose(Bt, nb, M + j0 * n + k0, n, nb, kb);
        gemmUpdate(M + i0 * n + j0, n, M + i0 * n + k0, n, Bt, nb, Ap, rows, nb, kb);
    };

    // The tiles of the panel are split into this many row chunks so that the phases on the
    // critical path keep more threads busy than there are tiles left in the panel.
    const size_t chunks = 4;
    // Rows per chunk, a multiple of the micro kernel height.
    const size_t chunkRows = ((bs + chunks - 1) / chunks + MR - 1) / MR * MR;

    // Right-looking blocked factorization with one step of lookahead: the tiles of the next panel
    // are updated first so that its (sequential) diagonal factorization can run concurrently with
    // the bulk of the trailing update.
    #pragma omp parallel default(none) shared(M, failed, updateTile) \
        firstprivate(n, bs, nt, chunks, chunkRows) num_threads(usefulThreads(nt))
    {
        #pragma omp single
        {
            if (!potrfDiag(M, n, std::min(bs, n), 0)) {
                failed = true;
            }
        }
        #pragma omp for schedule(dynamic, 1)
        for (size_t u = 0; u < (nt - 1) * chunks; ++u) {
            const size_t i0 = (1 + u / chunks) * bs;
            const size_t r0 = (u % chunks) * chunkRows;
            const size_t mb = std::min(bs, n - i0);
            if (r0 < mb) {
                trsmTile(M + (i0 + r0) * n, n, M, n, std::min(chunkRows, mb - r0), std::min(bs, n));
            }
        }

        for (size_t k = 0; k + 1 < nt && !failed; ++k) {
            const size_t p = k + 1;       // panel factorized during this step
            const size_t p0 = p * bs;
            const size_t pb = std::min(bs, n - p0);
            const size_t rest = nt - p - 1; // trailing tiles per dimension, excluding panel p

            // Update the tiles of panel p first; it is on the critical path.
            #pragma omp for schedule(dynamic, 1)
            for (size_t u = 0; u < (nt - p) * chunks; ++u) {
                const size_t i = p + u / chunks;
                const size_t r0 = (u % chunks) * chunkRows;
                const size_t mb = std::min(bs, n - i * bs);
                if (r0 < mb) {
                    updateTile(i, p, k, r0, std::min(chunkRows, mb - r0));
                }
            }

            // Factorize the diagonal tile of panel p while the remaining tiles are updated.
            #pragma omp single nowait
            {
                if (!potrfDiag(M + p0 * n + p0, n, pb, p0)) {
                    failed = true;
                }
            }

            // Remaining trailing update: A(i,j) -= L(i,k) * L(j,k)^T for p < j <= i.
            const size_t pairs = rest * (rest + 1) / 2;
            #pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < pairs; ++t) {
                // Invert the triangular numbering: column c holds rest - c pairs.
                const double disc = (double)(2 * rest + 1);
                size_t c = (size_t)((disc - sqrt(disc * disc - 8.0 * (double)t)) * 0.5);
                while (c * rest - c * (c - 1) / 2 > t) --c;
                while ((c + 1) * rest - (c + 1) * c / 2 <= t) ++c;

                const size_t j = p + 1 + c;
                const size_t i = j + (t - (c * rest - c * (c - 1) / 2));
                updateTile(i, j, k, 0, std::min(bs, n - i * bs));
            }

            // Panel solves: L(i,p) = A(i,p) * L(p,p)^-T
            #pragma omp for schedule(dynamic, 1)
            for (size_t u = 0; u < rest * chunks; ++u) {
                const size_t i0 = (p + 1 + u / chunks) * bs;
                const size_t r0 = (u % chunks) * chunkRows;
                const size_t mb = std::min(bs, n - i0);
                if (r0 < mb) {
                    trsmTile(M + (i0 + r0) * n + p0, n, M + p0 * n + p0, n,
                             std::min(chunkRows, mb - r0), pb);
                }
            }
        }

        // Zero out upper triangular part
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            double* const row = M + i * n;
            for (size_t j = i + 1; j < n; ++j) {
                row[j] = 0.0;
            }
        }
    }

    return !failed;
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
    
    // Compute A = B * B^T
    // The product is symmetric, so only the lower triangle is computed and mirrored;
    // the summation order of each element is unchanged.
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        const double* const bi = B.data() + i * n;
        size_t j = 0;
        for (; j + 4 <= i + 1; j += 4) {
            const double* const b0 = B.data() + (j + 0) * n;
            const double* const b1 = B.data() + (j + 1) * n;
            const double* const b2 = B.data() + (j + 2) * n;
            const double* const b3 = B.data() + (j + 3) * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double v = bi[k];
                s0 += v * b0[k];
                s1 += v * b1[k];
                s2 += v * b2[k];
                s3 += v * b3[k];
            }
            A[i * n + j + 0] = s0;
            A[i * n + j + 1] = s1;
            A[i * n + j + 2] = s2;
            A[i * n + j + 3] = s3;
        }
        for (; j <= i; ++j) {
            const double* const bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            A[i * n + j] = sum;
        }
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = A[j * n + i];
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T; L is lower triangular, so terms beyond column min(i, j) vanish
    // and the product is symmetric.
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        const double* const li = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* const lj = L.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) {
                sum += li[k] * lj[k];
            }
            reconstructed[i * n + j] = sum;
            reconstructed[j * n + i] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for schedule(static) reduction(max: maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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

// Spread the pages of the (large) matrices over all NUMA nodes instead of filling the node of the
// allocating thread, which would limit the factorization to the memory bandwidth of a single node.
// Equivalent to running under "numactl --interleave=all"; done through the raw syscalls to avoid a
// dependency on libnuma.
void interleaveMemoryNodes() {
#if defined(__linux__) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)
    constexpr int MPOL_INTERLEAVE = 3;
    constexpr int MPOL_F_MEMS_ALLOWED = 1 << 2;
    constexpr unsigned long MAX_NODES = 1024;

    unsigned long nodes[MAX_NODES / (8 * sizeof(unsigned long))] = {0};
    if (syscall(SYS_get_mempolicy, nullptr, nodes, MAX_NODES, nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
        return;
    }
    syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, nodes, MAX_NODES);
#endif
}

// Number of physical cores (counting SMT siblings only once), 0 if it cannot be determined.
int physicalCoreCount() {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        return 0;
    }
    // A core is identified by the pair (physical package id, core id).
    std::vector<long long> cores;
    char line[512];
    long long package = -1;
    long long core = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        long long value;
        if (sscanf(line, "physical id : %lld", &value) == 1) {
            package = value;
        } else if (sscanf(line, "core id : %lld", &value) == 1) {
            core = value;
        }
        if (package >= 0 && core >= 0) {
            cores.push_back(package * 100000 + core);
            package = -1;
            core = -1;
        }
    }
    fclose(f);
    std::sort(cores.begin(), cores.end());
    return (int)(std::unique(cores.begin(), cores.end()) - cores.begin());
}

// This benchmark is compute bound and its kernels already saturate the vector units of a core,
// so running more than one thread per physical core only adds synchronization overhead.
void configureThreads() {
    const int cores = physicalCoreCount();
    const int threads = omp_get_max_threads();
    if (cores > 0 && cores < threads) {
        omp_set_num_threads(cores);
    }
    printf("OpenMP threads: %d\n", omp_get_max_threads());
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
    configureThreads();
    interleaveMemoryNodes();

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
