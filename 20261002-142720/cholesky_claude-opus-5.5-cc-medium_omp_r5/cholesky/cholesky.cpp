#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (blocked right-looking algorithm with OpenMP)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {

// Register-blocking parameters of the GEMM micro-kernel
constexpr size_t MR = 6;
constexpr size_t NR = 8;
typedef double v4d __attribute__((vector_size(32), aligned(8)));

// Unblocked Cholesky of a diagonal tile (lower part), row-major with leading dim ld.
// Returns the local index of the failing diagonal element, or -1 on success.
long potrfTile(double* a, const size_t ld, const size_t m) {
    for (size_t j = 0; j < m; ++j) {
        double* rj = a + j * ld;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < j; ++k) {
            sum += rj[k] * rj[k];
        }
        const double val = rj[j] - sum;
        if (val <= 0.0) {
            return (long)j;
        }
        const double d = sqrt(val);
        rj[j] = d;
        for (size_t i = j + 1; i < m; ++i) {
            double* ri = a + i * ld;
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (size_t k = 0; k < j; ++k) {
                s += ri[k] * rj[k];
            }
            ri[j] = (ri[j] - s) / d;
        }
    }
    return -1;
}

// B := B * L^{-T}, where L is the m x m lower-triangular diagonal tile and B is mb x m.
void trsmTile(const double* l, double* b, const size_t ld, const size_t mb, const size_t m) {
    for (size_t r = 0; r < mb; ++r) {
        double* br = b + r * ld;
        for (size_t j = 0; j < m; ++j) {
            const double* lj = l + j * ld;
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (size_t k = 0; k < j; ++k) {
                s += br[k] * lj[k];
            }
            br[j] = (br[j] - s) / lj[j];
        }
    }
}

// C := C - A * B^T, A is mi x kb, B is mj x kb, C is mi x mj (all with leading dim ld).
// diagOff is the global row index of C's first row minus the global column index of its
// first column; micro-tiles lying entirely above the global diagonal are skipped.
void gemmTile(const double* A, const double* B, double* C, const size_t ld,
              const size_t mi, const size_t mj, const size_t kb, const long diagOff) {
    static thread_local std::vector<double> apack, bpack;
    const size_t mip = (mi + MR - 1) / MR * MR;
    const size_t mjp = (mj + NR - 1) / NR * NR;
    if (apack.size() < mip * kb) apack.resize(mip * kb);
    if (bpack.size() < mjp * kb) bpack.resize(mjp * kb);
    double* __restrict ap = apack.data();
    double* __restrict bp = bpack.data();

    // Pack A into MR-row panels: ap[panel][k][r]
    for (size_t i0 = 0; i0 < mip; i0 += MR) {
        double* dst = ap + i0 * kb;
        for (size_t r = 0; r < MR; ++r) {
            const size_t i = i0 + r;
            if (i < mi) {
                const double* src = A + i * ld;
                for (size_t k = 0; k < kb; ++k) dst[k * MR + r] = src[k];
            } else {
                for (size_t k = 0; k < kb; ++k) dst[k * MR + r] = 0.0;
            }
        }
    }
    // Pack B into NR-row panels: bp[panel][k][c]
    for (size_t j0 = 0; j0 < mjp; j0 += NR) {
        double* dst = bp + j0 * kb;
        for (size_t c = 0; c < NR; ++c) {
            const size_t j = j0 + c;
            if (j < mj) {
                const double* src = B + j * ld;
                for (size_t k = 0; k < kb; ++k) dst[k * NR + c] = src[k];
            } else {
                for (size_t k = 0; k < kb; ++k) dst[k * NR + c] = 0.0;
            }
        }
    }

    for (size_t j0 = 0; j0 < mjp; j0 += NR) {
        const double* __restrict bpp = bp + j0 * kb;
        for (size_t i0 = 0; i0 < mip; i0 += MR) {
            if ((long)(i0 + MR) + diagOff <= (long)j0) continue;
            const double* __restrict app = ap + i0 * kb;
            // MR x NR accumulator block held in vector registers
            v4d acc[MR][2] = {};
            for (size_t k = 0; k < kb; ++k) {
                const double* a = app + k * MR;
                const v4d b0 = *(const v4d*)(bpp + k * NR);
                const v4d b1 = *(const v4d*)(bpp + k * NR + 4);
                for (size_t r = 0; r < MR; ++r) {
                    const double ar = a[r];
                    acc[r][0] += ar * b0;
                    acc[r][1] += ar * b1;
                }
            }
            const size_t rmax = std::min(MR, mi - std::min(mi, i0));
            const size_t cmax = std::min(NR, mj - std::min(mj, j0));
            for (size_t r = 0; r < rmax; ++r) {
                double* crow = C + (i0 + r) * ld + j0;
                for (size_t c = 0; c < cmax; ++c) {
                    crow[c] -= acc[r][c / 4][c % 4];
                }
            }
        }
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;
    // Panel width (inner dimension of the trailing updates) and update chunk size
    const size_t nb = n >= 6144 ? 192 : (n >= 3072 ? 128 : 96);
    const size_t cb = nb;
    double* a = A.data();
    long failIndex = -1;
    // Small problems do not have enough work per step to feed every thread
    const int nthreads = (int)std::max<size_t>(1, std::min<size_t>(omp_get_max_threads(), n / 32));

    #pragma omp parallel num_threads(nthreads)
    {
        // Factor the first diagonal block
        #pragma omp single
        {
            const long f = potrfTile(a, n, std::min(nb, n));
            if (f >= 0) failIndex = f;
        }

        std::vector<std::pair<size_t, size_t>> chunks;
        for (size_t k0 = 0; k0 < n; k0 += nb) {
            // Barrier above guarantees failIndex and diagonal block k0 are final
            if (failIndex >= 0) break;
            const size_t kb = std::min(nb, n - k0);
            const size_t r0 = k0 + kb;  // first row of the trailing matrix
            if (r0 >= n) break;
            const double* akk = a + k0 * n + k0;

            // Panel solve: rows below the diagonal block are independent
            #pragma omp for schedule(static)
            for (size_t i = r0; i < n; ++i) {
                trsmTile(akk, a + i * n + k0, n, 1, kb);
            }

            // Trailing update: A22 -= L21 * L21^T (lower triangle only).
            // Chunk 0 is the next diagonal block, which is factored right after its
            // update (lookahead), overlapping with the remaining update chunks.
            const size_t kb2 = std::min(nb, n - r0);
            const size_t r1 = r0 + kb2;
            chunks.clear();
            chunks.emplace_back(r0, r0);
            for (size_t i0 = r1; i0 < n; i0 += cb) {
                const size_t i1 = std::min(n, i0 + cb);
                for (size_t j0 = r0; j0 < i1; j0 += cb) {
                    chunks.emplace_back(i0, j0);
                }
            }
            const size_t nchunks = chunks.size();
            #pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < nchunks; ++t) {
                const size_t i0 = chunks[t].first, j0 = chunks[t].second;
                if (t == 0) {
                    double* d = a + r0 * n + r0;
                    gemmTile(a + r0 * n + k0, a + r0 * n + k0, d, n, kb2, kb2, kb, 0);
                    const long f = potrfTile(d, n, kb2);
                    if (f >= 0) failIndex = (long)r0 + f;
                } else {
                    const size_t mi = std::min(cb, n - i0), mj = std::min(cb, n - j0);
                    gemmTile(a + i0 * n + k0, a + j0 * n + k0, a + i0 * n + j0, n,
                             mi, mj, kb, (long)i0 - (long)j0);
                }
            }
        }

        // Zero out upper triangular part (failIndex is consistent after the last barrier)
        if (failIndex < 0) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
            }
        }
    }

    if (failIndex >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)failIndex);
        return false;
    }

    return true;
}

// Compute C[i][j] = sum_k X[i*n+k] * X[j*n+k] for all i, j in parallel.
// Each element is accumulated in the same sequential k order as a plain triple loop.
static void multiplyByTranspose(const std::vector<double>& X, std::vector<double>& C, const size_t n) {
    constexpr size_t BI = 4, BJ = 4;
    const double* x = X.data();
    double* c = C.data();
    const size_t nbi = (n + BI - 1) / BI;
    const size_t nbj = (n + BJ - 1) / BJ;
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (size_t bi = 0; bi < nbi; ++bi) {
        for (size_t bj = 0; bj < nbj; ++bj) {
            const size_t i0 = bi * BI, j0 = bj * BJ;
            if (i0 + BI <= n && j0 + BJ <= n) {
                double acc[BI][BJ] = {};
                for (size_t k = 0; k < n; ++k) {
                    for (size_t r = 0; r < BI; ++r) {
                        const double xi = x[(i0 + r) * n + k];
                        for (size_t s = 0; s < BJ; ++s) {
                            acc[r][s] += xi * x[(j0 + s) * n + k];
                        }
                    }
                }
                for (size_t r = 0; r < BI; ++r)
                    for (size_t s = 0; s < BJ; ++s)
                        c[(i0 + r) * n + j0 + s] = acc[r][s];
            } else {
                for (size_t i = i0; i < std::min(n, i0 + BI); ++i) {
                    for (size_t j = j0; j < std::min(n, j0 + BJ); ++j) {
                        double sum = 0.0;
                        for (size_t k = 0; k < n; ++k) {
                            sum += x[i * n + k] * x[j * n + k];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
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
    multiplyByTranspose(B, A, n);
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    multiplyByTranspose(L, reconstructed, n);
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for reduction(max:maxError, relError) schedule(static)
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
