#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define CHOLESKY_USE_AVX2 1
#endif

#include "../common/results_output.hpp"

// Blocked, OpenMP-parallel Cholesky decomposition (left-looking).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Every element L[i][j] is still computed as
//     L[i][j] = (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
// with the summation carried out in strictly increasing k order into a single
// accumulator, i.e. exactly the summation order of the sequential version. The
// sum is only *split* at a block boundary J: the leading part (k < J) is
// computed by a parallel panel-GEMM into S[][], and the trailing part
// (J <= k < j) continues to accumulate into that very same value. Results
// therefore differ from the sequential code only by the last-bit effects of
// fused multiply-add (the kernels here always fuse, whereas the auto-vectorized
// scalar loop fuses only part of its iterations).

namespace {

// Interleave allocations across all NUMA nodes. The matrix is streamed by every
// thread during the panel updates, so a single-node placement (the default,
// since std::vector zero-initializes the storage from one thread) would starve
// the remote socket. Equivalent to running under `numactl --interleave=all`.
void interleaveMemory() {
#if defined(__linux__) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)
    constexpr int MPOL_DEFAULT_MODE = 0;
    constexpr int MPOL_INTERLEAVE_MODE = 3;
    constexpr unsigned long MPOL_F_MEMS_ALLOWED_FLAG = 1 << 2;
    constexpr unsigned long MAX_NODES = 1024;

    int mode = 0;
    if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0 || mode != MPOL_DEFAULT_MODE) {
        return; // an explicit policy (e.g. from numactl) is already in effect
    }
    std::vector<unsigned long> mask(MAX_NODES / (8 * sizeof(unsigned long)), 0UL);
    if (syscall(SYS_get_mempolicy, nullptr, mask.data(), MAX_NODES, nullptr, MPOL_F_MEMS_ALLOWED_FLAG) != 0) {
        return;
    }
    syscall(SYS_set_mempolicy, MPOL_INTERLEAVE_MODE, mask.data(), MAX_NODES);
#endif
}

// Pin each OpenMP thread to its own CPU. Thread migration between cores (and
// especially between NUMA nodes) costs an order of magnitude here, and the
// benchmark is usually launched without OMP_PROC_BIND set. An explicit binding
// policy from the environment always wins.
void bindThreads() {
#ifdef __linux__
    if (omp_get_proc_bind() != omp_proc_bind_false || getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) {
            cpus.push_back(c);
        }
    }
    if (cpus.empty()) {
        return;
    }
    #pragma omp parallel
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
}

constexpr size_t JB = 32;  // column block width
constexpr size_t KC = 128; // k cache block of the panel update
constexpr size_t MI = 6;   // rows handled per GEMM register tile
constexpr size_t NV = 8;   // columns handled per GEMM register tile (2 AVX2 vectors)

// Accumulates S[i][jj] += sum_{kBeg <= k < kEnd} A[i][k] * Bt[k][jj] for M
// consecutive rows i, where Bt[k][jj] = L[J+jj][k] is the transposed
// (k-contiguous -> jj-contiguous) panel of the block columns. The k range is a
// cache block of the full [0, J) range and the ranges are visited in increasing
// order, so every (i, jj) accumulator still sees its k in ascending order -
// exactly the sequential summation order.
template <size_t M>
inline void gemmTile(const double* __restrict Ap, const size_t n, const double* __restrict Bt,
                     double* __restrict S, const size_t i0, const size_t kBeg, const size_t kEnd,
                     const size_t jbPad) {
#ifdef CHOLESKY_USE_AVX2
    for (size_t jt = 0; jt < jbPad; jt += NV) {
        __m256d acc0[M], acc1[M];
        for (size_t m = 0; m < M; ++m) {
            if (kBeg == 0) {
                acc0[m] = _mm256_setzero_pd();
                acc1[m] = _mm256_setzero_pd();
            } else {
                acc0[m] = _mm256_loadu_pd(S + (i0 + m) * JB + jt);
                acc1[m] = _mm256_loadu_pd(S + (i0 + m) * JB + jt + 4);
            }
        }
        const double* bt = Bt + kBeg * JB + jt;
        for (size_t k = kBeg; k < kEnd; ++k, bt += JB) {
            const __m256d b0 = _mm256_loadu_pd(bt);
            const __m256d b1 = _mm256_loadu_pd(bt + 4);
            for (size_t m = 0; m < M; ++m) {
                const __m256d a = _mm256_set1_pd(Ap[(i0 + m) * n + k]);
                acc0[m] = _mm256_fmadd_pd(a, b0, acc0[m]);
                acc1[m] = _mm256_fmadd_pd(a, b1, acc1[m]);
            }
        }
        for (size_t m = 0; m < M; ++m) {
            _mm256_storeu_pd(S + (i0 + m) * JB + jt, acc0[m]);
            _mm256_storeu_pd(S + (i0 + m) * JB + jt + 4, acc1[m]);
        }
    }
#else
    for (size_t jt = 0; jt < jbPad; jt += NV) {
        double acc[M][NV];
        for (size_t m = 0; m < M; ++m) {
            for (size_t v = 0; v < NV; ++v) {
                acc[m][v] = (kBeg == 0) ? 0.0 : S[(i0 + m) * JB + jt + v];
            }
        }
        for (size_t k = kBeg; k < kEnd; ++k) {
            for (size_t m = 0; m < M; ++m) {
                const double a = Ap[(i0 + m) * n + k];
                for (size_t v = 0; v < NV; ++v) {
                    acc[m][v] += a * Bt[k * JB + jt + v];
                }
            }
        }
        for (size_t m = 0; m < M; ++m) {
            for (size_t v = 0; v < NV; ++v) {
                S[(i0 + m) * JB + jt + v] = acc[m][v];
            }
        }
    }
#endif
}

// Dispatches a row tile of runtime height (1..MI) to the register kernel.
inline void gemmTileDispatch(const double* __restrict Ap, const size_t n, const double* __restrict Bt,
                             double* __restrict S, const size_t i0, const size_t rows, const size_t kBeg,
                             const size_t kEnd, const size_t jbPad) {
    switch (rows) {
        case 6: gemmTile<6>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
        case 5: gemmTile<5>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
        case 4: gemmTile<4>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
        case 3: gemmTile<3>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
        case 2: gemmTile<2>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
        default: gemmTile<1>(Ap, n, Bt, S, i0, kBeg, kEnd, jbPad); break;
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) {
        return true;
    }

    double* const __restrict Ap = A.data();
    // Bt[k * JB + jj] = L[J + jj][k]: transposed panel of the active block columns.
    std::vector<double> BtStore(n * JB, 0.0);
    // S[i * JB + jj]: partial sum over k < J for element (i, J + jj).
    std::vector<double> SStore(n * JB, 0.0);
    double* const __restrict Bt = BtStore.data();
    double* const __restrict S = SStore.data();

    bool ok = true;

    // Each column block ends in a barrier, so for small matrices a huge team
    // costs more in synchronization than it contributes in compute. Roughly one
    // thread per 16 columns keeps the barriers amortized; large matrices always
    // use the full team.
    const int teamSize = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(std::max(omp_get_max_threads(), 1)), std::max<size_t>(n / 16, 1)));

    #pragma omp parallel num_threads(teamSize)
    {
        const int tid = omp_get_thread_num();
        const int nThreads = omp_get_num_threads();

        for (size_t J = 0; J < n; J += JB) {
            const size_t jb = std::min(JB, n - J);
            const size_t jbPad = (jb + NV - 1) / NV * NV;

            // --- Transpose the block-column panel of the already computed part ---
            #pragma omp for schedule(static) nowait
            for (size_t k = 0; k < J; ++k) {
                for (size_t jj = 0; jj < jb; ++jj) {
                    Bt[k * JB + jj] = Ap[(J + jj) * n + k];
                }
                for (size_t jj = jb; jj < jbPad; ++jj) {
                    Bt[k * JB + jj] = 0.0; // padding lanes must not carry stale data
                }
            }
            #pragma omp barrier

            // --- Panel GEMM: leading part (k < J) of every remaining element ---
            // Row tiles are distributed statically, and each thread walks its own
            // tiles inside a k cache block, so no synchronization is needed between
            // the k blocks: the transposed panel slice stays hot in L2 while the
            // rows of A are fetched from memory only once per block.
            const size_t nTiles = (n - J + MI - 1) / MI;
            const size_t tLo = nTiles * static_cast<size_t>(tid) / static_cast<size_t>(nThreads);
            const size_t tHi = nTiles * static_cast<size_t>(tid + 1) / static_cast<size_t>(nThreads);
            for (size_t kBeg = 0; kBeg < J || kBeg == 0; kBeg += KC) {
                const size_t kEnd = std::min(kBeg + KC, J);
                for (size_t t = tLo; t < tHi; ++t) {
                    const size_t i0 = J + t * MI;
                    gemmTileDispatch(Ap, n, Bt, S, i0, std::min(MI, n - i0), kBeg, kEnd, jbPad);
                }
                if (kEnd >= J) {
                    break;
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                // --- Factor the diagonal block sequentially (O(jb^3) work) ---
                for (size_t j = J; j < J + jb; ++j) {
                    const double* const Aj = Ap + j * n;
                    double sum = S[j * JB + (j - J)];
                    for (size_t k = J; k < j; ++k) {
                        sum += Aj[k] * Aj[k];
                    }
                    const double val = Aj[j] - sum;
                    if (val <= 0.0) {
                        // Matrix is not positive definite
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        ok = false;
                        break;
                    }
                    Ap[j * n + j] = sqrt(val);

                    const double djj = Ap[j * n + j];
                    const size_t jj = j - J;
                    size_t i = j + 1;
                    for (; i + 4 <= J + jb; i += 4) { // four independent FMA chains
                        double* const A0 = Ap + i * n;
                        double* const A1 = A0 + n;
                        double* const A2 = A0 + 2 * n;
                        double* const A3 = A0 + 3 * n;
                        const double* const Si = S + i * JB + jj;
                        double s0 = Si[0], s1 = Si[JB], s2 = Si[2 * JB], s3 = Si[3 * JB];
                        for (size_t k = J; k < j; ++k) {
                            const double b = Aj[k];
                            s0 += A0[k] * b;
                            s1 += A1[k] * b;
                            s2 += A2[k] * b;
                            s3 += A3[k] * b;
                        }
                        A0[j] = (A0[j] - s0) / djj;
                        A1[j] = (A1[j] - s1) / djj;
                        A2[j] = (A2[j] - s2) / djj;
                        A3[j] = (A3[j] - s3) / djj;
                    }
                    for (; i < J + jb; ++i) {
                        double* const Ai = Ap + i * n;
                        double s = S[i * JB + jj];
                        for (size_t k = J; k < j; ++k) {
                            s += Ai[k] * Aj[k];
                        }
                        Ai[j] = (Ai[j] - s) / djj;
                    }
                }
            } // implicit barrier: publishes ok and the diagonal block

            if (!ok) {
                break;
            }

            // --- Triangular solve for the rows below the diagonal block ---
            const size_t iStart = J + jb;
            const size_t nQuads = (n - iStart) / 4;
            #pragma omp for schedule(static) nowait
            for (size_t q = 0; q < nQuads; ++q) {
                const size_t i0 = iStart + q * 4;
                double* const A0 = Ap + i0 * n;
                double* const A1 = A0 + n;
                double* const A2 = A0 + 2 * n;
                double* const A3 = A0 + 3 * n;
                const double* const S0 = S + i0 * JB;
                for (size_t j = J; j < J + jb; ++j) {
                    const double* const Aj = Ap + j * n;
                    const size_t jj = j - J;
                    double s0 = S0[jj];
                    double s1 = S0[JB + jj];
                    double s2 = S0[2 * JB + jj];
                    double s3 = S0[3 * JB + jj];
                    for (size_t k = J; k < j; ++k) {
                        const double b = Aj[k];
                        s0 += A0[k] * b;
                        s1 += A1[k] * b;
                        s2 += A2[k] * b;
                        s3 += A3[k] * b;
                    }
                    const double djj = Aj[j];
                    A0[j] = (A0[j] - s0) / djj;
                    A1[j] = (A1[j] - s1) / djj;
                    A2[j] = (A2[j] - s2) / djj;
                    A3[j] = (A3[j] - s3) / djj;
                }
            }
            #pragma omp for schedule(static)
            for (size_t i = iStart + nQuads * 4; i < n; ++i) {
                double* const Ai = Ap + i * n;
                for (size_t j = J; j < J + jb; ++j) {
                    const double* const Aj = Ap + j * n;
                    double s = S[i * JB + (j - J)];
                    for (size_t k = J; k < j; ++k) {
                        s += Ai[k] * Aj[k];
                    }
                    Ai[j] = (Ai[j] - s) / Aj[j];
                }
            }
        }

        // --- Zero out upper triangular part ---
        if (ok) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    Ap[i * n + j] = 0.0;
                }
            }
        }
    }

    return ok;
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
    
    // Compute A = B * B^T (rows are independent; the k-order of every dot
    // product is preserved, so the generated matrix is bit-identical)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double* const Bi = B.data() + i * n;
        double* const Ai = A.data() + i * n;
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const double* const B0 = B.data() + j * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double a = Bi[k];
                s0 += a * B0[k];
                s1 += a * B0[n + k];
                s2 += a * B0[2 * n + k];
                s3 += a * B0[3 * n + k];
            }
            Ai[j] = s0;
            Ai[j + 1] = s1;
            Ai[j + 2] = s2;
            Ai[j + 3] = s3;
        }
        for (; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * B[j * n + k];
            }
            Ai[j] = sum;
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

    // Compute L * L^T
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double* const Li = L.data() + i * n;
        double* const Ri = reconstructed.data() + i * n;
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const double* const L0 = L.data() + j * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double a = Li[k];
                s0 += a * L0[k];
                s1 += a * L0[n + k];
                s2 += a * L0[2 * n + k];
                s3 += a * L0[3 * n + k];
            }
            Ri[j] = s0;
            Ri[j + 1] = s1;
            Ri[j + 2] = s2;
            Ri[j + 3] = s3;
        }
        for (; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * L[j * n + k];
            }
            Ri[j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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
    
    interleaveMemory();
    bindThreads();

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
