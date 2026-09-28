#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>

#include "../common/results_output.hpp"

// Cholesky decomposition (OpenMP-parallel, blocked left-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The left-looking formulation is used because it keeps, for every output element,
// the exact same accumulation order as the original scalar code:
//   L[i][j] = (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
// with the inner sum accumulated in a single accumulator for ascending k. Blocking
// only splits that ascending k range into consecutive chunks that are added into the
// same running accumulator, so the results are bit-for-bit identical to the original.

namespace {

constexpr size_t BS = 32;   // width of a column panel (number of columns per block)
constexpr size_t KBB = 96;  // k-blocking factor for the trailing-matrix update
constexpr size_t TI = 4;    // rows per tile in the update phase
constexpr size_t PKB = 64;  // k-blocking factor for the transposing pack

// The scalar reduction "for (k) sum += a[k] * b[k];" of the original code is compiled
// (-O3 -march=native) into groups of four separately rounded products that are added
// into the accumulator one after another, plus a fused multiply-add for the remaining
// len % 4 elements. Reproducing that rounding pattern explicitly - together with
// -ffp-contract=off, which keeps the compiler from fusing the products - makes this
// parallel implementation produce bit-for-bit the same results as the original.
// All blocking factors used below are multiples of four, so splitting the ascending
// k range into consecutive chunks never breaks a group of four apart.
inline double dotAccumulate(const double* __restrict a, const double* __restrict b, const size_t len, double s) {
    const size_t m = len & ~static_cast<size_t>(3);
    for (size_t k = 0; k < m; k += 4) {
        const double p0 = a[k] * b[k];
        const double p1 = a[k + 1] * b[k + 1];
        const double p2 = a[k + 2] * b[k + 2];
        const double p3 = a[k + 3] * b[k + 3];
        s += p0;
        s += p1;
        s += p2;
        s += p3;
    }
    for (size_t k = m; k < len; ++k) {
        s = std::fma(a[k], b[k], s);
    }
    return s;
}

// Four dot products against a common row, computed with the rounding pattern of
// dotAccumulate. The independent accumulator chains hide the latency of the serial
// additions, which a single dot product cannot.
inline void dotAccumulate4(const double* __restrict a, const double* __restrict b0, const double* __restrict b1,
                           const double* __restrict b2, const double* __restrict b3, const size_t len, double* out) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    const size_t m = len & ~static_cast<size_t>(3);
    for (size_t k = 0; k < m; ++k) {
        const double av = a[k];
        const double p0 = av * b0[k];
        const double p1 = av * b1[k];
        const double p2 = av * b2[k];
        const double p3 = av * b3[k];
        s0 += p0;
        s1 += p1;
        s2 += p2;
        s3 += p3;
    }
    for (size_t k = m; k < len; ++k) {
        const double av = a[k];
        s0 = std::fma(av, b0[k], s0);
        s1 = std::fma(av, b1[k], s1);
        s2 = std::fma(av, b2[k], s2);
        s3 = std::fma(av, b3[k], s3);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// Pin each OpenMP thread to one of the CPUs the process is allowed to run on.
// Thread migration between cores (and especially between NUMA nodes) otherwise
// destroys the cache and memory locality this blocked algorithm relies on.
// An explicit user binding request is left untouched.
// Small matrices do not provide enough work per thread to amortise the barriers
// between the phases of the factorisation, so the thread count is limited to what can
// be kept busy with at least a few row tiles each (never more than the user asked for).
// This has to happen before the first parallel region, otherwise the OpenMP runtime has
// to tear down the threads it has already created.
void limitThreads(const size_t n) {
    const int useful = static_cast<int>(std::max<size_t>(1, n / (4 * TI)));
    if (useful < omp_get_max_threads()) {
        omp_set_num_threads(useful);
    }
}

void bindThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("GOMP_CPU_AFFINITY") != nullptr) {
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
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) {
        return true;
    }

    // P: panel of the current block, transposed and packed as P[k * BS + jj] = L[J + jj][k].
    // The transposed layout makes the innermost loop of the update a contiguous, and hence
    // vectorizable, update of BS independent accumulators.
    std::vector<double> P(n * BS);
    // ACC[i * BS + jj] holds sum_{k < J} L[i][k] * L[J + jj][k], i.e. the part of the dot
    // product that has already been accumulated for the current block (zero for J == 0).
    std::vector<double> ACC(n * BS, 0.0);

    // Every phase below is executed inside a single parallel region: the panel loop
    // only needs barriers between its phases, which is considerably cheaper than
    // opening a new parallel region per phase and per panel.
    double diag[BS];
    bool failed = false;
    size_t failColumn = 0;

    #pragma omp parallel
    for (size_t J = 0; J < n; J += BS) {
        const size_t jb = std::min(BS, n - J);

        // ---- Pack the already-computed part of the panel rows (transposed) ----
        // Blocked transpose: each thread reads contiguous stretches of a panel row and
        // scatters them into a block of P that stays in cache.
        #pragma omp for schedule(static)
        for (size_t kt = 0; kt < J; kt += PKB) {
            const size_t ktend = std::min(kt + PKB, J);
            for (size_t jj = 0; jj < jb; ++jj) {
                const double* __restrict Arow = &A[(J + jj) * n];
                for (size_t k = kt; k < ktend; ++k) {
                    P[k * BS + jj] = Arow[k];
                }
            }
        }

        // ---- Update: accumulate the k < J contributions for all rows of the block ----
        if (J > 0) {
            const double* __restrict Abase = A.data();
            const double* __restrict Pbase = P.data();
            double* __restrict AccBase = ACC.data();

            #pragma omp for schedule(static)
            for (size_t it = J; it < n; it += TI) {
                const size_t iend = std::min(it + TI, n);
                for (size_t kb = 0; kb < J; kb += KBB) {
                    const size_t kend = std::min(kb + KBB, J);
                    for (size_t i = it; i < iend; ++i) {
                        double acc[BS];
                        double* __restrict accMem = &AccBase[i * BS];
                        if (kb == 0) {
                            for (size_t jj = 0; jj < BS; ++jj) {
                                acc[jj] = 0.0;
                            }
                        } else {
                            for (size_t jj = 0; jj < BS; ++jj) {
                                acc[jj] = accMem[jj];
                            }
                        }
                        const double* __restrict Ai = &Abase[i * n];
                        for (size_t k = kb; k < kend; k += 4) {
                            const double a0 = Ai[k];
                            const double a1 = Ai[k + 1];
                            const double a2 = Ai[k + 2];
                            const double a3 = Ai[k + 3];
                            const double* __restrict P0 = &Pbase[k * BS];
                            const double* __restrict P1 = P0 + BS;
                            const double* __restrict P2 = P0 + 2 * BS;
                            const double* __restrict P3 = P0 + 3 * BS;
                            // BS independent accumulator chains, each still summing in
                            // ascending k order -> vectorizes without reassociation.
                            for (size_t jj = 0; jj < BS; ++jj) {
                                const double p0 = a0 * P0[jj];
                                const double p1 = a1 * P1[jj];
                                const double p2 = a2 * P2[jj];
                                const double p3 = a3 * P3[jj];
                                double s = acc[jj];
                                s += p0;
                                s += p1;
                                s += p2;
                                s += p3;
                                acc[jj] = s;
                            }
                        }
                        for (size_t jj = 0; jj < BS; ++jj) {
                            accMem[jj] = acc[jj];
                        }
                    }
                }
            }
        }

        // ---- Factor the diagonal block (small, serial: it carries the dependencies) ----
        #pragma omp single
        for (size_t jj = 0; jj < jb; ++jj) {
            const size_t j = J + jj;

            const double sum = dotAccumulate(&A[j * n + J], &A[j * n + J], j - J, ACC[j * BS + jj]);
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                failed = true;
                failColumn = j;
                break;
            }
            const double d = sqrt(val);
            A[j * n + j] = d;
            diag[jj] = d;

            for (size_t i = j + 1; i < J + jb; ++i) {
                const double s = dotAccumulate(&A[i * n + J], &A[j * n + J], j - J, ACC[i * BS + jj]);
                A[i * n + j] = (A[i * n + j] - s) / d;
            }
        }
        // The implicit barrier of the single construct makes the flag visible to all
        // threads, so every thread leaves the panel loop at the same iteration.
        if (failed) {
            break;
        }

        // ---- Solve for the trailing rows of the panel (rows are independent) ----
        {
            const size_t ifirst = J + jb;
            double* __restrict Abase = A.data();
            const double* __restrict AccBase = ACC.data();

            #pragma omp for schedule(static)
            for (size_t it = ifirst; it < n; it += 4) {
                const size_t iend = std::min(it + 4, n);
                const size_t cnt = iend - it;

                if (cnt == 4) {
                    double* const r0 = &Abase[it * n];
                    double* const r1 = &Abase[(it + 1) * n];
                    double* const r2 = &Abase[(it + 2) * n];
                    double* const r3 = &Abase[(it + 3) * n];
                    for (size_t jj = 0; jj < jb; ++jj) {
                        const size_t j = J + jj;
                        const double* Lj = &Abase[j * n];
                        // Four independent dot products give instruction-level parallelism
                        // while each accumulator still runs over ascending k.
                        double s0 = AccBase[it * BS + jj];
                        double s1 = AccBase[(it + 1) * BS + jj];
                        double s2 = AccBase[(it + 2) * BS + jj];
                        double s3 = AccBase[(it + 3) * BS + jj];
                        const size_t len = j - J;
                        const size_t m = J + (len & ~static_cast<size_t>(3));
                        for (size_t k = J; k < m; ++k) {
                            const double lj = Lj[k];
                            const double q0 = r0[k] * lj;
                            const double q1 = r1[k] * lj;
                            const double q2 = r2[k] * lj;
                            const double q3 = r3[k] * lj;
                            s0 += q0;
                            s1 += q1;
                            s2 += q2;
                            s3 += q3;
                        }
                        for (size_t k = m; k < j; ++k) {
                            const double lj = Lj[k];
                            s0 = std::fma(r0[k], lj, s0);
                            s1 = std::fma(r1[k], lj, s1);
                            s2 = std::fma(r2[k], lj, s2);
                            s3 = std::fma(r3[k], lj, s3);
                        }
                        const double d = diag[jj];
                        r0[j] = (r0[j] - s0) / d;
                        r1[j] = (r1[j] - s1) / d;
                        r2[j] = (r2[j] - s2) / d;
                        r3[j] = (r3[j] - s3) / d;
                    }
                } else {
                    for (size_t jj = 0; jj < jb; ++jj) {
                        const size_t j = J + jj;
                        const double* Lj = &Abase[j * n];
                        const double d = diag[jj];
                        for (size_t i = it; i < iend; ++i) {
                            double* Ri = &Abase[i * n];
                            const double s = dotAccumulate(&Ri[J], &Lj[J], j - J, AccBase[i * BS + jj]);
                            Ri[j] = (Ri[j] - s) / d;
                        }
                    }
                }
            }
        }
    }

    if (failed) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failColumn);
        return false;
    }

    // ---- Zero out the upper triangular part ----
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double* __restrict Ai = &A[i * n];
        for (size_t j = i + 1; j < n; ++j) {
            Ai[j] = 0.0;
        }
    }

    return true;
}

// Distribute the pages of a matrix over the NUMA nodes by first-touching them from
// all threads in a round-robin fashion (Linux places a page on the node of the thread
// that touches it first). Without this, all rows would end up on the node of whichever
// thread happened to fill them, and the memory controllers of that node become the
// bottleneck once all cores work on the matrix.
void firstTouch(std::vector<double>& M, const size_t n) {
    // std::vector zero-initialises on construction, so all pages are already faulted in
    // on the master thread's node. Drop them first; they are re-faulted (zeroed) by the
    // thread that touches them below.
    const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(M.data());
    const uintptr_t end = begin + M.size() * sizeof(double);
    const uintptr_t alignedBegin = (begin + pageSize - 1) & ~(static_cast<uintptr_t>(pageSize) - 1);
    const uintptr_t alignedEnd = end & ~(static_cast<uintptr_t>(pageSize) - 1);
    if (alignedEnd > alignedBegin) {
        madvise(reinterpret_cast<void*>(alignedBegin), alignedEnd - alignedBegin, MADV_DONTNEED);
    }

    #pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t nt = static_cast<size_t>(omp_get_num_threads());
        for (size_t i = t * TI; i < n; i += nt * TI) {
            const size_t iend = std::min(i + TI, n);
            for (size_t idx = i * n; idx < iend * n; ++idx) {
                M[idx] = 0.0;
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
    // Only the lower triangle is computed and then mirrored: the products are identical
    // (same operands, same order), so the result is bit-for-bit the same as computing
    // both halves explicitly.
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        const double* Bi = &B[i * n];
        size_t j = 0;
        for (; j + 4 <= i + 1; j += 4) {
            double sums[4];
            dotAccumulate4(Bi, &B[j * n], &B[(j + 1) * n], &B[(j + 2) * n], &B[(j + 3) * n], n, sums);
            for (size_t u = 0; u < 4; ++u) {
                A[i * n + j + u] = sums[u];
                A[(j + u) * n + i] = sums[u];
            }
        }
        for (; j <= i; ++j) {
            const double sum = dotAccumulate(Bi, &B[j * n], n, 0.0);
            A[i * n + j] = sum;
            A[j * n + i] = sum;
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
    
    // Compute L * L^T (symmetric, so only the lower triangle is computed and mirrored:
    // the two halves consist of the same products in the same order)
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        const double* Li = &L[i * n];
        size_t j = 0;
        for (; j + 4 <= i + 1; j += 4) {
            double sums[4];
            dotAccumulate4(Li, &L[j * n], &L[(j + 1) * n], &L[(j + 2) * n], &L[(j + 3) * n], n, sums);
            for (size_t u = 0; u < 4; ++u) {
                reconstructed[i * n + j + u] = sums[u];
                reconstructed[(j + u) * n + i] = sums[u];
            }
        }
        for (; j <= i; ++j) {
            const double sum = dotAccumulate(Li, &L[j * n], n, 0.0);
            reconstructed[i * n + j] = sum;
            reconstructed[j * n + i] = sum;
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
    
    limitThreads(n);
    bindThreads();

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    firstTouch(A, n);
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
