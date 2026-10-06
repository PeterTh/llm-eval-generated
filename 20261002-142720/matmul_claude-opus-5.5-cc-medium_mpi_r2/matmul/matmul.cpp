#include <mpi.h>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed matrix multiplication (MPI, 2D block decomposition of C)
//
// Ranks form a pr x pc grid. Rank (r, c) owns the block of C with rows
// [rowBegin, rowEnd) and columns [colBegin, colEnd); it needs the matching
// rows of A (all k) and columns of B (all k), which it generates locally.
// Each element of C is accumulated over k = 0..N-1 in ascending order with a
// single accumulator, exactly like the reference loop, so results are
// bit-identical to the sequential version.
// ---------------------------------------------------------------------------

typedef double v4d __attribute__((vector_size(32)));
typedef double v4du __attribute__((vector_size(32), aligned(8)));

constexpr size_t MR = 6;    // micro-tile rows
constexpr size_t NR = 8;    // micro-tile columns (2 x 4 doubles)
constexpr size_t KC = 256;  // k-block (A/B micro-panels stay in L1/L2)
constexpr size_t MC = 96;   // row block of A kept in L2
constexpr size_t NC = 1024; // column block of B kept in L3

static inline size_t blockStart(const size_t n, const int parts, const int idx) {
    const size_t base = n / parts, rem = n % parts;
    return idx * base + std::min<size_t>(idx, rem);
}

// Fused a*b+c (used only where the reference build contracts to FMA)
static inline v4d fusedMulAdd(const v4d a, const v4d b, const v4d c) {
#ifdef __FMA__
    return _mm256_fmadd_pd(a, b, c);
#else
    return a * b + c;
#endif
}

// C[MR x NR] (+)= Ap[kc x MR] * Bp[kc x NR]; packed operands, k ascending.
// Rounding mirrors the reference build: every term is sum + round(a*b)
// (unfused, see -ffp-contract=off), except the final term k = N-1 for odd N,
// which the reference evaluates with a fused multiply-add.
template <bool FusedLast>
static inline void microKernel(const size_t kc, const double* __restrict Ap,
                               const double* __restrict Bp, double* __restrict C,
                               const size_t ldc, const bool first) {
    v4d c[MR][2];
    if (first) {
        for (size_t r = 0; r < MR; ++r) c[r][0] = c[r][1] = v4d{0.0, 0.0, 0.0, 0.0};
    } else {
        for (size_t r = 0; r < MR; ++r) {
            c[r][0] = *reinterpret_cast<const v4du*>(C + r * ldc);
            c[r][1] = *reinterpret_cast<const v4du*>(C + r * ldc + 4);
        }
    }
    const size_t kEnd = FusedLast ? kc - 1 : kc;
    for (size_t k = 0; k < kEnd; ++k) {
        const v4d b0 = *reinterpret_cast<const v4d*>(Bp + k * NR);
        const v4d b1 = *reinterpret_cast<const v4d*>(Bp + k * NR + 4);
#pragma GCC unroll 6
        for (size_t r = 0; r < MR; ++r) {
            const double a = Ap[k * MR + r];
            c[r][0] += a * b0;
            c[r][1] += a * b1;
        }
    }
    if constexpr (FusedLast) {
        const size_t k = kc - 1;
        const v4d b0 = *reinterpret_cast<const v4d*>(Bp + k * NR);
        const v4d b1 = *reinterpret_cast<const v4d*>(Bp + k * NR + 4);
        for (size_t r = 0; r < MR; ++r) {
            const double a = Ap[k * MR + r];
            const v4d av = {a, a, a, a};
            c[r][0] = fusedMulAdd(av, b0, c[r][0]);
            c[r][1] = fusedMulAdd(av, b1, c[r][1]);
        }
    }
    for (size_t r = 0; r < MR; ++r) {
        *reinterpret_cast<v4du*>(C + r * ldc) = c[r][0];
        *reinterpret_cast<v4du*>(C + r * ldc + 4) = c[r][1];
    }
}

static inline void runKernel(const bool fusedLast, const size_t kc, const double* Ap,
                             const double* Bp, double* C, const size_t ldc, const bool first) {
    if (fusedLast) microKernel<true>(kc, Ap, Bp, C, ldc, first);
    else microKernel<false>(kc, Ap, Bp, C, ldc, first);
}

// Local block product: C (m x n) = A (m x K, row-major) * B (K x n, row-major)
void localMultiply(const double* A, const double* B, double* C,
                   const size_t m, const size_t n, const size_t K) {
    if (m == 0 || n == 0) return;
    if (K == 0) {
        std::fill(C, C + m * n, 0.0);
        return;
    }
    const size_t nPanels = (n + NR - 1) / NR;

    // Pack all of local B into NR-wide column panels: Bp[panel][k][NR]
    double* Bp = static_cast<double*>(std::aligned_alloc(64, ((nPanels * K * NR * sizeof(double) + 63) / 64) * 64));
    for (size_t p = 0; p < nPanels; ++p) {
        const size_t j0 = p * NR, w = std::min(NR, n - j0);
        double* dst = Bp + p * K * NR;
        for (size_t k = 0; k < K; ++k) {
            const double* src = B + k * n + j0;
            size_t c = 0;
            for (; c < w; ++c) dst[k * NR + c] = src[c];
            for (; c < NR; ++c) dst[k * NR + c] = 0.0;
        }
    }

    const size_t mcMax = std::min(MC, ((m + MR - 1) / MR) * MR);
    double* Ap = static_cast<double*>(std::aligned_alloc(64, ((mcMax * KC * sizeof(double) + 63) / 64) * 64));
    alignas(64) double tile[MR * NR];

    for (size_t jc = 0; jc < n; jc += NC) {
        const size_t nc = std::min(NC, n - jc);
        for (size_t pc = 0; pc < K; pc += KC) {
            const size_t kc = std::min(KC, K - pc);
            const bool first = (pc == 0);
            const bool fusedLast = (K % 2 == 1) && (pc + kc == K);
            for (size_t ic = 0; ic < m; ic += MC) {
                const size_t mc = std::min(MC, m - ic);
                // Pack A[ic:ic+mc, pc:pc+kc] into MR-row panels: Ap[panel][k][MR]
                for (size_t ip = 0; ip < mc; ip += MR) {
                    const size_t h = std::min(MR, mc - ip);
                    double* dst = Ap + ip * kc;
                    for (size_t r = 0; r < h; ++r) {
                        const double* src = A + (ic + ip + r) * K + pc;
                        for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = src[k];
                    }
                    for (size_t r = h; r < MR; ++r)
                        for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = 0.0;
                }
                for (size_t jr = 0; jr < nc; jr += NR) {
                    const size_t j = jc + jr;
                    const size_t w = std::min(NR, n - j);
                    const double* bp = Bp + (j / NR) * K * NR + pc * NR;
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        const size_t h = std::min(MR, mc - ir);
                        const double* ap = Ap + ir * kc;
                        double* cp = C + (ic + ir) * n + j;
                        if (h == MR && w == NR) {
                            runKernel(fusedLast, kc, ap, bp, cp, n, first);
                        } else {
                            if (!first)
                                for (size_t r = 0; r < h; ++r)
                                    for (size_t c = 0; c < w; ++c) tile[r * NR + c] = cp[r * n + c];
                            runKernel(fusedLast, kc, ap, bp, tile, NR, first);
                            for (size_t r = 0; r < h; ++r)
                                for (size_t c = 0; c < w; ++c) cp[r * n + c] = tile[r * NR + c];
                        }
                    }
                }
            }
        }
    }
    std::free(Ap);
    std::free(Bp);
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identically on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 2D process grid; rank (pRow, pCol) owns block C[rowBegin:rowEnd, colBegin:colEnd]
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    const int pRow = rank / dims[1], pCol = rank % dims[1];
    const size_t rowBegin = blockStart(N, dims[0], pRow), rowEnd = blockStart(N, dims[0], pRow + 1);
    const size_t colBegin = blockStart(N, dims[1], pCol), colEnd = blockStart(N, dims[1], pCol + 1);
    const size_t mLoc = rowEnd - rowBegin, nLoc = colEnd - colBegin;

    // Allocate local blocks: rows of A, columns of B, block of C
    std::vector<double> A(mLoc * N);
    std::vector<double> B(N * nLoc);
    std::vector<double> Cloc(mLoc * nLoc);
    std::vector<double> C(root ? N * N : 0);
    
    // Initialize matrices (each rank generates exactly the parts it needs)
    if (root) printf("Initializing matrices...\n");
    for (size_t i = 0; i < mLoc; ++i)
        for (size_t k = 0; k < N; ++k)
            A[i * N + k] = getPseudoRndValue(N, rowBegin + i, k);
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < nLoc; ++j)
            B[k * nLoc + j] = getPseudoRndValue(N, k, colBegin + j);
    
    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    localMultiply(A.data(), B.data(), Cloc.data(), mLoc, nLoc, N);

    // Assemble the full C on rank 0
    if (root) {
        std::vector<MPI_Request> reqs;
        std::vector<MPI_Datatype> types;
        for (int p = 0; p < nprocs; ++p) {
            const int r = p / dims[1], c = p % dims[1];
            const size_t r0 = blockStart(N, dims[0], r), r1 = blockStart(N, dims[0], r + 1);
            const size_t c0 = blockStart(N, dims[1], c), c1 = blockStart(N, dims[1], c + 1);
            if (r1 == r0 || c1 == c0) continue;
            if (p == 0) {
                for (size_t i = 0; i < mLoc; ++i)
                    std::copy_n(&Cloc[i * nLoc], nLoc, &C[(rowBegin + i) * N + colBegin]);
                continue;
            }
            MPI_Datatype blk;
            MPI_Type_vector(static_cast<int>(r1 - r0), static_cast<int>(c1 - c0), static_cast<int>(N),
                            MPI_DOUBLE, &blk);
            MPI_Type_commit(&blk);
            types.push_back(blk);
            reqs.emplace_back();
            MPI_Irecv(&C[r0 * N + c0], 1, blk, p, 0, MPI_COMM_WORLD, &reqs.back());
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        for (auto& t : types) MPI_Type_free(&t);
    } else if (mLoc > 0 && nLoc > 0) {
        MPI_Send(Cloc.data(), static_cast<int>(mLoc * nLoc), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> Afull(N * N), Bfull(N * N);
            initMatrix(Afull, N);
            initMatrix(Bfull, N);
            bool valid = validateResult(Afull, Bfull, C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
