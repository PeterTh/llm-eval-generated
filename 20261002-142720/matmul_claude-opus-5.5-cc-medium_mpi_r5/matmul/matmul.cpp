#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <immintrin.h>
#include <mpi.h>

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
// Blocked, packed matrix multiply kernel.
//
// Every element C[i][j] is accumulated exactly like the reference loop:
// starting at 0.0, adding the rounded product A[i][k]*B[k][j] for k = 0..N-1
// in ascending order (separate multiply and add, no reassociation). For odd N
// the final term is fused (fma), matching the reference binary's code
// generation, so results are bitwise identical. Blocking only changes the
// point in time at which partial sums are stored/reloaded (exact for double).
// ---------------------------------------------------------------------------
constexpr size_t MR = 6;    // rows per micro-tile
constexpr size_t NR = 8;    // columns per micro-tile (2 x AVX2 vectors)
constexpr size_t KC = 256;  // k-block (packed B micro-panel stays in L1)
constexpr size_t MC = 96;   // row block (packed A block stays in L2)
constexpr size_t NC = 1536; // column block (packed B panel in L3)

// C tile (MR x NR, row stride ldc) += Apanel * Bpanel over kc steps.
static inline void microKernel(const size_t kc, const double* __restrict__ a,
                               const double* __restrict__ b, double* __restrict__ c,
                               const size_t ldc, const bool fuseLast) {
    __m256d c00 = _mm256_loadu_pd(c + 0 * ldc), c01 = _mm256_loadu_pd(c + 0 * ldc + 4);
    __m256d c10 = _mm256_loadu_pd(c + 1 * ldc), c11 = _mm256_loadu_pd(c + 1 * ldc + 4);
    __m256d c20 = _mm256_loadu_pd(c + 2 * ldc), c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
    __m256d c30 = _mm256_loadu_pd(c + 3 * ldc), c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
    __m256d c40 = _mm256_loadu_pd(c + 4 * ldc), c41 = _mm256_loadu_pd(c + 4 * ldc + 4);
    __m256d c50 = _mm256_loadu_pd(c + 5 * ldc), c51 = _mm256_loadu_pd(c + 5 * ldc + 4);

    const size_t kn = fuseLast ? kc - 1 : kc;
    for (size_t k = 0; k < kn; ++k) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        __m256d av;
#define MM_ROW(r)                                              \
        av = _mm256_broadcast_sd(a + r);                       \
        c##r##0 = _mm256_add_pd(c##r##0, _mm256_mul_pd(av, b0)); \
        c##r##1 = _mm256_add_pd(c##r##1, _mm256_mul_pd(av, b1));
        MM_ROW(0) MM_ROW(1) MM_ROW(2) MM_ROW(3) MM_ROW(4) MM_ROW(5)
#undef MM_ROW
        a += MR;
        b += NR;
    }
    if (fuseLast) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        __m256d av;
#define MM_ROW(r)                                          \
        av = _mm256_broadcast_sd(a + r);                   \
        c##r##0 = _mm256_fmadd_pd(av, b0, c##r##0);        \
        c##r##1 = _mm256_fmadd_pd(av, b1, c##r##1);
        MM_ROW(0) MM_ROW(1) MM_ROW(2) MM_ROW(3) MM_ROW(4) MM_ROW(5)
#undef MM_ROW
    }

    _mm256_storeu_pd(c + 0 * ldc, c00); _mm256_storeu_pd(c + 0 * ldc + 4, c01);
    _mm256_storeu_pd(c + 1 * ldc, c10); _mm256_storeu_pd(c + 1 * ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20); _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30); _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40); _mm256_storeu_pd(c + 4 * ldc + 4, c41);
    _mm256_storeu_pd(c + 5 * ldc, c50); _mm256_storeu_pd(c + 5 * ldc + 4, c51);
}

// Pack B[pc:pc+kc, jc:jc+nc] (row stride ldb) into NR-wide strips (zero padded).
static void packB(const double* B, const size_t ldb, const size_t pc, const size_t kc,
                  const size_t jc, const size_t nc, double* __restrict__ bp) {
    for (size_t jr = 0; jr < nc; jr += NR) {
        const size_t nr = std::min(NR, nc - jr);
        for (size_t k = 0; k < kc; ++k) {
            const double* src = B + (pc + k) * ldb + jc + jr;
            size_t j = 0;
            for (; j < nr; ++j) bp[j] = src[j];
            for (; j < NR; ++j) bp[j] = 0.0;
            bp += NR;
        }
    }
}

// Pack A[ic:ic+mc, pc:pc+kc] (row stride lda) into MR-tall strips (zero padded).
static void packA(const double* A, const size_t lda, const size_t ic, const size_t mc,
                  const size_t pc, const size_t kc, double* __restrict__ ap) {
    for (size_t ir = 0; ir < mc; ir += MR) {
        const size_t mr = std::min(MR, mc - ir);
        for (size_t k = 0; k < kc; ++k) {
            size_t i = 0;
            for (; i < mr; ++i) ap[i] = A[(ic + ir + i) * lda + pc + k];
            for (; i < MR; ++i) ap[i] = 0.0;
            ap += MR;
        }
    }
}

// C (rows x cols) = A (rows x N) * B (N x cols); row-major with leading
// dimensions lda, ldb, ldc.
void matrixMultiply(const double* A, const size_t lda, const double* B, const size_t ldb,
                    double* C, const size_t ldc, const size_t rows, const size_t cols,
                    const size_t N) {
    if (rows == 0 || cols == 0) return;
    for (size_t i = 0; i < rows; ++i) std::fill(C + i * ldc, C + i * ldc + cols, 0.0);
    if (N == 0) return;

    const size_t ncMax = std::min(NC, (cols + NR - 1) / NR * NR);
    const size_t kcMax = std::min(KC, N);
    const size_t mcMax = std::min(MC, (rows + MR - 1) / MR * MR);
    auto* bp = static_cast<double*>(std::aligned_alloc(64, ((ncMax * kcMax * 8 + 63) / 64) * 64));
    auto* ap = static_cast<double*>(std::aligned_alloc(64, ((mcMax * kcMax * 8 + 63) / 64) * 64));
    alignas(32) double tile[MR * NR];

    for (size_t jc = 0; jc < cols; jc += NC) {
        const size_t nc = std::min(NC, cols - jc);
        for (size_t pc = 0; pc < N; pc += KC) {
            const size_t kc = std::min(KC, N - pc);
            const bool fuseLast = (N & 1) && (pc + kc == N);
            packB(B, ldb, pc, kc, jc, nc, bp);
            for (size_t ic = 0; ic < rows; ic += MC) {
                const size_t mc = std::min(MC, rows - ic);
                packA(A, lda, ic, mc, pc, kc, ap);
                for (size_t jr = 0; jr < nc; jr += NR) {
                    const size_t nr = std::min(NR, nc - jr);
                    const double* bstrip = bp + jr * kc;
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        const size_t mr = std::min(MR, mc - ir);
                        const double* astrip = ap + ir * kc;
                        double* cptr = C + (ic + ir) * ldc + jc + jr;
                        if (mr == MR && nr == NR) {
                            microKernel(kc, astrip, bstrip, cptr, ldc, fuseLast);
                        } else {
                            for (size_t i = 0; i < MR; ++i)
                                for (size_t j = 0; j < NR; ++j)
                                    tile[i * NR + j] = (i < mr && j < nr) ? cptr[i * ldc + j] : 0.0;
                            microKernel(kc, astrip, bstrip, tile, NR, fuseLast);
                            for (size_t i = 0; i < mr; ++i)
                                for (size_t j = 0; j < nr; ++j)
                                    cptr[i * ldc + j] = tile[i * NR + j];
                        }
                    }
                }
            }
        }
    }
    std::free(ap);
    std::free(bp);
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
    
    // Parse command line arguments (every rank parses identically)
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

    // 2D block decomposition of C over a pr x pc process grid. Rank (r, c)
    // computes C[rows_r, cols_c] and needs only A[rows_r, :] and B[:, cols_c],
    // which it generates locally (no input communication required).
    int pr = 1;
    for (int d = 1; d * d <= nprocs; ++d)
        if (nprocs % d == 0) pr = nprocs / d;   // pr >= pc, as square as possible
    const int pc = nprocs / pr;
    // Rows split evenly; columns split in units of NR to keep tiles full.
    auto split = [](size_t n, int parts, size_t unit, int idx, size_t& first, size_t& cnt) {
        const size_t units = (n + unit - 1) / unit;
        const size_t base = units / parts, rem = units % parts;
        const size_t u0 = base * idx + std::min(static_cast<size_t>(idx), rem);
        const size_t u1 = u0 + base + (static_cast<size_t>(idx) < rem ? 1 : 0);
        first = std::min(u0 * unit, n);
        cnt = std::min(u1 * unit, n) - first;
    };
    auto blockOf = [&](int p, size_t& r0, size_t& nr, size_t& c0, size_t& nc) {
        split(N, pr, 1, p / pc, r0, nr);
        split(N, pc, NR, p % pc, c0, nc);
    };
    size_t myR0, myRows, myC0, myCols;
    blockOf(rank, myR0, myRows, myC0, myCols);

    // Allocate matrices: root holds full A, B and C (for output/validation),
    // other ranks only the blocks they need.
    std::vector<double> A, B, C;
    if (root) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
    } else {
        A.resize(myRows * N);
        B.resize(N * myCols);
        C.resize(myRows * myCols);
    }
    
    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    if (root) {
        initMatrix(A, N);
        initMatrix(B, N);
    } else {
        for (size_t i = 0; i < myRows; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, myR0 + i, k);
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < myCols; ++j)
                B[k * myCols + j] = getPseudoRndValue(N, k, myC0 + j);
    }
    const size_t lda = N;
    const size_t ldb = root ? N : myCols;
    const size_t ldc = root ? N : myCols;
    const double* myA = A.data() + (root ? myR0 * N : 0);
    const double* myB = B.data() + (root ? myC0 : 0);
    double* myC = C.data() + (root ? myR0 * N + myC0 : 0);

    // Root receives each remote block straight into its place in C via a
    // strided vector datatype.
    std::vector<MPI_Datatype> blockTypes;
    std::vector<MPI_Request> reqs;
    if (root) {
        for (int p = 1; p < nprocs; ++p) {
            size_t r0, nr, c0, nc;
            blockOf(p, r0, nr, c0, nc);
            if (nr == 0 || nc == 0) continue;
            MPI_Datatype t;
            MPI_Type_vector(static_cast<int>(nr), static_cast<int>(nc), static_cast<int>(N),
                            MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            blockTypes.push_back(t);
        }
        reqs.reserve(blockTypes.size());
    }
    
    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Collect the distributed blocks of C on the root rank
    if (root) {
        size_t ti = 0;
        for (int p = 1; p < nprocs; ++p) {
            size_t r0, nr, c0, nc;
            blockOf(p, r0, nr, c0, nc);
            if (nr == 0 || nc == 0) continue;
            reqs.emplace_back();
            MPI_Irecv(C.data() + r0 * N + c0, 1, blockTypes[ti++], p, 0, MPI_COMM_WORLD,
                      &reqs.back());
        }
    }

    matrixMultiply(myA, lda, myB, ldb, myC, ldc, myRows, myCols, N);

    if (root) {
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else if (myRows > 0 && myCols > 0) {
        MPI_Send(myC, static_cast<int>(myRows * myCols), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    for (auto& t : blockTypes) MPI_Type_free(&t);

    int exitCode = 0;
    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
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
            bool valid = validateResult(A, B, C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        fflush(stdout);
    }
    
    MPI_Finalize();
    return exitCode;
}
