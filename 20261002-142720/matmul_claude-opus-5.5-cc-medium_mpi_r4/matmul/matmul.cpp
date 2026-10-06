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
// Distributed matrix multiplication (MPI).
//
// The ranks form a 2D Pr x Pc grid and C is partitioned into Pr x Pc blocks,
// one per rank. Rank (r, c) needs only the r-th row block of A and the c-th
// column block of B; both are generated locally from the deterministic
// initializer, so no input communication is required. The C blocks are
// received directly into place on rank 0. The 2D layout keeps per-rank input
// traffic at O(N^2 / sqrt(P)) instead of O(N^2) for a 1D row split.
//
// The local kernel is a packed, cache-blocked, register-tiled GEMM. For every
// element C[i][j] the products A[i][k]*B[k][j] are accumulated in strictly
// increasing k order starting from 0.0, exactly like the reference triple
// loop, so the results are bitwise identical to the serial code.
//
// Rounding: the file is built with -ffp-contract=off so every step is a
// separately rounded multiply and add, matching the reference build (GCC
// vectorizes the reference k loop in pairs with separate mul/add). For odd N
// the reference handles the final k term in a scalar epilogue that is
// contracted into an FMA on FMA-capable targets; that is reproduced here.
// ---------------------------------------------------------------------------

typedef double v4d __attribute__((vector_size(32)));

constexpr size_t MR = 6;     // rows per micro tile
constexpr size_t NR = 8;     // cols per micro tile (2 x 4 doubles)
constexpr size_t KC = 256;   // k block
constexpr size_t MC = 96;    // row block (multiple of MR)
constexpr size_t NC = 2048;  // column block (multiple of NR)

static inline size_t roundUp(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Packed-buffer sizes. Buffers must be zero-initialized once; packing only
// writes the non-padding entries, so the padding stays zero.
static inline size_t packedASize(size_t rows, size_t N) { return roundUp(rows, MR) * N; }
static inline size_t packedBSize(size_t cols, size_t N) { return roundUp(cols, NR) * N; }

// Pack B[0:N, 0:cols] (leading dim ldb) into column panels: Bp[panel][k][NR].
static void packB(const double* B, const size_t ldb, double* Bp, const size_t cols, const size_t N) {
    const size_t panels = roundUp(cols, NR) / NR;
    for (size_t p = 0; p < panels; ++p) {
        const size_t j0 = p * NR;
        const size_t w = std::min(NR, cols - j0);
        double* dst = Bp + p * N * NR;
        for (size_t k = 0; k < N; ++k) {
            const double* src = B + k * ldb + j0;
            for (size_t c = 0; c < w; ++c) dst[k * NR + c] = src[c];
        }
    }
}

// Pack A[0:rows, 0:N] (leading dim lda) into row panels: Ap[panel][k][MR].
static void packA(const double* A, const size_t lda, double* Ap, const size_t rows, const size_t N) {
    const size_t panels = roundUp(rows, MR) / MR;
    for (size_t p = 0; p < panels; ++p) {
        const size_t i0 = p * MR;
        const size_t h = std::min(MR, rows - i0);
        double* dst = Ap + p * N * MR;
        for (size_t r = 0; r < h; ++r) {
            const double* src = A + (i0 + r) * lda;
            for (size_t k = 0; k < N; ++k) dst[k * MR + r] = src[k];
        }
    }
}

// acc[MR x NR] (in C) += Ap[kc x MR] * Bp[kc x NR], k in increasing order.
static inline void microKernel(const size_t kc, const double* __restrict a, const double* __restrict b,
                               double* __restrict c, const size_t ldc, const size_t h, const size_t w,
                               const bool fuseLast) {
    v4d acc[MR][2];
    if (h == MR && w == NR) {
        for (size_t r = 0; r < MR; ++r) {
            memcpy(&acc[r][0], c + r * ldc, sizeof(v4d));
            memcpy(&acc[r][1], c + r * ldc + 4, sizeof(v4d));
        }
    } else {
        double tmp[MR][NR] = {};
        for (size_t r = 0; r < h; ++r)
            for (size_t q = 0; q < w; ++q) tmp[r][q] = c[r * ldc + q];
        for (size_t r = 0; r < MR; ++r) {
            memcpy(&acc[r][0], &tmp[r][0], sizeof(v4d));
            memcpy(&acc[r][1], &tmp[r][4], sizeof(v4d));
        }
    }

    const size_t kPlain = fuseLast ? kc - 1 : kc;
    for (size_t k = 0; k < kPlain; ++k) {
        v4d b0, b1;
        memcpy(&b0, b + k * NR, sizeof(v4d));
        memcpy(&b1, b + k * NR + 4, sizeof(v4d));
        const double* ak = a + k * MR;
#pragma GCC unroll 6
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = {ak[r], ak[r], ak[r], ak[r]};
            acc[r][0] += av * b0;
            acc[r][1] += av * b1;
        }
    }
    if (fuseLast) {
        const size_t k = kc - 1;
        v4d b0, b1;
        memcpy(&b0, b + k * NR, sizeof(v4d));
        memcpy(&b1, b + k * NR + 4, sizeof(v4d));
        const double* ak = a + k * MR;
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = {ak[r], ak[r], ak[r], ak[r]};
#ifdef __FMA__
            acc[r][0] = (v4d)_mm256_fmadd_pd((__m256d)av, (__m256d)b0, (__m256d)acc[r][0]);
            acc[r][1] = (v4d)_mm256_fmadd_pd((__m256d)av, (__m256d)b1, (__m256d)acc[r][1]);
#else
            acc[r][0] += av * b0;
            acc[r][1] += av * b1;
#endif
        }
    }

    if (h == MR && w == NR) {
        for (size_t r = 0; r < MR; ++r) {
            memcpy(c + r * ldc, &acc[r][0], sizeof(v4d));
            memcpy(c + r * ldc + 4, &acc[r][1], sizeof(v4d));
        }
    } else {
        double tmp[MR][NR];
        for (size_t r = 0; r < MR; ++r) {
            memcpy(&tmp[r][0], &acc[r][0], sizeof(v4d));
            memcpy(&tmp[r][4], &acc[r][1], sizeof(v4d));
        }
        for (size_t r = 0; r < h; ++r)
            for (size_t q = 0; q < w; ++q) c[r * ldc + q] = tmp[r][q];
    }
}

// Computes the local block C[0:rows, 0:cols] = A[0:rows, :] * B[:, 0:cols].
// Ap / Bp are zero-initialized workspaces of packedASize / packedBSize.
void matrixMultiplyLocal(const double* A, const size_t lda, const double* B, const size_t ldb,
                         double* C, const size_t ldc, const size_t rows, const size_t cols,
                         const size_t N, double* Ap, double* Bp) {
    if (rows == 0 || cols == 0 || N == 0) return;
    packA(A, lda, Ap, rows, N);
    packB(B, ldb, Bp, cols, N);

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) C[i * ldc + j] = 0.0;

    for (size_t jc = 0; jc < cols; jc += NC) {
        const size_t jcEnd = std::min(cols, jc + NC);
        for (size_t pc = 0; pc < N; pc += KC) {
            const size_t kc = std::min(KC, N - pc);
            const bool fuseLast = (N % 2 == 1) && (pc + kc == N);
            for (size_t ic = 0; ic < rows; ic += MC) {
                const size_t icEnd = std::min(rows, ic + MC);
                for (size_t jr = jc; jr < jcEnd; jr += NR) {
                    const size_t w = std::min(NR, cols - jr);
                    const double* bp = Bp + (jr / NR) * N * NR + pc * NR;
                    for (size_t ir = ic; ir < icEnd; ir += MR) {
                        const size_t h = std::min(MR, rows - ir);
                        const double* ap = Ap + (ir / MR) * N * MR + pc * MR;
                        microKernel(kc, ap, bp, C + ir * ldc + jr, ldc, h, w, fuseLast);
                    }
                }
            }
        }
    }
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

    // 2D process grid Pr x Pc (Pc = largest divisor of P not above sqrt(P))
    int Pc = 1;
    for (int d = 1; d * d <= nprocs; ++d)
        if (nprocs % d == 0) Pc = d;
    const int Pr = nprocs / Pc;
    auto blockLo = [N](int b, int nb) { return N * static_cast<size_t>(b) / static_cast<size_t>(nb); };

    const int gridRow = rank / Pc, gridCol = rank % Pc;
    const size_t row0 = blockLo(gridRow, Pr), rows = blockLo(gridRow + 1, Pr) - row0;
    const size_t col0 = blockLo(gridCol, Pc), cols = blockLo(gridCol + 1, Pc) - col0;

    // Rank 0 holds full A, B and C (for output/validation); other ranks hold
    // only their A row block, B column block and C block.
    std::vector<double> A(root ? N * N : rows * N);
    std::vector<double> B(root ? N * N : N * cols);
    std::vector<double> C(root ? N * N : rows * cols);
    
    // Initialize matrices (each rank generates the data it needs locally)
    if (root) printf("Initializing matrices...\n");
    if (root) {
        initMatrix(A, N);
        initMatrix(B, N);
    } else {
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, row0 + i, k);
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < cols; ++j)
                B[k * cols + j] = getPseudoRndValue(N, k, col0 + j);
    }
    const double* myA = root ? A.data() + row0 * N : A.data();
    const double* myB = root ? B.data() + col0 : B.data();
    const size_t ldb = root ? N : cols;
    double* myC = root ? C.data() + row0 * N + col0 : C.data();
    const size_t ldc = root ? N : cols;

    // Packing workspaces (zero-initialized once, outside the timed region)
    std::vector<double> Ap(packedASize(rows, N));
    std::vector<double> Bp(packedBSize(cols, N));

    // Datatypes for collecting the C blocks on rank 0
    MPI_Datatype rowType = MPI_DATATYPE_NULL;
    std::vector<MPI_Datatype> blockTypes;
    std::vector<MPI_Request> requests;
    std::vector<size_t> blockOffsets;
    std::vector<int> blockSources;
    if (root) {
        for (int p = 1; p < nprocs; ++p) {
            const size_t r0 = blockLo(p / Pc, Pr), nr = blockLo(p / Pc + 1, Pr) - r0;
            const size_t c0 = blockLo(p % Pc, Pc), nc = blockLo(p % Pc + 1, Pc) - c0;
            if (nr == 0 || nc == 0) continue;
            MPI_Datatype t;
            MPI_Type_vector(static_cast<int>(nr), static_cast<int>(nc), static_cast<int>(N), MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            blockTypes.push_back(t);
            blockOffsets.push_back(r0 * N + c0);
            blockSources.push_back(p);
            requests.push_back(MPI_REQUEST_NULL);
        }
    } else if (rows > 0 && cols > 0) {
        MPI_Type_contiguous(static_cast<int>(cols), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);
    }

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (root) {
        for (size_t b = 0; b < blockSources.size(); ++b)
            MPI_Irecv(C.data() + blockOffsets[b], 1, blockTypes[b], blockSources[b], 0,
                      MPI_COMM_WORLD, &requests[b]);
    }
    matrixMultiplyLocal(myA, N, myB, ldb, myC, ldc, rows, cols, N, Ap.data(), Bp.data());
    if (root) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    } else if (rows > 0 && cols > 0) {
        MPI_Send(C.data(), static_cast<int>(rows), rowType, 0, 0, MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(globalDuration);

    for (auto& t : blockTypes) MPI_Type_free(&t);
    if (rowType != MPI_DATATYPE_NULL) MPI_Type_free(&rowType);

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
            bool valid = validateResult(A, B, C, N);
            
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
