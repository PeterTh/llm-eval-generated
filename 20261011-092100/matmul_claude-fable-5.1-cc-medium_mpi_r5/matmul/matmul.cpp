#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
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

// Initialize rows [rowBegin, rowBegin + numRows) of an NxN matrix into a local buffer.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t numRows) {
    for (size_t li = 0; li < numRows; ++li) {
        const size_t i = rowBegin + li;
        double* row = &mat[li * N];
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Local block multiply: C_local (numRows x N) = A_local (numRows x N) * B (N x N)
//
// Cache blocking over (j, k) with the B block packed once into contiguous
// 8-column panels (so the micro-kernel streams B linearly and row strides of
// N*8 bytes never alias in L1). A 4x8 tile of C is kept in vector registers
// over a whole k block. For every C element the k-sum is accumulated strictly
// in increasing k, so the rounding is identical to a plain sequential dot
// product (see -ffp-contract=off in CMakeLists.txt).
// ---------------------------------------------------------------------------
typedef double v4d __attribute__((vector_size(32)));

constexpr size_t NR = 8;    // tile width (one cache line of doubles)
constexpr size_t MR = 4;    // tile height
constexpr size_t BK = 256;  // k extent of a block
constexpr size_t BJ = 512;  // column extent of a block (packed B block = 1 MiB)

static inline v4d loadv(const double* p) {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
static inline void storev(double* p, const v4d v) { __builtin_memcpy(p, &v, sizeof(v)); }
static inline v4d bcast(const double x) { return v4d{x, x, x, x}; }

// Pack B[kBegin:kEnd, jBegin:jEnd) into panels of NR columns:
//   packed[(panel * kLen + k) * NR + c] = B[(kBegin + k) * N + jBegin + panel*NR + c]
// The last panel is zero-padded when the column count is not a multiple of NR.
static void packB(const double* __restrict__ B, double* __restrict__ packed, const size_t N,
                  const size_t kBegin, const size_t kEnd, const size_t jBegin, const size_t jEnd) {
    const size_t kLen = kEnd - kBegin;
    const size_t jLen = jEnd - jBegin;
    const size_t fullPanels = jLen / NR;
    for (size_t p = 0; p < fullPanels; ++p) {
        double* dst = packed + p * kLen * NR;
        const double* src = B + kBegin * N + jBegin + p * NR;
        for (size_t k = 0; k < kLen; ++k) {
            storev(dst + k * NR, loadv(src + k * N));
            storev(dst + k * NR + 4, loadv(src + k * N + 4));
        }
    }
    const size_t rest = jLen - fullPanels * NR;
    if (rest > 0) {
        double* dst = packed + fullPanels * kLen * NR;
        const double* src = B + kBegin * N + jBegin + fullPanels * NR;
        for (size_t k = 0; k < kLen; ++k) {
            size_t c = 0;
            for (; c < rest; ++c) dst[k * NR + c] = src[k * N + c];
            for (; c < NR; ++c) dst[k * NR + c] = 0.0;
        }
    }
}

// Update an R x NR tile of C (R rows starting at C, row stride ldc) with the
// packed panel for kLen consecutive k values. A points at row 0 (row stride
// lda) of the current k block.
template <size_t R>
static inline void microKernel(const double* __restrict__ A, const size_t lda,
                               const double* __restrict__ Bp, double* __restrict__ C,
                               const size_t ldc, const size_t kLen) {
    v4d acc0[R], acc1[R];
    for (size_t r = 0; r < R; ++r) {
        acc0[r] = loadv(C + r * ldc);
        acc1[r] = loadv(C + r * ldc + 4);
    }
    for (size_t k = 0; k < kLen; ++k) {
        const v4d b0 = loadv(Bp + k * NR);
        const v4d b1 = loadv(Bp + k * NR + 4);
        for (size_t r = 0; r < R; ++r) {
            const v4d a = bcast(A[r * lda + k]);
            acc0[r] += a * b0;
            acc1[r] += a * b1;
        }
    }
    for (size_t r = 0; r < R; ++r) {
        storev(C + r * ldc, acc0[r]);
        storev(C + r * ldc + 4, acc1[r]);
    }
}

// Same as microKernel but for a partial (ncols < NR) last panel: the panel is
// zero-padded, so compute on a full-width temporary and copy back ncols values.
template <size_t R>
static inline void microKernelEdge(const double* __restrict__ A, const double* __restrict__ Bp,
                                   double* __restrict__ C, const size_t N, const size_t kLen,
                                   const size_t ncols) {
    alignas(32) double tmp[R][NR];
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < NR; ++c) tmp[r][c] = c < ncols ? C[r * N + c] : 0.0;
    microKernel<R>(A, N, Bp, &tmp[0][0], NR, kLen);
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < ncols; ++c) C[r * N + c] = tmp[r][c];
}

// Apply all panels of a packed block to R rows of C.
template <size_t R>
static inline void rowPanel(const double* __restrict__ A, const double* __restrict__ Bp,
                            double* __restrict__ C, const size_t N, const size_t jLen,
                            const size_t kLen) {
    const size_t fullPanels = jLen / NR;
    for (size_t p = 0; p < fullPanels; ++p)
        microKernel<R>(A, N, Bp + p * kLen * NR, C + p * NR, N, kLen);
    const size_t rest = jLen - fullPanels * NR;
    if (rest > 0)
        microKernelEdge<R>(A, Bp + fullPanels * kLen * NR, C + fullPanels * NR, N, kLen, rest);
}

void matrixMultiplyRows(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t N, const size_t numRows) {
    std::fill(C.begin(), C.begin() + numRows * N, 0.0);
    if (numRows == 0) return;

    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    // Packed B block buffer: BK x (BJ rounded up to a multiple of NR).
    std::vector<double> packed(BK * ((std::min(BJ, N) + NR - 1) / NR) * NR);

    for (size_t jj = 0; jj < N; jj += BJ) {
        const size_t jEnd = std::min(N, jj + BJ);
        const size_t jLen = jEnd - jj;
        for (size_t kk = 0; kk < N; kk += BK) {
            const size_t kEnd = std::min(N, kk + BK);
            const size_t kLen = kEnd - kk;
            packB(b, packed.data(), N, kk, kEnd, jj, jEnd);

            size_t i = 0;
            for (; i + MR <= numRows; i += MR)
                rowPanel<MR>(a + i * N + kk, packed.data(), c + i * N + jj, N, jLen, kLen);
            switch (numRows - i) {
                case 3: rowPanel<3>(a + i * N + kk, packed.data(), c + i * N + jj, N, jLen, kLen); break;
                case 2: rowPanel<2>(a + i * N + kk, packed.data(), c + i * N + jj, N, jLen, kLen); break;
                case 1: rowPanel<1>(a + i * N + kk, packed.data(), c + i * N + jj, N, jLen, kLen); break;
                default: break;
            }
        }
    }
}

// Simple validation: compute a single element and compare.
// A is reconstructed from the deterministic generator; B and C are full matrices.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    // Parse command line arguments (every rank sees the same argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // 1D row decomposition of A and C; the first (N % size) ranks get one extra row.
    const size_t P = static_cast<size_t>(size);
    const size_t base = N / P;
    const size_t rem = N % P;
    const size_t r = static_cast<size_t>(rank);
    const size_t localRows = base + (r < rem ? 1 : 0);
    const size_t rowBegin = r * base + std::min(r, rem);

    // Allocate matrices: local row blocks of A and C, full B on every rank.
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);

    // Initialize matrices. The generator is deterministic, so each rank builds
    // its own slice of A and its own copy of B without any communication.
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(A, N, rowBegin, localRows);
    initMatrixRows(B, N, 0, N);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    matrixMultiplyRows(A, B, C, N, localRows);

    const double tLocal = MPI_Wtime() - tStart;
    double tMax = tLocal;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(tMax * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C on rank 0 only if it is needed for output or validation.
    if (printResults || validate) {
        std::vector<double> Cfull;
        if (rank == 0) Cfull.resize(N * N);

        // Use a row datatype so counts/displacements are in rows (avoids int overflow).
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(P), displs(P);
        for (size_t p = 0; p < P; ++p) {
            counts[p] = static_cast<int>(base + (p < rem ? 1 : 0));
            displs[p] = static_cast<int>(p * base + std::min(p, rem));
        }

        MPI_Gatherv(C.data(), static_cast<int>(localRows), rowType,
                    rank == 0 ? Cfull.data() : nullptr, counts.data(), displs.data(), rowType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(Cfull, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(B, Cfull, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
