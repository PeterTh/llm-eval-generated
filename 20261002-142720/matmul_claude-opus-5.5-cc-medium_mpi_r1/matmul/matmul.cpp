#include <mpi.h>

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
// Distributed matrix multiplication
//
// The ranks form a 2D Pr x Pc process grid and C is distributed in 2D blocks:
// rank (pr, pc) computes C[rows(pr), cols(pc)] = A[rows(pr), :] * B[:, cols(pc)].
// Initialization is deterministic, so every rank generates exactly the parts of
// A and B it needs (no scatter/broadcast). The 2D layout minimizes the operand
// data each rank streams through memory (N^2 * (1/Pr + 1/Pc) instead of N^2 for
// a 1D row split), which keeps the computation compute-bound at high core
// counts. Operands are generated directly in packed panel layout so the
// register-blocked micro-kernel streams contiguous data. Finally the C blocks
// are gathered on rank 0 straight into the full matrix via strided datatypes.
//
// Each C element is accumulated as a multiply-then-add chain over k in
// ascending order starting from 0, exactly like the original serial loop, so
// results are bitwise identical.
// ---------------------------------------------------------------------------

constexpr size_t MR = 6;    // micro-tile rows (A panel width)
constexpr size_t NR = 8;    // micro-tile cols (B panel width)
constexpr size_t KC = 512;  // k-block (B micro-panel stays in L1/L2)
constexpr size_t MC = 24;   // row block (A block stays in L2)
constexpr size_t NC = 256;  // column block (B block stays in L2/L3)

typedef double v4d __attribute__((vector_size(32), aligned(8)));

static inline v4d loadu(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline void storeu(double* p, const v4d& v) { std::memcpy(p, &v, sizeof(v)); }
static inline v4d bcast(const double x) { return v4d{x, x, x, x}; }

// C[0:6, 0:8] (row stride ldc) += Ap[k][0:6] * Bp[k][0:8] for k in [0, kc)
static inline void microKernel(const size_t kc, const double* __restrict Ap,
                               const double* __restrict Bp, double* __restrict C,
                               const size_t ldc) {
    v4d c00 = loadu(C + 0 * ldc), c01 = loadu(C + 0 * ldc + 4);
    v4d c10 = loadu(C + 1 * ldc), c11 = loadu(C + 1 * ldc + 4);
    v4d c20 = loadu(C + 2 * ldc), c21 = loadu(C + 2 * ldc + 4);
    v4d c30 = loadu(C + 3 * ldc), c31 = loadu(C + 3 * ldc + 4);
    v4d c40 = loadu(C + 4 * ldc), c41 = loadu(C + 4 * ldc + 4);
    v4d c50 = loadu(C + 5 * ldc), c51 = loadu(C + 5 * ldc + 4);

    for (size_t k = 0; k < kc; ++k) {
        const v4d b0 = loadu(Bp);
        const v4d b1 = loadu(Bp + 4);
        v4d a;
        a = bcast(Ap[0]); c00 += a * b0; c01 += a * b1;
        a = bcast(Ap[1]); c10 += a * b0; c11 += a * b1;
        a = bcast(Ap[2]); c20 += a * b0; c21 += a * b1;
        a = bcast(Ap[3]); c30 += a * b0; c31 += a * b1;
        a = bcast(Ap[4]); c40 += a * b0; c41 += a * b1;
        a = bcast(Ap[5]); c50 += a * b0; c51 += a * b1;
        Ap += MR;
        Bp += NR;
    }

    storeu(C + 0 * ldc, c00); storeu(C + 0 * ldc + 4, c01);
    storeu(C + 1 * ldc, c10); storeu(C + 1 * ldc + 4, c11);
    storeu(C + 2 * ldc, c20); storeu(C + 2 * ldc + 4, c21);
    storeu(C + 3 * ldc, c30); storeu(C + 3 * ldc + 4, c31);
    storeu(C + 4 * ldc, c40); storeu(C + 4 * ldc + 4, c41);
    storeu(C + 5 * ldc, c50); storeu(C + 5 * ldc + 4, c51);
}

// Pack rows [row0, row0 + rows) of A into panels of MR rows:
// Ap[p * N * MR + k * MR + r] = A[row0 + p * MR + r][k] (zero padded)
void initPackedA(std::vector<double>& Ap, const size_t N, const size_t row0, const size_t rows) {
    const size_t panels = (rows + MR - 1) / MR;
    Ap.assign(std::max<size_t>(panels * N * MR, 1), 0.0);
    for (size_t p = 0; p < panels; ++p) {
        for (size_t r = 0; r < MR; ++r) {
            const size_t li = p * MR + r;
            if (li >= rows) break;
            const size_t i = row0 + li;
            double* dst = Ap.data() + p * N * MR + r;
            for (size_t k = 0; k < N; ++k) dst[k * MR] = getPseudoRndValue(N, i, k);
        }
    }
}

// Pack columns [col0, col0 + cols) of B into panels of NR columns:
// Bp[p * N * NR + k * NR + c] = B[k][col0 + p * NR + c] (zero padded)
void initPackedB(std::vector<double>& Bp, const size_t N, const size_t col0, const size_t cols) {
    const size_t panels = (cols + NR - 1) / NR;
    Bp.assign(std::max<size_t>(panels * N * NR, 1), 0.0);
    for (size_t p = 0; p < panels; ++p) {
        double* panel = Bp.data() + p * N * NR;
        const size_t nc = std::min(NR, cols - p * NR);
        for (size_t k = 0; k < N; ++k) {
            for (size_t c = 0; c < nc; ++c) {
                panel[k * NR + c] = getPseudoRndValue(N, k, col0 + p * NR + c);
            }
        }
    }
}

// C (rows x cols, row-major, leading dimension cols) = A_block * B_block
void matrixMultiplyLocal(const double* Ap, const double* Bp, double* C, const size_t N,
                         const size_t rows, const size_t cols) {
    std::fill(C, C + rows * cols, 0.0);
    double tmp[MR * NR];
    const size_t ldc = cols;

    for (size_t jc = 0; jc < cols; jc += NC) {
        const size_t jend = std::min(jc + NC, cols);
        for (size_t pc = 0; pc < N; pc += KC) {
            const size_t kc = std::min(KC, N - pc);
            for (size_t ic = 0; ic < rows; ic += MC) {
                const size_t iend = std::min(ic + MC, rows);
                for (size_t jr = jc; jr < jend; jr += NR) {
                    const size_t nr = std::min(NR, cols - jr);
                    const double* bp = Bp + (jr / NR) * N * NR + pc * NR;
                    for (size_t ir = ic; ir < iend; ir += MR) {
                        const size_t mr = std::min(MR, rows - ir);
                        const double* ap = Ap + (ir / MR) * N * MR + pc * MR;
                        double* c = C + ir * ldc + jr;
                        if (mr == MR && nr == NR) {
                            microKernel(kc, ap, bp, c, ldc);
                        } else {
                            for (size_t r = 0; r < MR; ++r)
                                for (size_t q = 0; q < NR; ++q)
                                    tmp[r * NR + q] = (r < mr && q < nr) ? c[r * ldc + q] : 0.0;
                            microKernel(kc, ap, bp, tmp, NR);
                            for (size_t r = 0; r < mr; ++r)
                                for (size_t q = 0; q < nr; ++q) c[r * ldc + q] = tmp[r * NR + q];
                        }
                    }
                }
            }
        }
    }
}

// Split `units` blocks of `unit` elements (total n elements) over `parts`; part i
// gets elements [lo, hi). Splitting in micro-panel units avoids partial tiles.
static void splitRange(const size_t n, const size_t unit, const int parts, const int i,
                       size_t& lo, size_t& hi) {
    const size_t units = (n + unit - 1) / unit;
    lo = std::min(n, units * i / parts * unit);
    hi = std::min(n, units * (i + 1) / parts * unit);
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

    // 2D process grid (dims[0] >= dims[1])
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    const int gridRows = dims[0], gridCols = dims[1];
    auto blockOf = [&](const int r, size_t& r0, size_t& nr, size_t& c0, size_t& nc) {
        size_t hi;
        splitRange(N, MR, gridRows, r / gridCols, r0, hi);
        nr = hi - r0;
        splitRange(N, NR, gridCols, r % gridCols, c0, hi);
        nc = hi - c0;
    };
    size_t row0, rows, col0, cols;
    blockOf(rank, row0, rows, col0, cols);

    // Local storage
    std::vector<double> Ap, Bp;
    std::vector<double> Clocal(std::max<size_t>(rows * cols, 1));
    std::vector<double> C;
    if (root) C.resize(N * N);
    
    // Initialize matrices (each rank generates only the blocks it needs)
    if (root) printf("Initializing matrices...\n");
    initPackedA(Ap, N, row0, rows);
    initPackedB(Bp, N, col0, cols);

    // Receive datatypes on rank 0: strided view of each rank's block in C
    std::vector<MPI_Datatype> recvTypes;
    std::vector<MPI_Request> requests;
    if (root) {
        for (int r = 1; r < nprocs; ++r) {
            size_t r0, nr, c0, nc;
            blockOf(r, r0, nr, c0, nc);
            if (nr == 0 || nc == 0) continue;
            MPI_Datatype t;
            MPI_Type_vector(static_cast<int>(nr), static_cast<int>(nc), static_cast<int>(N),
                            MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            recvTypes.push_back(t);
        }
        requests.reserve(recvTypes.size());
    }
    MPI_Datatype rowType = MPI_DATATYPE_NULL;
    if (!root && rows > 0 && cols > 0) {
        MPI_Type_contiguous(static_cast<int>(cols), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);
    }
    
    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (root) {
        // Post receives early so remote blocks land directly in C
        size_t t = 0;
        for (int r = 1; r < nprocs; ++r) {
            size_t r0, nr, c0, nc;
            blockOf(r, r0, nr, c0, nc);
            if (nr == 0 || nc == 0) continue;
            requests.emplace_back();
            MPI_Irecv(C.data() + r0 * N + c0, 1, recvTypes[t++], r, 0, MPI_COMM_WORLD,
                      &requests.back());
        }
    }

    matrixMultiplyLocal(Ap.data(), Bp.data(), Clocal.data(), N, rows, cols);

    if (root) {
        for (size_t i = 0; i < rows; ++i)
            std::memcpy(C.data() + (row0 + i) * N + col0, Clocal.data() + i * cols,
                        cols * sizeof(double));
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    } else if (rows > 0 && cols > 0) {
        MPI_Send(Clocal.data(), static_cast<int>(rows), rowType, 0, 0, MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    for (auto& t : recvTypes) MPI_Type_free(&t);
    if (rowType != MPI_DATATYPE_NULL) MPI_Type_free(&rowType);
    Clocal = std::vector<double>();
    Ap = std::vector<double>();
    Bp = std::vector<double>();

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", global_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (global_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            std::vector<double> B(N * N);
            initMatrix(A, N);
            initMatrix(B, N);
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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
