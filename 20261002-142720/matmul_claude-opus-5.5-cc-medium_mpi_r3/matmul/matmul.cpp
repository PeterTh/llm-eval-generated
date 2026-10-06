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

// Initialize the block [r0, r0+rows) x [c0, c0+cols) of the global NxN matrix (local row-major)
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t r0, const size_t rows,
                     const size_t c0, const size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[i * cols + j] = getPseudoRndValue(N, r0 + i, c0 + j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixBlock(mat, N, 0, N, 0, N);
}

// ---------------------------------------------------------------------------
// Cache-blocked, register-tiled local GEMM: C[m x n] = A[m x K] * B[K x n]
// Every C element is accumulated over k in strictly increasing order starting
// from 0.0 with separate multiply and add (no FMA contraction), matching the
// rounding of the reference triple loop.
// ---------------------------------------------------------------------------
typedef double v4d __attribute__((vector_size(32), aligned(8)));

constexpr size_t MR = 6;      // rows per micro-tile
constexpr size_t NR = 8;      // cols per micro-tile (2 x 4 doubles)
constexpr size_t KC = 256;    // k-block (packed B panel ~ L1)
constexpr size_t MC = 96;     // row block (packed A ~ L2)
constexpr size_t NC = 1024;   // column block (packed B ~ L3 share)

static inline v4d loadv(const double* p) { v4d v; std::memcpy(&v, p, sizeof(v)); return v; }
static inline void storev(double* p, const v4d v) { std::memcpy(p, &v, sizeof(v)); }

// Ap: packed kc x MR (MR consecutive per k), Bp: packed kc x NR (NR consecutive per k)
// Ct: MR x NR tile of C with leading dimension ldc
template <bool First>
static inline void microKernel(const size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                               double* __restrict Ct, const size_t ldc) {
    v4d c[MR][2];
    for (size_t r = 0; r < MR; ++r) {
        if constexpr (First) {
            c[r][0] = v4d{0.0, 0.0, 0.0, 0.0};
            c[r][1] = v4d{0.0, 0.0, 0.0, 0.0};
        } else {
            c[r][0] = loadv(Ct + r * ldc);
            c[r][1] = loadv(Ct + r * ldc + 4);
        }
    }
    for (size_t k = 0; k < kc; ++k) {
        const v4d b0 = loadv(Bp + k * NR);
        const v4d b1 = loadv(Bp + k * NR + 4);
        const double* a = Ap + k * MR;
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = v4d{a[r], a[r], a[r], a[r]};
            c[r][0] += av * b0;
            c[r][1] += av * b1;
        }
    }
    for (size_t r = 0; r < MR; ++r) {
        storev(Ct + r * ldc, c[r][0]);
        storev(Ct + r * ldc + 4, c[r][1]);
    }
}

template <bool First>
static inline void tileKernel(const size_t kc, const double* Ap, const double* Bp,
                              double* Ct, const size_t ldc, const size_t mr, const size_t nr) {
    if (mr == MR && nr == NR) {
        microKernel<First>(kc, Ap, Bp, Ct, ldc);
        return;
    }
    // Edge tile: compute through a full-size scratch tile
    alignas(64) double tmp[MR * NR] = {};
    if constexpr (!First) {
        for (size_t r = 0; r < mr; ++r)
            for (size_t c = 0; c < nr; ++c) tmp[r * NR + c] = Ct[r * ldc + c];
    }
    microKernel<First>(kc, Ap, Bp, tmp, NR);
    for (size_t r = 0; r < mr; ++r)
        for (size_t c = 0; c < nr; ++c) Ct[r * ldc + c] = tmp[r * NR + c];
}

void matrixMultiplyLocal(const double* A, const size_t lda, const double* B, const size_t ldb,
                         double* C, const size_t ldc, const size_t rows, const size_t cols, const size_t K) {
    if (rows == 0 || cols == 0 || K == 0) return;
    std::vector<double> Bpack(KC * (NC + NR));
    std::vector<double> Apack(KC * (MC + MR));

    for (size_t jc = 0; jc < cols; jc += NC) {
        const size_t nc = std::min(NC, cols - jc);
        for (size_t pc = 0; pc < K; pc += KC) {
            const size_t kc = std::min(KC, K - pc);
            const bool first = (pc == 0);

            // Pack B[pc:pc+kc, jc:jc+nc] into NR-wide panels (zero padded)
            for (size_t jp = 0; jp < nc; jp += NR) {
                const size_t nr = std::min(NR, nc - jp);
                double* dst = Bpack.data() + jp * kc;
                for (size_t k = 0; k < kc; ++k) {
                    const double* src = B + (pc + k) * ldb + jc + jp;
                    size_t c = 0;
                    for (; c < nr; ++c) dst[k * NR + c] = src[c];
                    for (; c < NR; ++c) dst[k * NR + c] = 0.0;
                }
            }

            for (size_t ic = 0; ic < rows; ic += MC) {
                const size_t mc = std::min(MC, rows - ic);

                // Pack A[ic:ic+mc, pc:pc+kc] into MR-tall panels (zero padded)
                for (size_t ip = 0; ip < mc; ip += MR) {
                    const size_t mr = std::min(MR, mc - ip);
                    double* dst = Apack.data() + ip * kc;
                    for (size_t r = 0; r < MR; ++r) {
                        if (r < mr) {
                            const double* src = A + (ic + ip + r) * lda + pc;
                            for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = src[k];
                        } else {
                            for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = 0.0;
                        }
                    }
                }

                for (size_t jp = 0; jp < nc; jp += NR) {
                    const size_t nr = std::min(NR, nc - jp);
                    const double* Bp = Bpack.data() + jp * kc;
                    for (size_t ip = 0; ip < mc; ip += MR) {
                        const size_t mr = std::min(MR, mc - ip);
                        const double* Ap = Apack.data() + ip * kc;
                        double* Ct = C + (ic + ip) * ldc + jc + jp;
                        if (first)
                            tileKernel<true>(kc, Ap, Bp, Ct, ldc, mr, nr);
                        else
                            tileKernel<false>(kc, Ap, Bp, Ct, ldc, mr, nr);
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

    // 2D block distribution of C over a Pr x Pc process grid. Rank (pr, pc) computes
    // C[rows_pr, cols_pc] = A[rows_pr, :] * B[:, cols_pc]. Its A row panel and B column
    // panel are generated locally, so the multiplication needs no communication.
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    const int Pr = dims[0], Pc = dims[1];
    auto blockRange = [N](const int parts, const int idx, size_t& begin, size_t& count) {
        const size_t base = N / parts, rem = N % parts, u = static_cast<size_t>(idx);
        count = base + (u < rem ? 1 : 0);
        begin = u * base + std::min(u, rem);
    };
    size_t rowBegin, localRows, colBegin, localCols;
    blockRange(Pr, rank / Pc, rowBegin, localRows);
    blockRange(Pc, rank % Pc, colBegin, localCols);

    // Row-group leaders (column index 0) hold a full, contiguous C row panel; root's panel
    // is the top of the full C. Every rank's block sits in place inside its leader's layout.
    const int myRow = rank / Pc, myCol = rank % Pc;
    const bool leader = (myCol == 0);
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * localCols);
    std::vector<double> C(root ? N * N : (leader ? localRows * N : localRows * localCols));
    const size_t ldc = leader ? N : localCols;
    
    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initMatrixBlock(A, N, rowBegin, localRows, 0, N);
    initMatrixBlock(B, N, 0, N, colBegin, localCols);
    
    // Row datatype so message counts never overflow int
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N > 0 ? N : 1), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);
    MPI_Comm rowComm;
    MPI_Comm_split(MPI_COMM_WORLD, leader ? 0 : MPI_UNDEFINED, myRow, &rowComm);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyLocal(A.data(), N, B.data(), localCols, C.data(), ldc, localRows, localCols, N);

    // Stage 1: assemble each row panel on its leader (all process rows in parallel)
    if (localRows > 0) {
        if (leader) {
            std::vector<MPI_Request> reqs;
            std::vector<MPI_Datatype> types;
            for (int q = 1; q < Pc; ++q) {
                size_t cb, cc;
                blockRange(Pc, q, cb, cc);
                if (cc == 0) continue;
                MPI_Datatype blk;
                MPI_Type_vector(static_cast<int>(localRows), static_cast<int>(cc), static_cast<int>(N),
                                MPI_DOUBLE, &blk);
                MPI_Type_commit(&blk);
                types.push_back(blk);
                reqs.emplace_back();
                MPI_Irecv(C.data() + cb, 1, blk, rank + q, 0, MPI_COMM_WORLD, &reqs.back());
            }
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
            for (auto& t : types) MPI_Type_free(&t);
        } else if (localCols > 0) {
            MPI_Datatype blkRow;
            MPI_Type_contiguous(static_cast<int>(localCols), MPI_DOUBLE, &blkRow);
            MPI_Type_commit(&blkRow);
            MPI_Send(C.data(), static_cast<int>(localRows), blkRow, rank - myCol, 0, MPI_COMM_WORLD);
            MPI_Type_free(&blkRow);
        }
    }

    // Stage 2: gather the contiguous row panels from the leaders onto root
    if (leader) {
        std::vector<int> counts(Pr), displs(Pr);
        for (int q = 0; q < Pr; ++q) {
            size_t rb, rc;
            blockRange(Pr, q, rb, rc);
            counts[q] = static_cast<int>(rc);
            displs[q] = static_cast<int>(rb);
        }
        if (root) {
            MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), counts.data(), displs.data(), rowType, 0, rowComm);
        } else {
            MPI_Gatherv(C.data(), static_cast<int>(localRows), rowType, nullptr, nullptr, nullptr, rowType, 0,
                        rowComm);
        }
        MPI_Comm_free(&rowComm);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);
    
    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", max_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Root only holds its own panels; regenerate the full inputs for checking
            A.assign(N * N, 0.0);
            B.assign(N * N, 0.0);
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
