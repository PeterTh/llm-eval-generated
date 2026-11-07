#include <mpi.h>

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

// Initialize a block of rows [rowBegin, rowBegin + numRows) of an NxN matrix.
// Initialization is fully deterministic, so every rank generates the parts it
// needs locally instead of receiving them over the network.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin,
                    const size_t numRows) {
    for (size_t r = 0; r < numRows; ++r) {
        const size_t i = rowBegin + r;
        for (size_t j = 0; j < N; ++j) {
            mat[r * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Register-tiled kernel: computes C[i][j] = sum_k A[i][k] * B[k][j] for a tile
// of MR rows x NR columns.  For every element the accumulation runs
// sequentially over k starting from 0.0, exactly like the reference triple
// loop, so results are bitwise identical to the original implementation.
template <size_t MR, size_t NR>
static inline void kernelTile(const double* __restrict A, const double* __restrict B,
                              double* __restrict C, const size_t N, const size_t i0,
                              const size_t j0) {
    double acc[MR][NR];
    for (size_t ii = 0; ii < MR; ++ii)
        for (size_t jj = 0; jj < NR; ++jj) acc[ii][jj] = 0.0;

    const double* Bp = B + j0;
    for (size_t k = 0; k < N; ++k) {
        const double* bk = Bp + k * N;
        for (size_t ii = 0; ii < MR; ++ii) {
            const double a = A[(i0 + ii) * N + k];
            for (size_t jj = 0; jj < NR; ++jj) {
                acc[ii][jj] += a * bk[jj];
            }
        }
    }

    for (size_t ii = 0; ii < MR; ++ii)
        for (size_t jj = 0; jj < NR; ++jj) C[(i0 + ii) * N + j0 + jj] = acc[ii][jj];
}

// Generic edge kernel for partial tiles (same accumulation order).
static inline void kernelEdge(const double* __restrict A, const double* __restrict B,
                              double* __restrict C, const size_t N, const size_t i0,
                              const size_t i1, const size_t j0, const size_t j1) {
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Multiply the local row block A (localRows x N) with the full matrix B (N x N)
// producing the local row block C (localRows x N).
void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t MR = 4;   // rows per register tile
    constexpr size_t NR = 8;   // columns per register tile (2 AVX2 vectors)
    constexpr size_t JB = 64;  // column panel width: B panel (N x JB) stays cache resident

    const double* Ap = A.data();
    const double* Bp = B.data();
    double* Cp = C.data();

    const size_t fullI = localRows - localRows % MR;
    const size_t fullJ = N - N % NR;

    for (size_t jp = 0; jp < fullJ; jp += JB) {
        const size_t jpEnd = std::min(jp + JB, fullJ);
        for (size_t i0 = 0; i0 < fullI; i0 += MR) {
            for (size_t j0 = jp; j0 < jpEnd; j0 += NR) {
                kernelTile<MR, NR>(Ap, Bp, Cp, N, i0, j0);
            }
        }
    }
    // Remaining columns for the full row tiles
    if (fullJ < N) kernelEdge(Ap, Bp, Cp, N, 0, fullI, fullJ, N);
    // Remaining rows (all columns)
    if (fullI < localRows) kernelEdge(Ap, Bp, Cp, N, fullI, localRows, 0, N);
}

// Simple validation: compute a single element and compare.
// Every rank validates the check points that fall into its own row block.
bool validateResultLocal(const std::vector<double>& A, const std::vector<double>& B,
                         const std::vector<double>& C, const size_t N, const size_t rowBegin,
                         const size_t localRows) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            if (i < rowBegin || i >= rowBegin + localRows) continue;
            const size_t li = i - rowBegin;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[li * N + k] * B[k * N + j];
            }

            const double actual = C[li * N + j];
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
    const bool isRoot = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (isRoot) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
    }

    // 1D row-block decomposition of A and C; B is replicated on every rank.
    std::vector<int> rowCounts(nprocs), rowOffsets(nprocs);
    {
        const size_t base = N / static_cast<size_t>(nprocs);
        const size_t rem = N % static_cast<size_t>(nprocs);
        size_t off = 0;
        for (int r = 0; r < nprocs; ++r) {
            const size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            rowCounts[r] = static_cast<int>(cnt);
            rowOffsets[r] = static_cast<int>(off);
            off += cnt;
        }
    }
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t rowBegin = static_cast<size_t>(rowOffsets[rank]);

    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Clocal(localRows * N);
    std::vector<double> C;  // full result, only on root
    if (isRoot) C.resize(N * N);

    // Initialize matrices
    if (isRoot) printf("Initializing matrices...\n");
    initMatrixRows(A, N, rowBegin, localRows);
    initMatrixRows(B, N, 0, N);

    // Row datatype so gather counts are in rows (avoids int overflow for large N)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Perform matrix multiplication
    if (isRoot) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A, B, Clocal, N, localRows);

    // Collect the distributed result rows on root
    MPI_Gatherv(Clocal.data(), rowCounts[rank], rowType, isRoot ? C.data() : nullptr,
                rowCounts.data(), rowOffsets.data(), rowType, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Type_free(&rowType);

    if (isRoot) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDurationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        const int localValid = validateResultLocal(A, B, Clocal, N, rowBegin, localRows) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (isRoot) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
