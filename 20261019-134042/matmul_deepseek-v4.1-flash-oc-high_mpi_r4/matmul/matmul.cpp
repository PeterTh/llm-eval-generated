#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Index/size helpers for uneven block distributions
inline size_t blockStart(const size_t idx, const size_t parts, const size_t N) noexcept {
    const size_t base = N / parts;
    const size_t rem = N % parts;
    return idx * base + (idx < rem ? idx : rem);
}

inline size_t blockSize(const size_t idx, const size_t parts, const size_t N) noexcept {
    const size_t base = N / parts;
    const size_t rem = N % parts;
    return base + (idx < rem ? 1 : 0);
}

// Multiply a local row panel of A by a local column panel of B.
// The k-accumulation order matches the reference implementation exactly
// (including the fused final term for odd N), so results are bit-for-bit
// identical to the serial version while remaining cache friendly.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const size_t nrows, const size_t ncols) {
    constexpr size_t JB = 128;
    const size_t kMain = N - (N & 1);

    for (size_t i = 0; i < nrows; ++i) {
        const double* __restrict aRow = A.data() + i * N;
        for (size_t jb = 0; jb < ncols; jb += JB) {
            const size_t jw = (ncols - jb < JB) ? (ncols - jb) : JB;
            double* __restrict cRow = C.data() + i * ncols + jb;
            for (size_t k = 0; k < kMain; ++k) {
                const double aik = aRow[k];
                const double* __restrict bRow = B.data() + k * ncols + jb;
                for (size_t j = 0; j < jw; ++j) {
                    cRow[j] += aik * bRow[j];
                }
            }
            if (kMain < N) {
                const double aik = aRow[kMain];
                const double* __restrict bRow = B.data() + kMain * ncols + jb;
                for (size_t j = 0; j < jw; ++j) {
                    cRow[j] = std::fma(aik, bRow[j], cRow[j]);
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

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    }

    // Choose a near-square 2D process grid whose dimensions multiply to nprocs.
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    const size_t pr = static_cast<size_t>(dims[0]);
    const size_t pc = static_cast<size_t>(dims[1]);

    // Rank ordering is row-major over the process grid.
    const size_t rowIdx = static_cast<size_t>(rank) / pc;
    const size_t colIdx = static_cast<size_t>(rank) % pc;

    const size_t rowStart = blockStart(rowIdx, pr, N);
    const size_t colStart = blockStart(colIdx, pc, N);
    const size_t nrows = blockSize(rowIdx, pr, N);
    const size_t ncols = blockSize(colIdx, pc, N);

    // Each rank only stores the panels it needs: nrows x N of A and N x ncols of B.
    std::vector<double> A(nrows * N);
    std::vector<double> B(N * ncols);
    std::vector<double> C(nrows * ncols, 0.0);

    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    for (size_t i = 0; i < nrows; ++i) {
        const size_t gi = rowStart + i;
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = getPseudoRndValue(N, gi, j);
        }
    }
    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < ncols; ++j) {
            const size_t gj = colStart + j;
            B[k * ncols + j] = getPseudoRndValue(N, k, gj);
        }
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, nrows, ncols);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
    }

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) {
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather all C blocks on rank 0, reconstructing the full row-major matrix.
    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);
    int offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        const size_t rr = static_cast<size_t>(r) / pc;
        const size_t cc = static_cast<size_t>(r) % pc;
        const int cnt = static_cast<int>(blockSize(rr, pr, N) * blockSize(cc, pc, N));
        counts[r] = cnt;
        displs[r] = offset;
        offset += cnt;
    }

    std::vector<double> Cfull;
    if (rank == 0) {
        Cfull.assign(N * N, 0.0);
    }

    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(static_cast<size_t>(offset));
    }

    MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            const size_t rr = static_cast<size_t>(r) / pc;
            const size_t cc = static_cast<size_t>(r) % pc;
            const size_t rs = blockStart(rr, pr, N);
            const size_t cs = blockStart(cc, pc, N);
            const size_t nr = blockSize(rr, pr, N);
            const size_t nc = blockSize(cc, pc, N);
            const double* src = gathered.data() + displs[r];
            for (size_t i = 0; i < nr; ++i) {
                memcpy(Cfull.data() + (rs + i) * N + cs, src + i * nc, nc * sizeof(double));
            }
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(Cfull, "MatrixC");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            std::vector<double> Afull(N * N);
            std::vector<double> Bfull(N * N);
            initMatrix(Afull, N);
            initMatrix(Bfull, N);
            const bool valid = validateResult(Afull, Bfull, Cfull, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    MPI_Finalize();
    return rc;
}
