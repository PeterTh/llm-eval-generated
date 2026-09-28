#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// SIMD vector of doubles (compiler vector extension, no external dependency)
using v4d = double __attribute__((vector_size(32)));
constexpr size_t VL = sizeof(v4d) / sizeof(double);

// Cache blocking parameters for the local matrix multiplication kernel
constexpr size_t KC = 256;       // k-dimension block (packed B panel height)
constexpr size_t NC = 512;       // j-dimension block (packed B panel width)
constexpr size_t MR = 6;         // rows of C held in registers by the micro-kernel
constexpr size_t NRV = 2;        // vectors of columns held in registers
constexpr size_t NR = NRV * VL;  // columns of C held in registers

static inline v4d loadv(const double* p) noexcept {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

static inline void storev(double* p, const v4d v) noexcept {
    __builtin_memcpy(p, &v, sizeof(v));
}

static inline v4d broadcast(const double x) noexcept {
    return v4d{x, x, x, x};
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize the rows [rowBegin, rowEnd) of a matrix into dst, which holds those
// rows contiguously starting at index 0.
void initMatrixRows(double* dst, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = dst + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Multiply the locally owned rows of A (localRows x N) with the full matrix B (N x N),
// producing the corresponding rows of C (localRows x N). Apack/Bpack are scratch
// buffers for the packed blocks.
void matrixMultiplyLocal(const double* __restrict A, const double* __restrict B,
                         double* __restrict C, const size_t localRows, const size_t N,
                         std::vector<double>& Apack, std::vector<double>& Bpack) {
    for (size_t jc = 0; jc < N; jc += NC) {
        const size_t nc = std::min(NC, N - jc);

        for (size_t kc = 0; kc < N; kc += KC) {
            const size_t kcLen = std::min(KC, N - kc);

            const size_t jFull = nc - nc % NR;
            const size_t numPanels = jFull / NR;

            // Pack the current B block into NR-wide micro-panels, so that the
            // micro-kernel reads its B operand as one contiguous stream.
            for (size_t p = 0; p < numPanels; ++p) {
                double* const __restrict dst = Bpack.data() + p * kcLen * NR;
                for (size_t k = 0; k < kcLen; ++k) {
                    const double* const __restrict src = B + (kc + k) * N + jc + p * NR;
                    for (size_t t = 0; t < NR; ++t) {
                        dst[k * NR + t] = src[t];
                    }
                }
            }

            const double* const __restrict Bp = Bpack.data();

            size_t i = 0;
            for (; i + MR <= localRows; i += MR) {
                // Pack the A rows of this block so the micro-kernel reads them
                // contiguously (layout: [k][r])
                for (size_t k = 0; k < kcLen; ++k) {
                    for (size_t r = 0; r < MR; ++r) {
                        Apack[k * MR + r] = A[(i + r) * N + kc + k];
                    }
                }
                const double* const __restrict Ap = Apack.data();

                // Register-blocked micro-kernel: an MR x NR tile of C is kept in
                // registers while streaming over the k dimension.
                for (size_t p = 0; p < numPanels; ++p) {
                    const double* __restrict bp = Bp + p * kcLen * NR;
                    v4d acc[MR][NRV];
                    for (size_t r = 0; r < MR; ++r) {
                        for (size_t t = 0; t < NRV; ++t) {
                            acc[r][t] = v4d{0.0, 0.0, 0.0, 0.0};
                        }
                    }
                    for (size_t k = 0; k < kcLen; ++k, bp += NR) {
                        v4d b[NRV];
                        for (size_t t = 0; t < NRV; ++t) {
                            b[t] = loadv(bp + t * VL);
                        }
                        for (size_t r = 0; r < MR; ++r) {
                            const v4d av = broadcast(Ap[k * MR + r]);
                            for (size_t t = 0; t < NRV; ++t) {
                                acc[r][t] += av * b[t];
                            }
                        }
                    }
                    for (size_t r = 0; r < MR; ++r) {
                        double* const __restrict crow = C + (i + r) * N + jc + p * NR;
                        for (size_t t = 0; t < NRV; ++t) {
                            storev(crow + t * VL, loadv(crow + t * VL) + acc[r][t]);
                        }
                    }
                }
                // Remainder columns
                for (size_t r = 0; r < MR; ++r) {
                    double* const __restrict crow = C + (i + r) * N + jc;
                    for (size_t k = 0; k < kcLen; ++k) {
                        const double av = A[(i + r) * N + kc + k];
                        const double* const __restrict brow = B + (kc + k) * N + jc;
                        for (size_t j = jFull; j < nc; ++j) {
                            crow[j] += av * brow[j];
                        }
                    }
                }
            }
            // Remainder rows
            for (; i < localRows; ++i) {
                double* const __restrict crow = C + i * N + jc;
                for (size_t k = 0; k < kcLen; ++k) {
                    const double av = A[i * N + kc + k];
                    const double* const __restrict brow = B + (kc + k) * N + jc;
                    for (size_t j = 0; j < nc; ++j) {
                        crow[j] += av * brow[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
// A and B are regenerated on the fly, since only a few of their entries are needed.
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("MPI ranks: %d\n", numRanks);
    }

    // Distribute the rows of A (and thus of C) evenly across the ranks
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowOffsets(numRanks);
    {
        const size_t base = N / static_cast<size_t>(numRanks);
        const size_t rem = N % static_cast<size_t>(numRanks);
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t count = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            rowCounts[r] = static_cast<int>(count);
            rowOffsets[r] = static_cast<int>(offset);
            offset += count;
        }
    }
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t rowBegin = static_cast<size_t>(rowOffsets[rank]);

    // Every rank owns one row block of A and of C. B is needed in full by every
    // rank, so it is held once per compute node in an MPI-3 shared memory window
    // that all ranks of that node read directly - this keeps the memory per node
    // at O(N^2) and requires no communication at all during the multiplication.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    // The rows of B that this rank contributes to the node-local copy
    const size_t bRowBegin = (N * static_cast<size_t>(nodeRank)) / static_cast<size_t>(nodeSize);
    const size_t bRowEnd = (N * static_cast<size_t>(nodeRank + 1)) / static_cast<size_t>(nodeSize);

    MPI_Win bWin;
    double* bShared = nullptr;
    {
        const MPI_Aint localBytes = static_cast<MPI_Aint>((bRowEnd - bRowBegin) * N * sizeof(double));
        double* myPart = nullptr;
        MPI_Win_allocate_shared(localBytes, sizeof(double), MPI_INFO_NULL, nodeComm, &myPart, &bWin);

        // The window segments of all node ranks form one contiguous array of B
        MPI_Aint segSize = 0;
        int segDisp = 0;
        MPI_Win_shared_query(bWin, 0, &segSize, &segDisp, &bShared);
    }

    std::vector<double> A(localRows * N);
    std::vector<double> C(localRows * N, 0.0);

    // Initialize matrices; every rank generates the data it needs itself,
    // so no communication of the inputs is required.
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRows(A.data(), N, rowBegin, rowBegin + localRows);
    initMatrixRows(bShared + bRowBegin * N, N, bRowBegin, bRowEnd);
    MPI_Barrier(nodeComm);

    std::vector<double> Apack(KC * MR);
    std::vector<double> Bpack(KC * NC);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A.data(), bShared, C.data(), localRows, N, Apack, Bpack);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the full result on rank 0 if it is needed there
    int exitStatus = 0;
    if (printResults || validate) {
        MPI_Datatype gatherRowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &gatherRowType);
        MPI_Type_commit(&gatherRowType);

        std::vector<double> fullC;
        if (rank == 0) {
            fullC.resize(N * N);
        }
        MPI_Gatherv(C.data(), rowCounts[rank], gatherRowType,
                    rank == 0 ? fullC.data() : nullptr, rowCounts.data(), rowOffsets.data(),
                    gatherRowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&gatherRowType);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(fullC, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullC, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitStatus = 1;
                }
            }
        }

        if (validate) {
            MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Win_free(&bWin);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitStatus;
}
