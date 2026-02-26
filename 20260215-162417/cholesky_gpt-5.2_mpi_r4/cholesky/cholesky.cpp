#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static bool choleskyDecompositionBlock(std::vector<double>& A, const size_t n) {
    // In-place unblocked Cholesky on a small dense block (row-major).
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (size_t k = 0; k < j; ++k) {
                    const double v = A[j * n + k];
                    sum += v * v;
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    return false;
                }
                A[j * n + j] = std::sqrt(val);
            } else {
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix
static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // Then add identity-scaled diagonal to make it strictly positive definite
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (std::fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void computeRowDistribution(const size_t n, const int nranks, std::vector<int>& rowsPerRank, std::vector<int>& rowDispls) {
    rowsPerRank.assign(nranks, 0);
    rowDispls.assign(nranks, 0);

    const size_t base = n / (size_t)nranks;
    const size_t rem = n % (size_t)nranks;

    size_t disp = 0;
    for (int r = 0; r < nranks; ++r) {
        const size_t rows = base + ((size_t)r < rem ? 1 : 0);
        rowsPerRank[r] = (int)rows;
        rowDispls[r] = (int)disp;
        disp += rows;
    }
}

static bool mpiCholeskyRowBlocked(std::vector<double>& localA,
                                 const size_t n,
                                 const size_t startRow,
                                 const size_t localRows,
                                 const std::vector<int>& rowsPerRank,
                                 const std::vector<int>& rowDispls,
                                 MPI_Comm comm) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    constexpr size_t kBlockSize = 64;

    std::vector<int> recvCounts(nranks), displs(nranks);

    for (size_t k = 0; k < n; k += kBlockSize) {
        const size_t kb = std::min(kBlockSize, n - k);

        // --- Gather diagonal block rows to rank 0 and factorize ---
        const size_t diagStart = k;
        const size_t diagEnd = k + kb;

        const size_t localStart = startRow;
        const size_t localEnd = startRow + localRows;
        const size_t segStart = std::max(diagStart, localStart);
        const size_t segEnd = std::min(diagEnd, localEnd);
        const size_t localDiagRows = (segEnd > segStart) ? (segEnd - segStart) : 0;
        const size_t localDiagOff = (segStart > localStart) ? (segStart - localStart) : 0;

        std::vector<double> diagSend(localDiagRows * kb);
        for (size_t r = 0; r < localDiagRows; ++r) {
            const double* src = &localA[(localDiagOff + r) * n + k];
            std::memcpy(&diagSend[r * kb], src, kb * sizeof(double));
        }

        int total = 0;
        for (int r = 0; r < nranks; ++r) {
            const size_t rs = (size_t)rowDispls[r];
            const size_t re = rs + (size_t)rowsPerRank[r];
            const size_t s = std::max(diagStart, rs);
            const size_t e = std::min(diagEnd, re);
            const size_t cntRows = (e > s) ? (e - s) : 0;
            recvCounts[r] = (int)(cntRows * kb);
            displs[r] = total;
            total += recvCounts[r];
        }

        std::vector<double> diagBlock;
        if (rank == 0) {
            diagBlock.assign(kb * kb, 0.0);
        } else {
            diagBlock.resize(kb * kb);
        }

        MPI_Gatherv(diagSend.data(),
                    (int)(localDiagRows * kb),
                    MPI_DOUBLE,
                    diagBlock.data(),
                    recvCounts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    comm);

        int ok = 1;
        if (rank == 0) {
            ok = choleskyDecompositionBlock(diagBlock, kb) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        if (!ok) {
            return false;
        }

        MPI_Bcast(diagBlock.data(), (int)(kb * kb), MPI_DOUBLE, 0, comm);

        // Copy diagonal block rows back into local storage
        for (size_t r = 0; r < localDiagRows; ++r) {
            const size_t globalRow = segStart + r;
            const size_t ii = globalRow - k;
            double* dst = &localA[(localDiagOff + r) * n + k];
            std::memcpy(dst, &diagBlock[ii * kb], kb * sizeof(double));
        }

        // --- TRSM: compute panel below the diagonal block for local rows ---
        const size_t panelStartRow = k + kb;
        if (panelStartRow < n) {
            for (size_t iLocal = 0; iLocal < localRows; ++iLocal) {
                const size_t gi = startRow + iLocal;
                if (gi < panelStartRow) {
                    continue;
                }
                double* rowPanel = &localA[iLocal * n + k];
                for (size_t j = 0; j < kb; ++j) {
                    double sum = 0.0;
                    const double* Lj = &diagBlock[j * kb];
                    for (size_t t = 0; t < j; ++t) {
                        sum += rowPanel[t] * Lj[t];
                    }
                    rowPanel[j] = (rowPanel[j] - sum) / Lj[j];
                }
            }

            // --- Allgather the full panel (rows >= panelStartRow, cols k..k+kb-1) ---
            const size_t sendFirst = (startRow < panelStartRow) ? panelStartRow : startRow;
            const size_t sendLast = startRow + localRows;
            const size_t panelRowsSend = (sendLast > sendFirst) ? (sendLast - sendFirst) : 0;
            const size_t panelOffLocal = (sendFirst > startRow) ? (sendFirst - startRow) : 0;

            std::vector<double> panelSend(panelRowsSend * kb);
            for (size_t r = 0; r < panelRowsSend; ++r) {
                const double* src = &localA[(panelOffLocal + r) * n + k];
                std::memcpy(&panelSend[r * kb], src, kb * sizeof(double));
            }

            int totalPanel = 0;
            for (int r = 0; r < nranks; ++r) {
                const size_t rs = (size_t)rowDispls[r];
                const size_t re = rs + (size_t)rowsPerRank[r];
                const size_t s = std::max(panelStartRow, rs);
                const size_t cntRows = (re > s) ? (re - s) : 0;
                recvCounts[r] = (int)(cntRows * kb);
                displs[r] = totalPanel;
                totalPanel += recvCounts[r];
            }

            std::vector<double> panelBuf((size_t)totalPanel);
            MPI_Allgatherv(panelSend.data(),
                           (int)(panelRowsSend * kb),
                           MPI_DOUBLE,
                           panelBuf.data(),
                           recvCounts.data(),
                           displs.data(),
                           MPI_DOUBLE,
                           comm);

            // --- Trailing update for local rows (lower triangle only) ---
            for (size_t iLocal = 0; iLocal < localRows; ++iLocal) {
                const size_t gi = startRow + iLocal;
                if (gi < panelStartRow) {
                    continue;
                }

                double* row = &localA[iLocal * n];
                const double* Li = &row[k];

                const size_t jEnd = gi;
                for (size_t gj = panelStartRow; gj <= jEnd; ++gj) {
                    const double* Lj = &panelBuf[(gj - panelStartRow) * kb];
                    double sum = 0.0;
                    for (size_t t = 0; t < kb; ++t) {
                        sum += Li[t] * Lj[t];
                    }
                    row[gj] -= sum;
                }
            }
        }
    }

    // Zero upper triangle for local rows
    for (size_t iLocal = 0; iLocal < localRows; ++iLocal) {
        const size_t gi = startRow + iLocal;
        double* row = &localA[iLocal * n];
        for (size_t j = gi + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    int earlyExitCode = -1;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                earlyExitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&earlyExitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (earlyExitCode != -1) {
        MPI_Finalize();
        return earlyExitCode;
    }

    uint64_t n64 = (uint64_t)n;
    int v = validate ? 1 : 0;
    int r = printResults ? 1 : 0;
    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = (size_t)n64;
    validate = (v != 0);
    printResults = (r != 0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> rowsPerRank, rowDispls;
    computeRowDistribution(n, nranks, rowsPerRank, rowDispls);

    const size_t localRows = (size_t)rowsPerRank[rank];
    const size_t startRow = (size_t)rowDispls[rank];

    std::vector<int> countsD(nranks), displsD(nranks);
    for (int rr = 0; rr < nranks; ++rr) {
        const size_t c = (size_t)rowsPerRank[rr] * n;
        const size_t d = (size_t)rowDispls[rr] * n;
        countsD[rr] = (int)c;
        displsD[rr] = (int)d;
    }

    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A;
        }
        printf("Computing Cholesky decomposition...\n");
    }

    std::vector<double> localA(localRows * n);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 countsD.data(),
                 displsD.data(),
                 MPI_DOUBLE,
                 localA.data(),
                 (int)(localRows * n),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const bool success = mpiCholeskyRowBlocked(localA, n, startRow, localRows, rowsPerRank, rowDispls, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    int ok = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!ok) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> L;
    if (validate || printResults) {
        if (rank == 0) {
            L.resize(n * n);
        }
        MPI_Gatherv(localA.data(),
                    (int)(localRows * n),
                    MPI_DOUBLE,
                    rank == 0 ? L.data() : nullptr,
                    countsD.data(),
                    displsD.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long ms = (long)std::llround(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double seconds = (maxTime > 0.0) ? maxTime : 1e-9;
        const double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(L, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            }
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
