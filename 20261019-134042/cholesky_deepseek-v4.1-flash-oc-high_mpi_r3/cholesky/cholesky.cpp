#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition.
//
// The matrix is partitioned over the MPI ranks with a block-cyclic row
// distribution, so every rank stores only the rows it owns.  The factorization
// is a blocked left-looking algorithm: for each panel only the rows of that
// panel are broadcast, and all ranks update their own rows concurrently.  The
// accumulation order is identical to the sequential algorithm, therefore the
// result is bit-for-bit independent of the number of processes.

namespace {

constexpr size_t kBlockSize = 16;
constexpr size_t kRowTile = 16;

inline size_t blockBegin(size_t b, size_t nb) { return b * nb; }

inline size_t blockEnd(size_t b, size_t n, size_t nb) {
    return std::min(n, (b + 1) * nb);
}

inline size_t chooseBlockSize(size_t n) {
    if (n == 0) {
        return 1;
    }
    return std::min(n, kBlockSize);
}

// Reproduces the floating-point reduction performed by the original scalar
// loop: the bulk is accumulated with separately rounded products, followed by
// a fused multiply-add tail.
inline double dotSequential(const double* a, const double* b, size_t len) {
    double s = 0.0;
    const size_t body = (len >= 4) ? (len & ~static_cast<size_t>(3)) : 0;
    for (size_t t = 0; t < body; ++t) {
        s += a[t] * b[t];
    }
    for (size_t t = body; t < len; ++t) {
        s = std::fma(a[t], b[t], s);
    }
    return s;
}

} // namespace

// Decomposes positive definite matrix A into L * L^T where L is lower
// triangular. Only the locally owned rows of A are stored, in row-major order.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, const size_t nb,
                           const std::vector<int>& l2g, const int rank, const int numProcs,
                           int& failIndex) {
    const size_t lrows = l2g.size();

    std::vector<double> panel(nb * n, 0.0);
    std::vector<double> panelT(nb * n, 0.0);

    failIndex = -1;

    for (size_t k = 0; k < n; k += nb) {
        const size_t kb = std::min(nb, n - k);
        const size_t rowLen = k + kb;
        const size_t diagBlock = k / nb;
        const int owner = static_cast<int>(diagBlock % static_cast<size_t>(numProcs));

        size_t ownerRow = 0;
        bool localOk = true;
        if (rank == owner) {
            ownerRow = static_cast<size_t>(
                std::lower_bound(l2g.begin(), l2g.end(), static_cast<int>(k)) - l2g.begin());

            for (size_t jj = 0; jj < kb && localOk; ++jj) {
                const size_t j = k + jj;
                const double* Aj = &A[(ownerRow + jj) * n];
                const double sum = dotSequential(Aj, Aj, j);
                const double val = Aj[j] - sum;
                if (val <= 0.0) {
                    localOk = false;
                    failIndex = static_cast<int>(j);
                    break;
                }
                A[(ownerRow + jj) * n + j] = sqrt(val);

                const double diag = A[(ownerRow + jj) * n + j];
                for (size_t ii = jj + 1; ii < kb; ++ii) {
                    const double* Ai = &A[(ownerRow + ii) * n];
                    const double s = dotSequential(Ai, Aj, j);
                    A[(ownerRow + ii) * n + j] = (Ai[j] - s) / diag;
                }
            }
        }

        int okFlag = localOk ? 1 : 0;
        MPI_Bcast(&okFlag, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!okFlag) {
            MPI_Bcast(&failIndex, 1, MPI_INT, owner, MPI_COMM_WORLD);
            return false;
        }

        if (rank == owner) {
            for (size_t ii = 0; ii < kb; ++ii) {
                std::memcpy(&panel[ii * rowLen], &A[(ownerRow + ii) * n],
                            rowLen * sizeof(double));
            }
        }
        MPI_Bcast(panel.data(), static_cast<int>(kb * rowLen), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        for (size_t t = 0; t < k; ++t) {
            double* bt = &panelT[t * kb];
            for (size_t jj = 0; jj < kb; ++jj) {
                bt[jj] = panel[jj * rowLen + t];
            }
        }

        const size_t threshold = k + kb;
        const size_t start = static_cast<size_t>(
            std::lower_bound(l2g.begin(), l2g.end(), static_cast<int>(threshold)) - l2g.begin());

        for (size_t r0 = start; r0 < lrows; r0 += kRowTile) {
            const size_t mr = std::min(kRowTile, lrows - r0);
            double c[kRowTile][kBlockSize];

            for (size_t r = 0; r < mr; ++r) {
                std::memset(c[r], 0, kb * sizeof(double));
            }

            for (size_t t = 0; t < k; ++t) {
                const double* bt = &panelT[t * kb];
                for (size_t r = 0; r < mr; ++r) {
                    const double a = A[(r0 + r) * n + t];
                    double* cr = c[r];
                    for (size_t jj = 0; jj < kb; ++jj) {
                        cr[jj] += a * bt[jj];
                    }
                }
            }

            for (size_t r = 0; r < mr; ++r) {
                double* Ai = &A[(r0 + r) * n];
                double* cr = c[r];
                for (size_t jj = 0; jj < kb; ++jj) {
                    double acc = cr[jj];
                    const size_t len = k + jj;
                    const size_t qin = (len >= 4) ? (jj & ~static_cast<size_t>(3)) : 0;
                    const double* Pj = &panel[jj * rowLen];
                    for (size_t t = k; t < k + qin; ++t) {
                        acc += Ai[t] * Pj[t];
                    }
                    for (size_t t = k + qin; t < len; ++t) {
                        acc = std::fma(Ai[t], Pj[t], acc);
                    }
                    Ai[len] = (Ai[len] - acc) / Pj[len];
                }
            }
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix distributed over row blocks.
// Each rank computes the upper-triangular part of the rows it owns (symmetry
// halves the work); the lower part is then exchanged. The accumulation order
// matches the sequential version exactly.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const size_t nb,
                                    const std::vector<int>& l2g, const int rank,
                                    const int numProcs, const std::vector<int>& rowCounts,
                                    const std::vector<int>& rowDispls) {
    const size_t lrows = l2g.size();
    const size_t nblocks = (n + nb - 1) / nb;

    std::vector<double> BT(n * n, 0.0);
    {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        const size_t tile = 32;
        for (size_t i0 = 0; i0 < n; i0 += tile) {
            const size_t ie = std::min(n, i0 + tile);
            for (size_t j0 = 0; j0 < n; j0 += tile) {
                const size_t je = std::min(n, j0 + tile);
                for (size_t i = i0; i < ie; ++i) {
                    for (size_t j = j0; j < je; ++j) {
                        BT[j * n + i] = B[i * n + j];
                    }
                }
            }
        }
    }

    const size_t body = (n >= 4) ? (n & ~static_cast<size_t>(3)) : 0;
    constexpr size_t kGenRows = 16;
    constexpr size_t kGenCols = 128;

    size_t localBase = 0;
    for (size_t b = static_cast<size_t>(rank); b < nblocks; b += static_cast<size_t>(numProcs)) {
        const size_t s = blockBegin(b, nb);
        const size_t e = blockEnd(b, n, nb);
        for (size_t r0 = s; r0 < e; r0 += kGenRows) {
            const size_t mr = std::min(kGenRows, e - r0);
            for (size_t j0 = s; j0 < n; j0 += kGenCols) {
                const size_t nc = std::min(kGenCols, n - j0);
                if (j0 + nc <= r0) {
                    continue;
                }
                double c[kGenRows][kGenCols];
                for (size_t r = 0; r < mr; ++r) {
                    std::memset(c[r], 0, nc * sizeof(double));
                }
                for (size_t t = 0; t < body; ++t) {
                    const double* row = &BT[t * n];
                    for (size_t r = 0; r < mr; ++r) {
                        const double a = row[r0 + r];
                        double* cr = c[r];
                        for (size_t jj = 0; jj < nc; ++jj) {
                            cr[jj] += a * row[j0 + jj];
                        }
                    }
                }
                for (size_t t = body; t < n; ++t) {
                    const double* row = &BT[t * n];
                    for (size_t r = 0; r < mr; ++r) {
                        const double a = row[r0 + r];
                        double* cr = c[r];
                        for (size_t jj = 0; jj < nc; ++jj) {
                            cr[jj] = std::fma(a, row[j0 + jj], cr[jj]);
                        }
                    }
                }
                for (size_t r = 0; r < mr; ++r) {
                    const size_t gi = r0 + r;
                    double* Ai = &A[(localBase + (gi - s)) * n];
                    const size_t jstart = std::max(j0, gi);
                    for (size_t j = jstart; j < j0 + nc; ++j) {
                        Ai[j] = c[r][j - j0];
                    }
                }
            }
        }
        localBase += e - s;
    }

    std::vector<double>().swap(BT);

    std::vector<int> counts(numProcs), displs(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        counts[p] = rowCounts[p] * static_cast<int>(n);
        displs[p] = rowDispls[p] * static_cast<int>(n);
    }
    std::vector<double> recv(n * n);
    MPI_Allgatherv(A.empty() ? nullptr : A.data(), static_cast<int>(A.size()), MPI_DOUBLE,
                   recv.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> upper(n * n);
    for (int p = 0; p < numProcs; ++p) {
        size_t c = static_cast<size_t>(rowDispls[p]);
        for (size_t b = static_cast<size_t>(p); b < nblocks; b += static_cast<size_t>(numProcs)) {
            for (size_t g = blockBegin(b, nb); g < blockEnd(b, n, nb); ++g, ++c) {
                std::memcpy(&upper[g * n], &recv[c * n], n * sizeof(double));
            }
        }
    }

    for (size_t li = 0; li < lrows; ++li) {
        const size_t gi = static_cast<size_t>(l2g[li]);
        double* Ai = &A[li * n];
        for (size_t j = 0; j < gi; ++j) {
            Ai[j] = upper[j * n + gi];
        }
        Ai[gi] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double sum = dotSequential(&L[i * n], &L[j * n], n);
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool badArgs = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
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
            badArgs = true;
        }
    }

    if (badArgs) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t nb = chooseBlockSize(n);
    const size_t nblocks = (n + nb - 1) / nb;

    std::vector<int> l2g;
    for (size_t b = static_cast<size_t>(rank); b < nblocks; b += static_cast<size_t>(numProcs)) {
        for (size_t g = blockBegin(b, nb); g < blockEnd(b, n, nb); ++g) {
            l2g.push_back(static_cast<int>(g));
        }
    }
    const size_t lrows = l2g.size();

    std::vector<int> rowCounts(numProcs, 0);
    std::vector<int> rowDispls(numProcs, 0);
    const int localRows = static_cast<int>(lrows);
    MPI_Allgather(&localRows, 1, MPI_INT, rowCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int p = 1; p < numProcs; ++p) {
        rowDispls[p] = rowDispls[p - 1] + rowCounts[p - 1];
    }

    std::vector<double> A(lrows * n, 0.0);
    std::vector<double> A_orig;
    std::vector<double> gathered;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, nb, l2g, rank, numProcs, rowCounts, rowDispls);

    auto gatherFull = [&](const std::vector<double>& local, std::vector<double>& full) {
        std::vector<int> counts(numProcs), displs(numProcs);
        for (int p = 0; p < numProcs; ++p) {
            counts[p] = rowCounts[p] * static_cast<int>(n);
            displs[p] = rowDispls[p] * static_cast<int>(n);
        }
        std::vector<double> recv;
        if (rank == 0) {
            recv.resize(n * n);
        }
        MPI_Gatherv(local.empty() ? nullptr : local.data(),
                    static_cast<int>(local.size()), MPI_DOUBLE,
                    rank == 0 ? recv.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            full.resize(n * n);
            for (int p = 0; p < numProcs; ++p) {
                size_t c = static_cast<size_t>(rowDispls[p]);
                for (size_t b = static_cast<size_t>(p); b < nblocks; b += static_cast<size_t>(numProcs)) {
                    for (size_t g = blockBegin(b, nb); g < blockEnd(b, n, nb); ++g, ++c) {
                        std::memcpy(&full[g * n], &recv[c * n], n * sizeof(double));
                    }
                }
            }
        }
    };

    if (validate) {
        gatherFull(A, A_orig);
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int failIndex = -1;
    bool success = choleskyDecomposition(A, n, nb, l2g, rank, numProcs, failIndex);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failIndex);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    for (size_t li = 0; li < lrows; ++li) {
        const size_t gi = static_cast<size_t>(l2g[li]);
        for (size_t j = gi + 1; j < n; ++j) {
            A[li * n + j] = 0.0;
        }
    }

    if (printResults || validate) {
        gatherFull(A, gathered);
    }

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(gathered, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(gathered, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
