#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition parallelized with MPI.
// The global n x n matrix is distributed by rows using a 1D block-cyclic
// distribution (block size nb): row i is owned by rank (i / nb) % P.
// This keeps the trailing-matrix update - the dominant O(n^3) cost - load
// balanced across ranks even as the active submatrix shrinks.

// Unblocked sequential Cholesky, used locally to factor each small diagonal
// block (nb x nb) owned by a single rank.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Row -> owning rank under 1D block-cyclic distribution with block size nb.
static inline int ownerOfRow(size_t row, size_t nb, int P) {
    return static_cast<int>((row / nb) % static_cast<size_t>(P));
}

// Global row indices (ascending) owned by a given rank.
static std::vector<size_t> rowsOwnedByRank(size_t n, size_t nb, int P, int rank) {
    std::vector<size_t> rows;
    rows.reserve(n / static_cast<size_t>(P) + nb);
    for (size_t i = 0; i < n; ++i) {
        if (ownerOfRow(i, nb, P) == rank) {
            rows.push_back(i);
        }
    }
    return rows;
}

// Generate a symmetric positive definite matrix, distributed by rows.
// B is generated identically (and redundantly) on every rank so that the
// resulting distributed A is bit-identical to the sequential reference
// regardless of the number of ranks used.
void generatePositiveDefiniteMatrix(std::vector<double>& localA, const size_t n,
                                     const std::vector<size_t>& localRows) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t nLocal = localRows.size();
    localA.assign(nLocal * n, 0.0);

    for (size_t li = 0; li < nLocal; ++li) {
        const size_t i = localRows[li];
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            localA[li * n + j] = sum;
        }
        localA[li * n + i] += n;
    }
}

// Gather a row-distributed matrix into a full n x n matrix on `root`.
static void gatherFullMatrix(const std::vector<double>& localA, std::vector<double>& fullA,
                              size_t n, size_t nb, int rank, int P, int root) {
    std::vector<std::vector<size_t>> allRows(P);
    std::vector<int> recvcounts(P), displs(P);
    int total = 0;
    for (int r = 0; r < P; ++r) {
        allRows[r] = rowsOwnedByRank(n, nb, P, r);
        recvcounts[r] = static_cast<int>(allRows[r].size() * n);
        displs[r] = total;
        total += recvcounts[r];
    }

    std::vector<double> gatherBuf;
    if (rank == root) {
        gatherBuf.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE,
                rank == root ? gatherBuf.data() : nullptr, recvcounts.data(), displs.data(),
                MPI_DOUBLE, root, MPI_COMM_WORLD);

    if (rank == root) {
        fullA.assign(n * n, 0.0);
        for (int r = 0; r < P; ++r) {
            const auto& rows = allRows[r];
            const double* src = gatherBuf.data() + displs[r];
            for (size_t li = 0; li < rows.size(); ++li) {
                std::memcpy(fullA.data() + rows[li] * n, src + li * n, n * sizeof(double));
            }
        }
    }
}

// Distributed right-looking blocked Cholesky decomposition.
// localA holds, for each rank, the full-width (n columns) rows it owns,
// in ascending global-row order, matching `localRows`.
bool choleskyDecompositionMPI(std::vector<double>& localA, size_t n, size_t nb,
                               const std::vector<size_t>& localRows,
                               const std::vector<long long>& localIndexOf, int rank, int P) {
    const size_t nblk = (n + nb - 1) / nb;

    // Precompute the full row ownership map once (needed to build the
    // Allgatherv layout for the panel broadcast at every step).
    std::vector<std::vector<size_t>> allRows(P);
    for (int r = 0; r < P; ++r) {
        allRows[r] = rowsOwnedByRank(n, nb, P, r);
    }

    for (size_t kb = 0; kb < nblk; ++kb) {
        const size_t k0 = kb * nb;
        const size_t k1 = std::min(k0 + nb, n);
        const size_t kbs = k1 - k0;
        const int ownerK = static_cast<int>(kb % static_cast<size_t>(P));

        // 1) Factor the diagonal block (kbs x kbs) on its owner rank.
        std::vector<double> diagBlock(kbs * kbs);
        int successFlag = 1;
        if (rank == ownerK) {
            for (size_t r = 0; r < kbs; ++r) {
                const long long li = localIndexOf[k0 + r];
                for (size_t c = 0; c < kbs; ++c) {
                    diagBlock[r * kbs + c] = localA[static_cast<size_t>(li) * n + k0 + c];
                }
            }
            successFlag = choleskyDecomposition(diagBlock, kbs) ? 1 : 0;
        }

        MPI_Bcast(&successFlag, 1, MPI_INT, ownerK, MPI_COMM_WORLD);
        if (!successFlag) {
            return false;
        }

        MPI_Bcast(diagBlock.data(), static_cast<int>(kbs * kbs), MPI_DOUBLE, ownerK,
                  MPI_COMM_WORLD);

        if (rank == ownerK) {
            for (size_t r = 0; r < kbs; ++r) {
                const long long li = localIndexOf[k0 + r];
                for (size_t c = 0; c < kbs; ++c) {
                    localA[static_cast<size_t>(li) * n + k0 + c] = diagBlock[r * kbs + c];
                }
            }
        }

        // 2) Panel triangular solve: every rank updates the rows >= k1 that
        // it owns, using only the just-broadcast diagonal block.
        auto startIt = std::lower_bound(localRows.begin(), localRows.end(), k1);
        const size_t startIdx = static_cast<size_t>(startIt - localRows.begin());
        const size_t nLocal = localRows.size();

        for (size_t li = startIdx; li < nLocal; ++li) {
            double* row = &localA[li * n];
            for (size_t c = 0; c < kbs; ++c) {
                double sum = 0.0;
                for (size_t cc = 0; cc < c; ++cc) {
                    sum += row[k0 + cc] * diagBlock[c * kbs + cc];
                }
                row[k0 + c] = (row[k0 + c] - sum) / diagBlock[c * kbs + c];
            }
        }

        if (k1 >= n) {
            continue; // no trailing submatrix left
        }

        // 3) Gather the freshly computed panel (rows [k1, n), cols [k0, k1))
        // from all ranks so every rank can perform its share of the
        // trailing-matrix update.
        const size_t nrem = n - k1;
        std::vector<int> recvcounts(P), displs(P);
        int total = 0;
        for (int r = 0; r < P; ++r) {
            auto it = std::lower_bound(allRows[r].begin(), allRows[r].end(), k1);
            const size_t cnt = static_cast<size_t>(allRows[r].end() - it);
            recvcounts[r] = static_cast<int>(cnt * kbs);
            displs[r] = total;
            total += recvcounts[r];
        }

        std::vector<double> sendbuf((nLocal - startIdx) * kbs);
        for (size_t li = startIdx; li < nLocal; ++li) {
            const double* row = &localA[li * n + k0];
            std::memcpy(sendbuf.data() + (li - startIdx) * kbs, row, kbs * sizeof(double));
        }

        std::vector<double> recvbuf(static_cast<size_t>(total));
        MPI_Allgatherv(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE,
                        recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                        MPI_COMM_WORLD);

        // Map each global row in [k1, n) to its position in recvbuf.
        std::vector<size_t> sortedPos(nrem);
        std::vector<size_t> counters(P, 0);
        for (size_t t = 0; t < nrem; ++t) {
            const size_t g = k1 + t;
            const int r = ownerOfRow(g, nb, P);
            sortedPos[t] = static_cast<size_t>(displs[r]) / kbs + counters[r];
            counters[r]++;
        }

        // 4) Trailing update: A[i][j] -= dot(panel[i], panel[j]) for
        // k1 <= j <= i, restricted to rows i owned locally.
        for (size_t li = startIdx; li < nLocal; ++li) {
            const size_t i = localRows[li];
            const double* rowI = &recvbuf[sortedPos[i - k1] * kbs];
            double* localRow = &localA[li * n];
            for (size_t j = k1; j <= i; ++j) {
                const double* rowJ = &recvbuf[sortedPos[j - k1] * kbs];
                double sum = 0.0;
                for (size_t c = 0; c < kbs; ++c) {
                    sum += rowI[c] * rowJ[c];
                }
                localRow[j] -= sum;
            }
        }
    }

    // Zero out the upper triangular part of every owned row.
    for (size_t li = 0; li < localRows.size(); ++li) {
        const size_t i = localRows[li];
        for (size_t j = i + 1; j < n; ++j) {
            localA[li * n + j] = 0.0;
        }
    }

    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
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

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", P);
    }

    // Choose a block size that gives each rank several blocks (for load
    // balance under the cyclic distribution) while staying cache friendly.
    size_t nb = n / std::max<size_t>(1, static_cast<size_t>(P) * 4);
    nb = std::max<size_t>(nb, static_cast<size_t>(32));
    nb = std::min<size_t>(nb, static_cast<size_t>(256));
    if (nb == 0 || nb > n) {
        nb = std::max<size_t>(n, static_cast<size_t>(1));
    }

    const std::vector<size_t> localRows = rowsOwnedByRank(n, nb, P, rank);
    std::vector<long long> localIndexOf(n, -1);
    for (size_t li = 0; li < localRows.size(); ++li) {
        localIndexOf[localRows[li]] = static_cast<long long>(li);
    }

    // Generate positive definite matrix (distributed by rows)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    std::vector<double> localA;
    generatePositiveDefiniteMatrix(localA, n, localRows);

    std::vector<double> A_orig;
    if (validate) {
        gatherFullMatrix(localA, A_orig, n, nb, rank, P, 0);
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionMPI(localA, n, nb, localRows, localIndexOf, rank, P);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    long long ms = duration.count();
    long long maxMs = 0;
    MPI_Reduce(&ms, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;

    // Print results for external validation
    if (printResults || validate) {
        std::vector<double> fullA;
        gatherFullMatrix(localA, fullA, n, nb, rank, P, 0);

        if (rank == 0) {
            if (printResults) {
                print_results(fullA, "CholeskyL");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(fullA, A_orig, n);

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
