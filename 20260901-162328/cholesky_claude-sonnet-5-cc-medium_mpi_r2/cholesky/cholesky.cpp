#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Blocked, distributed-memory Cholesky decomposition (right-looking variant).
// Rows of the n x n matrix are distributed across MPI ranks using a 1D
// block-cyclic mapping (block size nb). Each rank stores only the rows it
// owns, so memory scales with n^2 / numRanks. Decomposes a symmetric
// positive definite matrix A into L * L^T where L is lower triangular.

namespace {

int g_rank = 0;
int g_numRanks = 1;

size_t blockOfRow(size_t row, size_t nb) { return row / nb; }

int ownerOfBlock(size_t block, int numRanks) { return static_cast<int>(block % static_cast<size_t>(numRanks)); }

int ownerOfRow(size_t row, size_t nb, int numRanks) { return ownerOfBlock(blockOfRow(row, nb), numRanks); }

// Choose a block size that balances load across ranks while keeping blocks
// large enough to amortize communication overhead.
size_t chooseBlockSize(size_t n, int numRanks) {
    if (n == 0) return 1;
    if (n <= 64) return std::max<size_t>(1, n);

    const size_t targetBlocksPerRank = 4;
    const size_t targetBlocks = std::max<size_t>(1, static_cast<size_t>(numRanks) * targetBlocksPerRank);
    size_t nb = n / targetBlocks;
    nb = std::clamp(nb, static_cast<size_t>(16), static_cast<size_t>(128));
    return nb;
}

} // namespace

// Distributed blocked Cholesky decomposition.
// localA holds only the rows owned by this rank (in ascending global-row
// order), each stored as a full-length row of n doubles (row-major).
bool choleskyDecompositionMPI(std::vector<double>& localA, const std::vector<size_t>& localRows, const size_t n,
                               const size_t nb) {
    const int numRanks = g_numRanks;
    const size_t nBlocks = (n + nb - 1) / nb;

    // Map global row -> local storage index (only valid for owned rows).
    std::vector<long long> globalToLocal(n, -1);
    for (size_t li = 0; li < localRows.size(); ++li) {
        globalToLocal[localRows[li]] = static_cast<long long>(li);
    }

    std::vector<double> diagBlock; // kb * kb, row-major
    std::vector<double> sendBuf;
    std::vector<double> recvBuf;
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    std::vector<std::vector<size_t>> buckets(numRanks);
    std::vector<long long> panelIndexOf(n, -1);

    for (size_t k = 0; k < nBlocks; ++k) {
        const size_t kstart = k * nb;
        const size_t kend = std::min(kstart + nb, n);
        const size_t kb = kend - kstart;
        const int ownerK = ownerOfBlock(k, numRanks);

        // ---- Step 1: factorize the diagonal block (owner only) ----
        int localSuccess = 1;
        if (g_rank == ownerK) {
            for (size_t i = kstart; i < kend; ++i) {
                const long long iLocal = globalToLocal[i];
                double* rowI = &localA[static_cast<size_t>(iLocal) * n];
                for (size_t j = kstart; j <= i; ++j) {
                    const long long jLocal = globalToLocal[j];
                    const double* rowJ = &localA[static_cast<size_t>(jLocal) * n];
                    double sum = 0.0;
                    for (size_t p = kstart; p < j; ++p) {
                        sum += rowI[p] * rowJ[p];
                    }
                    if (i == j) {
                        const double val = rowI[j] - sum;
                        if (val <= 0.0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                            localSuccess = 0;
                            break;
                        }
                        rowI[j] = std::sqrt(val);
                    } else {
                        rowI[j] = (rowI[j] - sum) / rowJ[j];
                    }
                }
                if (!localSuccess) break;
            }
        }

        MPI_Bcast(&localSuccess, 1, MPI_INT, ownerK, MPI_COMM_WORLD);
        if (!localSuccess) return false;

        // ---- Step 2: broadcast the factorized diagonal block ----
        diagBlock.assign(kb * kb, 0.0);
        if (g_rank == ownerK) {
            for (size_t i = 0; i < kb; ++i) {
                const long long iLocal = globalToLocal[kstart + i];
                const double* rowI = &localA[static_cast<size_t>(iLocal) * n];
                for (size_t j = 0; j <= i; ++j) {
                    diagBlock[i * kb + j] = rowI[kstart + j];
                }
            }
        }
        MPI_Bcast(diagBlock.data(), static_cast<int>(kb * kb), MPI_DOUBLE, ownerK, MPI_COMM_WORLD);

        // ---- Step 3: panel triangular solve (each rank, own rows below block) ----
        for (size_t i = kend; i < n; ++i) {
            if (ownerOfRow(i, nb, numRanks) != g_rank) continue;
            const long long iLocal = globalToLocal[i];
            double* rowI = &localA[static_cast<size_t>(iLocal) * n];
            for (size_t j = kstart; j < kend; ++j) {
                const size_t jr = j - kstart;
                double sum = 0.0;
                for (size_t p = kstart; p < j; ++p) {
                    sum += rowI[p] * diagBlock[jr * kb + (p - kstart)];
                }
                rowI[j] = (rowI[j] - sum) / diagBlock[jr * kb + jr];
            }
        }

        if (kend >= n) continue; // last block: no trailing matrix left

        // ---- Step 4: gather the panel (columns [kstart,kend)) for all rows below the block ----
        for (int r = 0; r < numRanks; ++r) buckets[r].clear();
        for (size_t b = k + 1; b < nBlocks; ++b) {
            const int owner = ownerOfBlock(b, numRanks);
            const size_t rowStart = b * nb;
            const size_t rowEnd = std::min(rowStart + nb, n);
            for (size_t i = rowStart; i < rowEnd; ++i) {
                buckets[owner].push_back(i);
            }
        }

        size_t total = 0;
        for (int r = 0; r < numRanks; ++r) {
            displs[r] = static_cast<int>(total * kb);
            recvCounts[r] = static_cast<int>(buckets[r].size() * kb);
            for (size_t idx = 0; idx < buckets[r].size(); ++idx) {
                panelIndexOf[buckets[r][idx]] = static_cast<long long>(total + idx);
            }
            total += buckets[r].size();
        }

        sendBuf.resize(buckets[g_rank].size() * kb);
        for (size_t idx = 0; idx < buckets[g_rank].size(); ++idx) {
            const size_t i = buckets[g_rank][idx];
            const long long iLocal = globalToLocal[i];
            const double* rowI = &localA[static_cast<size_t>(iLocal) * n];
            std::memcpy(&sendBuf[idx * kb], &rowI[kstart], kb * sizeof(double));
        }

        recvBuf.resize(total * kb);
        MPI_Allgatherv(sendBuf.data(), static_cast<int>(sendBuf.size()), MPI_DOUBLE, recvBuf.data(),
                        recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // ---- Step 5: trailing (Schur complement) update on owned rows ----
        for (size_t idx = 0; idx < buckets[g_rank].size(); ++idx) {
            const size_t i = buckets[g_rank][idx];
            const long long iLocal = globalToLocal[i];
            double* rowI = &localA[static_cast<size_t>(iLocal) * n];
            const double* panelI = &recvBuf[static_cast<size_t>(panelIndexOf[i]) * kb];
            for (size_t j = kend; j <= i; ++j) {
                const double* panelJ = &recvBuf[static_cast<size_t>(panelIndexOf[j]) * kb];
                double sum = 0.0;
                for (size_t p = 0; p < kb; ++p) {
                    sum += panelI[p] * panelJ[p];
                }
                rowI[j] -= sum;
            }
        }
    }

    // Zero out the upper triangular part of each owned row.
    for (size_t li = 0; li < localRows.size(); ++li) {
        const size_t i = localRows[li];
        double* rowI = &localA[li * n];
        for (size_t j = i + 1; j < n; ++j) {
            rowI[j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix, computing only the rows
// owned by this rank. Every rank regenerates the (replicated) helper matrix
// B deterministically so that no communication is required, and each row of
// A is computed with the exact same summation order as the original
// sequential algorithm.
void generatePositiveDefiniteMatrixMPI(std::vector<double>& localA, const std::vector<size_t>& localRows,
                                        const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (deterministic, identical on every rank).
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T, only for the rows owned by this rank.
    for (size_t li = 0; li < localRows.size(); ++li) {
        const size_t i = localRows[li];
        double* rowOut = &localA[li * n];
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            rowOut[j] = sum;
        }
        // Add diagonal dominance to ensure positive definiteness.
        rowOut[i] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    // (only ever invoked on rank 0 with the fully gathered matrices).

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

// Gather a row-distributed matrix (local rows in ascending global order)
// into a full n x n row-major buffer on rank 0.
void gatherToRank0(const std::vector<double>& localA, const std::vector<size_t>& localRows, const size_t n,
                    const size_t nb, std::vector<double>& full) {
    const int numRanks = g_numRanks;

    // Build, for every rank, the ascending list of global rows it owns (this
    // is fully determined by the block-cyclic mapping, no communication
    // needed to discover it).
    std::vector<std::vector<size_t>> rowsByRank(numRanks);
    for (size_t i = 0; i < n; ++i) {
        rowsByRank[ownerOfRow(i, nb, numRanks)].push_back(i);
    }

    std::vector<int> recvCounts(numRanks), displs(numRanks);
    size_t total = 0;
    for (int r = 0; r < numRanks; ++r) {
        displs[r] = static_cast<int>(total * n);
        recvCounts[r] = static_cast<int>(rowsByRank[r].size() * n);
        total += rowsByRank[r].size();
    }

    std::vector<double> gathered;
    if (g_rank == 0) gathered.resize(total * n);

    MPI_Gatherv(localA.data(), static_cast<int>(localRows.size() * n), MPI_DOUBLE, gathered.data(),
                recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        full.resize(n * n);
        for (int r = 0; r < numRanks; ++r) {
            const double* base = &gathered[static_cast<size_t>(displs[r])];
            for (size_t idx = 0; idx < rowsByRank[r].size(); ++idx) {
                const size_t globalRow = rowsByRank[r][idx];
                std::memcpy(&full[globalRow * n], &base[idx * n], n * sizeof(double));
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_numRanks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank).
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", g_numRanks);
    }

    const size_t nb = chooseBlockSize(n, g_numRanks);

    // Determine which rows this rank owns (block-cyclic distribution).
    std::vector<size_t> localRows;
    for (size_t i = 0; i < n; ++i) {
        if (ownerOfRow(i, nb, g_numRanks) == g_rank) localRows.push_back(i);
    }

    std::vector<double> localA(localRows.size() * n);
    std::vector<double> localA_orig;

    // Generate positive definite matrix (distributed across ranks).
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrixMPI(localA, localRows, n);

    if (validate) {
        localA_orig = localA; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionMPI(localA, localRows, n, nb);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;

    if (printResults || validate) {
        std::vector<double> fullA;
        gatherToRank0(localA, localRows, n, nb, fullA);

        if (g_rank == 0 && printResults) {
            print_results(fullA, "CholeskyL");
        }

        if (validate) {
            std::vector<double> fullA_orig;
            gatherToRank0(localA_orig, localRows, n, nb, fullA_orig);

            if (g_rank == 0) {
                printf("Validating result...\n");
                bool valid = validateCholesky(fullA, fullA_orig, n);

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
