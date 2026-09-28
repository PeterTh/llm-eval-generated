#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory (MPI) blocked, right-looking Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is partitioned into nb x nb blocks which are assigned to MPI
// ranks in a 1D block-cyclic fashion over block-rows. At each step the owner
// of the current diagonal block factorizes it and broadcasts it; every rank
// then updates its own owned rows of the panel below the diagonal (no
// communication needed, since a rank owns full rows). The freshly computed
// panel is exchanged with an allgather so that every rank can apply the
// trailing-submatrix update to the blocks it owns.

namespace {

size_t numBlocks(size_t n, size_t nb) { return (n + nb - 1) / nb; }

size_t blockSizeOf(size_t I, size_t n, size_t nb) {
    return std::min(nb, n - I * nb);
}

int ownerOf(size_t I, int P) { return static_cast<int>(I % static_cast<size_t>(P)); }

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (n == 0) {
        return true;
    }

    // Pick a block size that gives several blocks per rank (for load
    // balance/pipelining) while staying in a cache-friendly range.
    size_t nb = std::max<size_t>(32, std::min<size_t>(256, n / std::max<size_t>(1, 4 * static_cast<size_t>(P))));
    nb = std::min(nb, n);
    if (nb == 0) nb = 1;

    const size_t nblocks = numBlocks(n, nb);

    // Blocks owned by this rank (block-cyclic over block-rows), increasing order.
    std::vector<size_t> myBlocks;
    for (size_t I = static_cast<size_t>(rank); I < nblocks; I += static_cast<size_t>(P)) {
        myBlocks.push_back(I);
    }

    std::vector<size_t> myBlockRowStart(myBlocks.size());
    size_t localNRows = 0;
    for (size_t k = 0; k < myBlocks.size(); ++k) {
        myBlockRowStart[k] = localNRows;
        localNRows += blockSizeOf(myBlocks[k], n, nb);
    }

    std::vector<double> localA(localNRows * n);

    // Scatter the full matrix (held on rank 0) into the block-cyclic distributed layout.
    if (rank == 0) {
        for (size_t k = 0; k < myBlocks.size(); ++k) {
            const size_t I = myBlocks[k];
            const size_t bs = blockSizeOf(I, n, nb);
            std::memcpy(&localA[myBlockRowStart[k] * n], &A[I * nb * n], bs * n * sizeof(double));
        }
        for (int p = 1; p < P; ++p) {
            size_t rows = 0;
            for (size_t I = static_cast<size_t>(p); I < nblocks; I += static_cast<size_t>(P)) {
                rows += blockSizeOf(I, n, nb);
            }
            if (rows == 0) continue;
            std::vector<double> sendBuf(rows * n);
            size_t off = 0;
            for (size_t I = static_cast<size_t>(p); I < nblocks; I += static_cast<size_t>(P)) {
                const size_t bs = blockSizeOf(I, n, nb);
                std::memcpy(&sendBuf[off * n], &A[I * nb * n], bs * n * sizeof(double));
                off += bs;
            }
            MPI_Send(sendBuf.data(), static_cast<int>(sendBuf.size()), MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else if (localNRows > 0) {
        MPI_Recv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    bool success = true;
    std::vector<double> diagBlock(nb * nb);
    std::vector<double> panelBuf;

    for (size_t I = 0; I < nblocks; ++I) {
        const size_t bs = blockSizeOf(I, n, nb);
        const int owner = ownerOf(I, P);

        // Step 1: factorize the diagonal block (owner only).
        if (rank == owner) {
            const size_t k = I / static_cast<size_t>(P);
            const size_t rs = myBlockRowStart[k];
            const size_t colBase = I * nb;

            for (size_t ii = 0; ii < bs && success; ++ii) {
                for (size_t jj = 0; jj <= ii; ++jj) {
                    double sum = 0.0;
                    double* rowI = &localA[(rs + ii) * n + colBase];
                    double* rowJ = &localA[(rs + jj) * n + colBase];
                    if (ii == jj) {
                        for (size_t kk = 0; kk < jj; ++kk) {
                            sum += rowJ[kk] * rowJ[kk];
                        }
                        const double val = rowJ[jj] - sum;
                        if (val <= 0.0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", I * nb + jj);
                            success = false;
                            break;
                        }
                        rowJ[jj] = std::sqrt(val);
                    } else {
                        for (size_t kk = 0; kk < jj; ++kk) {
                            sum += rowI[kk] * rowJ[kk];
                        }
                        rowI[jj] = (rowI[jj] - sum) / rowJ[jj];
                    }
                }
            }
            if (success) {
                for (size_t ii = 0; ii < bs; ++ii) {
                    for (size_t jj = ii + 1; jj < bs; ++jj) {
                        localA[(rs + ii) * n + colBase + jj] = 0.0;
                    }
                    std::memcpy(&diagBlock[ii * nb], &localA[(rs + ii) * n + colBase], bs * sizeof(double));
                }
            }
        }

        int successFlag = success ? 1 : 0;
        MPI_Bcast(&successFlag, 1, MPI_INT, owner, MPI_COMM_WORLD);
        success = successFlag != 0;
        if (!success) break;

        MPI_Bcast(diagBlock.data(), static_cast<int>(bs * nb), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Step 2: panel update (triangular solve) for locally owned rows below the diagonal.
        for (size_t k = 0; k < myBlocks.size(); ++k) {
            const size_t J = myBlocks[k];
            if (J <= I) continue;
            const size_t bsJ = blockSizeOf(J, n, nb);
            const size_t rs = myBlockRowStart[k];
            const size_t colBase = I * nb;
            for (size_t r = 0; r < bsJ; ++r) {
                double* row = &localA[(rs + r) * n + colBase];
                for (size_t c = 0; c < bs; ++c) {
                    double sum = 0.0;
                    for (size_t kk = 0; kk < c; ++kk) {
                        sum += row[kk] * diagBlock[c * nb + kk];
                    }
                    row[c] = (row[c] - sum) / diagBlock[c * nb + c];
                }
            }
        }

        // Step 3: gather the freshly computed panel column so every rank can
        // apply the trailing-submatrix update to the blocks it owns.
        size_t rowsBelow = 0;
        for (size_t J = I + 1; J < nblocks; ++J) rowsBelow += blockSizeOf(J, n, nb);

        std::vector<int> recvRows(P, 0), displsRows(P, 0);
        int myRowsBelow = 0;
        for (size_t k = 0; k < myBlocks.size(); ++k) {
            if (myBlocks[k] > I) myRowsBelow += static_cast<int>(blockSizeOf(myBlocks[k], n, nb));
        }
        MPI_Allgather(&myRowsBelow, 1, MPI_INT, recvRows.data(), 1, MPI_INT, MPI_COMM_WORLD);
        {
            int acc = 0;
            for (int p = 0; p < P; ++p) { displsRows[p] = acc; acc += recvRows[p]; }
        }

        // Row offset (within the gathered panel buffer) of the first row of each block J>I.
        std::vector<size_t> blockPanelRowStart(nblocks, 0);
        for (int p = 0; p < P; ++p) {
            size_t cum = 0;
            for (size_t J = static_cast<size_t>(p); J < nblocks; J += static_cast<size_t>(P)) {
                if (J <= I) continue;
                blockPanelRowStart[J] = static_cast<size_t>(displsRows[p]) + cum;
                cum += blockSizeOf(J, n, nb);
            }
        }

        std::vector<double> sendPanel(static_cast<size_t>(myRowsBelow) * bs);
        {
            size_t off = 0;
            for (size_t k = 0; k < myBlocks.size(); ++k) {
                const size_t J = myBlocks[k];
                if (J <= I) continue;
                const size_t bsJ = blockSizeOf(J, n, nb);
                const size_t rs = myBlockRowStart[k];
                const size_t colBase = I * nb;
                for (size_t r = 0; r < bsJ; ++r) {
                    std::memcpy(&sendPanel[(off + r) * bs], &localA[(rs + r) * n + colBase], bs * sizeof(double));
                }
                off += bsJ;
            }
        }

        std::vector<int> recvCounts(P), displs(P);
        for (int p = 0; p < P; ++p) {
            recvCounts[p] = recvRows[p] * static_cast<int>(bs);
            displs[p] = displsRows[p] * static_cast<int>(bs);
        }
        panelBuf.assign(rowsBelow * bs, 0.0);
        MPI_Allgatherv(sendPanel.data(), static_cast<int>(sendPanel.size()), MPI_DOUBLE, panelBuf.data(),
                        recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Step 4: trailing-submatrix update for locally owned blocks.
        for (size_t k = 0; k < myBlocks.size(); ++k) {
            const size_t J = myBlocks[k];
            if (J <= I) continue;
            const size_t bsJ = blockSizeOf(J, n, nb);
            const size_t rs = myBlockRowStart[k];
            const double* LJI = &panelBuf[blockPanelRowStart[J] * bs];

            for (size_t K = I + 1; K <= J; ++K) {
                const size_t bsK = blockSizeOf(K, n, nb);
                const double* LKI = &panelBuf[blockPanelRowStart[K] * bs];
                const size_t colBase = K * nb;

                if (K == J) {
                    for (size_t r = 0; r < bsJ; ++r) {
                        double* dst = &localA[(rs + r) * n + colBase];
                        const double* Lr = &LJI[r * bs];
                        for (size_t c = 0; c <= r; ++c) {
                            const double* Lc = &LJI[c * bs];
                            double sum = 0.0;
                            for (size_t kk = 0; kk < bs; ++kk) sum += Lr[kk] * Lc[kk];
                            dst[c] -= sum;
                        }
                    }
                } else {
                    for (size_t r = 0; r < bsJ; ++r) {
                        double* dst = &localA[(rs + r) * n + colBase];
                        const double* Lr = &LJI[r * bs];
                        for (size_t c = 0; c < bsK; ++c) {
                            const double* Lc = &LKI[c * bs];
                            double sum = 0.0;
                            for (size_t kk = 0; kk < bs; ++kk) sum += Lr[kk] * Lc[kk];
                            dst[c] -= sum;
                        }
                    }
                }
            }
        }
    }

    if (success) {
        // Zero out the (dead) upper-triangular entries beyond each owned
        // block's diagonal so the final layout matches the sequential result.
        for (size_t k = 0; k < myBlocks.size(); ++k) {
            const size_t I = myBlocks[k];
            const size_t bs = blockSizeOf(I, n, nb);
            const size_t rs = myBlockRowStart[k];
            const size_t colEnd = I * nb + bs;
            if (colEnd < n) {
                for (size_t r = 0; r < bs; ++r) {
                    std::memset(&localA[(rs + r) * n + colEnd], 0, (n - colEnd) * sizeof(double));
                }
            }
        }

        // Gather the distributed result back into the full matrix on rank 0.
        if (rank == 0) {
            for (size_t k = 0; k < myBlocks.size(); ++k) {
                const size_t I = myBlocks[k];
                const size_t bs = blockSizeOf(I, n, nb);
                std::memcpy(&A[I * nb * n], &localA[myBlockRowStart[k] * n], bs * n * sizeof(double));
            }
            for (int p = 1; p < P; ++p) {
                size_t rows = 0;
                for (size_t I = static_cast<size_t>(p); I < nblocks; I += static_cast<size_t>(P)) {
                    rows += blockSizeOf(I, n, nb);
                }
                if (rows == 0) continue;
                std::vector<double> recvBuf(rows * n);
                MPI_Recv(recvBuf.data(), static_cast<int>(recvBuf.size()), MPI_DOUBLE, p, 1, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE);
                size_t off = 0;
                for (size_t I = static_cast<size_t>(p); I < nblocks; I += static_cast<size_t>(P)) {
                    const size_t bs = blockSizeOf(I, n, nb);
                    std::memcpy(&A[I * nb * n], &recvBuf[off * n], bs * n * sizeof(double));
                    off += bs;
                }
            }
        } else if (localNRows > 0) {
            MPI_Send(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
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
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // mpirun/mpiexec delivers identical argv to every rank, so every rank
    // can parse the command line independently and stay in sync without
    // any extra communication.
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
        printf("MPI ranks: %d\n", numRanks);
    }

    // Allocate matrix (only rank 0 needs the full matrix; other ranks work
    // on their own distributed shard inside choleskyDecomposition).
    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        A.assign(n * n, 0.0);

        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);

        if (validate) {
            A_orig = A; // Save original for validation
        }

        printf("Computing Cholesky decomposition...\n");
    }

    // Perform Cholesky decomposition (collective across all MPI ranks)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    int retcode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
                retcode = 0;
            } else {
                printf("Validation: FAILED\n");
                retcode = 1;
            }
        }
    }

    MPI_Bcast(&retcode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return retcode;
}
