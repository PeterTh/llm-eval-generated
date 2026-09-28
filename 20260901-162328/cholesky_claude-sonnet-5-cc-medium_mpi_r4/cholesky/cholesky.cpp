#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (right-looking blocked algorithm).
//
// The n x n matrix is partitioned into column-blocks of width NB, which are
// assigned to MPI ranks in a 1D block-cyclic fashion (block k is owned by
// rank k % size). Each rank only stores the lower-triangular portion of the
// column-blocks it owns (rows [k*NB, n) x cols [k*NB, k*NB+bw)), so memory
// scales down with the number of ranks. At each step k, the owner factorizes
// its diagonal/panel block and the result is broadcast to all ranks, which
// then apply the trailing Schur-complement update to the column-blocks they
// own. This mirrors the structure of ScaLAPACK's 1D PDPOTRF.

namespace {

inline size_t numBlocksFor(size_t n, size_t NB) {
    return (n + NB - 1) / NB;
}

inline size_t blockWidth(size_t k, size_t n, size_t NB) {
    const size_t rowStart = k * NB;
    return std::min(NB, n - rowStart);
}

inline size_t blockRowCount(size_t k, size_t n, size_t NB) {
    return n - k * NB;
}

inline int blockOwner(size_t k, int size) {
    return static_cast<int>(k % static_cast<size_t>(size));
}

// Local storage for the column-blocks owned by this rank. Block with global
// index k (owned by this rank) is stored at localBlocks[k / size], as a
// row-major (rowCount x bw) matrix covering rows [k*NB, n).
using LocalBlocks = std::vector<std::vector<double>>;

LocalBlocks allocateLocalBlocks(size_t n, size_t NB, size_t numBlocks, int rank, int size) {
    LocalBlocks blocks;
    size_t localCount = 0;
    for (size_t k = static_cast<size_t>(rank); k < numBlocks; k += static_cast<size_t>(size)) {
        ++localCount;
    }
    blocks.resize(localCount);
    for (size_t k = static_cast<size_t>(rank); k < numBlocks; k += static_cast<size_t>(size)) {
        const size_t idx = k / static_cast<size_t>(size);
        blocks[idx].resize(blockRowCount(k, n, NB) * blockWidth(k, n, NB));
    }
    return blocks;
}

// Generate the same random matrix B as the sequential reference implementation.
// This part is O(n^2) so it is cheap enough to replicate identically on every
// rank (keeps the PRNG sequence, and thus the generated matrix, bit-identical
// to the original single-process code).
std::vector<double> generateB(size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    return B;
}

// Fills the locally-owned blocks with A = B * B^T + n*I, restricted to the
// lower-triangular region each block stores. This reproduces
// generatePositiveDefiniteMatrix() exactly, but each rank only computes the
// entries it owns (O(n^3 / size) work per rank instead of O(n^3)).
void generateLocalBlocks(LocalBlocks& blocks, const std::vector<double>& B, size_t n, size_t NB,
                          size_t numBlocks, int rank, int size) {
    for (size_t k = static_cast<size_t>(rank); k < numBlocks; k += static_cast<size_t>(size)) {
        const size_t idx = k / static_cast<size_t>(size);
        const size_t bw = blockWidth(k, n, NB);
        const size_t rowCount = blockRowCount(k, n, NB);
        const size_t rowStart = k * NB;
        std::vector<double>& data = blocks[idx];

        for (size_t c = 0; c < bw; ++c) {
            const size_t globalCol = k * NB + c;
            const double* colPtr = &B[globalCol * n];
            for (size_t i = 0; i < rowCount; ++i) {
                const size_t globalRow = rowStart + i;
                const double* rowPtr = &B[globalRow * n];
                double sum = 0.0;
                for (size_t p = 0; p < n; ++p) {
                    sum += rowPtr[p] * colPtr[p];
                }
                if (globalRow == globalCol) {
                    sum += static_cast<double>(n);
                }
                data[i * bw + c] = sum;
            }
        }
    }
}

// Right-looking blocked Cholesky factorization over the 1D block-cyclic
// distribution described above. Returns false (consistently on all ranks) if
// the matrix is not positive definite.
bool choleskyDecompositionMPI(LocalBlocks& blocks, size_t n, size_t NB, size_t numBlocks, int rank,
                               int size, MPI_Comm comm) {
    std::vector<double> panelBuf;

    for (size_t k = 0; k < numBlocks; ++k) {
        const int owner = blockOwner(k, size);
        const size_t bwK = blockWidth(k, n, NB);
        const size_t rowCountK = blockRowCount(k, n, NB);
        const size_t rowStartK = k * NB;
        const size_t panelCount = rowCountK * bwK;

        int localSuccess = 1;
        double* panelPtr = nullptr;

        if (rank == owner) {
            const size_t idx = k / static_cast<size_t>(size);
            std::vector<double>& data = blocks[idx];

            for (size_t jcol = 0; jcol < bwK && localSuccess; ++jcol) {
                double sum = 0.0;
                for (size_t p = 0; p < jcol; ++p) {
                    const double v = data[jcol * bwK + p];
                    sum += v * v;
                }
                const double val = data[jcol * bwK + jcol] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           rowStartK + jcol);
                    localSuccess = 0;
                    break;
                }
                const double Ljj = std::sqrt(val);
                data[jcol * bwK + jcol] = Ljj;

                for (size_t irow = jcol + 1; irow < rowCountK; ++irow) {
                    double s = 0.0;
                    for (size_t p = 0; p < jcol; ++p) {
                        s += data[irow * bwK + p] * data[jcol * bwK + p];
                    }
                    data[irow * bwK + jcol] = (data[irow * bwK + jcol] - s) / Ljj;
                }
            }

            // Zero the strict upper triangle of the diagonal block, matching
            // the sequential reference implementation's output.
            for (size_t i = 0; i < bwK; ++i) {
                for (size_t j = i + 1; j < bwK; ++j) {
                    data[i * bwK + j] = 0.0;
                }
            }

            panelPtr = data.data();
        }

        MPI_Bcast(&localSuccess, 1, MPI_INT, owner, comm);
        if (!localSuccess) {
            return false;
        }

        if (rank != owner) {
            panelBuf.resize(panelCount);
            panelPtr = panelBuf.data();
        }
        MPI_Bcast(panelPtr, static_cast<int>(panelCount), MPI_DOUBLE, owner, comm);

        // Trailing update: for every column-block j > k owned by this rank,
        // apply the rank-bwK Schur-complement update using the broadcast panel.
        for (size_t j = k + 1; j < numBlocks; ++j) {
            if (blockOwner(j, size) != rank) {
                continue;
            }
            const size_t bwJ = blockWidth(j, n, NB);
            const size_t rowCountJ = blockRowCount(j, n, NB);
            const size_t rowStartJ = j * NB;
            const size_t offset = rowStartJ - rowStartK;

            const double* Pj = panelPtr + offset * bwK;
            std::vector<double>& dataJ = blocks[j / static_cast<size_t>(size)];

            for (size_t i = 0; i < rowCountJ; ++i) {
                const double* rowI = Pj + i * bwK;
                double* outRow = &dataJ[i * bwJ];
                for (size_t c = 0; c < bwJ; ++c) {
                    const double* rowC = Pj + c * bwK;
                    double s = 0.0;
                    for (size_t p = 0; p < bwK; ++p) {
                        s += rowI[p] * rowC[p];
                    }
                    outRow[c] -= s;
                }
            }
        }
    }

    return true;
}

// Gathers the distributed lower-triangular blocks into a full n x n matrix on
// rank 0 (row-major). Non-root ranks return an empty vector. If `mirror` is
// set, the upper triangle is filled with the transpose of the lower triangle
// (used to reconstruct the full symmetric original matrix); otherwise the
// upper triangle is left at zero (matching the sequential algorithm's output).
std::vector<double> assembleGlobalMatrix(const LocalBlocks& blocks, size_t n, size_t NB,
                                          size_t numBlocks, int rank, int size, MPI_Comm comm,
                                          bool mirror) {
    std::vector<double> global;
    if (rank == 0) {
        global.assign(n * n, 0.0);
    }

    for (size_t k = 0; k < numBlocks; ++k) {
        const int owner = blockOwner(k, size);
        const size_t bw = blockWidth(k, n, NB);
        const size_t rowCount = blockRowCount(k, n, NB);
        const size_t rowStart = k * NB;
        const size_t count = rowCount * bw;

        const double* src = nullptr;
        std::vector<double> tmp;

        if (rank == 0 && owner == 0) {
            src = blocks[k / static_cast<size_t>(size)].data();
        } else if (rank == 0 && owner != 0) {
            tmp.resize(count);
            MPI_Recv(tmp.data(), static_cast<int>(count), MPI_DOUBLE, owner, static_cast<int>(k), comm,
                     MPI_STATUS_IGNORE);
            src = tmp.data();
        } else if (rank == owner && owner != 0) {
            MPI_Send(blocks[k / static_cast<size_t>(size)].data(), static_cast<int>(count), MPI_DOUBLE, 0,
                      static_cast<int>(k), comm);
        }

        if (rank == 0) {
            for (size_t i = 0; i < rowCount; ++i) {
                const size_t globalRow = rowStart + i;
                for (size_t c = 0; c < bw; ++c) {
                    const size_t globalCol = k * NB + c;
                    const double val = src[i * bw + c];
                    global[globalRow * n + globalCol] = val;
                    if (mirror && globalRow != globalCol) {
                        global[globalCol * n + globalRow] = val;
                    }
                }
            }
        }
    }

    return global;
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

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on every rank under mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Block size for the 1D block-cyclic column distribution: capped for
    // cache friendliness, but small enough that every rank gets several
    // blocks so work stays balanced.
    const size_t NB = std::max<size_t>(1, std::min<size_t>(128, n / static_cast<size_t>(std::max(1, size))));
    const size_t numBlocks = numBlocksFor(n, NB);

    LocalBlocks blocks = allocateLocalBlocks(n, NB, numBlocks, rank, size);

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> B = generateB(n);
    generateLocalBlocks(blocks, B, n, NB, numBlocks, rank, size);
    B.clear();
    B.shrink_to_fit();

    LocalBlocks origBlocks;
    if (validate) {
        origBlocks = blocks; // deep copy, preserved for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionMPI(blocks, n, NB, numBlocks, rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMs = localDuration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults || validate) {
        std::vector<double> L = assembleGlobalMatrix(blocks, n, NB, numBlocks, rank, size, MPI_COMM_WORLD,
                                                       /*mirror=*/false);

        if (rank == 0 && printResults) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            std::vector<double> A_orig = assembleGlobalMatrix(origBlocks, n, NB, numBlocks, rank, size,
                                                                MPI_COMM_WORLD, /*mirror=*/true);
            if (rank == 0) {
                printf("Validating result...\n");
                bool valid = validateCholesky(L, A_orig, n);

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
