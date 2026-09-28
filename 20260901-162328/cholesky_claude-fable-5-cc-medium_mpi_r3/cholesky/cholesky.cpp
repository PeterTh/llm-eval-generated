#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (distributed memory)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization: rows are distributed block-cyclically across ranks with
// block size NB. The factorization proceeds panel by panel (left-looking):
// the rank owning a panel of NB consecutive rows finalizes those rows locally,
// broadcasts the panel, and then all ranks update the corresponding columns of
// their local rows below the panel. Dot products use the same ascending-k
// order as the sequential algorithm, so results are bitwise identical.

static constexpr size_t NB = 64; // panel/block size for block-cyclic distribution

static int g_rank = 0;
static int g_size = 1;

static inline int ownerOfRow(const size_t i) {
    return (int)((i / NB) % (size_t)g_size);
}

// Number of rows owned by rank r
static size_t numLocalRows(const size_t n, const int r) {
    size_t count = 0;
    const size_t nBlocks = (n + NB - 1) / NB;
    for (size_t b = (size_t)r; b < nBlocks; b += (size_t)g_size) {
        count += std::min(NB, n - b * NB);
    }
    return count;
}

// Local storage index of global row i on its owner rank
static inline size_t localRowIndex(const size_t i) {
    const size_t b = i / NB;
    return (b / (size_t)g_size) * NB + (i % NB);
}

bool choleskyDecomposition(std::vector<double>& localA, const size_t n) {
    // localA holds this rank's block-cyclic rows, each of length n
    std::vector<double> panel(NB * n);
    const size_t nLocal = localA.size() / n;

    for (size_t j0 = 0; j0 < n; j0 += NB) {
        const size_t j1 = std::min(j0 + NB, n);
        const size_t pnb = j1 - j0; // panel height
        const int owner = ownerOfRow(j0);
        int ok = 1;

        if (g_rank == owner) {
            // Factor the panel rows locally: columns 0..j0-1 of these rows
            // were finalized by previous panel updates.
            for (size_t j = j0; j < j1 && ok; ++j) {
                double* rowJ = &localA[localRowIndex(j) * n];
                // Off-diagonal elements within the panel columns
                for (size_t m = j0; m < j; ++m) {
                    const double* rowM = &localA[localRowIndex(m) * n];
                    double sum = 0.0;
                    for (size_t k = 0; k < m; ++k) {
                        sum += rowJ[k] * rowM[k];
                    }
                    rowJ[m] = (rowJ[m] - sum) / rowM[m];
                }
                // Diagonal element
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += rowJ[k] * rowJ[k];
                }
                const double val = rowJ[j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    ok = 0;
                } else {
                    rowJ[j] = sqrt(val);
                }
            }
            // Pack panel rows (columns 0..j1-1)
            for (size_t j = j0; j < j1; ++j) {
                memcpy(&panel[(j - j0) * j1], &localA[localRowIndex(j) * n], j1 * sizeof(double));
            }
        }

        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) {
            return false;
        }
        MPI_Bcast(panel.data(), (int)(pnb * j1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Update columns j0..j1-1 of all local rows below the panel
        for (size_t li = 0; li < nLocal; ++li) {
            const size_t blk = (li / NB) * (size_t)g_size + (size_t)g_rank;
            const size_t i = blk * NB + (li % NB);
            if (i < j1) continue;
            double* rowI = &localA[li * n];
            for (size_t j = j0; j < j1; ++j) {
                const double* rowJ = &panel[(j - j0) * j1];
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += rowI[k] * rowJ[k];
                }
                rowI[j] = (rowI[j] - sum) / rowJ[j];
            }
        }
    }

    // Zero out upper triangular part of local rows
    for (size_t li = 0; li < nLocal; ++li) {
        const size_t blk = (li / NB) * (size_t)g_size + (size_t)g_rank;
        const size_t i = blk * NB + (li % NB);
        for (size_t j = i + 1; j < n; ++j) {
            localA[li * n + j] = 0.0;
        }
    }

    return true;
}

// Generate this rank's block-cyclic rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& localA, const size_t n) {
    // Method: Create A = B * B^T where B is random (same B on every rank,
    // identical to the sequential generator). Each rank computes its rows.

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute local rows of A = B * B^T
    const size_t nLocal = localA.size() / n;
    for (size_t li = 0; li < nLocal; ++li) {
        const size_t blk = (li / NB) * (size_t)g_size + (size_t)g_rank;
        const size_t i = blk * NB + (li % NB);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            localA[li * n + j] = sum;
        }
        // Add diagonal dominance to ensure positive definiteness
        localA[li * n + i] += n;
    }
}

// Assemble the full matrix on every rank from the block-cyclic local rows
void gatherFullMatrix(const std::vector<double>& localA, std::vector<double>& full, const size_t n) {
    std::vector<int> counts(g_size), displs(g_size);
    int off = 0;
    for (int r = 0; r < g_size; ++r) {
        counts[r] = (int)(numLocalRows(n, r) * n);
        displs[r] = off;
        off += counts[r];
    }

    std::vector<double> gathered((size_t)off);
    MPI_Allgatherv(localA.data(), (int)localA.size(), MPI_DOUBLE,
                   gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Reorder block-cyclic rows into the full matrix
    full.assign(n * n, 0.0);
    std::vector<size_t> pos(g_size);
    for (int r = 0; r < g_size; ++r) {
        pos[r] = (size_t)displs[r];
    }
    const size_t nBlocks = (n + NB - 1) / NB;
    for (size_t b = 0; b < nBlocks; ++b) {
        const int r = (int)(b % (size_t)g_size);
        const size_t rows = std::min(NB, n - b * NB);
        memcpy(&full[b * NB * n], &gathered[pos[r]], rows * n * sizeof(double));
        pos[r] += rows * n;
    }
}

bool validateCholesky(const std::vector<double>& Lfull, const std::vector<double>& localOrig,
                      const size_t n) {
    // Validate by computing L * L^T for local rows and comparing with the
    // original matrix rows; error maxima are reduced across ranks.
    const size_t nLocal = localOrig.size() / n;

    double maxError = 0.0;
    double relError = 0.0;

    std::vector<double> reconstructedRow(n);
    for (size_t li = 0; li < nLocal; ++li) {
        const size_t blk = (li / NB) * (size_t)g_size + (size_t)g_rank;
        const size_t i = blk * NB + (li % NB);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Lfull[i * n + k] * Lfull[j * n + k];
            }
            reconstructedRow[j] = sum;
        }
        for (size_t j = 0; j < n; ++j) {
            const double error = fabs(reconstructedRow[j] - localOrig[li * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(localOrig[li * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (g_rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", g_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate this rank's block-cyclic rows
    std::vector<double> localA(numLocalRows(n, g_rank) * n);
    std::vector<double> localOrig;

    // Generate positive definite matrix
    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(localA, n);

    if (validate) {
        localOrig = localA; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(localA, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
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

    // Assemble the full factor where needed
    std::vector<double> Lfull;
    if (validate || printResults) {
        gatherFullMatrix(localA, Lfull, n);
    }

    // Print results for external validation
    if (printResults && g_rank == 0) {
        print_results(Lfull, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (g_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(Lfull, localOrig, n);

        if (g_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
