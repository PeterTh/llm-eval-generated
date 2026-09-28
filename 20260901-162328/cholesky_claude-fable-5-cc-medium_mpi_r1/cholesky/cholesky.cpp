#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (left-looking, panel-blocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Distribution: rows are assigned to ranks in a block-cyclic fashion with block
// size NB (row block b belongs to rank b % nprocs). Column panel k shares its
// index range with row block k, so the diagonal block of every panel is fully
// owned by a single rank. Per panel:
//   1. The owner factors the NB x NB diagonal block (left-looking, so all
//      columns to the left of the panel are already final in its rows).
//   2. The panel rows (their full prefix up to the panel end) are broadcast.
//   3. Every rank finalizes the panel columns of its own trailing rows using a
//      register-blocked kernel over the transposed panel prefix.
// Each matrix element is computed with the exact same accumulation order as
// the sequential algorithm (a single sum over k ascending, then subtract and
// divide), so the numerical behavior matches the original code.

constexpr size_t NB = 64; // row block height == column panel width

static inline int blockOwner(const size_t b, const int nprocs) {
    return (int)(b % (size_t)nprocs);
}

// Finalize columns [j0, j1) of rows [r0, r1) of A (rows owned by this rank).
// prow points at the broadcast panel rows: pw rows of length j1 each, holding
// rows j0..j1 of L (columns 0..j1). Bt is the transposed panel prefix,
// Bt[t * pw + c] = L[j0 + c][t] for t < j0.
static void updatePanelRows(double* A, const size_t n, const size_t r0, const size_t r1,
                            const size_t j0, const size_t j1, const double* prow,
                            const double* Bt) {
    const size_t pw = j1 - j0;
    constexpr size_t RB = 4; // rows per register block
    constexpr size_t CB = 8; // columns per register block

    for (size_t r = r0; r < r1; r += RB) {
        const size_t rcnt = std::min(RB, r1 - r);
        for (size_t c0 = 0; c0 < pw; c0 += CB) {
            const size_t ccnt = std::min(CB, pw - c0);
            double acc[RB][CB];

            if (rcnt == RB && ccnt == CB) {
                // Main kernel: 4 rows x 8 columns of accumulators held across
                // the t loop; inner loop vectorizes over the contiguous Bt row.
                const double* a0 = A + (r + 0) * n;
                const double* a1 = A + (r + 1) * n;
                const double* a2 = A + (r + 2) * n;
                const double* a3 = A + (r + 3) * n;
                double acc0[CB] = {0}, acc1[CB] = {0}, acc2[CB] = {0}, acc3[CB] = {0};
                for (size_t t = 0; t < j0; ++t) {
                    const double* bt = Bt + t * pw + c0;
                    const double v0 = a0[t], v1 = a1[t], v2 = a2[t], v3 = a3[t];
                    for (size_t c = 0; c < CB; ++c) {
                        acc0[c] += v0 * bt[c];
                        acc1[c] += v1 * bt[c];
                        acc2[c] += v2 * bt[c];
                        acc3[c] += v3 * bt[c];
                    }
                }
                for (size_t c = 0; c < CB; ++c) {
                    acc[0][c] = acc0[c];
                    acc[1][c] = acc1[c];
                    acc[2][c] = acc2[c];
                    acc[3][c] = acc3[c];
                }
            } else {
                for (size_t rr = 0; rr < rcnt; ++rr) {
                    const double* ar = A + (r + rr) * n;
                    for (size_t c = 0; c < ccnt; ++c) acc[rr][c] = 0.0;
                    for (size_t t = 0; t < j0; ++t) {
                        const double* bt = Bt + t * pw + c0;
                        const double v = ar[t];
                        for (size_t c = 0; c < ccnt; ++c) acc[rr][c] += v * bt[c];
                    }
                }
            }

            // Finish the triangular tail (t in [j0, c)) and write the results.
            // Columns must be finalized in ascending order because the tail
            // reads panel columns of the same row that were just written.
            for (size_t rr = 0; rr < rcnt; ++rr) {
                double* ar = A + (r + rr) * n;
                for (size_t cc = 0; cc < ccnt; ++cc) {
                    const size_t cl = c0 + cc;   // column index within panel
                    const size_t c = j0 + cl;    // global column index
                    double sum = acc[rr][cc];
                    const double* lrow = prow + cl * j1;
                    for (size_t t = j0; t < c; ++t) {
                        sum += ar[t] * lrow[t];
                    }
                    ar[c] = (ar[c] - sum) / lrow[c];
                }
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, const int rank,
                           const int nprocs) {
    const size_t nblocks = (n + NB - 1) / NB;
    std::vector<double> buf;
    std::vector<double> Bt;

    for (size_t kb = 0; kb < nblocks; ++kb) {
        const size_t j0 = kb * NB;
        const size_t j1 = std::min(j0 + NB, n);
        const size_t pw = j1 - j0;
        const int owner = blockOwner(kb, nprocs);

        // buf[0] is a status flag; the rest holds rows j0..j1, columns 0..j1
        buf.assign(1 + pw * j1, 0.0);

        if (rank == owner) {
            // Factor the diagonal block in place (rows j0..j1 are local and
            // their columns < j0 are already final)
            double status = 1.0;
            for (size_t c = j0; c < j1; ++c) {
                double sum = 0.0;
                for (size_t t = 0; t < c; ++t) {
                    sum += A[c * n + t] * A[c * n + t];
                }
                const double val = A[c * n + c] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", c);
                    status = -1.0;
                }
                if (status < 0.0) {
                    break;
                }
                A[c * n + c] = sqrt(val);

                for (size_t r = c + 1; r < j1; ++r) {
                    double s = 0.0;
                    for (size_t t = 0; t < c; ++t) {
                        s += A[r * n + t] * A[c * n + t];
                    }
                    A[r * n + c] = (A[r * n + c] - s) / A[c * n + c];
                }
            }
            buf[0] = status;
            if (status > 0.0) {
                for (size_t c = j0; c < j1; ++c) {
                    memcpy(&buf[1 + (c - j0) * j1], &A[c * n], j1 * sizeof(double));
                }
            }
        }

        MPI_Bcast(buf.data(), (int)buf.size(), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (buf[0] <= 0.0) {
            return false;
        }
        const double* prow = buf.data() + 1;

        // Transpose the panel prefix for the vectorized update kernel
        Bt.assign(j0 * pw, 0.0);
        for (size_t c = 0; c < pw; ++c) {
            for (size_t t = 0; t < j0; ++t) {
                Bt[t * pw + c] = prow[c * j1 + t];
            }
        }

        // Finalize the panel columns of all locally owned trailing rows
        for (size_t b = kb + 1; b < nblocks; ++b) {
            if (blockOwner(b, nprocs) != rank) continue;
            const size_t r0 = b * NB;
            const size_t r1 = std::min(r0 + NB, n);
            updatePanelRows(A.data(), n, r0, r1, j0, j1, prow, Bt.data());
        }
    }

    // Zero out upper triangular part of the locally owned rows
    for (size_t b = 0; b < nblocks; ++b) {
        if (blockOwner(b, nprocs) != rank) continue;
        const size_t r0 = b * NB;
        const size_t r1 = std::min(r0 + NB, n);
        for (size_t r = r0; r < r1; ++r) {
            if (r + 1 < n) {
                memset(&A[r * n + r + 1], 0, (n - r - 1) * sizeof(double));
            }
        }
    }

    return true;
}

// Generate the locally owned rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const int rank,
                                    const int nprocs) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (identical on every rank, so the distributed
    // rows match the sequential reference exactly)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute the owned rows of A = B * B^T
    const size_t nblocks = (n + NB - 1) / NB;
    for (size_t b = 0; b < nblocks; ++b) {
        if (blockOwner(b, nprocs) != rank) continue;
        const size_t r0 = b * NB;
        const size_t r1 = std::min(r0 + NB, n);
        for (size_t i = r0; i < r1; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                A[i * n + j] = sum;
            }
            // Add diagonal dominance to ensure positive definiteness
            A[i * n + i] += n;
        }
    }
}

// Gather the block-cyclically distributed rows so every rank holds the full matrix
void gatherFullMatrix(std::vector<double>& A, const size_t n, const int rank,
                      const int nprocs) {
    const size_t nblocks = (n + NB - 1) / NB;
    std::vector<int> counts(nprocs, 0), displs(nprocs, 0);
    for (size_t b = 0; b < nblocks; ++b) {
        const size_t rows = std::min(NB, n - b * NB);
        counts[blockOwner(b, nprocs)] += (int)(rows * n);
    }
    for (int r = 1; r < nprocs; ++r) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    // Pack the owned row blocks in ascending order
    std::vector<double> send((size_t)counts[rank]);
    size_t off = 0;
    for (size_t b = 0; b < nblocks; ++b) {
        if (blockOwner(b, nprocs) != rank) continue;
        const size_t rows = std::min(NB, n - b * NB);
        memcpy(&send[off], &A[b * NB * n], rows * n * sizeof(double));
        off += rows * n;
    }

    std::vector<double> recv(n * n);
    MPI_Allgatherv(send.data(), counts[rank], MPI_DOUBLE, recv.data(), counts.data(),
                   displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Unpack into the full matrix
    std::vector<size_t> offsets(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        offsets[r] = (size_t)displs[r];
    }
    for (size_t b = 0; b < nblocks; ++b) {
        const int o = blockOwner(b, nprocs);
        const size_t rows = std::min(NB, n - b * NB);
        memcpy(&A[b * NB * n], &recv[offsets[o]], rows * n * sizeof(double));
        offsets[o] += rows * n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n, const int rank, const int nprocs) {
    // Validate by computing L * L^T on the owned rows and comparing with the
    // original matrix; L must hold the full gathered result on every rank

    double maxError = 0.0;
    double relError = 0.0;

    const size_t nblocks = (n + NB - 1) / NB;
    for (size_t b = 0; b < nblocks; ++b) {
        if (blockOwner(b, nprocs) != rank) continue;
        const size_t r0 = b * NB;
        const size_t r1 = std::min(r0 + NB, n);
        for (size_t i = r0; i < r1; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                const double error = fabs(sum - A_orig[i * n + j]);
                maxError = std::max(maxError, error);

                const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (rank == 0) {
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI ranks: %d\n", nprocs);
    }

    // Allocate matrix (full size; each rank fills only its block-cyclic rows)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank, nprocs);

    if (validate) {
        A_orig = A; // Save original (owned rows) for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
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

    // Collect the full result wherever it is needed
    if (printResults || validate) {
        gatherFullMatrix(A, n, rank, nprocs);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, nprocs);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
