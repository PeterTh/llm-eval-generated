#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI, left-looking by column)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Rows are distributed cyclically: rank r owns global rows r, r+P, r+2P, ...
// Local rows are stored k-major: element (global row i, column k) lives at
// A[k * nown + li] with li = (i - r) / P and nown the number of rows owned by
// this rank. Contiguous local rows per column let the update loops vectorize.
//
// Every L[i][j] is still computed as sum_{k<j} L[i][k] * L[j][k], accumulated in
// increasing k exactly like the sequential algorithm, so the result is
// bit-identical to the sequential program (and the -r hash is unchanged).

constexpr size_t kTile = 256; // rows per cache block in the Cholesky update

// Number of rows of an n-row matrix owned by `rank` in the cyclic distribution
size_t ownedRows(size_t n, size_t rank, size_t size) {
    return n > rank ? (n - rank - 1) / size + 1 : 0;
}

// Copy the local k-major block into row-major order (nown rows of length n)
std::vector<double> toRowMajor(const std::vector<double>& A, size_t nown, size_t n) {
    std::vector<double> R(nown * n);
    for (size_t k = 0; k < n; ++k) {
        for (size_t li = 0; li < nown; ++li) {
            R[li * n + k] = A[k * nown + li];
        }
    }
    return R;
}

// Pass every rank's block one step around the ring (receive from left, send to right)
void ringShift(std::vector<double>& block, int rank, int size, MPI_Comm comm) {
    const int right = (rank + 1) % size;
    const int left = (rank + size - 1) % size;
    MPI_Sendrecv_replace(block.data(), static_cast<int>(block.size()), MPI_DOUBLE,
                         right, 0, left, 0, comm, MPI_STATUS_IGNORE);
}

// Generate a symmetric positive definite matrix
// Every rank replays the sequential random stream but keeps only its own rows of B.
// Blocks of B circulate around a ring so each rank computes its own rows of A.
// Only the lower triangle is computed unless `full` is set (needed for validation).
std::vector<double> generatePositiveDefiniteMatrix(size_t n, bool full, int rank, int size, MPI_Comm comm) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    const size_t r = rank, P = size;
    const size_t nown = ownedRows(n, r, P);
    const size_t nmax = ownedRows(n, 0, P);

    // Own rows of B, generated in the same order as the sequential code
    std::vector<double> Bown(nown * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (i % P == r) {
                Bown[(i / P) * n + k] = v;
            }
        }
    }

    // Ring buffer holding the block of B currently being processed
    std::vector<double> Bring(nmax * n);
    std::copy(Bown.begin(), Bown.end(), Bring.begin());

    // Own rows of A, k-major
    std::vector<double> A(n * nown, 0.0);
    for (size_t t = 0; t < P; ++t) {
        const size_t s = (r + P - t) % P; // owner of the block in Bring
        const size_t cs = ownedRows(n, s, P);
        for (size_t li = 0; li < nown; ++li) {
            const size_t i = r + li * P;
            for (size_t lj = 0; lj < cs; ++lj) {
                const size_t j = s + lj * P;
                if (!full && j > i) {
                    continue;
                }
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += Bown[li * n + k] * Bring[lj * n + k];
                }
                if (i == j) {
                    sum += static_cast<double>(n);
                }
                A[j * nown + li] = sum;
            }
        }
        if (t + 1 < P) {
            ringShift(Bring, rank, size, comm);
        }
    }

    return A;
}

// Cholesky decomposition, computed in place on the local k-major block
// Left-looking over columns: the owner of row j finishes L[j][j] and broadcasts row j,
// then every rank updates the column j entries of its own rows below the diagonal.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size, MPI_Comm comm) {
    const size_t r = rank, P = size;
    const size_t nown = ownedRows(n, r, P);

    std::vector<double> row(n + 1); // L[j][0..j], then failure flag
    std::vector<double> sums(nown);

    for (size_t j = 0; j < n; ++j) {
        const size_t owner = j % P;
        const size_t lj = j / P; // local index of row j on its owner

        if (r == owner) {
            // Diagonal element
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[k * nown + lj] * A[k * nown + lj];
            }
            const double val = A[j * nown + lj] - sum;
            if (val <= 0.0) {
                row[j + 1] = 1.0;
            } else {
                A[j * nown + lj] = sqrt(val);
                row[j + 1] = 0.0;
            }
            for (size_t k = 0; k <= j; ++k) {
                row[k] = A[k * nown + lj];
            }
        }
        MPI_Bcast(row.data(), static_cast<int>(j + 2), MPI_DOUBLE, static_cast<int>(owner), comm);

        if (row[j + 1] != 0.0) {
            // Matrix is not positive definite
            if (r == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // Off-diagonal elements of column j for the local rows i > j
        const double diag = row[j];
        const size_t li0 = (j < r) ? 0 : (j - r) / P + 1;
        double* colj = A.data() + j * nown;
        for (size_t t0 = li0; t0 < nown; t0 += kTile) {
            const size_t t1 = std::min(nown, t0 + kTile);
            for (size_t li = t0; li < t1; ++li) {
                sums[li] = 0.0;
            }
            for (size_t k = 0; k < j; ++k) {
                const double lk = row[k];
                const double* colk = A.data() + k * nown;
                for (size_t li = t0; li < t1; ++li) {
                    sums[li] += colk[li] * lk;
                }
            }
            for (size_t li = t0; li < t1; ++li) {
                colj[li] = (colj[li] - sums[li]) / diag;
            }
        }
    }

    // Zero out upper triangular part
    for (size_t li = 0; li < nown; ++li) {
        const size_t i = r + li * P;
        for (size_t j = i + 1; j < n; ++j) {
            A[j * nown + li] = 0.0;
        }
    }

    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n,
                      int rank, int size, MPI_Comm comm) {
    // Validate by computing L * L^T and comparing with original matrix
    const size_t r = rank, P = size;
    const size_t nown = ownedRows(n, r, P);
    const size_t nmax = ownedRows(n, 0, P);

    const std::vector<double> Lrow = toRowMajor(L, nown, n);
    std::vector<double> Lring(nmax * n);
    std::copy(Lrow.begin(), Lrow.end(), Lring.begin());

    double maxError = 0.0;
    double relError = 0.0;

    // Compute L * L^T for the local rows, one block of L rows at a time
    for (size_t t = 0; t < P; ++t) {
        const size_t s = (r + P - t) % P;
        const size_t cs = ownedRows(n, s, P);
        for (size_t li = 0; li < nown; ++li) {
            for (size_t lj = 0; lj < cs; ++lj) {
                const size_t j = s + lj * P;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += Lrow[li * n + k] * Lring[lj * n + k];
                }

                // Compare with original
                const double a = A_orig[j * nown + li];
                const double error = fabs(sum - a);
                maxError = std::max(maxError, error);

                const double rel = error / (fabs(a) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
        if (t + 1 < P) {
            ringShift(Lring, rank, size, comm);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, comm);

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

// Gather the full row-major L on rank 0 (returns an empty vector on other ranks)
std::vector<double> gatherRowMajor(const std::vector<double>& L, size_t n, int rank, int size, MPI_Comm comm) {
    const size_t P = size;
    const size_t nown = ownedRows(n, rank, P);
    const std::vector<double> Lrow = toRowMajor(L, nown, n);

    if (rank != 0) {
        MPI_Send(Lrow.data(), static_cast<int>(Lrow.size()), MPI_DOUBLE, 0, 0, comm);
        return {};
    }

    std::vector<double> full(n * n);
    std::vector<double> block;
    for (size_t s = 0; s < P; ++s) {
        const size_t cs = ownedRows(n, s, P);
        const double* src = Lrow.data();
        if (s != 0) {
            block.resize(cs * n);
            MPI_Recv(block.data(), static_cast<int>(cs * n), MPI_DOUBLE, static_cast<int>(s), 0, comm,
                     MPI_STATUS_IGNORE);
            src = block.data();
        }
        for (size_t li = 0; li < cs; ++li) {
            const size_t i = s + li * P;
            std::copy(src + li * n, src + (li + 1) * n, full.begin() + i * n);
        }
    }
    return full;
}

void printUsage(const char* progName, int rank) {
    if (rank != 0) {
        return;
    }
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int size) {
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
            printUsage(argv[0], rank);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            printUsage(argv[0], rank);
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A = generatePositiveDefiniteMatrix(n, validate, rank, size, MPI_COMM_WORLD);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = choleskyDecomposition(A, n, rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(end - start));

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        const std::vector<double> full = gatherRowMajor(A, n, rank, size, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(full, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, size, MPI_COMM_WORLD);

        if (valid) {
            if (rank == 0) {
                printf("Validation: PASSED\n");
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
