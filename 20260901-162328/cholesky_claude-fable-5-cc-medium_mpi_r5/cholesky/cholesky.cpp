#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (left-looking, column-by-column)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Distribution: rows are assigned cyclically to ranks (global row i belongs to
// rank i % p and is stored as local row i / p). At step j the owner of row j
// computes the diagonal element and broadcasts row j; every rank then computes
// column j entries for the rows it owns. Each element is computed with exactly
// the same sequence of floating-point operations as the sequential algorithm,
// so results are bit-identical to the original code.

static int g_rank = 0;
static int g_nprocs = 1;

static inline size_t numLocalRows(const size_t n, const int rank, const int nprocs) {
    return (n - (size_t)rank + (size_t)nprocs - 1) / (size_t)nprocs;
}

bool choleskyDecomposition(std::vector<double>& A_local, const size_t n) {
    // A_local holds the cyclically distributed rows of A in row-major order:
    // local row l corresponds to global row l * nprocs + rank.
    const int rank = g_rank;
    const int nprocs = g_nprocs;
    const size_t nlocal = numLocalRows(n, rank, nprocs);

    std::vector<double> rowj(n); // broadcast buffer for row j (entries 0..j)

    for (size_t j = 0; j < n; ++j) {
        const int owner = (int)(j % (size_t)nprocs);

        if (rank == owner) {
            double* Lj = &A_local[(j / nprocs) * n];
            // Diagonal element
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Lj[k] * Lj[k];
            }
            const double val = Lj[j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                rowj[j] = -1.0; // sentinel: signals failure to all ranks
            } else {
                Lj[j] = sqrt(val);
                memcpy(rowj.data(), Lj, (j + 1) * sizeof(double));
            }
        }

        MPI_Bcast(rowj.data(), (int)(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rowj[j] <= 0.0) {
            return false; // owner reported a non-positive-definite matrix
        }
        if (rank == owner) {
            A_local[(j / nprocs) * n + j] = rowj[j];
        }

        // Off-diagonal elements: rows i > j owned by this rank
        // First owned global row strictly greater than j:
        size_t lstart = j / nprocs + ((size_t)rank <= j % (size_t)nprocs ? 1 : 0);
        for (size_t l = lstart; l < nlocal; ++l) {
            double* Li = &A_local[l * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Li[k] * rowj[k];
            }
            Li[j] = (Li[j] - sum) / rowj[j];
        }
    }

    // Zero out upper triangular part of owned rows
    for (size_t l = 0; l < nlocal; ++l) {
        const size_t i = l * nprocs + rank;
        for (size_t j = i + 1; j < n; ++j) {
            A_local[l * n + j] = 0.0;
        }
    }

    return true;
}

// Generate the locally owned rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A_local, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    //
    // Every rank generates the identical matrix B (same seed and draw order as
    // the sequential code) and computes only its own rows of A.

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute owned rows of A = B * B^T
    const size_t nlocal = numLocalRows(n, g_rank, g_nprocs);
    for (size_t l = 0; l < nlocal; ++l) {
        const size_t i = l * g_nprocs + g_rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A_local[l * n + j] = sum;
        }
        // Add diagonal dominance to ensure positive definiteness
        A_local[l * n + i] += n;
    }
}

// Gather the cyclically distributed rows into a full matrix on rank 0
void gatherMatrix(const std::vector<double>& A_local, std::vector<double>& A_full, const size_t n) {
    const size_t nlocal = numLocalRows(n, g_rank, g_nprocs);
    if (g_rank == 0) {
        A_full.resize(n * n);
        for (size_t l = 0; l < nlocal; ++l) {
            memcpy(&A_full[(l * g_nprocs) * n], &A_local[l * n], n * sizeof(double));
        }
        std::vector<double> buf;
        for (int r = 1; r < g_nprocs; ++r) {
            const size_t rlocal = numLocalRows(n, r, g_nprocs);
            if (rlocal == 0) continue;
            buf.resize(rlocal * n);
            MPI_Recv(buf.data(), (int)(rlocal * n), MPI_DOUBLE, r, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (size_t l = 0; l < rlocal; ++l) {
                memcpy(&A_full[(l * g_nprocs + r) * n], &buf[l * n], n * sizeof(double));
            }
        }
    } else if (nlocal > 0) {
        MPI_Send(A_local.data(), (int)(nlocal * n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

bool validateCholesky(const std::vector<double>& L_full, const std::vector<double>& A_orig_local,
                      const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix.
    // L_full holds the complete factor on every rank; each rank checks the
    // rows it owns and the maximum errors are combined with a reduction.

    const size_t nlocal = numLocalRows(n, g_rank, g_nprocs);

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t l = 0; l < nlocal; ++l) {
        const size_t i = l * g_nprocs + g_rank;
        for (size_t j = 0; j < n; ++j) {
            // Compute (L * L^T)(i, j)
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L_full[i * n + k] * L_full[j * n + k];
            }
            const double error = fabs(sum - A_orig_local[l * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(A_orig_local[l * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double globalErrors[2];
    const double localErrors[2] = {maxError, relError};
    MPI_Allreduce(localErrors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = globalErrors[0];
    relError = globalErrors[1];

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (g_rank == 0) printf("Validation failed: relative error too large\n");
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
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

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
        printf("MPI ranks: %d\n", g_nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate owned rows (cyclic row distribution)
    const size_t nlocal = numLocalRows(n, g_rank, g_nprocs);
    std::vector<double> A_local(nlocal * n);
    std::vector<double> A_orig_local;

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A_local, n);

    if (validate) {
        A_orig_local = A_local; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A_local, n);

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

    // Assemble the full factor on rank 0 when it is needed for output/validation
    std::vector<double> L_full;
    if (printResults || validate) {
        gatherMatrix(A_local, L_full, n);
    }

    // Print results for external validation
    if (printResults && g_rank == 0) {
        print_results(L_full, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (g_rank == 0) printf("Validating result...\n");

        // Every rank needs the full factor to reconstruct its rows of L * L^T
        if (g_rank != 0) L_full.resize(n * n);
        MPI_Bcast(L_full.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        bool valid = validateCholesky(L_full, A_orig_local, n);

        if (g_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
