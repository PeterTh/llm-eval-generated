#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel blocked right-looking Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Uses replicated storage with distributed computation

static const size_t NB = 128; // Block size for trailing update

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int nprocs) {
    std::vector<double> col_buf(n);

    for (size_t jb = 0; jb < n; jb += NB) {
        const size_t bs = std::min(NB, n - jb);

        // Panel factorization: columns jb..jb+bs-1
        for (size_t j = jb; j < jb + bs; ++j) {
            // Diagonal element: sum within panel (prior blocks already applied)
            double sum = 0.0;
            const double* Lj = &A[j * n];
            for (size_t k = jb; k < j; ++k) {
                sum += Lj[k] * Lj[k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0)
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            A[j * n + j] = sqrt(val);
            const double diag_inv = 1.0 / A[j * n + j];

            // Off-diagonal elements: distribute rows cyclically among ranks
            const size_t nrows = n - j - 1;
            if (nrows > 0) {
                std::fill(col_buf.begin(), col_buf.begin() + nrows, 0.0);

                for (size_t idx = (size_t)rank; idx < nrows; idx += (size_t)nprocs) {
                    const size_t i = j + 1 + idx;
                    const double* Li = &A[i * n];
                    double s = 0.0;
                    for (size_t k = jb; k < j; ++k) {
                        s += Li[k] * Lj[k];
                    }
                    col_buf[idx] = (A[i * n + j] - s) * diag_inv;
                }

                MPI_Allreduce(MPI_IN_PLACE, col_buf.data(), (int)nrows,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                for (size_t idx = 0; idx < nrows; ++idx) {
                    A[(j + 1 + idx) * n + j] = col_buf[idx];
                }
            }
        }

        // Zero upper triangular part for processed rows
        for (size_t i = jb; i < jb + bs; ++i) {
            for (size_t jj = i + 1; jj < n; ++jj) {
                A[i * n + jj] = 0.0;
            }
        }

        // Trailing matrix update: A[i][j] -= sum_{k=jb}^{jb+bs-1} L[i][k]*L[j][k]
        // for i >= jb+bs, j >= jb+bs, j <= i
        const size_t ts = jb + bs;
        const size_t tn = n - ts;
        if (tn > 0) {
            // Distribute rows of trailing submatrix among ranks
            for (size_t ri = (size_t)rank; ri < tn; ri += (size_t)nprocs) {
                const size_t i = ts + ri;
                const double* Li = &A[i * n + jb];
                for (size_t rj = 0; rj <= ri; ++rj) {
                    const size_t j = ts + rj;
                    const double* Ljr = &A[j * n + jb];
                    double s = 0.0;
                    for (size_t k = 0; k < bs; ++k) {
                        s += Li[k] * Ljr[k];
                    }
                    A[i * n + j] -= s;
                }
            }

            // Synchronize trailing submatrix: broadcast each row from its owner
            for (size_t ri = 0; ri < tn; ++ri) {
                const int owner = (int)(ri % (size_t)nprocs);
                const size_t i = ts + ri;
                MPI_Bcast(&A[i * n + ts], (int)(ri + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
            }
            // Also sync diagonal entries (symmetric: A[j][j] was updated by the row owner of ri==rj)
            // Already covered since ri includes all rows and we broadcast columns ts..ts+ri
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (parallelized)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank, int nprocs) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // All ranks generate identical B (deterministic seed)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T, distributing rows among ranks
    std::fill(A.begin(), A.end(), 0.0);
    for (size_t i = (size_t)rank; i < n; i += (size_t)nprocs) {
        const double* Bi = &B[i * n];
        for (size_t j = 0; j < n; ++j) {
            const double* Bj = &B[j * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            A[i * n + j] = sum;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, A.data(), (int)(n * n),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

    int rank, nprocs;
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix (replicated on all ranks)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix (parallel)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, rank, nprocs);

    if (validate) {
        A_orig = A;
    }

    // Perform Cholesky decomposition
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
