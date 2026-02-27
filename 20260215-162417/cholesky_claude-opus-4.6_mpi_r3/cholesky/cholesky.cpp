#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallelized Cholesky decomposition using left-looking algorithm
// with cyclic row distribution for bit-identical results to original

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

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
        printf("Generating positive definite matrix...\n");
    }

    // All ranks generate the same matrix (deterministic with same seed)
    std::vector<double> A_full(n * n);
    generatePositiveDefiniteMatrix(A_full, n);

    std::vector<double> A_orig;
    if (validate && rank == 0) {
        A_orig = A_full;
    }

    // Cyclic row distribution: rank r owns rows r, r+P, r+2P, ...
    size_t nloc = 0;
    for (size_t i = (size_t)rank; i < n; i += (size_t)nprocs) nloc++;

    // Extract local rows (row-major, each row is n doubles)
    std::vector<double> local(nloc * n);
    for (size_t lr = 0; lr < nloc; ++lr) {
        size_t gi = (size_t)rank + lr * (size_t)nprocs;
        memcpy(&local[lr * n], &A_full[gi * n], n * sizeof(double));
    }
    A_full.clear();
    A_full.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> row_j(n);

    // Left-looking Cholesky: iterate over columns
    // For each column j, broadcast row j, then each rank computes L[i,j] for its local rows i > j
    for (size_t j = 0; j < n; ++j) {
        int owner = (int)(j % (size_t)nprocs);
        size_t lrj = j / (size_t)nprocs;

        if (rank == owner) {
            double* rj = &local[lrj * n];
            // Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2, k<j))
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rj[k] * rj[k];
            }
            double val = rj[j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            rj[j] = sqrt(val);
            memcpy(row_j.data(), rj, (j + 1) * sizeof(double));
        }

        MPI_Bcast(row_j.data(), (int)(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Off-diagonal: L[i,j] for local rows i > j
        double Ljj = row_j[j];
        for (size_t lr = 0; lr < nloc; ++lr) {
            size_t gi = (size_t)rank + lr * (size_t)nprocs;
            if (gi <= j) continue;
            double* ri = &local[lr * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += ri[k] * row_j[k];
            }
            ri[j] = (ri[j] - sum) / Ljj;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Zero upper triangular part
    for (size_t lr = 0; lr < nloc; ++lr) {
        size_t gi = (size_t)rank + lr * (size_t)nprocs;
        for (size_t col = gi + 1; col < n; ++col) {
            local[lr * n + col] = 0.0;
        }
    }

    // Gather results to rank 0
    std::vector<double> result;
    if (rank == 0) {
        result.resize(n * n);
        for (size_t lr = 0; lr < nloc; ++lr) {
            size_t gi = lr * (size_t)nprocs;
            memcpy(&result[gi * n], &local[lr * n], n * sizeof(double));
        }
        for (int r = 1; r < nprocs; ++r) {
            size_t r_nloc = 0;
            for (size_t i = (size_t)r; i < n; i += (size_t)nprocs) r_nloc++;
            if (r_nloc == 0) continue;
            std::vector<double> buf(r_nloc * n);
            MPI_Recv(buf.data(), (int)(r_nloc * n), MPI_DOUBLE, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t lr = 0; lr < r_nloc; ++lr) {
                size_t gi = (size_t)r + lr * (size_t)nprocs;
                memcpy(&result[gi * n], &buf[lr * n], n * sizeof(double));
            }
        }
    } else {
        if (nloc > 0) {
            MPI_Send(local.data(), (int)(nloc * n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(result, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(result, A_orig, n);
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
