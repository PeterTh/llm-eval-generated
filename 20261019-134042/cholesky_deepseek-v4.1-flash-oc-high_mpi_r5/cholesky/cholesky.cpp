#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static size_t rowsForRank(const size_t n, const int rank, const int size) {
    if ((size_t)rank >= n) {
        return 0;
    }
    return (n - (size_t)rank + (size_t)size - 1) / (size_t)size;
}

static bool choleskyDecompositionMPI(std::vector<double>& A, const size_t localRows,
                                     const size_t n, const int rank, const int size) {
    std::vector<double> bcast(n + 2);

    for (size_t j = 0; j < n; ++j) {
        const int owner = (int)(j % (size_t)size);
        const double* rowj = bcast.data() + 1;

        if (rank == owner) {
            const size_t lr = j / (size_t)size;
            double* row = A.data() + lr * n;

            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }

            const double val = row[j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                bcast[0] = 0.0;
            } else {
                row[j] = sqrt(val);
                for (size_t c = j + 1; c < n; ++c) {
                    row[c] = 0.0;
                }
                bcast[0] = 1.0;
                for (size_t k = 0; k <= j; ++k) {
                    bcast[k + 1] = row[k];
                }
            }
        }

        MPI_Bcast(bcast.data(), (int)(j + 2), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        if (bcast[0] == 0.0) {
            return false;
        }

        for (size_t lr = 0; lr < localRows; ++lr) {
            const size_t i = (size_t)rank + lr * (size_t)size;
            if (i <= j) {
                continue;
            }
            double* row = A.data() + lr * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * rowj[k];
            }
            row[j] = (row[j] - sum) / rowj[j];
        }
    }

    return true;
}

static void generatePositiveDefiniteMatrixLocal(std::vector<double>& A, const size_t localRows,
                                                const size_t n, const int rank, const int size) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t lr = 0; lr < localRows; ++lr) {
        const size_t i = (size_t)rank + lr * (size_t)size;
        double* row = A.data() + lr * n;
        const double* Bi = B.data() + i * n;

        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* Bj = B.data() + j * n;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            row[j] = sum;
        }

        row[i] += (double)n;
    }
}

static void gatherLocalToFull(const std::vector<double>& A, const size_t localRows,
                              const size_t n, const int rank, const int size,
                              std::vector<double>& full) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows = rowsForRank(n, r, size);
        counts[r] = (int)(rows * n);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> recv;
    if (rank == 0) {
        recv.resize(n * n);
    }

    MPI_Gatherv(A.data(), (int)(localRows * n), MPI_DOUBLE,
                rank == 0 ? recv.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        full.assign(n * n, 0.0);
        for (int r = 0; r < size; ++r) {
            size_t lr = 0;
            for (size_t i = (size_t)r; i < n; i += (size_t)size, ++lr) {
                std::memcpy(full.data() + i * n,
                            recv.data() + (size_t)displs[r] + lr * n,
                            n * sizeof(double));
            }
        }
    }
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                             const size_t n) {
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    }

    const size_t localRows = rowsForRank(n, rank, size);
    std::vector<double> A(localRows * n);
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrixLocal(A, localRows, n, rank, size);

    if (validate) {
        gatherLocalToFull(A, localRows, n, rank, size, A_orig);
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = choleskyDecompositionMPI(A, localRows, n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    const long duration_ms = (long)((end - start) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / ((double)duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if (printResults || validate) {
        gatherLocalToFull(A, localRows, n, rank, size, fullL);
    }

    if (printResults && rank == 0) {
        print_results(fullL, "CholeskyL");
    }

    if (validate) {
        int valid = 0;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateCholesky(fullL, A_orig, n) ? 1 : 0;

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
