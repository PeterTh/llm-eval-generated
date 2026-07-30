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
// with 1D block-row distribution across MPI processes.
//
// Algorithm: For each column j, the owner of row j computes L[j][j],
// broadcasts row j (cols 0..j) to all ranks, and all ranks compute
// L[i][j] for their local rows i > j using dot products.

static void computeRowDistribution(size_t n, int numRanks, int rank,
                                   size_t& localRowStart, size_t& localRows) {
    const size_t baseRows = n / numRanks;
    const size_t remainder = n % numRanks;
    localRowStart = 0;
    for (int r = 0; r < rank; ++r) {
        localRowStart += baseRows + (r < static_cast<int>(remainder) ? 1 : 0);
    }
    localRows = baseRows + (rank < static_cast<int>(remainder) ? 1 : 0);
}

static void buildRowOwnerTable(size_t n, int numRanks, std::vector<int>& rowOwner) {
    rowOwner.resize(n);
    const size_t baseRows = n / numRanks;
    const size_t remainder = n % numRanks;
    size_t cumRows = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows_r = baseRows + (r < static_cast<int>(remainder) ? 1 : 0);
        for (size_t i = cumRows; i < cumRows + rows_r; ++i) {
            rowOwner[i] = r;
        }
        cumRows += rows_r;
    }
}

bool choleskyDecomposition(std::vector<double>& A_local, const size_t n,
                           const size_t localRows, const size_t localRowStart) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::vector<int> rowOwner;
    buildRowOwnerTable(n, numRanks, rowOwner);

    // Buffer for broadcasting row j, columns 0..j
    std::vector<double> rowBuf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = rowOwner[j];

        // Owner computes L[j][j]
        if (rank == owner) {
            const size_t localIdx = j - localRowStart;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A_local[localIdx * n + k] * A_local[localIdx * n + k];
            }
            const double val = A_local[localIdx * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            A_local[localIdx * n + j] = sqrt(val);
        }

        // Broadcast row j, columns 0..j to all ranks
        if (rank == owner) {
            const size_t localIdx = j - localRowStart;
            std::memcpy(rowBuf.data(), A_local.data() + localIdx * n,
                        sizeof(double) * (j + 1));
        }
        MPI_Bcast(rowBuf.data(), static_cast<int>(j + 1), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        // Compute L[i][j] for local rows i > j
        const double diagVal = rowBuf[j];
        for (size_t localI = 0; localI < localRows; ++localI) {
            const size_t i = localRowStart + localI;
            if (i <= j) continue;

            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A_local[localI * n + k] * rowBuf[k];
            }
            A_local[localI * n + j] = (A_local[localI * n + j] - sum) / diagVal;
        }

        // Zero out upper triangular part for local rows
        for (size_t localI = 0; localI < localRows; ++localI) {
            const size_t i = localRowStart + localI;
            for (size_t j2 = i + 1; j2 < n; ++j2) {
                A_local[localI * n + j2] = 0.0;
            }
        }
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A_local, const size_t n,
                                    const size_t localRows,
                                    int rank, int numRanks) {
    // Rank 0 generates the full matrix (same as original sequential code)
    // then scatters to all ranks.
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
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
                A_full[i * n + j] = sum;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            A_full[i * n + i] += static_cast<double>(n);
        }
    }

    // Scatter rows to all ranks
    std::vector<int> recvcounts(numRanks);
    std::vector<int> displs(numRanks);
    const size_t baseRows = n / numRanks;
    const size_t remainder = n % numRanks;
    size_t cumRows = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows_r = baseRows + (r < static_cast<int>(remainder) ? 1 : 0);
        recvcounts[r] = static_cast<int>(rows_r * n);
        displs[r] = static_cast<int>(cumRows * n);
        cumRows += rows_r;
    }

    MPI_Scatterv(A_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
}

bool validateCholesky(const std::vector<double>& L_local, const std::vector<double>& A_orig,
                      const size_t n, const size_t localRows) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Build recv/displ arrays for Gatherv
    std::vector<int> rcv(numRanks), dsp(numRanks);
    const size_t baseR = n / numRanks;
    const size_t rem = n % numRanks;
    size_t cum = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows_r = baseR + (r < static_cast<int>(rem) ? 1 : 0);
        rcv[r] = static_cast<int>(rows_r * n);
        dsp[r] = static_cast<int>(cum * n);
        cum += rows_r;
    }

    // Gather full L and A_orig on rank 0
    std::vector<double> L_full, A_full;
    if (rank == 0) {
        L_full.resize(n * n);
        A_full.resize(n * n);
    }

    MPI_Gatherv(L_local.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                L_full.data(), rcv.data(), dsp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(A_orig.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                A_full.data(), rcv.data(), dsp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Compute L * L^T
        std::vector<double> reconstructed(n * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L_full[i * n + k] * L_full[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        double maxError = 0.0, relError = 0.0;
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_full[i]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A_full[i]) + 1e-10);
            relError = std::max(relError, rel);
        }
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
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
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t localRows, localRowStart;
    computeRowDistribution(n, numRanks, rank, localRowStart, localRows);

    if (rank == 0) {
        printf("Local rows for rank %d: %zu (rows %zu-%zu)\n",
               rank, localRows, localRowStart,
               localRows > 0 ? localRowStart + localRows - 1 : 0);
    }

    std::vector<double> A_local(localRows * n);
    std::vector<double> A_orig_local;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A_local, n, localRows, rank, numRanks);

    if (validate) A_orig_local = A_local;

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A_local, n, localRows, localRowStart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    MPI_Barrier(MPI_COMM_WORLD);

    if (!success) { MPI_Finalize(); return 1; }

    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults && rank == 0) {
        print_results(A_local, "CholeskyL");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A_local, A_orig_local, n, localRows);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
