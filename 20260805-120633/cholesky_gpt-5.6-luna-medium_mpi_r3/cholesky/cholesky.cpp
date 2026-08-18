#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed in contiguous blocks.  At step k the owner of row k
// factors its diagonal and broadcasts the completed part of that row.  Every
// rank can then update its own rows independently.  The row-wise algorithm has
// the same arithmetic order as the original implementation for each element.
bool choleskyDecomposition(std::vector<double>& localA, const size_t n,
                           const size_t firstRow, const size_t localRows,
                           const int rank, const int size) {
    std::vector<double> panel(n);

    for (size_t k = 0; k < n; ++k) {
        // This inverse of firstRow/endRow also handles size > n, where some
        // ranks own no rows.
        const int owner = static_cast<int>(
            (((k + 1) * static_cast<size_t>(size)) - 1) / n);
        const size_t ownerFirst = (n * static_cast<size_t>(owner)) /
                                  static_cast<size_t>(size);
        int valid = 1;

        if (rank == owner) {
            const size_t localK = k - ownerFirst;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) {
                const double value = localA[localK * n + j];
                sum += value * value;
            }

            const double diagonal = localA[localK * n + k] - sum;
            valid = diagonal > 0.0 ? 1 : 0;
            MPI_Bcast(&valid, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (!valid) {
                if (rank == 0) {
                    std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
                }
                return false;
            }

            localA[localK * n + k] = std::sqrt(diagonal);
            // Only the lower part is needed by later steps.  Keeping the
            // complete row in panel makes the broadcast contiguous.
            std::copy_n(localA.begin() + localK * n, k + 1, panel.begin());
        } else {
            MPI_Bcast(&valid, 1, MPI_INT, owner, MPI_COMM_WORLD);
        }

        if (!valid) {
            return false;
        }

        MPI_Bcast(panel.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        const size_t localEnd = firstRow + localRows;
        for (size_t i = firstRow; i < localEnd; ++i) {
            if (i <= k) {
                continue;
            }

            const size_t localI = i - firstRow;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) {
                sum += localA[localI * n + j] * panel[j];
            }
            localA[localI * n + k] =
                (localA[localI * n + k] - sum) / panel[k];
        }

        // The upper triangle is not part of L and is cleared once its row is
        // complete, retaining the original output semantics.
        for (size_t i = firstRow; i < localEnd; ++i) {
            if (i == k) {
                const size_t localI = i - firstRow;
                std::fill(localA.begin() + localI * n + k + 1,
                          localA.begin() + localI * n + n, 0.0);
            }
        }
    }
    return true;
}

// Generate exactly the same matrix as the original benchmark.  It is created
// only on rank zero and scattered before factorization.
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

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
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
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(A_orig[i]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
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
    int parseOk = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseOk = 0;
        }
    }
    if (n == 0) parseOk = 0;
    MPI_Allreduce(MPI_IN_PLACE, &parseOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!parseOk) {
        MPI_Finalize();
        return 1;
    }

    const size_t firstRow = (n * static_cast<size_t>(rank)) /
                            static_cast<size_t>(size);
    const size_t endRow = (n * static_cast<size_t>(rank + 1)) /
                          static_cast<size_t>(size);
    const size_t localRows = endRow - firstRow;
    std::vector<int> counts(size), displacements(size);
    for (int p = 0; p < size; ++p) {
        const size_t pFirst = n * static_cast<size_t>(p) / static_cast<size_t>(size);
        const size_t pEnd = n * static_cast<size_t>(p + 1) / static_cast<size_t>(size);
        counts[p] = static_cast<int>((pEnd - pFirst) * n);
        displacements[p] = static_cast<int>(pFirst * n);
    }

    std::vector<double> A;
    std::vector<double> A_orig;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }

    std::vector<double> localA(localRows * n);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(),
                 MPI_DOUBLE, localA.data(), counts[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, firstRow, localRows,
                                               rank, size);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int globalSuccess = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        A.resize(n * n);
    }
    MPI_Gatherv(localA.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double gflops = (static_cast<double>(n) * n * n / 3.0) /
                              duration / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
