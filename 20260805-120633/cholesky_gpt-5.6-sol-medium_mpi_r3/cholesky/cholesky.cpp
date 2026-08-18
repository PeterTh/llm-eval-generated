#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are assigned cyclically.  This is important for Cholesky: work grows
// with the row number, so a contiguous row distribution is badly imbalanced.
static size_t localRowCount(size_t n, int rank, int processes) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(processes);
    return n > r ? 1 + (n - 1 - r) / p : 0;
}

// Distributed, left-looking Cholesky.  At column j the owner of row j forms
// the diagonal and broadcasts the already completed part of that row.  Every
// rank can then form L(i,j) for its rows independently.  Communication is
// O(n^2), while the O(n^3) arithmetic is balanced across ranks.
static bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                                  int rank, int processes, MPI_Comm comm) {
    std::vector<double> pivot(n);
    const size_t p = static_cast<size_t>(processes);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % p);

        if (rank == owner) {
            double* row = localA.data() + (j / p) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }

            const double value = row[j] - sum;
            if (value > 0.0 && std::isfinite(value)) {
                row[j] = std::sqrt(value);
                std::copy_n(row, j + 1, pivot.data());
            } else {
                // A non-positive value is also the failure flag received by
                // every rank, avoiding a second collective per column.
                pivot[j] = -1.0;
            }
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);
        if (!(pivot[j] > 0.0)) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // Skip local rows at or above the pivot.  With cyclic placement,
        // local index q represents global row rank + q * processes.
        size_t first = 0;
        if (j >= static_cast<size_t>(rank)) {
            first = (j - static_cast<size_t>(rank)) / p + 1;
        }
        const size_t rows = localRowCount(n, rank, processes);
        for (size_t q = first; q < rows; ++q) {
            double* row = localA.data() + q * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pivot[k];
            }
            row[j] = (row[j] - sum) / pivot[j];
        }
    }

    return true;
}

// Each rank generates only the lower triangle of its own rows.  B is
// deterministically replicated, so no root bottleneck or input scatter is
// needed and the generated coefficients exactly match the serial benchmark.
static void generatePositiveDefiniteMatrix(std::vector<double>& localA, size_t n,
                                           int rank, int processes) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    const size_t rows = localRowCount(n, rank, processes);
    const size_t p = static_cast<size_t>(processes);
    for (size_t q = 0; q < rows; ++q) {
        const size_t i = static_cast<size_t>(rank) + q * p;
        double* aRow = localA.data() + q * n;
        const double* bRow = B.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* other = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bRow[k] * other[k];
            }
            aRow[j] = sum;
        }
        aRow[i] += static_cast<double>(n);
    }
}

// Gather cyclic rows and restore ordinary row-major order on rank zero.
static bool gatherMatrix(const std::vector<double>& local, std::vector<double>& global,
                         size_t n, int rank, int processes, MPI_Comm comm) {
    const size_t localElements = local.size();
    // MPI_Gatherv uses int counts and displacements in the MPI-3 interface.
    const bool validCounts = localElements <= static_cast<size_t>(INT_MAX) &&
                             n * n <= static_cast<size_t>(INT_MAX);
    const int locallyValid = validCounts ? 1 : 0;
    int allValid = 0;
    MPI_Allreduce(&locallyValid, &allValid, 1, MPI_INT, MPI_MIN, comm);
    if (!allValid) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix is too large for MPI_Gatherv counts\n");
        }
        return false;
    }

    std::vector<int> counts(static_cast<size_t>(processes));
    std::vector<int> displacements(static_cast<size_t>(processes));
    size_t total = 0;
    for (int r = 0; r < processes; ++r) {
        const size_t count = localRowCount(n, r, processes) * n;
        counts[static_cast<size_t>(r)] = static_cast<int>(count);
        displacements[static_cast<size_t>(r)] = static_cast<int>(total);
        total += count;
    }

    std::vector<double> packed;
    if (rank == 0) {
        packed.resize(n * n);
    }
    MPI_Gatherv(local.data(), static_cast<int>(localElements), MPI_DOUBLE,
                rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        global.resize(n * n);
        for (int r = 0; r < processes; ++r) {
            const size_t rows = localRowCount(n, r, processes);
            const double* source = packed.data() + static_cast<size_t>(displacements[r]);
            for (size_t q = 0; q < rows; ++q) {
                const size_t globalRow = static_cast<size_t>(r) + q * static_cast<size_t>(processes);
                std::copy_n(source + q * n, n, global.data() + globalRow * n);
            }
        }
    }
    return true;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& AOrig, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t terms = std::min(i, j) + 1;
            for (size_t k = 0; k < terms; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = std::fabs(sum - AOrig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(AOrig[i * n + j]) + 1e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &processes);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(INT_MAX)) {
                parseStatus = 1;
            } else {
                n = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseStatus = 1;
        }
    }

    if (parseStatus != 0) {
        if (rank == 0) {
            if (parseStatus == 1) {
                std::printf("Invalid command line or matrix size\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    // Guard all n*n allocations against size_t overflow before any rank tries
    // them; allocation failures are handled by the MPI job launcher/runtime.
    if (n > std::numeric_limits<size_t>::max() / n) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size overflows addressable memory\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI processes: %d\n", processes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    const size_t rows = localRowCount(n, rank, processes);
    std::vector<double> localA(rows * n, 0.0);
    generatePositiveDefiniteMatrix(localA, n, rank, processes);
    std::vector<double> localOrig;
    if (validate) {
        localOrig = localA;
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, rank, processes, comm);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const double ops = static_cast<double>(n) * static_cast<double>(n) *
                           static_cast<double>(n) / 3.0;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }

    std::vector<double> A;
    std::vector<double> AOrig;
    bool gathered = true;
    if (printResults || validate) {
        gathered = gatherMatrix(localA, A, n, rank, processes, comm);
    }
    if (gathered && validate) {
        gathered = gatherMatrix(localOrig, AOrig, n, rank, processes, comm);
        if (rank == 0) {
            // Only the lower input triangle was generated; use symmetry to
            // reconstruct the full original matrix expected by validation.
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    AOrig[i * n + j] = AOrig[j * n + i];
                }
            }
        }
    }

    int result = gathered ? 0 : 1;
    if (rank == 0 && gathered) {
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateCholesky(A, AOrig, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return result;
}
