#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed cyclically so that the shrinking trailing matrix stays
// balanced throughout the factorization. Local row i has global index
// rank + i * ranks.
static size_t rowCount(size_t n, int rank, int ranks) {
    return n > static_cast<size_t>(rank)
        ? 1 + (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(ranks)
        : 0;
}

static size_t firstRowAtOrAfter(size_t start, int rank, int ranks) {
    const size_t p = static_cast<size_t>(ranks);
    return start + (static_cast<size_t>(rank) + p - start % p) % p;
}

// The original generator uses one fixed rand_r sequence for B. Each rank
// reproduces B and computes only its own rows of B * B^T, in the same order.
static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n,
                                           int rank, int ranks, bool needUpper) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    for (size_t local = 0; local < rowCount(n, rank, ranks); ++local) {
        const size_t i = static_cast<size_t>(rank) + local * static_cast<size_t>(ranks);
        for (size_t j = 0; j < (needUpper ? n : i + 1); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[local * n + j] = sum;
        }
        A[local * n + i] += n;
    }
}

static bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    constexpr size_t blockSize = 64;
    std::vector<int> counts(ranks), displacements(ranks);
    std::vector<double> diagonal(blockSize * blockSize);
    std::vector<double> send, received, panel;

    for (size_t base = 0; base < n; base += blockSize) {
        const size_t width = std::min(blockSize, n - base);
        const size_t end = base + width;
        std::fill(diagonal.begin(), diagonal.end(), 0.0);
        for (size_t i = 0; i < width; ++i) {
            const size_t row = base + i;
            if (row % static_cast<size_t>(ranks) == static_cast<size_t>(rank)) {
                const double* src = &A[(row / static_cast<size_t>(ranks)) * n + base];
                std::copy(src, src + i + 1, diagonal.begin() + i * width);
            }
        }
        // Every tile element has exactly one owner, so summing zero-filled
        // contributions replicates the diagonal tile without a root bottleneck.
        MPI_Allreduce(MPI_IN_PLACE, diagonal.data(), static_cast<int>(width * width),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i = 0; i < width; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += diagonal[i * width + k] * diagonal[j * width + k];
                if (i == j) {
                    const double value = diagonal[i * width + i] - sum;
                    if (value <= 0.0) {
                        if (rank == 0)
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", base + i);
                        return false;
                    }
                    diagonal[i * width + i] = std::sqrt(value);
                } else {
                    diagonal[i * width + j] = (diagonal[i * width + j] - sum) / diagonal[j * width + j];
                }
            }
        }

        // Solve each owned row of the panel against the factored diagonal tile.
        const size_t first = firstRowAtOrAfter(base, rank, ranks);
        for (size_t row = first; row < n; row += static_cast<size_t>(ranks)) {
            double* local = &A[(row / static_cast<size_t>(ranks)) * n + base];
            if (row < end) {
                const size_t i = row - base;
                std::copy_n(diagonal.data() + i * width, i + 1, local);
            } else {
                for (size_t j = 0; j < width; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += local[k] * diagonal[j * width + k];
                    local[j] = (local[j] - sum) / diagonal[j * width + j];
                }
            }
        }
        if (end == n) break;

        // Exchange the solved rows below the diagonal. Transpose the received
        // panel so the trailing update streams through consecutive columns.
        int total = 0;
        for (int p = 0; p < ranks; ++p) {
            const size_t f = firstRowAtOrAfter(end, p, ranks);
            const size_t rows = f < n ? 1 + (n - 1 - f) / static_cast<size_t>(ranks) : 0;
            counts[p] = static_cast<int>(rows * width);
            displacements[p] = total;
            total += counts[p];
        }
        send.resize(static_cast<size_t>(counts[rank]));
        size_t s = 0;
        for (size_t row = firstRowAtOrAfter(end, rank, ranks); row < n; row += static_cast<size_t>(ranks)) {
            const double* src = &A[(row / static_cast<size_t>(ranks)) * n + base];
            std::copy_n(src, width, send.data() + s);
            s += width;
        }
        received.resize(static_cast<size_t>(total));
        MPI_Allgatherv(send.data(), counts[rank], MPI_DOUBLE, received.data(), counts.data(),
                       displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        panel.resize(width * (n - end));
        for (int p = 0; p < ranks; ++p) {
            size_t offset = static_cast<size_t>(displacements[p]);
            for (size_t row = firstRowAtOrAfter(end, p, ranks); row < n; row += static_cast<size_t>(ranks)) {
                for (size_t k = 0; k < width; ++k)
                    panel[k * (n - end) + row - end] = received[offset + k];
                offset += width;
            }
        }
        for (size_t row = firstRowAtOrAfter(end, rank, ranks); row < n; row += static_cast<size_t>(ranks)) {
            double* local = &A[(row / static_cast<size_t>(ranks)) * n];
            // Tiling keeps the updated part of a row in cache while all panel
            // columns are applied; the innermost loop is contiguous and vectorizable.
            constexpr size_t columnTile = 128;
            for (size_t col = end; col <= row; col += columnTile) {
                const size_t stop = std::min(row + 1, col + columnTile);
                for (size_t k = 0; k < width; ++k) {
                    const double factor = local[base + k];
                    const double* pk = panel.data() + k * (n - end);
                    for (size_t j = col; j < stop; ++j) local[j] -= factor * pk[j - end];
                }
            }
        }
    }
    // The input matrix is symmetric; the benchmark reports a lower triangle.
    for (size_t local = 0; local < rowCount(n, rank, ranks); ++local) {
        const size_t row = static_cast<size_t>(rank) + local * static_cast<size_t>(ranks);
        std::fill(A.begin() + local * n + row + 1, A.begin() + (local + 1) * n, 0.0);
    }
    return true;
}

// Gather cyclic rows in rank order, then restore their original order. The
// complete result is needed only for optional reporting and validation.
static std::vector<double> gatherRows(const std::vector<double>& A, size_t n,
                                      int rank, int ranks, bool all) {
    std::vector<int> counts(ranks), displacements(ranks);
    int total = 0;
    for (int p = 0; p < ranks; ++p) {
        counts[p] = static_cast<int>(rowCount(n, p, ranks) * n);
        displacements[p] = total;
        total += counts[p];
    }
    std::vector<double> packed((all || rank == 0) ? static_cast<size_t>(total) : 0);
    if (all)
        MPI_Allgatherv(A.data(), counts[rank], MPI_DOUBLE, packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else
        MPI_Gatherv(A.data(), counts[rank], MPI_DOUBLE, packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> result((all || rank == 0) ? n * n : 0);
    if (all || rank == 0) {
        for (int p = 0; p < ranks; ++p) {
            for (size_t i = 0; i < rowCount(n, p, ranks); ++i) {
                const size_t row = static_cast<size_t>(p) + i * static_cast<size_t>(ranks);
                std::copy_n(packed.data() + displacements[p] + i * n, n, result.data() + row * n);
            }
        }
    }
    return result;
}

static bool validateCholesky(const std::vector<double>& localL, const std::vector<double>& original,
                             size_t n, int rank, int ranks, const std::vector<double>& L) {
    double errors[2] = {0.0, 0.0};
    for (size_t local = 0; local < rowCount(n, rank, ranks); ++local) {
        const size_t i = static_cast<size_t>(rank) + local * static_cast<size_t>(ranks);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k)
                sum += localL[local * n + k] * L[j * n + k];
            const double value = original[local * n + j];
            const double error = std::fabs(sum - value);
            errors[0] = std::max(errors[0], error);
            errors[1] = std::max(errors[1], error / (std::fabs(value) + 1e-10));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI's variable-count collectives use int counts and displacements.
    const size_t mpiLimit = static_cast<size_t>(std::numeric_limits<int>::max());
    if (n != 0 && (n > mpiLimit / std::min(n, size_t{64}) ||
                   ((validate || printResults) && n > mpiLimit / n))) {
        if (rank == 0) printf("Matrix size is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A(rowCount(n, rank, ranks) * n), original;
    generatePositiveDefiniteMatrix(A, n, rank, ranks, validate);
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    std::vector<double> complete;
    if (printResults || validate) complete = gatherRows(A, n, rank, ranks, validate);
    if (printResults && rank == 0) print_results(complete, "CholeskyL");
    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(A, original, n, rank, ranks, complete);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        result = valid ? 0 : 1;
    }
    MPI_Finalize();
    return result;
}
