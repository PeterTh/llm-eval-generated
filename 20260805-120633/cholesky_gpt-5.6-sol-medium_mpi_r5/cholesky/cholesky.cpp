#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Rows are distributed in cyclic blocks.  This keeps all ranks busy as the
// active lower-right submatrix shrinks, while every rank stores only O(n^2/P)
// matrix data.  A moderate block size also makes the trailing updates cache
// friendly without exposing it as a benchmark option.
constexpr size_t kBlockSize = 64;

static size_t blockRows(size_t n, size_t block, int rank, int ranks) {
    size_t rows = 0;
    const size_t blocks = (n + block - 1) / block;
    for (size_t b = static_cast<size_t>(rank); b < blocks; b += ranks) {
        rows += std::min(block, n - b * block);
    }
    return rows;
}

static size_t localRow(size_t globalRow, size_t block, int ranks) {
    const size_t globalBlock = globalRow / block;
    return (globalBlock / static_cast<size_t>(ranks)) * block + globalRow % block;
}

static int rowOwner(size_t globalRow, size_t block, int ranks) {
    return static_cast<int>((globalRow / block) % static_cast<size_t>(ranks));
}

// Generate exactly the same deterministic SPD matrix as the original
// benchmark.  Generation is outside the timed region; the matrix is scattered
// immediately afterwards, so the factorization itself has distributed memory.
static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
        A[i * n + i] += static_cast<double>(n);
    }
}

static bool mpiCountFits(size_t count) {
    return count <= static_cast<size_t>(std::numeric_limits<int>::max());
}

static bool scatterBlockRows(const std::vector<double>& global,
                             std::vector<double>& local, size_t n,
                             int rank, int ranks) {
    const size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t first = b * kBlockSize;
        const size_t rows = std::min(kBlockSize, n - first);
        const size_t count = rows * n;
        if (!mpiCountFits(count)) return false;
        const int owner = static_cast<int>(b % static_cast<size_t>(ranks));
        if (rank == 0 && owner == 0) {
            std::copy_n(global.data() + first * n, count,
                        local.data() + localRow(first, kBlockSize, ranks) * n);
        } else if (rank == 0) {
            MPI_Send(global.data() + first * n, static_cast<int>(count), MPI_DOUBLE,
                     owner, 100, MPI_COMM_WORLD);
        } else if (rank == owner) {
            MPI_Recv(local.data() + localRow(first, kBlockSize, ranks) * n,
                     static_cast<int>(count), MPI_DOUBLE, 0, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    return true;
}

static bool gatherBlockRows(const std::vector<double>& local,
                            std::vector<double>& global, size_t n,
                            int rank, int ranks) {
    const size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t first = b * kBlockSize;
        const size_t rows = std::min(kBlockSize, n - first);
        const size_t count = rows * n;
        if (!mpiCountFits(count)) return false;
        const int owner = static_cast<int>(b % static_cast<size_t>(ranks));
        if (rank == 0 && owner == 0) {
            std::copy_n(local.data() + localRow(first, kBlockSize, ranks) * n,
                        count, global.data() + first * n);
        } else if (rank == 0) {
            MPI_Recv(global.data() + first * n, static_cast<int>(count), MPI_DOUBLE,
                     owner, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == owner) {
            MPI_Send(local.data() + localRow(first, kBlockSize, ranks) * n,
                     static_cast<int>(count), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
        }
    }
    return true;
}

static bool distributedCholesky(std::vector<double>& A, size_t n,
                                int rank, int ranks) {
    std::vector<double> diagonal(kBlockSize * kBlockSize);
    // Transposed panel layout: panel[p*n+i] is L(i,k+p).  This layout turns
    // every local trailing update into contiguous vector operations.
    std::vector<double> panel(kBlockSize * n);
    std::vector<double> sendPanel;
    std::vector<double> receivePanel;
    std::vector<int> counts(ranks), displacements(ranks);

    for (size_t k = 0; k < n; k += kBlockSize) {
        const size_t width = std::min(kBlockSize, n - k);
        const size_t end = k + width;
        const int owner = rowOwner(k, kBlockSize, ranks);
        int positiveDefinite = 1;

        if (rank == owner) {
            const size_t firstLocal = localRow(k, kBlockSize, ranks);
            for (size_t i = 0; i < width && positiveDefinite; ++i) {
                double* row = A.data() + (firstLocal + i) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t p = 0; p < j; ++p) {
                        sum += row[k + p] * diagonal[j * width + p];
                    }
                    if (i == j) {
                        const double value = row[k + j] - sum;
                        if (value <= 0.0 || !std::isfinite(value)) {
                            positiveDefinite = 0;
                            break;
                        }
                        row[k + j] = std::sqrt(value);
                    } else {
                        row[k + j] = (row[k + j] - sum) /
                                     diagonal[j * width + j];
                    }
                    diagonal[i * width + j] = row[k + j];
                }
            }
        }

        MPI_Bcast(&positiveDefinite, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positiveDefinite) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite in block starting at %zu\n", k);
            }
            return false;
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        // Distributed triangular solve for the current block column.
        for (size_t b = static_cast<size_t>(rank);
             b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
            const size_t first = b * kBlockSize;
            const size_t rows = std::min(kBlockSize, n - first);
            if (first < end) continue;
            const size_t firstLocal = localRow(first, kBlockSize, ranks);
            for (size_t r = 0; r < rows; ++r) {
                double* row = A.data() + (firstLocal + r) * n;
                for (size_t j = 0; j < width; ++j) {
                    double sum = 0.0;
                    for (size_t p = 0; p < j; ++p) {
                        sum += row[k + p] * diagonal[j * width + p];
                    }
                    row[k + j] = (row[k + j] - sum) /
                                 diagonal[j * width + j];
                }
            }
        }

        // Pack each rank's portion of the panel in monotonically increasing
        // global-row order, then exchange it in one collective.
        size_t ownedBelow = 0;
        for (size_t b = static_cast<size_t>(rank);
             b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
            const size_t first = b * kBlockSize;
            if (first >= end) ownedBelow += std::min(kBlockSize, n - first);
        }
        sendPanel.resize(ownedBelow * width);
        size_t packedRow = 0;
        for (size_t b = static_cast<size_t>(rank);
             b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
            const size_t first = b * kBlockSize;
            const size_t rows = std::min(kBlockSize, n - first);
            if (first < end) continue;
            const size_t firstLocal = localRow(first, kBlockSize, ranks);
            for (size_t r = 0; r < rows; ++r, ++packedRow) {
                std::copy_n(A.data() + (firstLocal + r) * n + k, width,
                            sendPanel.data() + packedRow * width);
            }
        }

        int total = 0;
        bool countsFit = mpiCountFits(sendPanel.size());
        for (int q = 0; q < ranks; ++q) {
            const size_t rows = blockRows(n, kBlockSize, q, ranks);
            size_t rowsAtOrAbove = 0;
            for (size_t b = static_cast<size_t>(q);
                 b * kBlockSize < end; b += static_cast<size_t>(ranks)) {
                rowsAtOrAbove += std::min(kBlockSize, n - b * kBlockSize);
            }
            const size_t count = (rows - rowsAtOrAbove) * width;
            const bool countFits = mpiCountFits(count);
            countsFit = countsFit && countFits;
            const bool displacementFits = countFits &&
                static_cast<size_t>(total) + count <=
                    static_cast<size_t>(std::numeric_limits<int>::max());
            countsFit = countsFit && displacementFits;
            counts[q] = displacementFits ? static_cast<int>(count) : 0;
            displacements[q] = total;
            total += counts[q];
        }
        int allCountsFit = countsFit ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &allCountsFit, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (!allCountsFit) return false;
        receivePanel.resize(static_cast<size_t>(total));
        MPI_Allgatherv(sendPanel.data(), static_cast<int>(sendPanel.size()), MPI_DOUBLE,
                       receivePanel.data(), counts.data(), displacements.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        for (int q = 0; q < ranks; ++q) {
            size_t pos = static_cast<size_t>(displacements[q]);
            for (size_t b = static_cast<size_t>(q);
                 b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
                const size_t first = b * kBlockSize;
                const size_t rows = std::min(kBlockSize, n - first);
                if (first < end) continue;
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t p = 0; p < width; ++p) {
                        panel[p * n + first + r] = receivePanel[pos + r * width + p];
                    }
                }
                pos += rows * width;
            }
        }

        // Rank-width updates have unit-stride matrix and panel operands, which
        // the compiler can vectorize efficiently.
        for (size_t b = static_cast<size_t>(rank);
             b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
            const size_t first = b * kBlockSize;
            const size_t rows = std::min(kBlockSize, n - first);
            if (first < end) continue;
            const size_t firstLocal = localRow(first, kBlockSize, ranks);
            for (size_t r = 0; r < rows; ++r) {
                const size_t globalRow = first + r;
                double* row = A.data() + (firstLocal + r) * n;
                for (size_t p = 0; p < width; ++p) {
                    const double multiplier = row[k + p];
                    const double* source = panel.data() + p * n;
#pragma GCC ivdep
                    for (size_t j = end; j <= globalRow; ++j) {
                        row[j] -= multiplier * source[j];
                    }
                }
            }
        }
    }

    // Match the original representation: the upper triangle is explicitly 0.
    for (size_t b = static_cast<size_t>(rank);
         b * kBlockSize < n; b += static_cast<size_t>(ranks)) {
        const size_t first = b * kBlockSize;
        const size_t rows = std::min(kBlockSize, n - first);
        const size_t firstLocal = localRow(first, kBlockSize, ranks);
        for (size_t r = 0; r < rows; ++r) {
            double* row = A.data() + (firstLocal + r) * n;
            std::fill(row + first + r + 1, row + n, 0.0);
        }
    }
    return true;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t limit = std::min(i, j);
            for (size_t k = 0; k <= limit; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(original[i * n + j]) + 1e-10));
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
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
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
            parseStatus = 1;
        }
    }
    if (parseStatus || n == 0 || n > std::numeric_limits<size_t>::max() / n) {
        if (rank == 0 && !parseStatus) std::printf("Matrix size must be positive and representable\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> global;
    std::vector<double> original;
    if (rank == 0) {
        global.resize(n * n);
        generatePositiveDefiniteMatrix(global, n);
        if (validate) original = global;
    }
    std::vector<double> local(blockRows(n, kBlockSize, rank, ranks) * n);
    int setupOk = scatterBlockRows(global, local, n, rank, ranks) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &setupOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!setupOk) {
        if (rank == 0) std::printf("Matrix is too large for MPI message counts\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0 && !validate) std::vector<double>().swap(global);

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool localSuccess = distributedCholesky(local, n, rank, ranks);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int success = localSuccess ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / elapsed / 1e9);
    }

    int resultOk = 1;
    if (printResults || validate) {
        if (rank == 0 && global.empty()) global.resize(n * n);
        resultOk = gatherBlockRows(local, global, n, rank, ranks) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &resultOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!resultOk) {
            if (rank == 0) std::printf("Result is too large for MPI message counts\n");
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0 && printResults) print_results(global, "CholeskyL");
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        resultOk = validateCholesky(global, original, n) ? 1 : 0;
        std::printf("Validation: %s\n", resultOk ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&resultOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return resultOk ? 0 : 1;
}
