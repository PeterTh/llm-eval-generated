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

namespace {

// A row slab keeps the trailing update local and limits persistent matrix
// storage to O(n^2 / ranks).  The panel width is deliberately cache-sized:
// it substantially reduces synchronization relative to an unblocked method
// without requiring a dense, replicated matrix.
constexpr size_t kPanelWidth = 96;

struct RowDistribution {
    size_t first;
    size_t rows;
    std::vector<int> counts;
    std::vector<int> displacements;
};

RowDistribution distributeRows(size_t n, int ranks, int rank) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    RowDistribution distribution{};
    distribution.rows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    distribution.first = static_cast<size_t>(rank) * base +
                         std::min(static_cast<size_t>(rank), extra);
    distribution.counts.resize(ranks);
    distribution.displacements.resize(ranks);

    size_t offset = 0;
    for (int process = 0; process < ranks; ++process) {
        const size_t processRows = base + (static_cast<size_t>(process) < extra ? 1 : 0);
        distribution.counts[process] = static_cast<int>(processRows);
        distribution.displacements[process] = static_cast<int>(offset);
        offset += processRows;
    }
    return distribution;
}

int ownerOfRow(size_t row, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t largeRegion = (base + 1) * extra;
    if (row < largeRegion) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(extra + (row - largeRegion) / base);
}

// Generates exactly the same SPD input as the original program.  B is small
// compared with the distributed matrix and is replicated so every rank can
// create its assigned rows independently, avoiding a serial initialization.
void generatePositiveDefiniteRows(std::vector<double>& localA, size_t n,
                                  const RowDistribution& distribution) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t index = 0; index < B.size(); ++index) {
        B[index] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    for (size_t localRow = 0; localRow < distribution.rows; ++localRow) {
        const size_t globalRow = distribution.first + localRow;
        double* const output = localA.data() + localRow * n;
        const double* const left = B.data() + globalRow * n;
        for (size_t column = 0; column < n; ++column) {
            const double* const right = B.data() + column * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += left[k] * right[k];
            }
            output[column] = sum;
        }
        output[globalRow] += static_cast<double>(n);
    }
}

bool distributedCholesky(std::vector<double>& localA, size_t n,
                         const RowDistribution& distribution, int rank, int ranks) {
    std::vector<double> panel;
    std::vector<double> diagonalBlock;
    std::vector<int> panelCounts(ranks);
    std::vector<int> panelDisplacements(ranks);

    for (size_t panelStart = 0; panelStart < n;) {
        const int owner = ownerOfRow(panelStart, n, ranks);
        const size_t ownerEnd = distribution.displacements[owner] +
                                static_cast<size_t>(distribution.counts[owner]);
        const size_t width = std::min({kPanelWidth, n - panelStart, ownerEnd - panelStart});
        const size_t panelEnd = panelStart + width;
        int panelValid = 1;

        diagonalBlock.assign(width * width, 0.0);
        if (rank == owner) {
            const size_t localStart = panelStart - distribution.first;
            for (size_t i = 0; i < width; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double value = localA[(localStart + i) * n + panelStart + j];
                    for (size_t k = 0; k < j; ++k) {
                        value -= diagonalBlock[i * width + k] * diagonalBlock[j * width + k];
                    }
                    if (i == j) {
                        if (value <= 0.0) {
                            panelValid = 0;
                            break;
                        }
                        diagonalBlock[i * width + j] = std::sqrt(value);
                    } else {
                        diagonalBlock[i * width + j] = value / diagonalBlock[j * width + j];
                    }
                }
                if (!panelValid) {
                    break;
                }
            }
        }
        MPI_Bcast(&panelValid, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!panelValid) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", panelStart);
            }
            return false;
        }
        MPI_Bcast(diagonalBlock.data(), static_cast<int>(width * width), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        // Store the factored diagonal block in the distributed matrix.  This
        // is also what makes the gathered -r result a conventional lower L.
        if (rank == owner) {
            const size_t localStart = panelStart - distribution.first;
            for (size_t i = 0; i < width; ++i) {
                double* const row = localA.data() + (localStart + i) * n;
                for (size_t j = 0; j < width; ++j) {
                    row[panelStart + j] = j <= i ? diagonalBlock[i * width + j] : 0.0;
                }
                std::fill(row + panelStart + i + 1, row + n, 0.0);
            }
        }

        // Solve A_ik = L_ik L_kk^T for every locally owned trailing row.
        for (size_t localRow = 0; localRow < distribution.rows; ++localRow) {
            const size_t globalRow = distribution.first + localRow;
            if (globalRow < panelEnd) {
                continue;
            }
            double* const row = localA.data() + localRow * n;
            for (size_t column = 0; column < width; ++column) {
                double value = row[panelStart + column];
                for (size_t k = 0; k < column; ++k) {
                    value -= row[panelStart + k] * diagonalBlock[column * width + k];
                }
                row[panelStart + column] = value / diagonalBlock[column * width + column];
            }
        }

        // Complete panel rows are gathered once.  Thereafter each rank owns
        // its entire rank-k update, with no element-wise communication.
        panel.assign(n * width, 0.0);
        for (int process = 0; process < ranks; ++process) {
            panelCounts[process] = static_cast<int>(distribution.counts[process] * width);
            panelDisplacements[process] = static_cast<int>(distribution.displacements[process] * width);
        }
        std::vector<double> localPanel(distribution.rows * width);
        for (size_t localRow = 0; localRow < distribution.rows; ++localRow) {
            const size_t globalRow = distribution.first + localRow;
            double* const destination = localPanel.data() + localRow * width;
            if (globalRow >= panelStart && globalRow < panelEnd) {
                std::copy_n(diagonalBlock.data() + (globalRow - panelStart) * width, width, destination);
            } else if (globalRow >= panelEnd) {
                std::copy_n(localA.data() + localRow * n + panelStart, width, destination);
            }
        }
        MPI_Allgatherv(localPanel.data(), static_cast<int>(localPanel.size()), MPI_DOUBLE, panel.data(),
                       panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < distribution.rows; ++localRow) {
            const size_t globalRow = distribution.first + localRow;
            if (globalRow < panelEnd) {
                continue;
            }
            double* const row = localA.data() + localRow * n;
            const double* const rowPanel = panel.data() + globalRow * width;
            for (size_t k = 0; k < width; ++k) {
                const double multiplier = rowPanel[k];
                for (size_t column = panelEnd; column <= globalRow; ++column) {
                    row[column] -= multiplier * panel[column * width + k];
                }
            }
            // Preserve the original lower-triangular output convention.
            std::fill(row + globalRow + 1, row + n, 0.0);
        }
        panelStart = panelEnd;
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            for (size_t k = 0; k < n; ++k) {
                reconstructed += L[i * n + k] * L[j * n + k];
            }
            const double error = std::fabs(reconstructed - original[i * n + j]);
            maxError = std::max(maxError, error);
            relativeError = std::max(relativeError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int argumentStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            argumentStatus = 1;
        }
    }
    if (argumentStatus || n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            if (argumentStatus) std::printf("Unknown or incomplete option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const RowDistribution distribution = distributeRows(n, ranks, rank);
    std::vector<double> localA(distribution.rows * n);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n,
                    validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\nGenerating positive definite matrix...\n", ranks);
    }
    generatePositiveDefiniteRows(localA, n, distribution);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = distributedCholesky(localA, n, distribution, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();

    int globalSuccess = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    const double seconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&seconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> elementCounts(ranks), elementDisplacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        elementCounts[process] = static_cast<int>(distribution.counts[process] * n);
        elementDisplacements[process] = static_cast<int>(distribution.displacements[process] * n);
    }
    std::vector<double> result;
    if (rank == 0 && (validate || printResults)) result.resize(n * n);
    if (validate || printResults) {
        MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr, elementCounts.data(), elementDisplacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(std::llround(maxSeconds * 1000.0));
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / maxSeconds / 1e9);
        if (printResults) print_results(result, "CholeskyL");
        if (validate) {
            std::vector<double> original(n * n);
            // Regenerate on rank zero only for validation, keeping the normal
            // distributed path free of a global matrix allocation.
            RowDistribution rootDistribution{0, n, {static_cast<int>(n)}, {0}};
            generatePositiveDefiniteRows(original, n, rootDistribution);
            std::printf("Validating result...\n");
            if (validateCholesky(result, original, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
