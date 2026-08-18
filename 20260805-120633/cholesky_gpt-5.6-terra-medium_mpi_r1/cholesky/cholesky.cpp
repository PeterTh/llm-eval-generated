#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t kBlockSize = 96;

struct RowDistribution {
    size_t first;
    size_t count;
    std::vector<int> counts;
    std::vector<int> displacements;
};

RowDistribution distributeRows(const size_t n, const int ranks, const int rank) {
    RowDistribution distribution;
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    distribution.count = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    distribution.first = static_cast<size_t>(rank) * base +
                         std::min(static_cast<size_t>(rank), remainder);
    distribution.counts.resize(ranks);
    distribution.displacements.resize(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        distribution.counts[r] = static_cast<int>(rows);
        distribution.displacements[r] = static_cast<int>(offset);
        offset += rows;
    }
    return distribution;
}

int ownerOfRow(const size_t row, const RowDistribution& distribution) {
    const auto upper = std::upper_bound(distribution.displacements.begin(),
                                        distribution.displacements.end(),
                                        static_cast<int>(row));
    return static_cast<int>(upper - distribution.displacements.begin()) - 1;
}

void generatePositiveDefiniteMatrixRows(std::vector<double>& localA, const size_t n,
                                        const RowDistribution& distribution, const int rank) {
    // B is broadcast once, then each process forms only its assigned rows of B*B^T.
    // This preserves the original deterministic matrix while distributing its O(n^3) cost.
    std::vector<double> B(n * n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }
    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (size_t localRow = 0; localRow < distribution.count; ++localRow) {
        const size_t globalRow = distribution.first + localRow;
        double* const out = localA.data() + localRow * n;
        const double* const brow = B.data() + globalRow * n;
        for (size_t j = 0; j < n; ++j) {
            const double* const bcolumnRow = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += brow[k] * bcolumnRow[k];
            }
            out[j] = sum;
        }
        out[globalRow] += static_cast<double>(n);
    }
}

bool distributedCholesky(std::vector<double>& localA, const size_t n,
                         const RowDistribution& distribution, const int rank) {
    const size_t localEnd = distribution.first + distribution.count;

    for (size_t blockStart = 0; blockStart < n;) {
        const int owner = ownerOfRow(blockStart, distribution);
        // Keep each panel within one rank so the panel factorization has no communication.
        const size_t ownerEnd = static_cast<size_t>(distribution.displacements[owner]) +
                                static_cast<size_t>(distribution.counts[owner]);
        const size_t width = std::min({kBlockSize, n - blockStart, ownerEnd - blockStart});
        std::vector<double> diagonalPanel(width * width, 0.0);
        int positiveDefinite = 1;

        if (rank == owner) {
            for (size_t i = 0; i < width; ++i) {
                double* const row = localA.data() + (blockStart + i - distribution.first) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double value = row[blockStart + j];
                    for (size_t k = 0; k < j; ++k) {
                        value -= row[blockStart + k] *
                                 localA[(blockStart + j - distribution.first) * n + blockStart + k];
                    }
                    if (i == j) {
                        if (value <= 0.0) {
                            positiveDefinite = 0;
                            break;
                        }
                        row[blockStart + j] = std::sqrt(value);
                    } else {
                        row[blockStart + j] = value /
                                              localA[(blockStart + j - distribution.first) * n + blockStart + j];
                    }
                }
                if (!positiveDefinite) break;
            }
            if (positiveDefinite) {
                for (size_t i = 0; i < width; ++i) {
                    const double* const row = localA.data() + (blockStart + i - distribution.first) * n;
                    for (size_t j = 0; j <= i; ++j) diagonalPanel[i * width + j] = row[blockStart + j];
                }
            }
        }

        MPI_Bcast(&positiveDefinite, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!positiveDefinite) return false;
        MPI_Bcast(diagonalPanel.data(), static_cast<int>(diagonalPanel.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        const size_t trailingStart = blockStart + width;
        // Triangular solve: L_ik = A_ik * inv(L_kk^T), for rows owned by this rank.
        for (size_t globalRow = std::max(trailingStart, distribution.first); globalRow < localEnd; ++globalRow) {
            double* const row = localA.data() + (globalRow - distribution.first) * n;
            for (size_t j = 0; j < width; ++j) {
                double value = row[blockStart + j];
                for (size_t k = 0; k < j; ++k) value -= row[blockStart + k] * diagonalPanel[j * width + k];
                row[blockStart + j] = value / diagonalPanel[j * width + j];
            }
        }

        const size_t localPanelStart = std::max(trailingStart, distribution.first);
        const size_t localPanelRows = localEnd > localPanelStart ? localEnd - localPanelStart : 0;
        std::vector<double> sendPanel(localPanelRows * width);
        for (size_t i = 0; i < localPanelRows; ++i) {
            const double* const row = localA.data() + (localPanelStart + i - distribution.first) * n;
            std::copy_n(row + blockStart, width, sendPanel.data() + i * width);
        }

        std::vector<int> panelCounts(distribution.counts.size());
        std::vector<int> panelDisplacements(distribution.counts.size());
        int panelSize = 0;
        for (size_t r = 0; r < distribution.counts.size(); ++r) {
            const size_t rankStart = static_cast<size_t>(distribution.displacements[r]);
            const size_t rankEnd = rankStart + static_cast<size_t>(distribution.counts[r]);
            const size_t rows = rankEnd > std::max(rankStart, trailingStart)
                                    ? rankEnd - std::max(rankStart, trailingStart) : 0;
            panelCounts[r] = static_cast<int>(rows * width);
            panelDisplacements[r] = panelSize;
            panelSize += panelCounts[r];
        }
        std::vector<double> trailingPanel(static_cast<size_t>(panelSize));
        MPI_Allgatherv(sendPanel.data(), static_cast<int>(sendPanel.size()), MPI_DOUBLE,
                       trailingPanel.data(), panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Rank-local SYRK/GEMM-style update of the trailing lower triangle.
        for (size_t globalRow = std::max(trailingStart, distribution.first); globalRow < localEnd; ++globalRow) {
            double* const row = localA.data() + (globalRow - distribution.first) * n;
            const double* const left = trailingPanel.data() + (globalRow - trailingStart) * width;
            for (size_t j = trailingStart; j <= globalRow; ++j) {
                const double* const right = trailingPanel.data() + (j - trailingStart) * width;
                double update = 0.0;
                for (size_t k = 0; k < width; ++k) update += left[k] * right[k];
                row[j] -= update;
            }
        }
        blockStart = trailingStart;
    }

    for (size_t localRow = 0; localRow < distribution.count; ++localRow) {
        double* const row = localA.data() + localRow * n;
        const size_t globalRow = distribution.first + localRow;
        std::fill(row + globalRow + 1, row + n, 0.0);
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, const size_t n) {
    double maxAbsoluteError = 0.0;
    double maxRelativeError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            for (size_t k = 0; k < n; ++k) reconstructed += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(reconstructed - original[i * n + j]);
            maxAbsoluteError = std::max(maxAbsoluteError, error);
            maxRelativeError = std::max(maxRelativeError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxAbsoluteError);
    std::printf("Max relative error: %.10e\n", maxRelativeError);
    if (maxRelativeError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n"
                "  -r           Print results for external validation\n  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false, argumentsValid = true, showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) showHelp = true;
        else argumentsValid = false;
    }
    if (!argumentsValid || showHelp || n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) / n) {
        if (rank == 0) {
            if (!argumentsValid) std::printf("Invalid command-line option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid && !showHelp ? 1 : 0;
    }

    const RowDistribution distribution = distributeRows(n, ranks, rank);
    std::vector<double> localA(distribution.count * n);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nMPI ranks: %d\nValidation: %s\n"
                    "Generating positive definite matrix...\n", n, n, ranks, validate ? "enabled" : "disabled");
    }
    generatePositiveDefiniteMatrixRows(localA, n, distribution, rank);
    std::vector<double> localOriginal;
    if (validate) localOriginal = localA;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool localSuccess = distributedCholesky(localA, n, distribution, rank);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int success = localSuccess ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    std::vector<double> fullL;
    std::vector<double> fullOriginal;
    if ((printResults || validate) && rank == 0) fullL.resize(n * n);
    std::vector<int> elementCounts(ranks), elementDisplacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        elementCounts[r] = distribution.counts[r] * static_cast<int>(n);
        elementDisplacements[r] = distribution.displacements[r] * static_cast<int>(n);
    }
    if (printResults || validate) {
        MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE, fullL.data(),
                    elementCounts.data(), elementDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (validate) {
        if (rank == 0) fullOriginal.resize(n * n);
        MPI_Gatherv(localOriginal.data(), static_cast<int>(localOriginal.size()), MPI_DOUBLE, fullOriginal.data(),
                    elementCounts.data(), elementDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int result = success ? 0 : 1;
    if (rank == 0) {
        if (!success) std::printf("Cholesky decomposition failed\n");
        else {
            const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
            std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", milliseconds,
                        (static_cast<double>(n) * n * n / 3.0) / maxElapsed / 1e9);
            if (printResults) print_results(fullL, "CholeskyL");
            if (validate) {
                std::printf("Validating result...\n");
                result = validateCholesky(fullL, fullOriginal, n) ? 0 : 1;
                std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
