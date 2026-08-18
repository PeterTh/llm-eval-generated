#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous, block-aligned row ranges.  A rank
// stores all columns of its rows, which keeps the local Schur-complement update
// contiguous in memory.  At each block step the newly computed panel is
// all-gathered; every rank can then update its own lower-triangular rows without
// further fine-grained communication.
struct RowDistribution {
    std::vector<size_t> counts;
    std::vector<size_t> displacements;
    std::vector<int> blockOwners;
};

size_t chooseBlockSize(const size_t n, const int ranks) {
    if (n == 0) {
        return 1;
    }

    // Keep several panels per rank for load balance, while retaining a tile
    // large enough for the cache-friendly rank-b update below.
    const size_t targetPanels = std::max<size_t>(1, static_cast<size_t>(ranks) * 4);
    const size_t balancedSize = (n + targetPanels - 1) / targetPanels;
    return std::min<size_t>(128, std::max<size_t>(32, balancedSize));
}

RowDistribution makeRowDistribution(const size_t n, const size_t blockSize, const int ranks) {
    const size_t blockCount = (n + blockSize - 1) / blockSize;
    const size_t baseBlocks = blockCount / static_cast<size_t>(ranks);
    const size_t extraBlocks = blockCount % static_cast<size_t>(ranks);

    RowDistribution distribution;
    distribution.counts.resize(ranks);
    distribution.displacements.resize(ranks);
    distribution.blockOwners.resize(blockCount);

    size_t firstBlock = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        const size_t ownedBlocks = baseBlocks + (static_cast<size_t>(rank) < extraBlocks ? 1 : 0);
        const size_t firstRow = std::min(n, firstBlock * blockSize);
        const size_t endRow = std::min(n, (firstBlock + ownedBlocks) * blockSize);

        distribution.displacements[rank] = firstRow;
        distribution.counts[rank] = endRow - firstRow;
        for (size_t block = firstBlock; block < firstBlock + ownedBlocks; ++block) {
            distribution.blockOwners[block] = rank;
        }
        firstBlock += ownedBlocks;
    }
    return distribution;
}

bool makeMpiCounts(const RowDistribution& distribution, const size_t n,
                   std::vector<int>& elementCounts, std::vector<int>& elementDisplacements) {
    elementCounts.resize(distribution.counts.size());
    elementDisplacements.resize(distribution.displacements.size());

    if (n == 0) {
        std::fill(elementCounts.begin(), elementCounts.end(), 0);
        std::fill(elementDisplacements.begin(), elementDisplacements.end(), 0);
        return true;
    }

    for (size_t rank = 0; rank < distribution.counts.size(); ++rank) {
        if (distribution.counts[rank] > static_cast<size_t>(std::numeric_limits<int>::max()) / n ||
            distribution.displacements[rank] > static_cast<size_t>(std::numeric_limits<int>::max()) / n) {
            return false;
        }
        elementCounts[rank] = static_cast<int>(distribution.counts[rank] * n);
        elementDisplacements[rank] = static_cast<int>(distribution.displacements[rank] * n);
    }
    return true;
}

// Generate exactly the same B matrix as the original benchmark on rank zero,
// then compute A = B * B^T in parallel.  B row blocks circulate around the MPI
// ring, so each rank only retains its own rows after initialization.
void generatePositiveDefiniteMatrix(std::vector<double>& localA, const size_t n,
                                    const RowDistribution& distribution,
                                    const std::vector<int>& elementCounts,
                                    const std::vector<int>& elementDisplacements,
                                    const int rank, const int ranks) {
    const size_t localRows = distribution.counts[rank];
    std::vector<double> localB(localRows * n);
    std::vector<double> fullB;

    if (rank == 0) {
        fullB.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            fullB[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }

    MPI_Scatterv(rank == 0 ? fullB.data() : nullptr, elementCounts.data(), elementDisplacements.data(),
                 MPI_DOUBLE, localB.data(), elementCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double>().swap(fullB);

    std::vector<double> current = localB;
    int currentOwner = rank;
    for (int step = 0; step < ranks; ++step) {
        const size_t currentRows = distribution.counts[currentOwner];
        const size_t currentStart = distribution.displacements[currentOwner];

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            const double* const bRow = localB.data() + localRow * n;
            double* const aRow = localA.data() + localRow * n + currentStart;
            for (size_t otherRow = 0; otherRow < currentRows; ++otherRow) {
                const double* const otherBRow = current.data() + otherRow * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += bRow[k] * otherBRow[k];
                }
                aRow[otherRow] = sum;
            }
        }

        if (step + 1 < ranks) {
            const int nextRank = (rank + 1) % ranks;
            const int previousRank = (rank + ranks - 1) % ranks;
            const int nextOwner = (currentOwner + ranks - 1) % ranks;
            std::vector<double> next(distribution.counts[nextOwner] * n);

            MPI_Sendrecv(current.data(), static_cast<int>(currentRows * n), MPI_DOUBLE, nextRank, 0,
                         next.data(), static_cast<int>(distribution.counts[nextOwner] * n), MPI_DOUBLE,
                         previousRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            current = std::move(next);
            currentOwner = nextOwner;
        }
    }

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = distribution.displacements[rank] + localRow;
        localA[localRow * n + globalRow] += static_cast<double>(n);
    }
}

bool factorDiagonalBlock(std::vector<double>& localA, const size_t n, const size_t localOffset,
                         const size_t blockStart, const size_t blockWidth, int& failureIndex) {
    for (size_t column = 0; column < blockWidth; ++column) {
        const size_t diagonalRow = localOffset + column;
        double* const diagonal = localA.data() + diagonalRow * n + blockStart;

        double sum = 0.0;
        for (size_t k = 0; k < column; ++k) {
            sum += diagonal[k] * diagonal[k];
        }
        const double value = diagonal[column] - sum;
        if (value <= 0.0) {
            failureIndex = static_cast<int>(blockStart + column);
            return false;
        }
        diagonal[column] = std::sqrt(value);

        for (size_t row = column + 1; row < blockWidth; ++row) {
            double* const target = localA.data() + (localOffset + row) * n + blockStart;
            double dot = 0.0;
            for (size_t k = 0; k < column; ++k) {
                dot += target[k] * diagonal[k];
            }
            target[column] = (target[column] - dot) / diagonal[column];
        }
    }
    return true;
}

bool choleskyDecomposition(std::vector<double>& localA, const size_t n,
                           const RowDistribution& distribution, const size_t blockSize,
                           const int rank, const int ranks) {
    const size_t localRows = distribution.counts[rank];
    const size_t localStart = distribution.displacements[rank];
    const size_t blockCount = distribution.blockOwners.size();

    std::vector<int> panelCounts(ranks);
    std::vector<int> panelDisplacements(ranks);

    for (size_t block = 0; block < blockCount; ++block) {
        const size_t blockStart = block * blockSize;
        const size_t blockWidth = std::min(blockSize, n - blockStart);
        const size_t trailingStart = blockStart + blockWidth;
        const int owner = distribution.blockOwners[block];

        int failed = 0;
        int failureIndex = -1;
        if (rank == owner) {
            const size_t localOffset = blockStart - localStart;
            if (!factorDiagonalBlock(localA, n, localOffset, blockStart, blockWidth, failureIndex)) {
                failed = 1;
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&failureIndex, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed != 0) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %d\n", failureIndex);
            }
            return false;
        }

        // The triangular solve needs just the diagonal tile.  Packing it makes
        // the broadcast contiguous and avoids exposing MPI datatypes in the hot path.
        std::vector<double> diagonal(blockWidth * blockWidth, 0.0);
        if (rank == owner) {
            const size_t localOffset = blockStart - localStart;
            for (size_t row = 0; row < blockWidth; ++row) {
                const double* const source = localA.data() + (localOffset + row) * n + blockStart;
                std::copy_n(source, row + 1, diagonal.data() + row * blockWidth);
            }
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // A(i,k) = A(i,k) * inv(L(k,k)^T), for local rows below the panel.
        const size_t firstLocalTrailingRow = std::max(trailingStart, localStart);
        for (size_t globalRow = firstLocalTrailingRow; globalRow < localStart + localRows; ++globalRow) {
            double* const target = localA.data() + (globalRow - localStart) * n + blockStart;
            for (size_t column = 0; column < blockWidth; ++column) {
                double sum = target[column];
                const double* const diagonalRow = diagonal.data() + column * blockWidth;
                for (size_t k = 0; k < column; ++k) {
                    sum -= target[k] * diagonalRow[k];
                }
                target[column] = sum / diagonalRow[column];
            }
        }

        // Pack the local portion of the completed panel and exchange it once.
        // The transposed copy makes the innermost Schur update a contiguous SAXPY.
        std::vector<double> localPanel(localRows * blockWidth);
        for (size_t row = 0; row < localRows; ++row) {
            std::copy_n(localA.data() + row * n + blockStart, blockWidth,
                        localPanel.data() + row * blockWidth);
        }
        for (int process = 0; process < ranks; ++process) {
            panelCounts[process] = static_cast<int>(distribution.counts[process] * blockWidth);
            panelDisplacements[process] = static_cast<int>(distribution.displacements[process] * blockWidth);
        }
        std::vector<double> panel(n * blockWidth);
        MPI_Allgatherv(localPanel.data(), panelCounts[rank], MPI_DOUBLE, panel.data(), panelCounts.data(),
                       panelDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        if (trailingStart < n) {
            std::vector<double> transposedPanel(blockWidth * n);
            for (size_t row = trailingStart; row < n; ++row) {
                const double* const source = panel.data() + row * blockWidth;
                for (size_t k = 0; k < blockWidth; ++k) {
                    transposedPanel[k * n + row] = source[k];
                }
            }

            const size_t firstUpdateRow = std::max(trailingStart, localStart);
            for (size_t globalRow = firstUpdateRow; globalRow < localStart + localRows; ++globalRow) {
                double* const target = localA.data() + (globalRow - localStart) * n;
                const double* const rowPanel = panel.data() + globalRow * blockWidth;
                for (size_t k = 0; k < blockWidth; ++k) {
                    const double multiplier = rowPanel[k];
                    const double* const source = transposedPanel.data() + k * n + trailingStart;
                    double* const destination = target + trailingStart;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
                    for (size_t column = trailingStart; column <= globalRow; ++column) {
                        destination[column - trailingStart] -= multiplier * source[column - trailingStart];
                    }
                }
            }
        }
    }

    // Keep the result representation identical to the original program.
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = localStart + localRow;
        std::fill(localA.begin() + static_cast<std::ptrdiff_t>(localRow * n + globalRow + 1),
                  localA.begin() + static_cast<std::ptrdiff_t>((localRow + 1) * n), 0.0);
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& AOriginal, const size_t n) {
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
    double relativeError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - AOriginal[i]);
        maxError = std::max(maxError, error);
        relativeError = std::max(relativeError, error / (std::fabs(AOriginal[i]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
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
    int argumentStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            argumentStatus = 1;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
        }
    }
    if (argumentStatus != 0) {
        MPI_Finalize();
        return argumentStatus;
    }

    const size_t blockSize = chooseBlockSize(n, ranks);
    const RowDistribution distribution = makeRowDistribution(n, blockSize, ranks);
    std::vector<int> elementCounts;
    std::vector<int> elementDisplacements;
    const bool mpiCountsFit = makeMpiCounts(distribution, n, elementCounts, elementDisplacements);
    if (!mpiCountsFit) {
        if (rank == 0) {
            std::printf("Error: matrix is too large for this MPI implementation\n");
        }
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

    const size_t localRows = distribution.counts[rank];
    std::vector<double> localA(localRows * n);
    generatePositiveDefiniteMatrix(localA, n, distribution, elementCounts, elementDisplacements, rank, ranks);
    std::vector<double> localOriginal;
    if (validate) {
        localOriginal = localA;
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, distribution, blockSize, rank, ranks);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(duration * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = duration > 0.0 ? operations / duration / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    std::vector<double> original;
    if (rank == 0 && (printResults || validate)) {
        result.resize(n * n);
        if (validate) {
            original.resize(n * n);
        }
    }
    if (printResults || validate) {
        MPI_Gatherv(localA.data(), elementCounts[rank], MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                    elementCounts.data(), elementDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (validate) {
        MPI_Gatherv(localOriginal.data(), elementCounts[rank], MPI_DOUBLE, rank == 0 ? original.data() : nullptr,
                    elementCounts.data(), elementDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0) {
        if (printResults) {
            print_results(result, "CholeskyL");
        }
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateCholesky(result, original, n) ? 1 : 0;
            std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid != 0 ? 0 : 1;
}
