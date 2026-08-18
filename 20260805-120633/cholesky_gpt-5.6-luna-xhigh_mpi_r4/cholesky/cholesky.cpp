#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// A moderate tile size gives the local update kernels enough work to amortize
// MPI collectives while leaving enough tiles to distribute over a 2-D grid.
constexpr size_t BlockSize = 64;
constexpr int Root = 0;
constexpr int DistributionTag = 1701;
constexpr int GatherTag = 1702;

struct ProcessGrid {
    MPI_Comm comm = MPI_COMM_NULL;
    int size = 0;
    int rank = 0;
    int rows = 0;
    int cols = 0;
    int row = 0;
    int col = 0;

    ProcessGrid() {
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int dimensions[2] = {0, 0};
        MPI_Dims_create(size, 2, dimensions);
        rows = dimensions[0];
        cols = dimensions[1];

        const int periods[2] = {0, 0};
        // Reordering is disabled so Cartesian ranks retain the world-rank
        // mapping used by ownerRank().
        MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);

        int coordinates[2] = {0, 0};
        MPI_Cart_coords(comm, rank, 2, coordinates);
        row = coordinates[0];
        col = coordinates[1];
    }

    int ownerRank(const size_t blockRow, const size_t blockCol) const {
        return static_cast<int>((blockRow % static_cast<size_t>(rows)) *
                                static_cast<size_t>(cols) +
                                (blockCol % static_cast<size_t>(cols)));
    }
};

struct DistributedMatrix {
    size_t n = 0;
    size_t blockSize = 0;
    size_t tileElements = 0;
    size_t blockCount = 0;
    size_t localRowBlocks = 0;
    size_t localColBlocks = 0;
    const ProcessGrid& grid;
    std::vector<double> values;

    DistributedMatrix(const size_t matrixSize, const size_t tileSize,
                      const ProcessGrid& processGrid)
        : n(matrixSize),
          blockSize(tileSize),
          tileElements(tileSize * tileSize),
          blockCount((matrixSize + tileSize - 1) / tileSize),
          grid(processGrid) {
        localRowBlocks = grid.row < static_cast<int>(blockCount)
                             ? (blockCount - 1 - static_cast<size_t>(grid.row)) /
                                       static_cast<size_t>(grid.rows) +
                                   1
                             : 0;
        localColBlocks = grid.col < static_cast<int>(blockCount)
                             ? (blockCount - 1 - static_cast<size_t>(grid.col)) /
                                       static_cast<size_t>(grid.cols) +
                                   1
                             : 0;

        values.resize(localRowBlocks * localColBlocks * tileElements, 0.0);
    }

    bool owns(const size_t blockRow, const size_t blockCol) const {
        return blockRow % static_cast<size_t>(grid.rows) ==
                   static_cast<size_t>(grid.row) &&
               blockCol % static_cast<size_t>(grid.cols) ==
                   static_cast<size_t>(grid.col);
    }

    double* tile(const size_t blockRow, const size_t blockCol) {
        const size_t localRow = blockRow / static_cast<size_t>(grid.rows);
        const size_t localCol = blockCol / static_cast<size_t>(grid.cols);
        return values.data() +
               (localRow * localColBlocks + localCol) * tileElements;
    }

    const double* tile(const size_t blockRow, const size_t blockCol) const {
        const size_t localRow = blockRow / static_cast<size_t>(grid.rows);
        const size_t localCol = blockCol / static_cast<size_t>(grid.cols);
        return values.data() +
               (localRow * localColBlocks + localCol) * tileElements;
    }
};

size_t blockWidth(const size_t n, const size_t blockSize,
                  const size_t block) {
    const size_t begin = block * blockSize;
    return begin < n ? std::min(blockSize, n - begin) : 0;
}

// Generate the same matrix as the original benchmark.  Only rank zero needs
// the global matrix; the factorization itself stores only distributed tiles.
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

void packGlobalTile(const std::vector<double>& global, const size_t n,
                    const size_t blockSize, const size_t blockRow,
                    const size_t blockCol, std::vector<double>& packed) {
    std::fill(packed.begin(), packed.end(), 0.0);
    const size_t rowBegin = blockRow * blockSize;
    const size_t colBegin = blockCol * blockSize;
    const size_t rows = blockWidth(n, blockSize, blockRow);
    const size_t cols = blockWidth(n, blockSize, blockCol);

    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(global.data() + (rowBegin + i) * n + colBegin, cols,
                    packed.data() + i * blockSize);
    }
}

void distributeInitialMatrix(const std::vector<double>& global,
                             DistributedMatrix& matrix) {
    const ProcessGrid& grid = matrix.grid;
    std::vector<double> packed(matrix.tileElements, 0.0);

    for (size_t blockRow = 0; blockRow < matrix.blockCount; ++blockRow) {
        for (size_t blockCol = 0; blockCol <= blockRow; ++blockCol) {
            const int owner = grid.ownerRank(blockRow, blockCol);

            if (grid.rank == Root) {
                packGlobalTile(global, matrix.n, matrix.blockSize, blockRow,
                               blockCol, packed);
                if (owner == Root) {
                    std::copy(packed.begin(), packed.end(),
                              matrix.tile(blockRow, blockCol));
                } else {
                    MPI_Send(packed.data(), static_cast<int>(matrix.tileElements),
                             MPI_DOUBLE, owner, DistributionTag, grid.comm);
                }
            } else if (owner == grid.rank) {
                MPI_Recv(matrix.tile(blockRow, blockCol),
                         static_cast<int>(matrix.tileElements), MPI_DOUBLE,
                         Root, DistributionTag, grid.comm, MPI_STATUS_IGNORE);
            }
        }
    }
}

bool factorDiagonalTile(double* tile, const size_t blockSize,
                        const size_t width, size_t& badElement) {
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += tile[i * blockSize + k] * tile[j * blockSize + k];
            }

            if (i == j) {
                const double value = tile[j * blockSize + j] - sum;
                if (value <= 0.0) {
                    badElement = j;
                    return false;
                }
                tile[j * blockSize + j] = std::sqrt(value);
            } else {
                tile[i * blockSize + j] =
                    (tile[i * blockSize + j] - sum) /
                    tile[j * blockSize + j];
            }
        }

        for (size_t j = i + 1; j < width; ++j) {
            tile[i * blockSize + j] = 0.0;
        }
    }
    return true;
}

// Solve X * L^T = B, where L is the factored diagonal tile.
void solvePanelTile(double* panelTile, const double* diagonalTile,
                    const size_t blockSize, const size_t panelRows,
                    const size_t diagonalWidth) {
    for (size_t row = 0; row < panelRows; ++row) {
        for (size_t col = 0; col < diagonalWidth; ++col) {
            double value = panelTile[row * blockSize + col];
            for (size_t k = 0; k < col; ++k) {
                value -= panelTile[row * blockSize + k] *
                         diagonalTile[col * blockSize + k];
            }
            panelTile[row * blockSize + col] =
                value / diagonalTile[col * blockSize + col];
        }
    }
}

// Exchange the current block column and make both orientations available to
// every rank.  Allgatherv sends each panel tile exactly once; the received
// panel is O(n * blockSize), rather than a replicated O(n^2) matrix.
void exchangePanel(const DistributedMatrix& matrix, const size_t kBlock,
                   std::vector<double>& sendBuffer,
                   std::vector<double>& receiveBuffer,
                   std::vector<int>& counts, std::vector<int>& displacements,
                   std::vector<double>& panel,
                   std::vector<double>& transposedPanel) {
    const ProcessGrid& grid = matrix.grid;
    size_t localTiles = 0;

    for (size_t blockRow = kBlock + 1; blockRow < matrix.blockCount;
         ++blockRow) {
        if (matrix.owns(blockRow, kBlock)) {
            std::copy(matrix.tile(blockRow, kBlock),
                      matrix.tile(blockRow, kBlock) + matrix.tileElements,
                      sendBuffer.data() + localTiles * matrix.tileElements);
            ++localTiles;
        }
    }

    const size_t localElements = localTiles * matrix.tileElements;
    if (localElements > static_cast<size_t>(INT_MAX)) {
        if (grid.rank == Root) {
            std::fprintf(stderr, "MPI panel exceeds the supported message size\n");
        }
        MPI_Abort(grid.comm, 1);
    }
    const int sendCount = static_cast<int>(localElements);
    MPI_Allgather(&sendCount, 1, MPI_INT, counts.data(), 1, MPI_INT, grid.comm);

    int totalElements = 0;
    for (int rank = 0; rank < grid.size; ++rank) {
        displacements[rank] = totalElements;
        if (counts[rank] > INT_MAX - totalElements) {
            if (grid.rank == Root) {
                std::fprintf(stderr, "MPI panel exceeds the supported message size\n");
            }
            MPI_Abort(grid.comm, 1);
        }
        totalElements += counts[rank];
    }

    MPI_Allgatherv(sendBuffer.data(), sendCount, MPI_DOUBLE,
                   receiveBuffer.data(), counts.data(), displacements.data(),
                   MPI_DOUBLE, grid.comm);

    for (size_t blockRow = kBlock + 1; blockRow < matrix.blockCount;
         ++blockRow) {
        const int owner = grid.ownerRank(blockRow, kBlock);
        size_t tileIndex = 0;
        for (size_t previous = kBlock + 1; previous < blockRow; ++previous) {
            if (grid.ownerRank(previous, kBlock) == owner) {
                ++tileIndex;
            }
        }

        const double* source =
            receiveBuffer.data() + displacements[owner] +
            tileIndex * matrix.tileElements;
        double* normal = panel.data() + blockRow * matrix.tileElements;
        double* transpose =
            transposedPanel.data() + blockRow * matrix.tileElements;
        const size_t rows = blockWidth(matrix.n, matrix.blockSize, blockRow);
        const size_t cols = blockWidth(matrix.n, matrix.blockSize, kBlock);

        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                normal[row * matrix.blockSize + col] =
                    source[row * matrix.blockSize + col];
                transpose[col * matrix.blockSize + row] =
                    source[row * matrix.blockSize + col];
            }
        }
    }
}

// C -= A * B^T, with A and B taken from the current panel.  The k loop is
// outside the innermost loop so each update streams across contiguous rows.
void updateTile(double* C, const double* A, const double* transposedB,
                const size_t blockSize, const size_t rows,
                const size_t cols, const size_t inner) {
    for (size_t row = 0; row < rows; ++row) {
        double* c = C + row * blockSize;
        const double* a = A + row * blockSize;
        for (size_t k = 0; k < inner; ++k) {
            const double scalar = a[k];
            const double* b = transposedB + k * blockSize;
            for (size_t col = 0; col < cols; ++col) {
                c[col] -= scalar * b[col];
            }
        }
    }
}

bool distributedCholesky(DistributedMatrix& matrix) {
    const ProcessGrid& grid = matrix.grid;
    std::vector<double> diagonal(matrix.tileElements, 0.0);
    std::vector<double> panel(matrix.blockCount * matrix.tileElements, 0.0);
    std::vector<double> transposedPanel(matrix.blockCount * matrix.tileElements,
                                        0.0);
    const size_t maxLocalPanelTiles =
        (matrix.blockCount + static_cast<size_t>(grid.rows) - 1) /
        static_cast<size_t>(grid.rows);
    std::vector<double> sendBuffer(
        std::max<size_t>(1, maxLocalPanelTiles * matrix.tileElements), 0.0);
    std::vector<double> receiveBuffer(
        std::max<size_t>(1, matrix.blockCount * matrix.tileElements), 0.0);
    std::vector<int> counts(static_cast<size_t>(grid.size), 0);
    std::vector<int> displacements(static_cast<size_t>(grid.size), 0);

    for (size_t kBlock = 0; kBlock < matrix.blockCount; ++kBlock) {
        const size_t diagonalWidth =
            blockWidth(matrix.n, matrix.blockSize, kBlock);
        const int diagonalOwner = grid.ownerRank(kBlock, kBlock);

        bool localSuccess = true;
        size_t localBadElement = 0;
        if (grid.rank == diagonalOwner) {
            std::copy(matrix.tile(kBlock, kBlock),
                      matrix.tile(kBlock, kBlock) + matrix.tileElements,
                      diagonal.begin());
            localSuccess = factorDiagonalTile(
                matrix.tile(kBlock, kBlock), matrix.blockSize, diagonalWidth,
                localBadElement);
            if (localSuccess) {
                std::copy(matrix.tile(kBlock, kBlock),
                          matrix.tile(kBlock, kBlock) + matrix.tileElements,
                          diagonal.begin());
            }
        }

        int localStatus = localSuccess ? 1 : 0;
        int globalStatus = 0;
        MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MIN,
                      grid.comm);
        if (globalStatus == 0) {
            long long localBad = localSuccess
                                     ? std::numeric_limits<long long>::max()
                                     : static_cast<long long>(kBlock *
                                                              matrix.blockSize +
                                                              localBadElement);
            long long globalBad = 0;
            MPI_Allreduce(&localBad, &globalBad, 1, MPI_LONG_LONG, MPI_MIN,
                          grid.comm);
            if (grid.rank == Root) {
                std::printf(
                    "Error: Matrix is not positive definite at diagonal element %lld\n",
                    globalBad);
            }
            return false;
        }

        MPI_Bcast(diagonal.data(), static_cast<int>(matrix.tileElements),
                  MPI_DOUBLE, diagonalOwner, grid.comm);

        for (size_t blockRow = kBlock + 1; blockRow < matrix.blockCount;
             ++blockRow) {
            if (matrix.owns(blockRow, kBlock)) {
                solvePanelTile(matrix.tile(blockRow, kBlock), diagonal.data(),
                               matrix.blockSize,
                               blockWidth(matrix.n, matrix.blockSize, blockRow),
                               diagonalWidth);
            }
        }

        exchangePanel(matrix, kBlock, sendBuffer, receiveBuffer, counts,
                      displacements, panel, transposedPanel);

        for (size_t blockRow = static_cast<size_t>(grid.row);
             blockRow < matrix.blockCount;
             blockRow += static_cast<size_t>(grid.rows)) {
            if (blockRow <= kBlock) {
                continue;
            }
            const size_t rows =
                blockWidth(matrix.n, matrix.blockSize, blockRow);
            const double* leftPanel =
                panel.data() + blockRow * matrix.tileElements;

            for (size_t blockCol = static_cast<size_t>(grid.col);
                 blockCol <= blockRow;
                 blockCol += static_cast<size_t>(grid.cols)) {
                if (blockCol <= kBlock) {
                    continue;
                }
                updateTile(matrix.tile(blockRow, blockCol), leftPanel,
                           transposedPanel.data() +
                               blockCol * matrix.tileElements,
                           matrix.blockSize, rows,
                           blockWidth(matrix.n, matrix.blockSize, blockCol),
                           diagonalWidth);
            }
        }
    }

    return true;
}

void gatherLowerMatrix(const DistributedMatrix& matrix,
                       std::vector<double>& global) {
    const ProcessGrid& grid = matrix.grid;
    std::vector<double> packed(matrix.tileElements, 0.0);

    if (grid.rank == Root) {
        std::fill(global.begin(), global.end(), 0.0);
    }

    for (size_t blockRow = 0; blockRow < matrix.blockCount; ++blockRow) {
        for (size_t blockCol = 0; blockCol <= blockRow; ++blockCol) {
            const int owner = grid.ownerRank(blockRow, blockCol);
            if (grid.rank == Root) {
                if (owner == Root) {
                    std::copy(matrix.tile(blockRow, blockCol),
                              matrix.tile(blockRow, blockCol) +
                                  matrix.tileElements,
                              packed.begin());
                } else {
                    MPI_Recv(packed.data(), static_cast<int>(matrix.tileElements),
                             MPI_DOUBLE, owner, GatherTag, grid.comm,
                             MPI_STATUS_IGNORE);
                }

                const size_t rowBegin = blockRow * matrix.blockSize;
                const size_t colBegin = blockCol * matrix.blockSize;
                const size_t rows =
                    blockWidth(matrix.n, matrix.blockSize, blockRow);
                const size_t cols =
                    blockWidth(matrix.n, matrix.blockSize, blockCol);
                for (size_t row = 0; row < rows; ++row) {
                    std::copy_n(packed.data() + row * matrix.blockSize, cols,
                                global.data() + (rowBegin + row) * matrix.n +
                                    colBegin);
                }
            } else if (owner == grid.rank) {
                MPI_Send(matrix.tile(blockRow, blockCol),
                         static_cast<int>(matrix.tileElements), MPI_DOUBLE,
                         Root, GatherTag, grid.comm);
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& AOriginal, const size_t n) {
    std::vector<double> reconstructed(n * n, 0.0);

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
        relativeError = std::max(
            relativeError, error / (std::fabs(AOriginal[i]) + 1e-10));
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    ProcessGrid grid;
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed =
                std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0') {
                parseError = true;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (grid.rank == Root) {
            if (parseError) {
                std::printf("Unknown or invalid command line option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Comm_free(&grid.comm);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (grid.rank == Root) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d grid), block size: %zu\n",
                    grid.size, grid.rows, grid.cols, BlockSize);
    }

    std::vector<double> globalInput;
    std::vector<double> original;
    if (grid.rank == Root) {
        globalInput.resize(n * n);
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(globalInput, n);
        if (validate) {
            original = globalInput;
        }
    }

    DistributedMatrix matrix(n, BlockSize, grid);
    distributeInitialMatrix(globalInput, matrix);
    globalInput.clear();
    globalInput.shrink_to_fit();
    MPI_Barrier(grid.comm);

    if (grid.rank == Root) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(matrix);
    const double localSeconds = MPI_Wtime() - start;
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX,
               Root, grid.comm);

    if (!success) {
        if (grid.rank == Root) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&grid.comm);
        MPI_Finalize();
        return 1;
    }

    const long long durationMilliseconds = std::max<long long>(
        1, static_cast<long long>(std::llround(computationSeconds * 1000.0)));
    if (grid.rank == Root) {
        std::printf("Computation time: %lld ms\n", durationMilliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops =
            operations / std::max(computationSeconds, 1e-12) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    const bool needFullResult = validate || printResults;
    std::vector<double> result;
    if (needFullResult && grid.rank == Root) {
        result.resize(n * n, 0.0);
    }
    if (needFullResult) {
        gatherLowerMatrix(matrix, result);
    }

    bool valid = true;
    if (grid.rank == Root && printResults) {
        print_results(result, "CholeskyL");
    }
    if (grid.rank == Root && validate) {
        std::printf("Validating result...\n");
        valid = validateCholesky(result, original, n);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Comm_free(&grid.comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
