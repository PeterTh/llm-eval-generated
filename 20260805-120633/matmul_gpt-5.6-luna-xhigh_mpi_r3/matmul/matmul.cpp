#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.  Keeping this
// function deterministic lets every MPI rank initialize its own matrix tile
// without communicating the input matrices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

struct Partition {
    std::vector<size_t> counts;
    std::vector<size_t> starts;
};

Partition makePartition(const size_t N, const int parts) {
    Partition partition;
    partition.counts.resize(static_cast<size_t>(parts));
    partition.starts.resize(static_cast<size_t>(parts));

    const size_t base = N / static_cast<size_t>(parts);
    const size_t remainder = N % static_cast<size_t>(parts);
    size_t offset = 0;
    for (int part = 0; part < parts; ++part) {
        partition.counts[static_cast<size_t>(part)] =
            base + (static_cast<size_t>(part) < remainder ? 1 : 0);
        partition.starts[static_cast<size_t>(part)] = offset;
        offset += partition.counts[static_cast<size_t>(part)];
    }
    return partition;
}

void initLocalA(double* A, const size_t N,
                const size_t rowStart, const size_t colStart,
                const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < cols; ++k) {
            A[i * cols + k] = getPseudoRndValue(N, rowStart + i, colStart + k);
        }
    }
}

void initLocalB(double* B, const size_t N,
                const size_t rowStart, const size_t colStart,
                const size_t rows, const size_t cols) {
    for (size_t k = 0; k < rows; ++k) {
        for (size_t j = 0; j < cols; ++j) {
            B[k * cols + j] = getPseudoRndValue(N, rowStart + k, colStart + j);
        }
    }
}

// Simple validation: compute a single element and compare.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

int checkedMpiCount(const size_t count, const int rank) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Matrix tile is too large for this MPI implementation.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return static_cast<int>(count);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Reassemble the distributed C tiles in row-major order.  This collective is
// only used for the optional result/validation paths, so the normal benchmark
// does not centralize the O(N^2) output.
std::vector<double> gatherResult(const std::vector<double>& localC,
                                 const Partition& rowPartition,
                                 const Partition& colPartition,
                                 const int gridRows, const int gridCols,
                                 const MPI_Comm cartComm, const int cartRank,
                                 const size_t N) {
    const size_t localElements = localC.size();
    const int localCount = checkedMpiCount(localElements, cartRank);

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> packed;
    if (cartRank == 0) {
        const int worldSize = gridRows * gridCols;
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));

        size_t packedElements = 0;
        for (int rank = 0; rank < worldSize; ++rank) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(cartComm, rank, 2, coords);
            const size_t elements =
                rowPartition.counts[static_cast<size_t>(coords[0])] *
                colPartition.counts[static_cast<size_t>(coords[1])];
            counts[static_cast<size_t>(rank)] = checkedMpiCount(elements, cartRank);
            displacements[static_cast<size_t>(rank)] = checkedMpiCount(packedElements, cartRank);
            packedElements += elements;
        }
        packed.resize(packedElements);
    }

    MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE,
                cartRank == 0 ? packed.data() : nullptr,
                cartRank == 0 ? counts.data() : nullptr,
                cartRank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, cartComm);

    std::vector<double> result;
    if (cartRank == 0) {
        result.resize(N * N);
        const int worldSize = gridRows * gridCols;
        for (int rank = 0; rank < worldSize; ++rank) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(cartComm, rank, 2, coords);
            const size_t rows = rowPartition.counts[static_cast<size_t>(coords[0])];
            const size_t cols = colPartition.counts[static_cast<size_t>(coords[1])];
            const size_t rowStart = rowPartition.starts[static_cast<size_t>(coords[0])];
            const size_t colStart = colPartition.starts[static_cast<size_t>(coords[1])];
            const double* source = packed.data() + displacements[static_cast<size_t>(rank)];

            for (size_t i = 0; i < rows; ++i) {
                std::copy_n(source + i * cols, cols,
                            result.begin() + (rowStart + i) * N + colStart);
            }
        }
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool parseFailed = false;
    bool showHelp = false;

    // Parse command line arguments on every rank so all ranks follow the same
    // collective control flow.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseFailed = true;
        }
    }

    if (showHelp || parseFailed || N == 0) {
        if (worldRank == 0 && N == 0 && !showHelp && !parseFailed) {
            printf("Matrix size must be greater than zero.\n");
            printUsage(argv[0]);
        } else if (worldRank == 0 && showHelp) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseFailed || N == 0 ? 1 : 0;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cartComm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartComm);

    int cartRank = 0;
    int coords[2] = {0, 0};
    MPI_Comm_rank(cartComm, &cartRank);
    MPI_Cart_coords(cartComm, cartRank, 2, coords);

    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm colComm = MPI_COMM_NULL;
    MPI_Comm_split(cartComm, coords[0], coords[1], &rowComm);
    MPI_Comm_split(cartComm, coords[1], coords[0], &colComm);

    const Partition rowPartition = makePartition(N, dims[0]);
    const Partition colPartition = makePartition(N, dims[1]);
    // A rectangular process grid needs one common partition of the k
    // dimension.  Use lcm(Pr, Pc) panels and assign panel t to column t % Pc
    // for A and row t % Pr for B.  This retains SUMMA's O(N^2/P) storage and
    // works for any MPI process count, including non-square grids.
    const int kBlocks = std::lcm(dims[0], dims[1]);
    const Partition kPartition = makePartition(N, kBlocks);
    const size_t localRows = rowPartition.counts[static_cast<size_t>(coords[0])];
    const size_t localCols = colPartition.counts[static_cast<size_t>(coords[1])];
    const size_t rowStart = rowPartition.starts[static_cast<size_t>(coords[0])];
    const size_t colStart = colPartition.starts[static_cast<size_t>(coords[1])];

    constexpr size_t noOffset = std::numeric_limits<size_t>::max();
    std::vector<size_t> aOffsets(static_cast<size_t>(kBlocks), noOffset);
    std::vector<size_t> bOffsets(static_cast<size_t>(kBlocks), noOffset);
    size_t localAElements = 0;
    size_t localBElements = 0;
    for (int kBlock = 0; kBlock < kBlocks; ++kBlock) {
        const size_t kCount = kPartition.counts[static_cast<size_t>(kBlock)];
        if (kBlock % dims[1] == coords[1]) {
            aOffsets[static_cast<size_t>(kBlock)] = localAElements;
            localAElements += localRows * kCount;
        }
        if (kBlock % dims[0] == coords[0]) {
            bOffsets[static_cast<size_t>(kBlock)] = localBElements;
            localBElements += kCount * localCols;
        }
    }

    std::vector<double> localA(localAElements);
    std::vector<double> localB(localBElements);
    std::vector<double> localC(localRows * localCols, 0.0);

    for (int kBlock = 0; kBlock < kBlocks; ++kBlock) {
        const size_t kIndex = static_cast<size_t>(kBlock);
        const size_t kStart = kPartition.starts[kIndex];
        const size_t kCount = kPartition.counts[kIndex];
        if (aOffsets[kIndex] != noOffset && localRows != 0 && kCount != 0) {
            initLocalA(localA.data() + aOffsets[kIndex], N,
                       rowStart, kStart, localRows, kCount);
        }
        if (bOffsets[kIndex] != noOffset && localCols != 0 && kCount != 0) {
            initLocalB(localB.data() + bOffsets[kIndex], N,
                       kStart, colStart, kCount, localCols);
        }
    }

    size_t maxK = 0;
    for (const size_t count : kPartition.counts) {
        maxK = std::max(maxK, count);
    }
    std::vector<double> aPanel(localRows * maxK);
    std::vector<double> bPanel(maxK * localCols);

    if (cartRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    MPI_Barrier(cartComm);
    if (cartRank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(cartComm);
    const double start = MPI_Wtime();

    // SUMMA: A panels are broadcast across process rows and B panels across
    // process columns.  The k panels are visited in order, preserving the
    // original per-element accumulation order.
    for (int kBlock = 0; kBlock < kBlocks; ++kBlock) {
        const size_t kIndex = static_cast<size_t>(kBlock);
        const size_t panelK = kPartition.counts[kIndex];

        if (aOffsets[kIndex] != noOffset && localRows != 0 && panelK != 0) {
            std::copy_n(localA.data() + aOffsets[kIndex], localRows * panelK,
                        aPanel.data());
        }
        MPI_Bcast(aPanel.data(), checkedMpiCount(localRows * panelK, cartRank),
                  MPI_DOUBLE, kBlock % dims[1], rowComm);

        if (bOffsets[kIndex] != noOffset && localCols != 0 && panelK != 0) {
            std::copy_n(localB.data() + bOffsets[kIndex], panelK * localCols,
                        bPanel.data());
        }
        MPI_Bcast(bPanel.data(), checkedMpiCount(panelK * localCols, cartRank),
                  MPI_DOUBLE, kBlock % dims[0], colComm);

        for (size_t i = 0; i < localRows; ++i) {
            double* cRow = localC.data() + i * localCols;
            const double* aRow = aPanel.data() + i * panelK;
            for (size_t k = 0; k < panelK; ++k) {
                const double aValue = aRow[k];
                const double* bRow = bPanel.data() + k * localCols;
                for (size_t j = 0; j < localCols; ++j) {
                    cRow[j] += aValue * bRow[j];
                }
            }
        }
    }

    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

    if (cartRank == 0) {
        const long durationMs = std::max(1L, static_cast<long>(maximumElapsed * 1000.0));
        printf("Computation time: %ld ms\n", durationMs);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                               static_cast<double>(N)) /
                              (maximumElapsed > 0.0 ? maximumElapsed : 1e-9) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullC;
    if (printResults || validate) {
        fullC = gatherResult(localC, rowPartition, colPartition,
                             dims[0], dims[1], cartComm, cartRank, N);
    }

    if (printResults && cartRank == 0) {
        print_results(fullC, "MatrixC");
    }

    int validationPassed = 1;
    if (validate && cartRank == 0) {
        printf("Validating result...\n");
        // Validation is intentionally outside the timed region.  Recreating
        // the deterministic inputs here keeps the normal distributed path
        // from allocating full matrices on rank 0.
        std::vector<double> fullA(N * N);
        std::vector<double> fullB(N * N);
        initMatrix(fullA, N);
        initMatrix(fullB, N);
        if (validateResult(fullA, fullB, fullC, N)) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            validationPassed = 0;
        }
    }

    MPI_Bcast(&validationPassed, 1, MPI_INT, 0, cartComm);

    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&cartComm);
    MPI_Finalize();

    return validationPassed ? 0 : 1;
}
