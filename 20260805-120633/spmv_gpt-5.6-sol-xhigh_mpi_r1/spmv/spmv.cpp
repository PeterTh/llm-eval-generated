#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int VALUES_TAG = 100;
constexpr int COLUMNS_TAG = 101;
constexpr int ROWS_TAG = 102;
constexpr int OUTPUT_TAG = 103;

struct Config {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;
    int parseResult = 0;  // 0: run, 1: help, 2: invalid arguments
};

void fill(double* values, const index_t count, const double maxVal) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Preserve the original deterministic matrix construction, while doing all
// size arithmetic wide enough to avoid overflow for dimensions above 65535.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters,
                      const index_t nonzeros, const index_t dim) {
    index_t nnzAssigned = 0;
    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(dim) * static_cast<std::uint64_t>(dim);
    const double probability = static_cast<double>(nonzeros) /
                               static_cast<double>(matrixEntries);

    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = nnzAssigned;
        for (index_t col = 0; col < dim; ++col) {
            const std::uint64_t position =
                static_cast<std::uint64_t>(row) * dim + col;
            const std::uint64_t entriesLeft = matrixEntries - position;
            const std::uint64_t needed = nonzeros - nnzAssigned;
            if (entriesLeft <= needed) {
                fillRemaining = true;
            }

            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < nonzeros && randomValue <= probability) ||
                fillRemaining) {
                cols[nnzAssigned++] = col;
            }
        }
    }
    rowDelimiters[dim] = nonzeros;
}

// Each rank invokes this kernel only for the CSR rows assigned to that rank.
// Contiguous rows retain the exact accumulation order of the serial program.
void spmvCpu(const double* __restrict__ values,
             const index_t* __restrict__ cols,
             const index_t* __restrict__ rowDelimiters,
             const double* __restrict__ vec, const index_t rowCount,
             double* __restrict__ out) {
    for (index_t row = 0; row < rowCount; ++row) {
        double sum = 0.0;
        const index_t end = rowDelimiters[row + 1];
        for (index_t entry = rowDelimiters[row]; entry < end; ++entry) {
            sum += values[entry] * vec[cols[entry]];
        }
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result,
                   const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                            i, ref, res);
                return false;
            }
        } else {
            const double relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, ref, res, relativeError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseIndex(const char* text, index_t* value) {
    if (text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    *value = static_cast<index_t>(parsed);
    return true;
}

Config parseArguments(const int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], &config.numRows)) {
                config.parseResult = 2;
                break;
            }
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], &config.sparsity)) {
                config.parseResult = 2;
                break;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], &config.iterations)) {
                config.parseResult = 2;
                break;
            }
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            config.maxVal = std::strtod(argv[++i], &end);
            if (errno != 0 || end == argv[i] || *end != '\0' ||
                !std::isfinite(config.maxVal)) {
                config.parseResult = 2;
                break;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            config.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            config.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            config.parseResult = 1;
            break;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            config.parseResult = 2;
            break;
        }
    }

    if (config.parseResult == 0 &&
        (config.numRows == 0 || config.sparsity == 0 || config.iterations == 0)) {
        std::fprintf(stderr, "Matrix size, sparsity, and iterations must be positive.\n");
        config.parseResult = 2;
    }
    if (config.parseResult == 0) {
        const std::uint64_t matrixEntries =
            static_cast<std::uint64_t>(config.numRows) * config.numRows;
        if (matrixEntries / config.sparsity >
            std::numeric_limits<index_t>::max()) {
            std::fprintf(stderr,
                         "The requested matrix has too many nonzeros for the CSR index type.\n");
            config.parseResult = 2;
        }
    }
    return config;
}

// Choose contiguous row boundaries closest to equal cumulative-nonzero
// targets. This keeps the numerical ordering stable while balancing work even
// when the random matrix has uneven row densities.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowDelimiters,
                                   const index_t numRows,
                                   const index_t nonzeros,
                                   const int processCount) {
    std::vector<index_t> starts(static_cast<std::size_t>(processCount) + 1);
    starts.front() = 0;
    starts.back() = numRows;

    if (nonzeros == 0) {
        for (int rank = 1; rank < processCount; ++rank) {
            const auto rankIndex = static_cast<std::size_t>(rank);
            starts[rankIndex] = static_cast<index_t>(
                static_cast<std::uint64_t>(numRows) *
                static_cast<std::uint64_t>(rank) /
                static_cast<std::uint64_t>(processCount));
        }
        return starts;
    }

    for (int rank = 1; rank < processCount; ++rank) {
        const auto rankIndex = static_cast<std::size_t>(rank);
        const std::uint64_t target =
            static_cast<std::uint64_t>(nonzeros) *
            static_cast<std::uint64_t>(rank) /
            static_cast<std::uint64_t>(processCount);
        const index_t previous = starts[rankIndex - 1];
        const auto first = rowDelimiters.begin() + previous;
        const auto found = std::lower_bound(first, rowDelimiters.end(), target);
        index_t upper = static_cast<index_t>(found - rowDelimiters.begin());
        if (upper > numRows) {
            upper = numRows;
        }
        const index_t lower = upper > previous ? upper - 1 : upper;
        const std::uint64_t lowerDistance =
            target >= rowDelimiters[lower] ? target - rowDelimiters[lower]
                                           : rowDelimiters[lower] - target;
        const std::uint64_t upperDistance =
            target >= rowDelimiters[upper] ? target - rowDelimiters[upper]
                                           : rowDelimiters[upper] - target;
        starts[rankIndex] = lowerDistance <= upperDistance ? lower : upper;
    }
    return starts;
}

template <typename T>
void broadcastLarge(T* data, const std::uint64_t count,
                    const MPI_Datatype datatype, MPI_Comm communicator) {
    std::uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(
            std::min<std::uint64_t>(count - offset, INT_MAX));
        MPI_Bcast(data + offset, chunk, datatype, 0, communicator);
        offset += static_cast<std::uint64_t>(chunk);
    }
}

template <typename T>
void scatterContiguous(const std::vector<T>& global,
                       std::vector<T>& local,
                       const std::vector<index_t>& starts,
                       const MPI_Datatype datatype, const int tag,
                       const int rank, const int processCount,
                       MPI_Comm communicator) {
    const auto rankIndex = static_cast<std::size_t>(rank);
    const std::uint64_t localCount =
        static_cast<std::uint64_t>(starts[rankIndex + 1]) - starts[rankIndex];
    const std::uint64_t totalCount = starts.back();

    if (totalCount <= static_cast<std::uint64_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<std::size_t>(processCount));
            displacements.resize(static_cast<std::size_t>(processCount));
            for (int peer = 0; peer < processCount; ++peer) {
                const auto peerIndex = static_cast<std::size_t>(peer);
                counts[peerIndex] = static_cast<int>(
                    starts[peerIndex + 1] - starts[peerIndex]);
                displacements[peerIndex] =
                    static_cast<int>(starts[peerIndex]);
            }
        }
        MPI_Scatterv(rank == 0 ? global.data() : nullptr,
                     rank == 0 ? counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr, datatype,
                     local.data(), static_cast<int>(localCount), datatype, 0,
                     communicator);
        return;
    }

    // MPI-3 collectives use int counts/displacements. Retain support for the
    // full uint32_t CSR range by chunking only when those limits are exceeded.
    if (rank == 0) {
        std::copy_n(global.data() + starts[0],
                    static_cast<std::size_t>(localCount), local.data());
        for (int peer = 1; peer < processCount; ++peer) {
            const auto peerIndex = static_cast<std::size_t>(peer);
            std::uint64_t offset = starts[peerIndex];
            std::uint64_t remaining =
                static_cast<std::uint64_t>(starts[peerIndex + 1]) -
                starts[peerIndex];
            while (remaining != 0) {
                const int chunk = static_cast<int>(
                    std::min<std::uint64_t>(remaining, INT_MAX));
                MPI_Send(global.data() + offset, chunk, datatype, peer, tag,
                         communicator);
                offset += static_cast<std::uint64_t>(chunk);
                remaining -= static_cast<std::uint64_t>(chunk);
            }
        }
    } else {
        std::uint64_t offset = 0;
        while (offset < localCount) {
            const int chunk = static_cast<int>(
                std::min<std::uint64_t>(localCount - offset, INT_MAX));
            MPI_Recv(local.data() + offset, chunk, datatype, 0, tag,
                     communicator, MPI_STATUS_IGNORE);
            offset += static_cast<std::uint64_t>(chunk);
        }
    }
}

template <typename T>
void gatherContiguous(const std::vector<T>& local,
                      std::vector<T>& global,
                      const std::vector<index_t>& starts,
                      const MPI_Datatype datatype, const int tag,
                      const int rank, const int processCount,
                      MPI_Comm communicator) {
    const auto rankIndex = static_cast<std::size_t>(rank);
    const std::uint64_t localCount =
        static_cast<std::uint64_t>(starts[rankIndex + 1]) - starts[rankIndex];
    const std::uint64_t totalCount = starts.back();

    if (totalCount <= static_cast<std::uint64_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<std::size_t>(processCount));
            displacements.resize(static_cast<std::size_t>(processCount));
            for (int peer = 0; peer < processCount; ++peer) {
                const auto peerIndex = static_cast<std::size_t>(peer);
                counts[peerIndex] = static_cast<int>(
                    starts[peerIndex + 1] - starts[peerIndex]);
                displacements[peerIndex] =
                    static_cast<int>(starts[peerIndex]);
            }
        }
        MPI_Gatherv(local.data(), static_cast<int>(localCount), datatype,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, datatype, 0,
                    communicator);
        return;
    }

    if (rank == 0) {
        std::copy_n(local.data(), static_cast<std::size_t>(localCount),
                    global.data() + starts[0]);
        for (int peer = 1; peer < processCount; ++peer) {
            const auto peerIndex = static_cast<std::size_t>(peer);
            std::uint64_t offset = starts[peerIndex];
            const std::uint64_t end = starts[peerIndex + 1];
            while (offset < end) {
                const int chunk = static_cast<int>(
                    std::min<std::uint64_t>(end - offset, INT_MAX));
                MPI_Recv(global.data() + offset, chunk, datatype, peer, tag,
                         communicator, MPI_STATUS_IGNORE);
                offset += static_cast<std::uint64_t>(chunk);
            }
        }
    } else {
        std::uint64_t offset = 0;
        while (offset < localCount) {
            const int chunk = static_cast<int>(
                std::min<std::uint64_t>(localCount - offset, INT_MAX));
            MPI_Send(local.data() + offset, chunk, datatype, 0, tag,
                     communicator);
            offset += static_cast<std::uint64_t>(chunk);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm communicator = MPI_COMM_WORLD;
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processCount);

    Config config;
    if (rank == 0) {
        config = parseArguments(argc, argv);
    }
    MPI_Bcast(&config, static_cast<int>(sizeof(config)), MPI_BYTE, 0,
              communicator);

    if (config.parseResult != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return config.parseResult == 1 ? 0 : 1;
    }

    const index_t numRows = config.numRows;
    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(numRows) * numRows;
    const index_t nonzeros =
        static_cast<index_t>(matrixEntries / config.sparsity);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n",
                    config.sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonzeros,
                    100.0 * (1.0 - static_cast<double>(nonzeros) /
                                       static_cast<double>(matrixEntries)));
        std::printf("Iterations: %u\n", config.iterations);
        std::printf("Max value: %.2f\n", config.maxVal);
        std::printf("MPI processes: %d\n", processCount);
        std::printf("Validation: %s\n",
                    config.validate ? "enabled" : "disabled");
        std::printf("Initializing data structures...\n");
    }

    std::vector<double> vec(numRows);
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> reference;

    if (rank == 0) {
        globalValues.resize(nonzeros);
        globalColumns.resize(nonzeros);
        globalRowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);

        // C specifies the initial rand() state as if seeded with one. Make it
        // explicit so MPI/runtime initialization cannot affect reproducibility.
        srand(1);
        fill(vec.data(), numRows, config.maxVal);
        fill(globalValues.data(), nonzeros, config.maxVal);
        initRandomMatrix(globalColumns.data(), globalRowDelimiters.data(),
                         nonzeros, numRows);

        if (config.validate) {
            std::printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(globalValues.data(), globalColumns.data(),
                    globalRowDelimiters.data(), vec.data(), numRows,
                    reference.data());
        }
    }

    broadcastLarge(vec.data(), numRows, MPI_DOUBLE, communicator);

    std::vector<index_t> rowStarts(static_cast<std::size_t>(processCount) + 1);
    std::vector<index_t> nnzStarts(static_cast<std::size_t>(processCount) + 1);
    if (rank == 0) {
        rowStarts = partitionRows(globalRowDelimiters, numRows, nonzeros,
                                  processCount);
        for (int peer = 0; peer <= processCount; ++peer) {
            const auto peerIndex = static_cast<std::size_t>(peer);
            nnzStarts[peerIndex] =
                globalRowDelimiters[rowStarts[peerIndex]];
        }
    }
    MPI_Bcast(rowStarts.data(), processCount + 1, MPI_UINT32_T, 0,
              communicator);
    MPI_Bcast(nnzStarts.data(), processCount + 1, MPI_UINT32_T, 0,
              communicator);

    const auto rankIndex = static_cast<std::size_t>(rank);
    const index_t localRows =
        rowStarts[rankIndex + 1] - rowStarts[rankIndex];
    const index_t localNonzeros =
        nnzStarts[rankIndex + 1] - nnzStarts[rankIndex];
    std::vector<double> localValues(localNonzeros);
    std::vector<index_t> localColumns(localNonzeros);
    std::vector<index_t> localRowDelimiters(
        static_cast<std::size_t>(localRows) + 1);
    std::vector<double> localOutput(localRows);

    scatterContiguous(globalValues, localValues, nnzStarts, MPI_DOUBLE,
                      VALUES_TAG, rank, processCount, communicator);
    scatterContiguous(globalColumns, localColumns, nnzStarts, MPI_UINT32_T,
                      COLUMNS_TAG, rank, processCount, communicator);
    scatterContiguous(globalRowDelimiters, localRowDelimiters, rowStarts,
                      MPI_UINT32_T, ROWS_TAG, rank, processCount, communicator);

    const index_t globalNnzOffset = nnzStarts[rankIndex];
    for (index_t row = 0; row < localRows; ++row) {
        localRowDelimiters[row] -= globalNnzOffset;
    }
    localRowDelimiters[localRows] = localNonzeros;

    // The global CSR is no longer needed during the benchmark. Releasing it
    // avoids retaining an unnecessary full-matrix copy on rank zero.
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalRowDelimiters);
        std::printf("Computing SpMV...\n");
    }

    MPI_Barrier(communicator);
    const double start = MPI_Wtime();
    for (index_t iteration = 0; iteration < config.iterations; ++iteration) {
        spmvCpu(localValues.data(), localColumns.data(),
                localRowDelimiters.data(), vec.data(), localRows,
                localOutput.data());
    }
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               communicator);

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double averageMilliseconds =
            milliseconds / static_cast<double>(config.iterations);
        const double gflops = elapsed > 0.0
                                  ? (2.0 * static_cast<double>(nonzeros) *
                                     config.iterations) /
                                        elapsed / 1e9
                                  : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n",
                    averageMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOutput;
    if (config.printResults || config.validate) {
        if (rank == 0) {
            globalOutput.resize(numRows);
        }
        gatherContiguous(localOutput, globalOutput, rowStarts, MPI_DOUBLE,
                         OUTPUT_TAG, rank, processCount, communicator);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (config.printResults) {
            print_results(globalOutput, "OutputVector");
        }
        if (config.validate) {
            std::printf("Validating result...\n");
            if (verifyResults(reference.data(), globalOutput.data(), numRows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, communicator);
    MPI_Finalize();
    return exitCode;
}
