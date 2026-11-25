#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr offset_t TRANSFER_CHUNK = 1 << 20;

enum MessageTag { RowOffsets, Columns, Values, ValidationFailure };

struct RowPartition {
    index_t rows;
    int ranks;

    index_t begin(int rank) const {
        const auto r = static_cast<index_t>(rank);
        const auto p = static_cast<index_t>(ranks);
        return r * (rows / p) + std::min(r, rows % p);
    }

    index_t count(int rank) const {
        return begin(rank + 1) - begin(rank);
    }
};

void fill(double* data, offset_t count, double maxVal) {
    for (offset_t i = 0; i < count; ++i) {
        data[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Bound each MPI message and let rank zero reuse a small buffer for values.
template <typename T>
void sendBlock(const T* data, offset_t count, MPI_Datatype type, int dest, MessageTag tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, TRANSFER_CHUNK));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void receiveBlock(T* data, offset_t count, MPI_Datatype type, MessageTag tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, TRANSFER_CHUNK));
        MPI_Recv(data, chunk, type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// Generate consecutive CSR blocks from the original row-major random stream.
// Rank zero keeps its own block and reuses one staging block for the others.
std::vector<offset_t> distributeStructure(const RowPartition& partition, int rank,
                                          offset_t nItems, std::vector<offset_t>& rowOffsets,
                                          std::unique_ptr<index_t[]>& receivedColumns,
                                          std::vector<index_t>& rootColumns) {
    if (rank != 0) {
        if (partition.count(rank) != 0) {
            receiveBlock(rowOffsets.data(), rowOffsets.size(), MPI_UINT64_T, RowOffsets);
            receivedColumns = std::make_unique_for_overwrite<index_t[]>(rowOffsets.back());
            receiveBlock(receivedColumns.get(), rowOffsets.back(), MPI_UINT32_T, Columns);
        }
        return {};
    }

    const offset_t totalEntries = static_cast<offset_t>(partition.rows) * partition.rows;
    const double probability = totalEntries == 0 ? 0.0 : static_cast<double>(nItems) / totalEntries;
    // Adjust the integer cutoff to match the original floating-point comparison.
    int cutoff = static_cast<int>(probability * RAND_MAX);
    if (cutoff / static_cast<double>(RAND_MAX) > probability) {
        --cutoff;
    } else if (cutoff < RAND_MAX && (cutoff + 1) / static_cast<double>(RAND_MAX) <= probability) {
        ++cutoff;
    }

    srand(8675309);
    offset_t remaining = nItems;
    // Once this many entries have been skipped, every remaining entry is required.
    offset_t skipsLeft = totalEntries - nItems;
    const int activeRanks = std::max(1, std::min(partition.ranks, static_cast<int>(partition.rows)));
    std::vector<offset_t> blockSizes(activeRanks);
    std::vector<offset_t> stagingOffsets;
    std::vector<index_t> stagingColumns;
    for (int dest = 0; dest < activeRanks; ++dest) {
        const index_t rows = partition.count(dest);
        auto& offsets = dest == 0 ? rowOffsets : stagingOffsets;
        auto& columns = dest == 0 ? rootColumns : stagingColumns;
        offsets.resize(static_cast<size_t>(rows) + 1);
        columns.clear();
        columns.reserve(nItems / activeRanks);
        for (index_t row = 0; row < rows; ++row) {
            offsets[row] = columns.size();
            for (index_t col = 0; col < partition.rows && remaining != 0; ++col) {
                if (skipsLeft == 0 || rand() <= cutoff) {
                    columns.push_back(col);
                    --remaining;
                } else {
                    --skipsLeft;
                }
            }
        }
        offsets[rows] = columns.size();
        blockSizes[dest] = columns.size();
        if (dest != 0) {
            sendBlock(offsets.data(), offsets.size(), MPI_UINT64_T, dest, RowOffsets);
            sendBlock(columns.data(), columns.size(), MPI_UINT32_T, dest, Columns);
        }
    }
    return blockSizes;
}

void spmvCpu(const double* val, const index_t* cols, const offset_t* rowOffsets,
             const double* vec, index_t rows, double* out) {
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (offset_t j = rowOffsets[row]; j < rowOffsets[row + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
}

bool verifyResults(const std::vector<double>& reference, const std::vector<double>& result,
                   index_t firstRow, index_t numRows, int rank) {
    index_t localFailure = numRows;
    double failureValues[2] = {};
    for (index_t i = 0; i < result.size(); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            localFailure = firstRow + i;
            failureValues[0] = ref;
            failureValues[1] = res;
            break;
        }
    }

    index_t firstFailure;
    MPI_Allreduce(&localFailure, &firstFailure, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (firstFailure == numRows) {
        return true;
    }
    if (rank != 0 && firstFailure == localFailure) {
        MPI_Send(failureValues, 2, MPI_DOUBLE, 0, ValidationFailure, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        if (firstFailure != localFailure) {
            MPI_Recv(failureValues, 2, MPI_DOUBLE, MPI_ANY_SOURCE, ValidationFailure,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        const double ref = failureValues[0];
        const double res = failureValues[1];
        if (std::abs(ref) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   firstFailure, ref, res);
        } else {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   firstFailure, ref, res, std::abs((res - ref) / ref));
        }
    }
    return false;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (sparsity == 0 || numRows > INT_MAX) {
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be nonzero and the matrix must have at most %d rows.\n",
                    INT_MAX);
        }
        return 1;
    }

    const offset_t totalEntries = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = totalEntries / sparsity;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    const RowPartition partition{numRows, ranks};
    const index_t localRows = partition.count(rank);
    std::vector<offset_t> rowOffsets(static_cast<size_t>(localRows) + 1);
    std::unique_ptr<index_t[]> receivedColumns;
    std::vector<index_t> rootColumns;
    const auto blockSizes = distributeStructure(partition, rank, nItems, rowOffsets,
                                                receivedColumns, rootColumns);
    const auto* columns = rank == 0 ? rootColumns.data() : receivedColumns.get();

    std::vector<double> vec(numRows);
    if (rank == 0) {
        // Reproduce the original default-seeded stream: vector, then CSR values.
        srand(1);
        fill(vec.data(), numRows, maxVal);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const offset_t localItems = rowOffsets.back();
    auto values = std::make_unique_for_overwrite<double[]>(localItems);
    if (rank == 0) {
        fill(values.get(), localItems, maxVal);
        if (blockSizes.size() > 1) {
            const auto maxBlockSize = *std::max_element(blockSizes.begin() + 1, blockSizes.end());
            auto stagingValues =
                std::make_unique_for_overwrite<double[]>(std::min(maxBlockSize, TRANSFER_CHUNK));
            for (size_t dest = 1; dest < blockSizes.size(); ++dest) {
                offset_t remaining = blockSizes[dest];
                while (remaining != 0) {
                    const offset_t count = std::min(remaining, TRANSFER_CHUNK);
                    fill(stagingValues.get(), count, maxVal);
                    sendBlock(stagingValues.get(), count, MPI_DOUBLE, static_cast<int>(dest), Values);
                    remaining -= count;
                }
            }
        }
    } else {
        receiveBlock(values.get(), localItems, MPI_DOUBLE, Values);
    }

    std::vector<double> out(localRows);
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        reference.resize(localRows);
        spmvCpu(values.get(), columns, rowOffsets.data(), vec.data(), localRows, reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(values.get(), columns, rowOffsets.data(), vec.data(), localRows, out.data());
        // Make each result observable to the compiler even though all inputs are fixed.
        asm volatile("" : : "g"(out.data()) : "memory");
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops = elapsed > 0 ? (2.0 * nItems * iterations) / elapsed / 1e9 : 0.0;
        const double avgTime = iterations > 0 ? elapsed * 1000.0 / iterations : 0.0;
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> result;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(partition.count(r));
                displacements[r] = static_cast<int>(partition.begin(r));
            }
            result.resize(numRows);
        }
        MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE, result.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(result, "OutputVector");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = verifyResults(reference, out, partition.begin(rank), numRows, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return status;
}
