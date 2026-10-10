#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const uint64_t numEntries = static_cast<uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = numEntries - (static_cast<uint64_t>(i) * dim + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
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

// Balance the number of nonzeros plus the per-row output work. Whole rows
// preserve the original accumulation order and avoid partial-sum reductions.
void partitionRows(const std::vector<index_t>& rowDelimiters, index_t numRows,
                   int ranks, std::vector<index_t>& rowCounts,
                   std::vector<index_t>& rowOffsets, std::vector<index_t>& itemCounts,
                   std::vector<index_t>& itemOffsets, std::vector<index_t>& metadata) {
    rowCounts.resize(ranks);
    rowOffsets.resize(ranks);
    itemCounts.resize(ranks);
    itemOffsets.resize(ranks);
    metadata.resize(static_cast<size_t>(ranks) * 4);

    const uint64_t totalWork = static_cast<uint64_t>(rowDelimiters[numRows]) + numRows;
    index_t begin = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        index_t end = numRows;
        if (rank + 1 < ranks) {
            const uint64_t target = totalWork * static_cast<uint64_t>(rank + 1) / ranks;
            index_t low = begin;
            index_t high = numRows;
            while (low < high) {
                const index_t mid = low + (high - low) / 2;
                if (static_cast<uint64_t>(rowDelimiters[mid]) + mid < target) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            end = low;
            if (end > begin) {
                const uint64_t above = static_cast<uint64_t>(rowDelimiters[end]) + end;
                const uint64_t below = static_cast<uint64_t>(rowDelimiters[end - 1]) + end - 1;
                if (above - target > target - below) {
                    --end;
                }
            }
        }
        rowOffsets[rank] = begin;
        rowCounts[rank] = end - begin;
        itemOffsets[rank] = rowDelimiters[begin];
        itemCounts[rank] = rowDelimiters[end] - rowDelimiters[begin];
        const size_t base = static_cast<size_t>(rank) * 4;
        metadata[base] = rowOffsets[rank];
        metadata[base + 1] = rowCounts[rank];
        metadata[base + 2] = itemOffsets[rank];
        metadata[base + 3] = itemCounts[rank];
        begin = end;
    }
}

// Window large transfers so MPI implementations with int counts/displacements
// can handle the full range of the CSR index type without truncation.
constexpr uint64_t MPI_CHUNK = static_cast<uint64_t>(std::numeric_limits<int>::max());

void mpiLayout(const std::vector<index_t>& counts, const std::vector<index_t>& offsets,
               uint64_t windowBegin, uint64_t windowEnd, std::vector<int>& mpiCounts,
               std::vector<int>& mpiOffsets) {
    mpiCounts.resize(counts.size());
    mpiOffsets.resize(counts.size());
    for (size_t rank = 0; rank < counts.size(); ++rank) {
        const uint64_t begin = std::max(windowBegin, static_cast<uint64_t>(offsets[rank]));
        const uint64_t end = std::min(windowEnd, static_cast<uint64_t>(offsets[rank]) + counts[rank]);
        mpiCounts[rank] = end > begin ? static_cast<int>(end - begin) : 0;
        mpiOffsets[rank] = end > begin ? static_cast<int>(begin - windowBegin) : 0;
    }
}

template <typename T>
void scatterArray(std::vector<T>& data, index_t totalCount,
                  const std::vector<index_t>& counts, const std::vector<index_t>& offsets,
                  index_t localCount, index_t localOffset, MPI_Datatype datatype, int rank) {
    if (rank != 0) {
        data.resize(localCount);
    }
    std::vector<int> mpiCounts, mpiOffsets;
    for (uint64_t window = 0; window < totalCount; window += MPI_CHUNK) {
        const uint64_t windowEnd = std::min(window + MPI_CHUNK, static_cast<uint64_t>(totalCount));
        if (rank == 0) {
            mpiLayout(counts, offsets, window, windowEnd, mpiCounts, mpiOffsets);
        }
        const uint64_t begin = std::max(window, static_cast<uint64_t>(localOffset));
        const uint64_t end = std::min(windowEnd, static_cast<uint64_t>(localOffset) + localCount);
        const int count = end > begin ? static_cast<int>(end - begin) : 0;
        T* receive = count > 0 ? data.data() + (begin - localOffset) : nullptr;
        MPI_Scatterv(rank == 0 ? data.data() + window : nullptr,
                     mpiCounts.data(), mpiOffsets.data(), datatype,
                     rank == 0 ? MPI_IN_PLACE : receive, count, datatype, 0, MPI_COMM_WORLD);
    }
    // Rank zero owns the first slice, which is already in place.
    data.resize(localCount);
}

void gatherOutput(const std::vector<double>& local, std::vector<double>& global,
                  index_t numRows, const std::vector<index_t>& counts,
                  const std::vector<index_t>& offsets, index_t localOffset, int rank) {
    if (rank == 0) {
        global.resize(numRows);
    }
    std::vector<int> mpiCounts, mpiOffsets;
    for (uint64_t window = 0; window < numRows; window += MPI_CHUNK) {
        const uint64_t windowEnd = std::min(window + MPI_CHUNK, static_cast<uint64_t>(numRows));
        if (rank == 0) {
            mpiLayout(counts, offsets, window, windowEnd, mpiCounts, mpiOffsets);
        }
        const uint64_t begin = std::max(window, static_cast<uint64_t>(localOffset));
        const uint64_t end = std::min(windowEnd, static_cast<uint64_t>(localOffset) + local.size());
        const int count = end > begin ? static_cast<int>(end - begin) : 0;
        MPI_Gatherv(count > 0 ? local.data() + (begin - localOffset) : nullptr,
                    count, MPI_DOUBLE, rank == 0 ? global.data() + window : nullptr,
                    mpiCounts.data(), mpiOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

bool parseIndex(const char* text, index_t& value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        parsed > std::numeric_limits<index_t>::max()) {
        printf("Invalid integer argument: %s\n", text);
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse once and share the configuration, including help/error exits.
    int parseStatus = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                parseStatus = parseIndex(argv[++i], numRows) ? 0 : 1;
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                parseStatus = parseIndex(argv[++i], sparsity) ? 0 : 1;
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                parseStatus = parseIndex(argv[++i], iterations) ? 0 : 1;
            } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                maxVal = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
            }
            if (parseStatus != 0) {
                break;
            }
        }
        if (parseStatus == 0 && sparsity == 0) {
            printf("Sparsity must be greater than zero.\n");
            parseStatus = 1;
        }
        if (parseStatus == 0 && static_cast<uint64_t>(numRows) * numRows / sparsity >
            std::numeric_limits<index_t>::max()) {
            printf("Number of non-zero elements exceeds the CSR index range.\n");
            parseStatus = 1;
        }
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        return parseStatus == 2 ? 0 : 1;
    }
    index_t options[3] = {numRows, sparsity, iterations};
    int flags[2] = {validate, printResults};
    MPI_Bcast(options, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    numRows = options[0];
    sparsity = options[1];
    iterations = options[2];
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    // Calculate number of non-zero elements
    const uint64_t numEntries = static_cast<uint64_t>(numRows) * numRows;
    const index_t nItems = static_cast<index_t>(numEntries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, numEntries == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / numEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", ranks);
    }

    // Only rank zero generates the original random sequence and global CSR.
    // After distribution each rank retains only its own matrix rows.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        // Restore rand's original default seed even if MPI_Init used rand.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution before distributing CSR.
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    std::vector<index_t> rowCounts, rowOffsets, itemCounts, itemOffsets, metadata;
    if (rank == 0) {
        partitionRows(h_rowDelimiters, numRows, ranks, rowCounts, rowOffsets,
                      itemCounts, itemOffsets, metadata);
    }
    index_t localInfo[4] = {};
    MPI_Scatter(metadata.data(), 4, MPI_UINT32_T, localInfo, 4, MPI_UINT32_T,
                0, MPI_COMM_WORLD);
    const index_t firstRow = localInfo[0];
    const index_t localRows = localInfo[1];
    const index_t firstItem = localInfo[2];
    const index_t localItems = localInfo[3];

    for (uint64_t begin = 0; begin < numRows; begin += MPI_CHUNK) {
        const int count = static_cast<int>(std::min(MPI_CHUNK, static_cast<uint64_t>(numRows) - begin));
        MPI_Bcast(h_vec.data() + begin, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    scatterArray(h_val, nItems, itemCounts, itemOffsets, localItems, firstItem, MPI_DOUBLE, rank);
    scatterArray(h_cols, nItems, itemCounts, itemOffsets, localItems, firstItem, MPI_UINT32_T, rank);
    // Send row starts only, so adjacent ranks do not overlap the send buffer.
    scatterArray(h_rowDelimiters, numRows, rowCounts, rowOffsets, localRows,
                 firstRow, MPI_UINT32_T, rank);
    h_rowDelimiters.resize(static_cast<size_t>(localRows) + 1);
    for (index_t row = 0; row < localRows; ++row) {
        h_rowDelimiters[row] -= firstItem;
    }
    h_rowDelimiters[localRows] = localItems;
    if (rank == 0 && ranks > 1) {
        h_val.shrink_to_fit();
        h_cols.shrink_to_fit();
        h_rowDelimiters.shrink_to_fit();
    }
    std::vector<index_t>().swap(metadata);
    std::vector<index_t>().swap(itemCounts);
    std::vector<index_t>().swap(itemOffsets);
    std::vector<double> localOut(localRows);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // The input vector is unchanged across iterations: no communication is
    // needed in the timed loop, and every output row has exactly one owner.
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Assemble the result once, and only when it is actually consumed.
    if (printResults || validate) {
        gatherOutput(localOut, h_out, numRows, rowCounts, rowOffsets, firstRow, rank);
    }

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);

        // Use the slowest rank's elapsed time for global performance metrics.
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations > 0 ? duration * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
