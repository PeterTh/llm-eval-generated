#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const std::uint64_t numEntriesLeft =
                static_cast<std::uint64_t>(dim) * dim -
                (static_cast<std::uint64_t>(i) * dim + j);
            const std::uint64_t needToAssign =
                static_cast<std::uint64_t>(n) - nnzAssigned;
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

// Keep each benchmark iteration observable to the optimizer. Without this
// barrier, identical iterations that overwrite the same output may be folded
// into a single SpMV by whole-program optimization.
inline void keepResult(const double* result) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(result) : "memory");
#else
    (void)result;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// MPI-3 collectives use int element counts. Use the fast collective path for
// normal problem sizes and a chunked point-to-point fallback for larger CSR
// arrays, whose index type can represent more than INT_MAX elements.
void broadcastLarge(void* buffer, std::uint64_t count, MPI_Datatype datatype,
                    const int root, MPI_Comm communicator) {
    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    auto* bytes = static_cast<unsigned char*>(buffer);
    std::uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<std::uint64_t>(
            count - offset, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
        MPI_Bcast(bytes + offset * static_cast<std::uint64_t>(typeSize), chunk,
                  datatype, root, communicator);
        offset += static_cast<std::uint64_t>(chunk);
    }
}

void scatterContiguous(const void* rootData, void* localData,
                       const std::vector<std::uint64_t>& counts,
                       const std::vector<std::uint64_t>& displacements,
                       MPI_Datatype datatype, const int root, const int tag,
                       MPI_Comm communicator) {
    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processCount);

    bool useScatterv = true;
    for (int process = 0; process < processCount; ++process) {
        useScatterv = useScatterv &&
            counts[process] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
            displacements[process] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    }

    if (useScatterv) {
        std::vector<int> intCounts(processCount);
        std::vector<int> intDisplacements(processCount);
        for (int process = 0; process < processCount; ++process) {
            intCounts[process] = static_cast<int>(counts[process]);
            intDisplacements[process] = static_cast<int>(displacements[process]);
        }
        MPI_Scatterv(rootData, intCounts.data(), intDisplacements.data(), datatype,
                     localData, intCounts[rank], datatype, root, communicator);
        return;
    }

    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    const auto* source = static_cast<const unsigned char*>(rootData);
    auto* destination = static_cast<unsigned char*>(localData);
    constexpr std::uint64_t maxChunk = static_cast<std::uint64_t>(std::numeric_limits<int>::max());

    if (rank == root) {
        const std::uint64_t rootBytes = counts[root] * static_cast<std::uint64_t>(typeSize);
        if (rootBytes != 0) {
            std::memcpy(destination,
                        source + displacements[root] * static_cast<std::uint64_t>(typeSize),
                        static_cast<std::size_t>(rootBytes));
        }
        for (int process = 0; process < processCount; ++process) {
            if (process == root) {
                continue;
            }
            std::uint64_t sent = 0;
            while (sent < counts[process]) {
                const int chunk = static_cast<int>(std::min(maxChunk, counts[process] - sent));
                MPI_Send(source + (displacements[process] + sent) *
                                      static_cast<std::uint64_t>(typeSize),
                         chunk, datatype, process, tag, communicator);
                sent += static_cast<std::uint64_t>(chunk);
            }
        }
    } else {
        std::uint64_t received = 0;
        while (received < counts[rank]) {
            const int chunk = static_cast<int>(std::min(maxChunk, counts[rank] - received));
            MPI_Recv(destination + received * static_cast<std::uint64_t>(typeSize),
                     chunk, datatype, root, tag, communicator, MPI_STATUS_IGNORE);
            received += static_cast<std::uint64_t>(chunk);
        }
    }
}

void gatherContiguous(const void* localData, void* rootData,
                      const std::vector<std::uint64_t>& counts,
                      const std::vector<std::uint64_t>& displacements,
                      MPI_Datatype datatype, const int root, const int tag,
                      MPI_Comm communicator) {
    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processCount);

    bool useGatherv = true;
    for (int process = 0; process < processCount; ++process) {
        useGatherv = useGatherv &&
            counts[process] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
            displacements[process] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    }

    if (useGatherv) {
        std::vector<int> intCounts(processCount);
        std::vector<int> intDisplacements(processCount);
        for (int process = 0; process < processCount; ++process) {
            intCounts[process] = static_cast<int>(counts[process]);
            intDisplacements[process] = static_cast<int>(displacements[process]);
        }
        MPI_Gatherv(localData, intCounts[rank], datatype, rootData,
                    intCounts.data(), intDisplacements.data(), datatype, root, communicator);
        return;
    }

    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    const auto* source = static_cast<const unsigned char*>(localData);
    auto* destination = static_cast<unsigned char*>(rootData);
    constexpr std::uint64_t maxChunk = static_cast<std::uint64_t>(std::numeric_limits<int>::max());

    if (rank == root) {
        const std::uint64_t rootBytes = counts[root] * static_cast<std::uint64_t>(typeSize);
        if (rootBytes != 0) {
            std::memcpy(destination + displacements[root] * static_cast<std::uint64_t>(typeSize),
                        source, static_cast<std::size_t>(rootBytes));
        }
        for (int process = 0; process < processCount; ++process) {
            if (process == root) {
                continue;
            }
            std::uint64_t received = 0;
            while (received < counts[process]) {
                const int chunk = static_cast<int>(std::min(maxChunk, counts[process] - received));
                MPI_Recv(destination + (displacements[process] + received) *
                                           static_cast<std::uint64_t>(typeSize),
                         chunk, datatype, process, tag, communicator, MPI_STATUS_IGNORE);
                received += static_cast<std::uint64_t>(chunk);
            }
        }
    } else {
        std::uint64_t sent = 0;
        while (sent < counts[rank]) {
            const int chunk = static_cast<int>(std::min(maxChunk, counts[rank] - sent));
            MPI_Send(source + sent * static_cast<std::uint64_t>(typeSize),
                     chunk, datatype, root, tag, communicator);
            sent += static_cast<std::uint64_t>(chunk);
        }
    }
}

// Partition at CSR row boundaries using nonzeros plus one unit of per-row
// work. This balances irregular matrices while also handling empty rows well.
void makePartitions(const std::vector<index_t>& rowDelimiters, const index_t numRows,
                    const index_t nItems, const int processCount,
                    std::vector<index_t>& rowStarts,
                    std::vector<index_t>& nonzeroStarts) {
    rowStarts.assign(static_cast<std::size_t>(processCount) + 1, 0);
    nonzeroStarts.assign(static_cast<std::size_t>(processCount) + 1, 0);
    const std::uint64_t totalWork = static_cast<std::uint64_t>(nItems) + numRows;

    for (int process = 1; process < processCount; ++process) {
        const std::uint64_t processIndex = static_cast<std::uint64_t>(process);
        const std::uint64_t target =
            (totalWork / static_cast<std::uint64_t>(processCount)) * processIndex +
            ((totalWork % static_cast<std::uint64_t>(processCount)) * processIndex) /
                static_cast<std::uint64_t>(processCount);

        index_t low = rowStarts[process - 1];
        index_t high = numRows;
        while (low < high) {
            const index_t middle = low + (high - low) / 2;
            const std::uint64_t work = static_cast<std::uint64_t>(rowDelimiters[middle]) + middle;
            if (work < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        index_t boundary = low;
        if (low > rowStarts[process - 1]) {
            const index_t previous = low - 1;
            const std::uint64_t highWork = static_cast<std::uint64_t>(rowDelimiters[low]) + low;
            const std::uint64_t lowWork =
                static_cast<std::uint64_t>(rowDelimiters[previous]) + previous;
            if (target - lowWork <= highWork - target) {
                boundary = previous;
            }
        }
        rowStarts[process] = boundary;
        nonzeroStarts[process] = rowDelimiters[boundary];
    }

    rowStarts[processCount] = numRows;
    nonzeroStarts[processCount] = nItems;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    // Rank zero owns parsing and diagnostics; configuration is then broadcast
    // so every process is guaranteed to execute with identical parameters.
    if (rank == 0) {
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
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = -1;
                break;
            }
        }
        if (parseStatus == 0 && (numRows == 0 || sparsity == 0 || iterations == 0)) {
            fprintf(stderr, "Matrix size, sparsity, and iteration count must all be positive.\n");
            parseStatus = -1;
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus < 0 ? 1 : 0;
    }

    index_t integerConfiguration[3] = {numRows, sparsity, iterations};
    int booleanConfiguration[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(integerConfiguration, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(booleanConfiguration, 2, MPI_INT, 0, MPI_COMM_WORLD);
    numRows = integerConfiguration[0];
    sparsity = integerConfiguration[1];
    iterations = integerConfiguration[2];
    validate = booleanConfiguration[0] != 0;
    printResults = booleanConfiguration[1] != 0;

    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(numRows) * static_cast<std::uint64_t>(numRows);
    const std::uint64_t itemCount64 = matrixEntries / sparsity;
    if (itemCount64 > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            fprintf(stderr, "The requested matrix has too many nonzeros for the 32-bit CSR format.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(itemCount64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                         static_cast<double>(matrixEntries)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", processCount);
    }

    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> denseVector(numRows);
    std::vector<double> h_reference;

    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);

        printf("Initializing data structures...\n");
        // C specifies the initial rand() state as if srand(1) were called.
        // Set it explicitly because MPI implementations may use libc before
        // returning from MPI_Init.
        srand(1);
        fill(denseVector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(globalValues.data(), globalColumns.data(), globalRowDelimiters.data(),
                    denseVector.data(), numRows, h_reference.data());
        }
    }

    broadcastLarge(denseVector.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<index_t> rowStarts(static_cast<std::size_t>(processCount) + 1);
    std::vector<index_t> nonzeroStarts(static_cast<std::size_t>(processCount) + 1);
    if (rank == 0) {
        makePartitions(globalRowDelimiters, numRows, nItems, processCount,
                       rowStarts, nonzeroStarts);
    }
    broadcastLarge(rowStarts.data(), rowStarts.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    broadcastLarge(nonzeroStarts.data(), nonzeroStarts.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<std::uint64_t> rowCounts(processCount);
    std::vector<std::uint64_t> rowDisplacements(processCount);
    std::vector<std::uint64_t> delimiterCounts(processCount);
    std::vector<std::uint64_t> nonzeroCounts(processCount);
    std::vector<std::uint64_t> nonzeroDisplacements(processCount);
    for (int process = 0; process < processCount; ++process) {
        rowCounts[process] = static_cast<std::uint64_t>(rowStarts[process + 1]) -
                             rowStarts[process];
        rowDisplacements[process] = rowStarts[process];
        delimiterCounts[process] = rowCounts[process] + 1;
        nonzeroCounts[process] = static_cast<std::uint64_t>(nonzeroStarts[process + 1]) -
                                 nonzeroStarts[process];
        nonzeroDisplacements[process] = nonzeroStarts[process];
    }

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t localNonzeros = static_cast<index_t>(nonzeroCounts[rank]);
    std::vector<double> localValues(localNonzeros);
    std::vector<index_t> localColumns(localNonzeros);
    std::vector<index_t> localRowDelimiters(static_cast<std::size_t>(localRows) + 1);
    std::vector<double> localOutput(localRows);

    scatterContiguous(globalValues.data(), localValues.data(), nonzeroCounts,
                      nonzeroDisplacements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    scatterContiguous(globalColumns.data(), localColumns.data(), nonzeroCounts,
                      nonzeroDisplacements, MPI_UINT32_T, 0, 101, MPI_COMM_WORLD);
    scatterContiguous(globalRowDelimiters.data(), localRowDelimiters.data(), delimiterCounts,
                      rowDisplacements, MPI_UINT32_T, 0, 102, MPI_COMM_WORLD);
    const index_t localNonzeroStart = nonzeroStarts[rank];
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= localNonzeroStart;
    }

    // The global matrix is no longer needed for the parallel computation.
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalRowDelimiters);
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localColumns.data(), localRowDelimiters.data(),
                denseVector.data(), localRows, localOutput.data());
        keepResult(localOutput.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops =
            (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
            duration / 1e9;
        const double avgTime = duration * 1000.0 / static_cast<double>(iterations);
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOutput;
    if (rank == 0 && (printResults || validate)) {
        globalOutput.resize(numRows);
    }
    if (printResults || validate) {
        gatherContiguous(localOutput.data(), globalOutput.data(), rowCounts,
                         rowDisplacements, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD);
    }

    int returnCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(globalOutput, "OutputVector");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), globalOutput.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
