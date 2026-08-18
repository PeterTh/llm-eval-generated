#include <algorithm>
#include <cstdint>
#include <cmath>
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
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t firstRow = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       firstRow + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       firstRow + i, ref, res, relError);
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

// Partition contiguous rows so that each process receives approximately the
// same number of nonzeros.  A row is the smallest independently computable
// CSR unit, so a single unusually dense row remains intact.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowDelimiters,
                                   const index_t numRows, const int numProcesses) {
    std::vector<index_t> rowOffsets(static_cast<size_t>(numProcesses) + 1);
    const uint64_t totalNnz = rowDelimiters[numRows];

    rowOffsets.front() = 0;
    for (int rank = 1; rank < numProcesses; ++rank) {
        const uint64_t target = totalNnz * static_cast<uint64_t>(rank) /
                                static_cast<uint64_t>(numProcesses);
        const auto boundary = std::lower_bound(rowDelimiters.begin(), rowDelimiters.end(),
                                               static_cast<index_t>(target));
        index_t row = static_cast<index_t>(boundary - rowDelimiters.begin());

        // Use the nearest CSR row boundary to reduce the maximum per-rank work.
        if (row > 0 && row < numRows) {
            const uint64_t before = rowDelimiters[row - 1];
            const uint64_t after = rowDelimiters[row];
            if (target - before < after - target) {
                --row;
            }
        }
        rowOffsets[rank] = std::max(rowOffsets[rank - 1], row);
    }
    rowOffsets.back() = numRows;
    return rowOffsets;
}

int asMpiCount(const uint64_t count) {
    if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "MPI collective count exceeds INT_MAX\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcesses = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int exitAfterParsing = 0;
    int parseExitCode = 0;

    // Parse only on rank zero so diagnostics and benchmark output are emitted once.
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
                exitAfterParsing = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitAfterParsing = 1;
                parseExitCode = 1;
            }

            if (exitAfterParsing != 0) {
                break;
            }
        }
    }

    MPI_Bcast(&exitAfterParsing, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseExitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitAfterParsing != 0) {
        MPI_Finalize();
        return parseExitCode;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int options[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(options, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = options[0] != 0;
    printResults = options[1] != 0;

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Rank zero preserves the original deterministic initialization and scatters
    // independent row ranges.  The vector is replicated because every CSR row
    // may reference every column; it is read-only throughout the benchmark.
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowDelimiters;
    std::vector<index_t> rowOffsets;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    std::vector<double> denseVector(numRows);

    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // C guarantees the same sequence as the original program's implicit
        // initial rand() seed, independently of any MPI implementation details.
        srand(1);
        fill(denseVector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRowDelimiters.data(), nItems, numRows);

        rowOffsets = partitionRows(globalRowDelimiters, numRows, numProcesses);
        rowCounts.resize(numProcesses);
        rowDisplacements.resize(numProcesses);
        nnzCounts.resize(numProcesses);
        nnzDisplacements.resize(numProcesses);
        for (int process = 0; process < numProcesses; ++process) {
            const index_t firstRow = rowOffsets[process];
            const index_t lastRow = rowOffsets[process + 1];
            rowCounts[process] = asMpiCount(lastRow - firstRow);
            rowDisplacements[process] = asMpiCount(firstRow);
            nnzCounts[process] = asMpiCount(globalRowDelimiters[lastRow] -
                                             globalRowDelimiters[firstRow]);
            nnzDisplacements[process] = asMpiCount(globalRowDelimiters[firstRow]);
        }
    }

    MPI_Bcast(denseVector.data(), asMpiCount(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localRowCount = 0;
    int localRowStart = 0;
    int localNnzCount = 0;
    MPI_Scatter(rank == 0 ? rowCounts.data() : nullptr, 1, MPI_INT,
                &localRowCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? rowDisplacements.data() : nullptr, 1, MPI_INT,
                &localRowStart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnzCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localValues(localNnzCount);
    std::vector<index_t> localColumns(localNnzCount);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRowCount) + 1);
    std::vector<int> delimiterCounts;
    if (rank == 0) {
        delimiterCounts.resize(numProcesses);
        for (int process = 0; process < numProcesses; ++process) {
            delimiterCounts[process] = rowCounts[process] + 1;
        }
    }

    MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localValues.data(), localNnzCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalColumns.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localColumns.data(), localNnzCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rank == 0 ? delimiterCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRowDelimiters.data(), localRowCount + 1, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    const index_t localNnzStart = localRowDelimiters.front();
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= localNnzStart;
    }

    // The global matrix is no longer needed after distribution.  Releasing it
    // keeps rank zero's memory footprint comparable to the other ranks.
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalRowDelimiters);
    }

    std::vector<double> localOutput(localRowCount);
    std::vector<double> localReference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        localReference.resize(localRowCount);
        spmvCpu(localValues.data(), localColumns.data(), localRowDelimiters.data(),
                denseVector.data(), static_cast<index_t>(localRowCount), localReference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localColumns.data(), localRowDelimiters.data(),
                denseVector.data(), static_cast<index_t>(localRowCount), localOutput.data());
    }

    const double localElapsedSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        // Calculate global performance against the slowest process's local work.
        const double gflops = (2.0 * nItems * iterations) /
                              (durationMilliseconds / 1000.0) / 1e9;
        const double avgTime = durationMilliseconds / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather only when requested.  Normal benchmark runs leave the result
    // distributed and avoid a per-iteration or post-timing communication cost.
    if (printResults) {
        std::vector<double> globalOutput;
        if (rank == 0) {
            globalOutput.resize(numRows);
        }
        MPI_Gatherv(localOutput.data(), localRowCount, MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr,
                    rank == 0 ? rowCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalOutput, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResults(localReference.data(), localOutput.data(),
                                             static_cast<index_t>(localRowCount),
                                             static_cast<index_t>(localRowStart)) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid != 0 ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
