#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
                   const index_t indexOffset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       indexOffset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       indexOffset + i, ref, res, relError);
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

[[noreturn]] void abortMpi(const int rank, const char* message) {
    if (rank == 0) {
        fprintf(stderr, "Error: %s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    // Parse command line arguments
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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or incomplete command-line option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (sparsity == 0) {
        abortMpi(rank, "sparsity must be greater than zero");
    }

    // The original benchmark uses 32-bit CSR indices and computes a dense
    // matrix's number of positions in 32-bit arithmetic.  Keep that format,
    // but reject sizes which cannot be represented instead of silently
    // constructing a corrupt CSR matrix.
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    if (totalEntries > std::numeric_limits<index_t>::max()) {
        abortMpi(rank, "matrix dimensions exceed the 32-bit CSR index range");
    }

    // Calculate number of non-zero elements
    const index_t nItems = static_cast<index_t>(totalEntries / sparsity);
    if (nItems > static_cast<index_t>(INT_MAX)) {
        abortMpi(rank, "the MPI CSR payload exceeds the supported MPI count range");
    }

    // Contiguous row ownership gives every rank a private CSR slice.  The
    // remainder is assigned to the first ranks, keeping row counts balanced.
    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    const index_t rowsPerRank = numRows / static_cast<index_t>(worldSize);
    const index_t extraRows = numRows % static_cast<index_t>(worldSize);
    index_t nextRow = 0;
    for (int process = 0; process < worldSize; ++process) {
        const index_t processRows = rowsPerRank +
                                     (static_cast<index_t>(process) < extraRows ? 1 : 0);
        if (processRows > static_cast<index_t>(INT_MAX) ||
            nextRow > static_cast<index_t>(INT_MAX)) {
            abortMpi(rank, "the row partition exceeds the supported MPI count range");
        }
        rowCounts[process] = static_cast<int>(processRows);
        rowDisplacements[process] = static_cast<int>(nextRow);
        nextRow += processRows;
    }

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t firstRow = static_cast<index_t>(rowDisplacements[rank]);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                static_cast<double>(totalEntries)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // The input vector is replicated because each row can reference any
    // column.  The matrix itself remains distributed by row.
    std::vector<double> h_vec(numRows);
    std::vector<double> localVal;
    std::vector<index_t> localCols;
    std::vector<index_t> localRowDelimiters(localRows + 1);

    // Keep the original random initialization order and seed exactly.  Rank
    // zero creates the global CSR once, then scatters only each rank's rows;
    // no rank keeps a second copy after initialization.
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<int> nnzCounts(worldSize, 0);
    std::vector<int> nnzDisplacements(worldSize, 0);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(numRows + 1);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);

        for (int process = 0; process < worldSize; ++process) {
            const index_t processFirstRow = static_cast<index_t>(rowDisplacements[process]);
            const index_t processLastRow = processFirstRow +
                                           static_cast<index_t>(rowCounts[process]);
            const uint64_t processNnz =
                static_cast<uint64_t>(globalRowDelimiters[processLastRow]) -
                globalRowDelimiters[processFirstRow];
            if (processNnz > static_cast<uint64_t>(INT_MAX)) {
                abortMpi(rank, "a rank's CSR payload exceeds the MPI count range");
            }
            nnzCounts[process] = static_cast<int>(processNnz);
            nnzDisplacements[process] = static_cast<int>(globalRowDelimiters[processFirstRow]);
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);
    localVal.resize(localNnz);
    localCols.resize(localNnz);

    std::vector<int> rowDelimiterCounts(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        rowDelimiterCounts[process] = rowCounts[process] + 1;
    }

    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr,
                 nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 localVal.data(), nnzCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                 nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T,
                 localCols.data(), nnzCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rowDelimiterCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), rowCounts[rank] + 1, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // CSR row offsets received from rank zero are global offsets.  Convert
    // them to offsets relative to this rank's local value/column arrays.
    const index_t localNnzBase = localRowDelimiters[0];
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= localNnzBase;
    }

    globalVal.clear();
    globalVal.shrink_to_fit();
    globalCols.clear();
    globalCols.shrink_to_fit();
    globalRowDelimiters.clear();
    globalRowDelimiters.shrink_to_fit();

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    std::vector<double> h_out(localRows);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double elapsedMs = elapsed * 1000.0;
        const double avgTime = iterations == 0
                                   ? 0.0
                                   : elapsedMs / static_cast<double>(iterations);
        const double gflops = elapsed > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * iterations) /
                                        (elapsed * 1.0e9)
                                  : 0.0;
        printf("Computation time: %.3f ms\n", elapsedMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather only when requested.  This keeps the benchmark memory-scalable
    // and avoids an unnecessary collective after the timed kernel.
    if (printResults) {
        std::vector<double> globalOut;
        if (rank == 0) {
            globalOut.resize(numRows);
        }
        MPI_Gatherv(h_out.data(), rowCounts[rank], MPI_DOUBLE,
                    rank == 0 ? globalOut.data() : nullptr,
                    rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalOut, "OutputVector");
        }
    }

    // Validation
    int validationPassed = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool localValid = verifyResults(h_reference.data(), h_out.data(), localRows,
                                             firstRow);
        const int localValidation = localValid ? 1 : 0;
        MPI_Allreduce(&localValidation, &validationPassed, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
