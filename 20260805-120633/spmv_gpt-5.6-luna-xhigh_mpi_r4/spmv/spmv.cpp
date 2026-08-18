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

// Return the contiguous row range assigned to one MPI rank.  Rows, rather
// than individual non-zeroes, are distributed so each rank can evaluate its
// rows without communication in the timed loop.
void rowPartition(const index_t dim, const int rank, const int nranks,
                  index_t& firstRow, index_t& rowCount) {
    const index_t rowsPerRank = dim / static_cast<index_t>(nranks);
    const index_t extraRows = dim % static_cast<index_t>(nranks);
    firstRow = static_cast<index_t>(
        static_cast<uint64_t>(rank) * rowsPerRank +
        static_cast<uint64_t>(rank < static_cast<int>(extraRows) ? rank : extraRows));
    rowCount = rowsPerRank + (rank < static_cast<int>(extraRows) ? 1U : 0U);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse options on rank 0 and broadcast the resulting configuration.  A
    // single parser avoids divergent command-line handling across ranks.
    int commandResult = 0;
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
                commandResult = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                commandResult = -1;
                break;
            }
        }
    }

    MPI_Bcast(&commandResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (commandResult != 0) {
        MPI_Finalize();
        return commandResult > 0 ? 0 : 1;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    // MPI collectives use int counts/displacements.  These checks also avoid
    // the unsigned arithmetic overflow present in the original expression
    // for inputs too large to represent in this benchmark's index type.
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    int inputValid = 1;
    if (sparsity == 0) {
        inputValid = 0;
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be greater than zero.\n");
        }
    } else if (totalEntries > std::numeric_limits<index_t>::max()) {
        inputValid = 0;
        if (rank == 0) {
            fprintf(stderr, "Matrix is too large for the benchmark index type.\n");
        }
    }

    const uint64_t nItems64 = sparsity == 0 ? 0 : totalEntries / sparsity;
    if (nItems64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        inputValid = 0;
        if (rank == 0) {
            fprintf(stderr, "Matrix has too many non-zeroes for MPI counts.\n");
        }
    }
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        inputValid = 0;
        if (rank == 0) {
            fprintf(stderr, "Matrix dimension is too large for MPI counts.\n");
        }
    }

    MPI_Bcast(&inputValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!inputValid) {
        MPI_Finalize();
        return 1;
    }

    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        const double sparsePercent = totalEntries == 0
                                          ? 0.0
                                          : 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, sparsePercent);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate the exact original data on rank 0, then distribute only the
    // local CSR rows.  Keeping the full matrix off the other ranks provides
    // the distributed-memory behavior while preserving deterministic output.
    std::vector<double> h_val;                          // Rank 0: non-zero values
    std::vector<index_t> h_cols;                        // Rank 0: column indices
    std::vector<index_t> h_rowDelimiters;               // Rank 0: row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_reference;                    // Rank 0: reference result

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute the reference before releasing the global
        // CSR storage.  The local computation uses the same row-wise order.
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    index_t localRowStart = 0;
    index_t localRows = 0;
    rowPartition(numRows, rank, nranks, localRowStart, localRows);

    std::vector<index_t> nnzCounts;
    std::vector<int> valueCounts;
    std::vector<int> valueDisplacements;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> outputCounts;
    if (rank == 0) {
        nnzCounts.resize(nranks);
        valueCounts.resize(nranks);
        valueDisplacements.resize(nranks);
        rowCounts.resize(nranks);
        rowDisplacements.resize(nranks);
        outputCounts.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            index_t firstRow = 0;
            index_t rowCount = 0;
            rowPartition(numRows, r, nranks, firstRow, rowCount);
            const index_t firstNnz = h_rowDelimiters[firstRow];
            const index_t lastNnz = h_rowDelimiters[firstRow + rowCount];
            nnzCounts[r] = lastNnz - firstNnz;
            valueCounts[r] = static_cast<int>(nnzCounts[r]);
            valueDisplacements[r] = static_cast<int>(firstNnz);
            rowCounts[r] = static_cast<int>(rowCount) + 1;
            rowDisplacements[r] = static_cast<int>(firstRow);
            outputCounts[r] = static_cast<int>(rowCount);
        }
    }

    index_t localNnz = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_UINT32_T,
                &localNnz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? valueCounts.data() : nullptr,
                 rank == 0 ? valueDisplacements.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? valueCounts.data() : nullptr,
                 rank == 0 ? valueDisplacements.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // The received delimiters are global CSR offsets; make them local to the
    // rank's value array while retaining the exact per-row traversal order.
    const index_t localNnzStart = localRowDelimiters.front();
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] -= localNnzStart;
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // The global CSR arrays are no longer needed after Scatterv.  Release
        // them so rank 0 does not remain a memory bottleneck during timing.
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
        printf("Computing SpMV...\n");
        fflush(stdout);
    }

    std::vector<double> localOut(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localSeconds = MPI_Wtime() - start;
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    std::vector<double> h_out;
    const bool gatherResults = validate || printResults;
    if (gatherResults) {
        if (rank == 0) {
            h_out.resize(numRows);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? outputCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double computationMilliseconds = computationSeconds * 1000.0;
        const double gflops = computationSeconds > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * iterations) /
                                        computationSeconds / 1e9
                                  : 0.0;
        const double avgTime = iterations > 0
                                   ? computationMilliseconds / static_cast<double>(iterations)
                                   : 0.0;

        printf("Computation time: %.3f ms\n", computationMilliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation.
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
