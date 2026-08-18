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
    bool showHelp = false;
    bool parseSucceeded = true;

    // Parse command line arguments on rank 0 and broadcast the configuration.
    // All MPI ranks still enter and leave MPI collectively, including for -h
    // and malformed command lines.
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
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseSucceeded = false;
                break;
            }
        }
    }

    int parseStatus = parseSucceeded ? 1 : 0;
    int helpStatus = showHelp ? 1 : 0;
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&helpStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus == 0 || helpStatus != 0) {
        if (helpStatus != 0 && rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 0 ? 1 : 0;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int optionFlags = (validate ? 1 : 0) | (printResults ? 2 : 0);
    MPI_Bcast(&optionFlags, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (optionFlags & 1) != 0;
    printResults = (optionFlags & 2) != 0;

    // Calculate the number of non-zero elements without overflowing during
    // the intermediate square operation.
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = sparsity == 0 ? 0 : matrixEntries / sparsity;
    const bool validInput = sparsity != 0 && nItems64 <= std::numeric_limits<index_t>::max() &&
                            numRows <= static_cast<index_t>(std::numeric_limits<int>::max()) &&
                            nItems64 <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    int inputStatus = validInput ? 1 : 0;
    MPI_Bcast(&inputStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (inputStatus == 0) {
        if (rank == 0) {
            printf("Invalid input: matrix dimensions, sparsity, or non-zero count exceed supported limits.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        const double sparsePercent = matrixEntries == 0
                                         ? 100.0
                                         : 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries);
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, sparsePercent);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition contiguous rows as evenly as possible.  The matrix remains in
    // CSR form locally, so there is no communication in the timed SpMV loop.
    const index_t baseRows = numRows / static_cast<index_t>(worldSize);
    const index_t extraRows = numRows % static_cast<index_t>(worldSize);
    const index_t localRows = baseRows + (static_cast<index_t>(rank) < extraRows ? 1 : 0);

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    if (rank == 0) {
        rowCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        for (int p = 0; p < worldSize; ++p) {
            const index_t pRows = baseRows + (static_cast<index_t>(p) < extraRows ? 1 : 0);
            const index_t pStart = static_cast<index_t>(p) * baseRows +
                                   (static_cast<index_t>(p) < extraRows ? static_cast<index_t>(p)
                                                                        : extraRows);
            rowCounts[p] = static_cast<int>(pRows);
            rowDisplacements[p] = static_cast<int>(pStart);
        }
    }

    std::vector<double> h_val;                  // Global non-zero values (rank 0 only)
    std::vector<index_t> h_cols;                // Global column indices (rank 0 only)
    std::vector<index_t> h_rowDelimiters;       // Global row delimiters (rank 0 only)
    std::vector<double> h_vec(numRows);         // Replicated dense vector
    std::vector<double> h_out;                  // Global output vector (rank 0 only)

    if (rank == 0) {
        h_val.resize(nItems);                  // Non-zero values
        h_cols.resize(nItems);                 // Column indices
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);  // Row delimiters
        h_out.resize(numRows);                 // Output vector

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Distribute the CSR matrix.  Row pointers are rebased so each local
    // SpMV can use the same tight inner loop as the original implementation.
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    if (rank == 0) {
        nnzCounts.resize(worldSize);
        nnzDisplacements.resize(worldSize);
        for (int p = 0; p < worldSize; ++p) {
            const index_t pStart = static_cast<index_t>(rowDisplacements[p]);
            const index_t pEnd = pStart + static_cast<index_t>(rowCounts[p]);
            nnzCounts[p] = static_cast<int>(h_rowDelimiters[pEnd] - h_rowDelimiters[pStart]);
            nnzDisplacements[p] = static_cast<int>(h_rowDelimiters[pStart]);
        }
    }

    int localNnz = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1, 0);
    std::vector<double> localOut(static_cast<size_t>(localRows), 0.0);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr,
                 MPI_UINT32_T, localRowDelimiters.data(), static_cast<int>(localRows),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (localRows != 0) {
        const index_t rowOffset = localRowDelimiters[0];
        for (index_t i = 0; i < localRows; ++i) {
            localRowDelimiters[i] -= rowOffset;
        }
        localRowDelimiters[localRows] = static_cast<index_t>(localNnz);
    }

    // The root no longer needs the global CSR storage after distribution.
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    // Perform distributed SpMV computation.  Since vec is invariant across
    // iterations, it is broadcast once above and no MPI operation is needed
    // in this performance-critical loop.
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Only the final vector is needed, so gather once outside the timed loop.
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr,
                rank == 0 ? rowDisplacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double elapsedMs = elapsed * 1000.0;
        const double avgTime = iterations == 0 ? 0.0 : elapsedMs / static_cast<double>(iterations);
        const double gflops = elapsed > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * iterations) / elapsed / 1e9
                                  : 0.0;

        printf("Computation time: %.3f ms\n", elapsedMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid == 1 ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid == 1 ? 0 : 1;
}
