#include <chrono>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                            (static_cast<uint64_t>(i) * dim + j);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // MPI collectives use int element counts.  The CSR indices are uint32_t,
    // so reject inputs which cannot be represented instead of overflowing.
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        matrixEntries / sparsity > static_cast<uint64_t>(INT_MAX)) {
        if (rank == 0) {
            fprintf(stderr, "Invalid dimensions, sparsity, or iteration count\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixEntries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank zero creates the same deterministic problem as the serial program.
    // The matrix is then distributed by contiguous, nnz-balanced row ranges.
    std::vector<double> globalVal(rank == 0 ? nItems : 0);
    std::vector<index_t> globalCols(rank == 0 ? nItems : 0);
    std::vector<index_t> globalRows(rank == 0 ? numRows + 1 : 0);
    std::vector<double> h_vec(numRows);
    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(globalVal.data(), globalCols.data(), globalRows.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    std::vector<int> rowStarts(numRanks), rowCounts(numRanks);
    std::vector<int> nnzStarts(numRanks), nnzCounts(numRanks);
    if (rank == 0) {
        std::vector<index_t> boundaries(numRanks + 1);
        boundaries.front() = 0;
        boundaries.back() = numRows;
        for (int p = 1; p < numRanks; ++p) {
            const index_t target = static_cast<index_t>(
                (static_cast<uint64_t>(nItems) * p) / numRanks);
            boundaries[p] = static_cast<index_t>(
                std::lower_bound(globalRows.begin(), globalRows.end(), target) - globalRows.begin());
            if (boundaries[p] > numRows) boundaries[p] = numRows;
        }
        for (int p = 0; p < numRanks; ++p) {
            rowStarts[p] = static_cast<int>(boundaries[p]);
            rowCounts[p] = static_cast<int>(boundaries[p + 1] - boundaries[p]);
            nnzStarts[p] = static_cast<int>(globalRows[boundaries[p]]);
            nnzCounts[p] = static_cast<int>(globalRows[boundaries[p + 1]] -
                                            globalRows[boundaries[p]]);
        }
    }

    int localRowStart = 0, localRows = 0, localNnzStart = 0, localNnz = 0;
    MPI_Scatter(rowStarts.data(), 1, MPI_INT, &localRowStart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzStarts.data(), 1, MPI_INT, &localNnzStart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);
    std::vector<int> rowDelimiterCounts(numRanks), rowDelimiterStarts(numRanks);
    if (rank == 0) {
        for (int p = 0; p < numRanks; ++p) {
            rowDelimiterCounts[p] = rowCounts[p] + 1;
            rowDelimiterStarts[p] = rowStarts[p];
        }
    }
    MPI_Scatterv(globalVal.data(), nnzCounts.data(), nnzStarts.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalCols.data(), nnzCounts.data(), nnzStarts.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalRows.data(), rowDelimiterCounts.data(), rowDelimiterStarts.data(),
                 MPI_UINT32_T, localRowDelimiters.data(), localRows + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= static_cast<index_t>(localNnzStart);
    }

    // The global matrix is not retained during the benchmark.
    globalVal.clear();
    globalCols.clear();
    globalRows.clear();
    globalVal.shrink_to_fit();
    globalCols.shrink_to_fit();
    globalRows.shrink_to_fit();

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), static_cast<index_t>(localRows), localOut.data());
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_out(rank == 0 && (validate || printResults) ? numRows : 0);
    if (validate || printResults) {
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, h_out.data(),
                    rowCounts.data(), rowStarts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double durationMs = elapsed * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);
    
        // Calculate aggregate performance metrics from the slowest rank.
        const double gflops = elapsed > 0.0 ? (2.0 * nItems * iterations) / elapsed / 1e9 : 0.0;
        const double avgTime = durationMs / static_cast<double>(iterations);
    
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
