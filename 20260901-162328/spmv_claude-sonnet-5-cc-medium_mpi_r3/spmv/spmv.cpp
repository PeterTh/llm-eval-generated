#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
#define INDEX_MPI_T MPI_UINT32_T

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

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits the rows [0, dim) among `numRanks` MPI ranks so that each rank
//   receives (as close as possible to) an equal number of non-zero elements,
//   using the global row delimiters. This keeps the SpMV computation load
//   balanced across ranks even though rows may vary widely in nnz count.
//
// ****************************************************************************
std::vector<index_t> computeRowPartition(const index_t* rowDelimiters, const index_t dim,
                                          const int numRanks) {
    std::vector<index_t> rowOffsets(numRanks + 1);
    rowOffsets[0] = 0;
    rowOffsets[numRanks] = dim;

    const index_t totalNnz = rowDelimiters[dim];
    for (int r = 1; r < numRanks; ++r) {
        const index_t target = static_cast<index_t>(
            (static_cast<uint64_t>(totalNnz) * r) / static_cast<uint64_t>(numRanks));
        index_t row = static_cast<index_t>(
            std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target) - rowDelimiters);
        row = std::max(row, rowOffsets[r - 1]);
        row = std::min(row, dim);
        rowOffsets[r] = row;
    }
    return rowOffsets;
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
    const bool isRoot = (rank == 0);

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
            if (isRoot) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // The dense vector is small (O(dim)) and needed in full by every rank,
    // since column indices touched by a rank's rows may be arbitrary.
    std::vector<double> h_vec(numRows);
    // Row delimiters are also O(dim) and are needed in full by every rank so
    // that the row -> rank partition can be computed identically everywhere.
    std::vector<index_t> h_rowDelimiters(numRows + 1);

    // Only the root generates the (potentially large, O(nnz)) matrix data,
    // using the exact same sequential RNG stream as the original single
    // process implementation. This guarantees bit-for-bit identical matrix
    // and vector contents regardless of the number of MPI ranks used.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    if (isRoot) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), INDEX_MPI_T, 0, MPI_COMM_WORLD);

    // Partition rows across ranks so that each rank gets (approximately) an
    // equal share of non-zero elements. This partition is computed
    // identically and independently on every rank from the broadcast row
    // delimiters, avoiding an extra communication step.
    const std::vector<index_t> rowOffsets = computeRowPartition(h_rowDelimiters.data(), numRows, numRanks);

    const index_t rowStart = rowOffsets[rank];
    const index_t rowEnd = rowOffsets[rank + 1];
    const index_t localNumRows = rowEnd - rowStart;
    const index_t localNItems = h_rowDelimiters[rowEnd] - h_rowDelimiters[rowStart];
    const index_t localOffset = h_rowDelimiters[rowStart];

    // Scatter the non-zero values/columns to each rank according to the
    // row partition computed above.
    std::vector<int> nnzCounts(numRanks), nnzDispls(numRanks);
    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowOffsets[r]]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowOffsets[r + 1]] - h_rowDelimiters[rowOffsets[r]]);
        rowDispls[r] = static_cast<int>(rowOffsets[r]);
        rowCounts[r] = static_cast<int>(rowOffsets[r + 1] - rowOffsets[r]);
    }

    std::vector<double> h_localVal(localNItems);
    std::vector<index_t> h_localCols(localNItems);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 h_localVal.data(), static_cast<int>(localNItems), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), INDEX_MPI_T,
                 h_localCols.data(), static_cast<int>(localNItems), INDEX_MPI_T, 0, MPI_COMM_WORLD);

    // Root no longer needs the full matrix data once it has been scattered,
    // except for computing the reference solution when validation is enabled.
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }
    // Build local row delimiters, shifted to be relative to the local nnz buffer.
    std::vector<index_t> h_localRowDelimiters(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[rowStart + i] - localOffset;
    }

    std::vector<double> h_localOut(localNumRows);

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_localVal.data(), h_localCols.data(), h_localRowDelimiters.data(),
                h_vec.data(), localNumRows, h_localOut.data());
    }

    const double endTime = MPI_Wtime();
    const double localElapsed = endTime - startTime;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed result vector back onto the root for reporting,
    // result printing, and validation.
    std::vector<double> h_out;
    if (isRoot) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(h_localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (isRoot) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / maxElapsed / 1e9;
        const double avgTime = (maxElapsed * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int validationResult = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                validationResult = 0;
            } else {
                printf("Validation: FAILED\n");
                validationResult = 1;
            }
        }
        MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validationResult;
    }

    MPI_Finalize();
    return 0;
}
