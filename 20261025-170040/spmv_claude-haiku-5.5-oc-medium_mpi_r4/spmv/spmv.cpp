#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Function: partitionRows
//
// Purpose:
//   Splits the matrix rows into contiguous blocks, one per MPI rank, with the
//   number of nonzeros in each block approximately balanced.
//
// Arguments:
//   rowDelimiters: CSR row delimiters (size dim+1), valid on root only
//   dim:           number of rows/columns in the matrix
//   nItems:        total number of nonzero elements
//   nprocs:        number of MPI ranks
//
// Returns:
//   vector of size nprocs+1; rank r owns rows [result[r], result[r+1])
//
// ****************************************************************************
std::vector<index_t> partitionRows(const index_t* rowDelimiters, const index_t dim,
                                   const index_t nItems, const int nprocs) {
    std::vector<index_t> rowStart(nprocs + 1, 0);
    rowStart[nprocs] = dim;
    for (int r = 1; r < nprocs; ++r) {
        const index_t target = static_cast<index_t>(static_cast<uint64_t>(r) * nItems / nprocs);
        const index_t* it = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        index_t row = static_cast<index_t>(it - rowDelimiters);
        row = std::min(row, dim);
        rowStart[r] = std::max(row, rowStart[r - 1]);
    }
    return rowStart;
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = 0;
    bool exitEarly = false;
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
            if (isRoot) printUsage(argv[0]);
            exitCode = 0;
            exitEarly = true;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            exitEarly = true;
            break;
        }
    }
    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Row-delimiter and nonzero offsets are exchanged as 32-bit values and
    // per-rank counts are passed to MPI_Scatterv as int.
    if (static_cast<uint64_t>(nItems) > static_cast<uint64_t>(INT_MAX) ||
        static_cast<uint64_t>(numRows) > static_cast<uint64_t>(INT_MAX)) {
        if (isRoot) printf("Problem size exceeds MPI int count limit\n");
        MPI_Finalize();
        return 1;
    }

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Dense vector is replicated on every rank (used by all local rows)
    std::vector<double> h_vec(numRows);

    // Full matrix data exists only on root, which generates it with the same
    // sequence of rand() calls as the serial code.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition rows across ranks (balanced by nonzero count) and broadcast the
    // partition so every rank knows its block and its nonzero offsets.
    std::vector<index_t> rowStart(nprocs + 1, 0);   // rank r owns rows [rowStart[r], rowStart[r+1])
    std::vector<index_t> nzStart(nprocs + 1, 0);    // rank r owns nonzeros [nzStart[r], nzStart[r+1])
    if (isRoot) {
        rowStart = partitionRows(h_rowDelimiters.data(), numRows, nItems, nprocs);
        for (int r = 0; r <= nprocs; ++r) {
            nzStart[r] = h_rowDelimiters[rowStart[r]];
        }
    }
    MPI_Bcast(rowStart.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nzStart.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<int> rowCounts(nprocs), rowDispls(nprocs), nzCounts(nprocs), nzDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        rowDispls[r] = static_cast<int>(rowStart[r]);
        rowCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        nzDispls[r] = static_cast<int>(nzStart[r]);
        nzCounts[r] = static_cast<int>(nzStart[r + 1] - nzStart[r]);
    }

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t localNnz = static_cast<index_t>(nzCounts[rank]);

    // Scatter the local nonzero values and column indices
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(h_val.data(), nzCounts.data(), nzDispls.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nzCounts.data(), nzDispls.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Scatter row delimiters rebased so each rank's local nonzero indices start at 0
    std::vector<index_t> rebasedDelims;
    if (isRoot) {
        rebasedDelims.resize(numRows);
        for (int r = 0; r < nprocs; ++r) {
            for (index_t i = rowStart[r]; i < rowStart[r + 1]; ++i) {
                rebasedDelims[i] = h_rowDelimiters[i] - nzStart[r];
            }
        }
    }
    std::vector<index_t> localRowDelimiters(localRows + 1);
    MPI_Scatterv(rebasedDelims.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    localRowDelimiters[localRows] = localNnz;

    // For validation, compute reference solution on root
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation on local rows
    std::vector<double> localOut(localRows);
    if (isRoot) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double tLocal = MPI_Wtime() - tStart;
    double tMax = 0.0;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Results (and validation) are only needed on root
    std::vector<double> h_out;
    if (printResults || validate) {
        if (isRoot) h_out.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (isRoot) {
        const long long durationMs = static_cast<long long>(tMax * 1000.0);

        printf("Computation time: %lld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int validFlag = 1;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            validFlag = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
        }
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (isRoot) {
            printf(validFlag ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        exitCode = validFlag ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
