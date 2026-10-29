#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (!sparsity || !numRows || entries > UINT32_MAX || numRows > INT_MAX || entries / sparsity > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity (MPI counts and CSR indices must fit).\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);
    std::vector<int> rowCounts(ranks), rowDispls(ranks), nzCounts(ranks), nzDispls(ranks);
    for (int p = 0; p < ranks; ++p) {
        rowCounts[p] = numRows / ranks + (static_cast<unsigned>(p) < numRows % ranks);
        rowDispls[p] = p == 0 ? 0 : rowDispls[p-1] + rowCounts[p-1];
    }
    std::vector<double> vec(numRows), values, reference, output;
    std::vector<index_t> cols, rows;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        values.resize(nItems); cols.resize(nItems); rows.resize(size_t(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
        }
        for (int p = 0; p < ranks; ++p) {
            nzDispls[p] = rows[rowDispls[p]];
            nzCounts[p] = rows[rowDispls[p] + rowCounts[p]] - nzDispls[p];
        }
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localNnz = 0, nzOffset = 0;
    MPI_Scatter(nzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nzDispls.data(), 1, MPI_INT, &nzOffset, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const int localRows = rowCounts[rank];
    std::vector<index_t> localRow(size_t(localRows) + 1), localCols(localNnz);
    std::vector<double> localValues(localNnz), localOut(localRows);
    // Scatter row starts; the last delimiter is supplied by the nonzero count.
    MPI_Scatterv(rows.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 localRow.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (int i = 0; i < localRows; ++i) localRow[i] -= nzOffset;
    localRow[localRows] = localNnz;
    MPI_Scatterv(cols.data(), nzCounts.data(), nzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(values.data(), nzCounts.data(), nzDispls.data(), MPI_DOUBLE,
                 localValues.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Release the root's full matrix before benchmarking.
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(cols); std::vector<index_t>().swap(rows);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter)
        spmvCpu(localValues.data(), localCols.data(), localRow.data(), vec.data(), localRows, localOut.data());
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (printResults || validate) {
        if (rank == 0) output.resize(numRows);
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, output.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), output.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
