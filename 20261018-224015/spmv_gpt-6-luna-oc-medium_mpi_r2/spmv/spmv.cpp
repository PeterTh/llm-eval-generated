#include <chrono>
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!numRows || !sparsity || !iterations) {
        if (rank == 0) fprintf(stderr, "Matrix size, sparsity, and iterations must be positive.\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = (numRows * numRows) / sparsity;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\n",
               numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> vals(rank == 0 ? nItems : 0), vec(numRows), result(rank == 0 ? numRows : 0);
    std::vector<index_t> cols(rank == 0 ? nItems : 0), delimiters(rank == 0 ? numRows + 1 : 0);
    if (rank == 0) {
        fill(vec.data(), numRows, maxVal);
        fill(vals.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), delimiters.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> rowCounts(size), rowDispls(size), nzCounts(size), nzDispls(size);
    int rowOffset = 0;
    for (int p = 0; p < size; ++p) {
        rowCounts[p] = static_cast<int>(numRows / size + (p < static_cast<int>(numRows % size)));
        rowDispls[p] = rowOffset;
        rowOffset += rowCounts[p];
    }
    std::vector<index_t> lengths(rank == 0 ? numRows : 0);
    if (rank == 0) {
        for (index_t r = 0; r < numRows; ++r) lengths[r] = delimiters[r + 1] - delimiters[r];
        for (int p = 0; p < size; ++p) {
            const index_t first = static_cast<index_t>(rowDispls[p]);
            nzDispls[p] = static_cast<int>(delimiters[first]);
            nzCounts[p] = static_cast<int>(delimiters[first + rowCounts[p]] - delimiters[first]);
        }
    }
    MPI_Bcast(nzCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nzDispls.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    const int localRows = rowCounts[rank], localNnz = nzCounts[rank];
    std::vector<index_t> localLengths(localRows), localCols(localNnz), localDelimiters(localRows + 1);
    std::vector<double> localVals(localNnz), localResult(localRows);
    MPI_Scatterv(rank == 0 ? lengths.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_UINT32_T, localLengths.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? cols.data() : nullptr, nzCounts.data(), nzDispls.data(), MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? vals.data() : nullptr, nzCounts.data(), nzDispls.data(), MPI_DOUBLE, localVals.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    localDelimiters[0] = 0;
    for (int r = 0; r < localRows; ++r) localDelimiters[r + 1] = localDelimiters[r] + localLengths[r];

    std::vector<double> reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        reference.resize(numRows);
        spmvCpu(vals.data(), cols.data(), delimiters.data(), vec.data(), numRows, reference.data());
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter)
        spmvCpu(localVals.data(), localCols.data(), localDelimiters.data(), vec.data(), static_cast<index_t>(localRows), localResult.data());
    MPI_Gatherv(localResult.data(), localRows, MPI_DOUBLE, rank == 0 ? result.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count(), seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double ms = seconds * 1000.0;
        printf("Computation time: %ld ms\n", static_cast<long>(ms));
        printf("Average time per iteration: %.3f ms\n", ms / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / seconds / 1e9);
        if (printResults) print_results(result, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), result.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
