#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        if (rank == 0) fprintf(stderr, "n, sparsity, and iterations must be positive\n");
        MPI_Finalize(); return 1;
    }
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
    }

    std::vector<int> rowCounts(ranks), rowDispls(ranks), nnzCounts(ranks), nnzDispls(ranks);
    for (int p = 0; p < ranks; ++p) {
        const index_t first = static_cast<index_t>((uint64_t(numRows) * p) / ranks);
        const index_t last = static_cast<index_t>((uint64_t(numRows) * (p + 1)) / ranks);
        rowDispls[p] = static_cast<int>(first); rowCounts[p] = static_cast<int>(last - first);
        nnzDispls[p] = 0;
    }
    std::vector<double> allVal, allVec;
    std::vector<index_t> allCols, allRows;
    std::vector<int> allRowLengths;
    if (rank == 0) {
        allVal.resize(nItems); allCols.resize(nItems); allRows.resize(numRows + 1); allVec.resize(numRows); allRowLengths.resize(numRows);
        fill(allVec.data(), numRows, maxVal); fill(allVal.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        for (index_t i=0;i<numRows;++i) allRowLengths[i]=static_cast<int>(allRows[i+1]-allRows[i]);
        for (int p = 0; p < ranks; ++p) {
            const int first = rowDispls[p], last = first + rowCounts[p];
            nnzDispls[p] = static_cast<int>(allRows[first]);
            nnzCounts[p] = static_cast<int>(allRows[last] - allRows[first]);
        }
    }
    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> localRowLengths(rowCounts[rank]);
    const int localNnz = nnzCounts[rank];
    std::vector<double> h_val(localNnz), h_vec(numRows), h_out(rowCounts[rank]);
    std::vector<index_t> h_cols(localNnz);
    if (rank == 0) std::copy(allVec.begin(), allVec.end(), h_vec.begin());
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? allRowLengths.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_INT,
                 localRowLengths.data(), rowCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? allVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 h_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? allCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UNSIGNED,
                 h_cols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<index_t> localDelims(rowCounts[rank] + 1, 0);
    for (int i=0;i<rowCounts[rank];++i) localDelims[i+1]=localDelims[i]+static_cast<index_t>(localRowLengths[i]);
    std::vector<double> gathered(numRows);

    if (rank == 0) printf("Initializing data structures...\n");

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        if (rank == 0) { h_reference.resize(numRows); spmvCpu(allVal.data(), allCols.data(), allRows.data(), allVec.data(), numRows, h_reference.data()); }
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (index_t i = 0; i < static_cast<index_t>(h_out.size()); ++i) {
            double t=0.0; for (index_t j=localDelims[i]; j<localDelims[i+1]; ++j) t += h_val[j]*h_vec[h_cols[j]]; h_out[i]=t;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end-start).count(), maxSeconds=0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(h_out.data(), static_cast<int>(h_out.size()), MPI_DOUBLE, gathered.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %.3f ms\n", maxSeconds*1000.0);
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / maxSeconds / 1e9;
    if (rank == 0) { printf("Average time per iteration: %.3f ms\n", maxSeconds*1000.0/iterations); printf("Performance: %.3f GFLOPS\n", gflops); }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(gathered, "OutputVector");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = rank != 0 || verifyResults(h_reference.data(), gathered.data(), numRows);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }

    MPI_Finalize(); return 0;
}
