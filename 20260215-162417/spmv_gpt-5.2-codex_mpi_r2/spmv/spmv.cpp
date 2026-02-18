#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int runBenchmark = 1;
    int exitCode = 0;

    // Parse command line arguments
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
                runBenchmark = 0;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                runBenchmark = 0;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&runBenchmark, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!runBenchmark) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validateInt != 0);
    printResults = (printInt != 0);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    const index_t baseRows = numRows / static_cast<index_t>(size);
    const index_t remainder = numRows % static_cast<index_t>(size);
    const index_t rankIndex = static_cast<index_t>(rank);
    const index_t rowsLocal = baseRows + (rankIndex < remainder ? 1u : 0u);
    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

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

    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;
    std::vector<int> rowDelimCounts;
    std::vector<int> rowDelimDispls;
    std::vector<int> gatherCounts;
    std::vector<int> gatherDispls;
    std::vector<index_t> baseNnz;
    if (rank == 0) {
        nnzCounts.resize(size);
        nnzDispls.resize(size);
        rowDelimCounts.resize(size);
        rowDelimDispls.resize(size);
        gatherCounts.resize(size);
        gatherDispls.resize(size);
        baseNnz.resize(size);

        for (int r = 0; r < size; ++r) {
            const index_t rIndex = static_cast<index_t>(r);
            const index_t rRows = baseRows + (rIndex < remainder ? 1u : 0u);
            const index_t rStart = rIndex * baseRows + std::min(rIndex, remainder);
            const index_t rEnd = rStart + rRows;
            const index_t base = h_rowDelimiters[rStart];
            const index_t end = h_rowDelimiters[rEnd];

            baseNnz[r] = base;
            nnzCounts[r] = static_cast<int>(end - base);
            nnzDispls[r] = static_cast<int>(base);
            rowDelimCounts[r] = static_cast<int>(rRows + 1);
            rowDelimDispls[r] = static_cast<int>(rStart);
            gatherCounts[r] = static_cast<int>(rRows);
            gatherDispls[r] = static_cast<int>(rStart);
        }
    }

    int nnzLocalInt = 0;
    index_t baseNnzLocal = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &nnzLocalInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? baseNnz.data() : nullptr, 1, MPI_UINT32_T,
                &baseNnzLocal, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t nnzLocal = static_cast<index_t>(nnzLocalInt);
    std::vector<double> localVal(nnzLocal);
    std::vector<index_t> localCols(nnzLocal);
    std::vector<index_t> localRowDelimiters(rowsLocal + 1);
    std::vector<double> localOut(rowsLocal);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_DOUBLE,
                 nnzLocal ? localVal.data() : nullptr,
                 nnzLocalInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_UINT32_T,
                 nnzLocal ? localCols.data() : nullptr,
                 nnzLocalInt, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowDelimCounts.data() : nullptr,
                 rank == 0 ? rowDelimDispls.data() : nullptr,
                 MPI_UINT32_T,
                 localRowDelimiters.data(),
                 static_cast<int>(rowsLocal + 1),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    for (index_t i = 0; i < rowsLocal + 1; ++i) {
        localRowDelimiters[i] -= baseNnzLocal;
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), rowsLocal, localOut.data());
    }

    const double end = MPI_Wtime();
    const double localTimeSec = end - start;
    double maxTimeSec = 0.0;
    MPI_Reduce(&localTimeSec, &maxTimeSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(rowsLocal ? localOut.data() : nullptr,
                static_cast<int>(rowsLocal), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? gatherCounts.data() : nullptr,
                rank == 0 ? gatherDispls.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxTimeSec * 1000.0;
        const double gflops = maxTimeSec > 0.0
                                  ? (2.0 * nItems * iterations) / maxTimeSec / 1e9
                                  : 0.0;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Computation time: %.3f ms\n", durationMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int finalCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            finalCode = 1;
        }
    }

    MPI_Bcast(&finalCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return finalCode;
}
