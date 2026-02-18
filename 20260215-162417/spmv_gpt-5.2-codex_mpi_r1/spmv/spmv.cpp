#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
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

void computeRowBounds(const index_t numRows, const int size, const int rank, index_t& rowStart, index_t& rowEnd) {
    const index_t baseRows = numRows / static_cast<index_t>(size);
    const index_t remainder = numRows % static_cast<index_t>(size);
    rowStart = baseRows * static_cast<index_t>(rank) + static_cast<index_t>(rank < static_cast<int>(remainder) ? rank : remainder);
    rowEnd = rowStart + baseRows + static_cast<index_t>(rank < static_cast<int>(remainder) ? 1 : 0);
}

int checkedCount(const index_t value, const char* label, const int rank) {
    if (value > static_cast<index_t>(INT_MAX)) {
        if (rank == 0) {
            fprintf(stderr, "Value too large for MPI int count (%s=%u)\n", label, value);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(value);
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
    bool showHelp = false;
    bool parseError = false;
    const char* errorOption = nullptr;

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
            errorOption = argv[i];
            break;
        }

    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (parseError) {
        if (rank == 0) {
            printf("Unknown option: %s\n", errorOption ? errorOption : "");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
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
        printf("MPI ranks: %d\n", size);
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(static_cast<size_t>(numRows));

    if (rank == 0) {
        h_val.resize(static_cast<size_t>(nItems));
        h_cols.resize(static_cast<size_t>(nItems));
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    const int numRowsInt = checkedCount(numRows, "numRows", rank);
    MPI_Bcast(h_vec.data(), numRowsInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    index_t rowStart = 0;
    index_t rowEnd = 0;
    computeRowBounds(numRows, size, rank, rowStart, rowEnd);
    const index_t localRows = rowEnd - rowStart;
    const int localRowsInt = checkedCount(localRows, "localRows", rank);
    const index_t localRowDelims = localRows + 1;
    const int localRowDelimsInt = checkedCount(localRowDelims, "localRowDelims", rank);

    std::vector<index_t> h_rowDelimiters_local(static_cast<size_t>(localRowDelims));

    int localNnzCount = 0;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    std::vector<int> outCounts;
    std::vector<int> outDispls;

    if (rank == 0) {
        nnzCounts.resize(size);
        nnzDispls.resize(size);
        rowCounts.resize(size);
        rowDispls.resize(size);
        outCounts.resize(size);
        outDispls.resize(size);

        for (int r = 0; r < size; ++r) {
            index_t rStart = 0;
            index_t rEnd = 0;
            computeRowBounds(numRows, size, r, rStart, rEnd);
            const index_t rRows = rEnd - rStart;
            rowCounts[r] = checkedCount(rRows + 1, "rowCounts", rank);
            rowDispls[r] = checkedCount(rStart, "rowDispls", rank);
            outCounts[r] = checkedCount(rRows, "outCounts", rank);
            outDispls[r] = checkedCount(rStart, "outDispls", rank);

            const index_t nnzStart = h_rowDelimiters[rStart];
            const index_t nnzEnd = h_rowDelimiters[rEnd];
            const index_t nnzCount = nnzEnd - nnzStart;
            nnzCounts[r] = checkedCount(nnzCount, "nnzCounts", rank);
            nnzDispls[r] = checkedCount(nnzStart, "nnzDispls", rank);
        }
    }

    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnzCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> h_val_local(static_cast<size_t>(localNnzCount));
    std::vector<index_t> h_cols_local(static_cast<size_t>(localNnzCount));

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_DOUBLE,
                 h_val_local.data(), localNnzCount, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_UINT32_T,
                 h_cols_local.data(), localNnzCount, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDispls.data() : nullptr,
                 MPI_UINT32_T,
                 h_rowDelimiters_local.data(), localRowDelimsInt, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    const index_t rowOffset = h_rowDelimiters_local.empty() ? 0 : h_rowDelimiters_local[0];
    for (auto& entry : h_rowDelimiters_local) {
        entry -= rowOffset;
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(static_cast<size_t>(numRows));
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    std::vector<double> h_out_local(static_cast<size_t>(localRows));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val_local.data(), h_cols_local.data(), h_rowDelimiters_local.data(),
                h_vec.data(), localRows, h_out_local.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);

        // Calculate performance metrics
        const double avgTime = durationMs / static_cast<double>(iterations);
        const double gflops = maxTime > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / maxTime / 1e9
                                  : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> h_out;
    if (validate || printResults) {
        if (rank == 0) {
            h_out.resize(static_cast<size_t>(numRows));
        }
        MPI_Gatherv(h_out_local.data(), localRowsInt, MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? outCounts.data() : nullptr,
                    rank == 0 ? outDispls.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int resultCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                resultCode = 1;
            }
        }
        MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return resultCode;
}
