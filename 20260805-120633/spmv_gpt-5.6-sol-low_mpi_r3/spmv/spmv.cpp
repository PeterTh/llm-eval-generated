#include <chrono>
#include <cmath>
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

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        if (rank == 0) fprintf(stderr, "Error: -n, -s, and -i must be greater than zero\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI processes: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Contiguous row ownership gives balanced row counts and permits a single
    // final Gatherv in global row order.
    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    for (int p = 0; p < numRanks; ++p) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * p / numRanks;
        const uint64_t end = static_cast<uint64_t>(numRows) * (p + 1) / numRanks;
        rowDispls[p] = static_cast<int>(begin);
        rowCounts[p] = static_cast<int>(end - begin);
    }
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);

    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRows;
    std::vector<double> h_vec(numRows);
    if (rank == 0) {
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRows.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> nnzCounts(numRanks), nnzDispls(numRanks);
    if (rank == 0) {
        for (int p = 0; p < numRanks; ++p) {
            const index_t first = static_cast<index_t>(rowDispls[p]);
            const index_t last = first + static_cast<index_t>(rowCounts[p]);
            const uint64_t offset = globalRows[first];
            const uint64_t count = static_cast<uint64_t>(globalRows[last]) - offset;
            if (offset > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "Error: MPI distribution exceeds MPI count limits\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            nnzDispls[p] = static_cast<int>(offset);
            nnzCounts[p] = static_cast<int>(count);
        }
    }

    int localNnz = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> h_val(localNnz);
    std::vector<index_t> h_cols(localNnz);
    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, h_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, h_cols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<index_t> h_rowDelimiters(localRows + 1);
    if (rank == 0) {
        for (int p = 0; p < numRanks; ++p) {
            const index_t first = static_cast<index_t>(rowDispls[p]);
            std::vector<index_t> delimiters(static_cast<size_t>(rowCounts[p]) + 1);
            for (int r = 0; r <= rowCounts[p]; ++r) {
                delimiters[static_cast<size_t>(r)] =
                    globalRows[first + static_cast<index_t>(r)] - globalRows[first];
            }
            if (p == 0) h_rowDelimiters.swap(delimiters);
            else MPI_Send(delimiters.data(), rowCounts[p] + 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(h_rowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<double> h_out(localRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(globalVal.data(), globalCols.data(), globalRows.data(),
                h_vec.data(), numRows, h_reference.data());
    }
    if (rank == 0) {
        // Release the root-only global matrix before benchmark execution.
        std::vector<double>().swap(globalVal);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRows);
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> globalOut;
    if (rank == 0 && (validate || printResults)) globalOut.resize(numRows);
    if (validate || printResults) {
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOut.data() : nullptr, rowCounts.data(), rowDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Calculate performance metrics
    if (rank == 0) {
        const double milliseconds = duration * 1000.0;
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = milliseconds / static_cast<double>(iterations);
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalOut, "OutputVector");
    }

    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), globalOut.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
