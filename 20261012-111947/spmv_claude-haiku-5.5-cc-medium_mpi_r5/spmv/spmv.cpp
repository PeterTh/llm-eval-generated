#include <mpi.h>

#include <chrono>
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
// Function: initRandomMatrixRows
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. Only the column indexes of the rows in
//   [rowBegin, rowEnd) are stored, so each MPI rank keeps just its slice.
//   The generated matrix is identical to the one produced for the full
//   matrix, because the random stream is consumed for every entry.
//
// Arguments:
//   cols:          receives the column indexes of nonzeros in rows
//                  [rowBegin, rowEnd), in CSR order
//   rowDelimiters: array of size dim+1 holding global indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row stored by this rank
//   rowEnd:        one past the last row stored by this rank
//
// ****************************************************************************
void initRandomMatrixRows(std::vector<index_t>& cols, index_t* rowDelimiters, const index_t n,
                          const index_t dim, const index_t rowBegin, const index_t rowEnd) {
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
                if (i >= rowBegin && i < rowEnd) {
                    cols.push_back(j);
                }
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
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically)
    int exitCode = 0;
    bool parsed = true;
    for (int i = 1; i < argc && parsed; ++i) {
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
            if (root) printUsage(argv[0]);
            parsed = false;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            parsed = false;
        }
    }
    if (!parsed) {
        MPI_Finalize();
        return exitCode;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block-row partition: rank r owns rows [rowDispls[r], rowDispls[r] + rowCounts[r])
    const index_t procs = static_cast<index_t>(numProcs);
    const index_t baseRows = numRows / procs;
    const index_t extraRows = numRows % procs;
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowDispls(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        const index_t owned = baseRows + (static_cast<index_t>(r) < extraRows ? 1 : 0);
        rowCounts[r] = static_cast<int>(owned);
        rowDispls[r] = (r == 0) ? 0 : rowDispls[r - 1] + rowCounts[r - 1];
    }
    const index_t rowBegin = static_cast<index_t>(rowDispls[rank]);
    const index_t rowEnd = rowBegin + static_cast<index_t>(rowCounts[rank]);
    const index_t localRows = rowEnd - rowBegin;

    // Allocate and initialize data structures
    if (root) printf("Initializing data structures...\n");

    // Every rank replays the serial program's random streams and keeps only the
    // slice it owns, so the distributed run sees the same matrix and vector.
    std::vector<index_t> globalRowDelimiters(numRows + 1);
    std::vector<index_t> h_cols;
    initRandomMatrixRows(h_cols, globalRowDelimiters.data(), nItems, numRows, rowBegin, rowEnd);

    // Local CSR row delimiters, rebased so that this rank's nonzeros start at zero
    const index_t valBegin = globalRowDelimiters[rowBegin];
    const index_t valEnd = globalRowDelimiters[rowEnd];
    std::vector<index_t> localRowDelimiters(localRows + 1);
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] = globalRowDelimiters[rowBegin + i] - valBegin;
    }

    // The vector and nonzero values use the default rand() stream (seed 1); the
    // nonzero values follow the vector entries, so skip to this rank's slice
    srand(1);
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    fill(h_vec.data(), numRows, maxVal);
    for (index_t k = 0; k < valBegin; ++k) {
        rand();
    }
    std::vector<double> h_val(valEnd - valBegin);       // Local non-zero values
    fill(h_val.data(), valEnd - valBegin, maxVal);

    // For validation, compute reference solution for the local rows
    std::vector<double> h_reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation on the local rows. The vector is replicated,
    // so the iteration loop requires no communication.
    if (root) printf("Computing SpMV...\n");
    std::vector<double> h_out(localRows);               // Local output rows
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    const long long localDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // The slowest rank determines the time of the parallel computation
    long long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %lld ms\n", duration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration / 1000.0) / 1e9;
        const double avgTime = duration / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the distributed results on the root for printing and validation
    const bool needGlobal = printResults || validate;
    std::vector<double> globalOut;
    std::vector<double> globalReference;
    if (needGlobal) {
        if (root) globalOut.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    globalOut.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    if (validate) {
        if (root) globalReference.resize(numRows);
        MPI_Gatherv(h_reference.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    globalReference.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (root && printResults) {
        print_results(globalOut, "OutputVector");
    }

    // Validation
    if (validate) {
        int valid = 1;
        if (root) {
            printf("Validating result...\n");
            valid = verifyResults(globalReference.data(), globalOut.data(), numRows) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (root) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
