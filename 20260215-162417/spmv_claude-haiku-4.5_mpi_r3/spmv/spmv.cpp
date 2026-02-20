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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0)
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all processes
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

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
        printf("MPI processes: %d\n", size);
    }

    // Allocate data structures for all processes
    std::vector<double> h_vec(numRows);                 // Dense vector (same on all processes)
    std::vector<double> h_out(numRows);                 // Output vector (computed distributed)
    
    // Allocate matrix data (full on rank 0, distributed on others)
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    
    // Broadcast vector to all processes
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Calculate row distribution for each process
    index_t rowsPerProcess = numRows / size;
    index_t extraRows = numRows % size;
    
    // Calculate local row range for this process
    index_t localRowStart = rank * rowsPerProcess + std::min(static_cast<index_t>(rank), extraRows);
    index_t localRowEnd = localRowStart + rowsPerProcess + (rank < extraRows ? 1 : 0);
    index_t localNumRows = localRowEnd - localRowStart;
    
    // Prepare data for ScatterV: send each process its rows
    std::vector<int> sendCounts(size);
    std::vector<int> sendDispls(size);
    std::vector<int> valSendCounts(size);
    std::vector<int> valSendDispls(size);
    
    if (rank == 0) {
        index_t rowStart = 0;
        for (int p = 0; p < size; ++p) {
            index_t rowsForProcess = rowsPerProcess + (static_cast<index_t>(p) < extraRows ? 1 : 0);
            index_t rowEnd = rowStart + rowsForProcess;
            
            // Count non-zero elements in this process's rows
            index_t nnzCount = (rowEnd < numRows) ? h_rowDelimiters[rowEnd] : nItems;
            if (rowStart > 0) {
                nnzCount -= h_rowDelimiters[rowStart];
            }
            
            valSendCounts[p] = static_cast<int>(nnzCount);
            valSendDispls[p] = static_cast<int>(h_rowDelimiters[rowStart]);
            
            sendCounts[p] = static_cast<int>(rowsForProcess + 1);  // +1 for row delimiters
            sendDispls[p] = static_cast<int>(rowStart);
            
            rowStart = rowEnd;
        }
    }
    
    // Broadcast send counts to all processes
    MPI_Bcast(sendCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendDispls.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valSendCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valSendDispls.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Receive local row delimiters
    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, sendCounts.data(), sendDispls.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localNumRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    
    // Adjust row delimiters for local indexing
    if (rank == 0) {
        index_t baseIdx = h_rowDelimiters[0];
        for (index_t i = 0; i <= localNumRows; ++i) {
            localRowDelimiters[i] -= baseIdx;
        }
    } else {
        index_t baseIdx = localRowDelimiters[0];
        for (index_t i = 0; i <= localNumRows; ++i) {
            localRowDelimiters[i] -= baseIdx;
        }
    }
    
    // Receive local non-zero values and column indices
    std::vector<double> localVal(valSendCounts[rank]);
    std::vector<index_t> localCols(valSendCounts[rank]);
    
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, valSendCounts.data(), valSendDispls.data(), MPI_DOUBLE,
                 localVal.data(), valSendCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, valSendCounts.data(), valSendDispls.data(), MPI_UINT32_T,
                 localCols.data(), valSendCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation (local portion on each process)
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    
    // Allocate local output vector
    std::vector<double> localOut(localNumRows);
    
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localNumRows, localOut.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    // Gather results back to rank 0
    std::vector<int> gatherCounts(size);
    std::vector<int> gatherDispls(size);
    
    for (int p = 0; p < size; ++p) {
        index_t rowsForProcess = rowsPerProcess + (static_cast<index_t>(p) < extraRows ? 1 : 0);
        gatherCounts[p] = static_cast<int>(rowsForProcess);
        if (p == 0) {
            gatherDispls[p] = 0;
        } else {
            gatherDispls[p] = gatherDispls[p-1] + gatherCounts[p-1];
        }
    }
    
    MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                h_out.data(), gatherCounts.data(), gatherDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
