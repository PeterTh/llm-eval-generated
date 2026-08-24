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
// Function: spmvMpi
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format with MPI
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array holding indices to rows for local rows
//   vec: dense vector of size dim to be used for multiplication
//   dim: total number of rows/columns in the matrix
//   localRowStart: starting row index for this process
//   localNumRows: number of rows this process handles
//   out: output - result from the spmv calculation (local portion)
//
// ****************************************************************************
void spmvMpi(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, const index_t localRowStart,
             const index_t localNumRows, double* out) {
    for (index_t i = 0; i < localNumRows; ++i) {
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
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI Processes: %d\n", numProcs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures on rank 0
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);
        
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    
    // Broadcast the vector to all processes
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Determine row distribution
    index_t localNumRows = numRows / numProcs;
    index_t remainder = numRows % numProcs;
    index_t localRowStart = rank * localNumRows + std::min(static_cast<index_t>(rank), remainder);
    
    if (rank < static_cast<int>(remainder)) {
        localNumRows++;
    }
    
    // Calculate local non-zero elements for each process
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowDispls(numProcs);
    std::vector<int> nnzCounts(numProcs, 0);
    std::vector<int> nnzDispls(numProcs, 0);
    
    if (rank == 0) {
        for (int p = 0; p < numProcs; ++p) {
            index_t pLocalNumRows = numRows / numProcs;
            index_t pRemainder = numRows % numProcs;
            index_t pLocalRowStart = p * pLocalNumRows + std::min(static_cast<index_t>(p), pRemainder);
            
            if (p < static_cast<int>(pRemainder)) {
                pLocalNumRows++;
            }
            
            rowCounts[p] = pLocalNumRows;
            rowDispls[p] = pLocalRowStart;
            
            // Calculate nnz for this process
            nnzCounts[p] = h_rowDelimiters[pLocalRowStart + pLocalNumRows] - h_rowDelimiters[pLocalRowStart];
            nnzDispls[p] = h_rowDelimiters[pLocalRowStart];
        }
    }
    
    // Broadcast counts and displacements
    MPI_Bcast(nnzCounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate local storage
    const index_t localNnz = nnzCounts[rank];
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localNumRows + 1);
    std::vector<double> local_out(localNumRows);
    
    // Scatter the matrix data
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), 
                 MPI_DOUBLE, local_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), 
                 MPI_UNSIGNED, local_cols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Prepare and scatter row delimiters (need localNumRows + 1 entries per process)
    std::vector<index_t> tempRowDelims;
    if (rank == 0) {
        tempRowDelims.resize(numRows + numProcs);
        int offset = 0;
        for (int p = 0; p < numProcs; ++p) {
            index_t pLocalNumRows = numRows / numProcs;
            index_t pRemainder = numRows % numProcs;
            index_t pLocalRowStart = p * pLocalNumRows + std::min(static_cast<index_t>(p), pRemainder);
            
            if (p < static_cast<int>(pRemainder)) {
                pLocalNumRows++;
            }
            
            index_t baseNnz = h_rowDelimiters[pLocalRowStart];
            for (index_t i = 0; i <= pLocalNumRows; ++i) {
                tempRowDelims[offset + i] = h_rowDelimiters[pLocalRowStart + i] - baseNnz;
            }
            offset += pLocalNumRows + 1;
        }
    }
    
    std::vector<int> rowDelimCounts(numProcs);
    std::vector<int> rowDelimDispls(numProcs);
    int offset = 0;
    for (int p = 0; p < numProcs; ++p) {
        index_t pLocalNumRows = numRows / numProcs;
        index_t pRemainder = numRows % numProcs;
        if (p < static_cast<int>(pRemainder)) {
            pLocalNumRows++;
        }
        rowDelimCounts[p] = pLocalNumRows + 1;
        rowDelimDispls[p] = offset;
        offset += pLocalNumRows + 1;
    }
    
    MPI_Scatterv(rank == 0 ? tempRowDelims.data() : nullptr, rowDelimCounts.data(), rowDelimDispls.data(),
                 MPI_UNSIGNED, local_rowDelimiters.data(), localNumRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvMpi(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), numRows, localRowStart, localNumRows, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localDurationMs = std::chrono::duration<double, std::milli>(end - start).count();
    double durationMs = 0.0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    MPI_Gatherv(local_out.data(), localNumRows, MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", durationMs);
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);
        
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
