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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution for this rank
    index_t rowsPerRank = numRows / size;
    index_t remainder = numRows % size;
    index_t startRow = rank * rowsPerRank + std::min(static_cast<index_t>(rank), remainder);
    index_t endRow = startRow + rowsPerRank + (rank < remainder ? 1 : 0);
    index_t localRows = endRow - startRow;
    
    // Full data structures on rank 0 for initialization
    std::vector<double> h_val_full;
    std::vector<index_t> h_cols_full;
    std::vector<index_t> h_rowDelimiters_full;
    std::vector<double> h_vec(numRows);  // All ranks need full vector
    
    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val_full.resize(nItems);
        h_cols_full.resize(nItems);
        h_rowDelimiters_full.resize(numRows + 1);
        
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val_full.data(), nItems, maxVal);
        initRandomMatrix(h_cols_full.data(), h_rowDelimiters_full.data(), nItems, numRows);
    }
    
    // Broadcast vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast row delimiters to all ranks (needed to compute local ranges)
    std::vector<index_t> h_rowDelimiters_full_bc(numRows + 1);
    if (rank == 0) {
        h_rowDelimiters_full_bc = h_rowDelimiters_full;
    }
    MPI_Bcast(h_rowDelimiters_full_bc.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Compute local data size
    index_t localStart = h_rowDelimiters_full_bc[startRow];
    index_t localEnd = h_rowDelimiters_full_bc[endRow];
    index_t localNnz = localEnd - localStart;
    
    // Allocate local data structures
    std::vector<double> h_val(localNnz);
    std::vector<index_t> h_cols(localNnz);
    std::vector<index_t> h_rowDelimiters(localRows + 1);
    std::vector<double> h_out(localRows);
    
    // Distribute matrix data
    if (rank == 0) {
        // Copy local portion for rank 0
        std::copy(h_val_full.begin() + localStart, h_val_full.begin() + localEnd, h_val.begin());
        std::copy(h_cols_full.begin() + localStart, h_cols_full.begin() + localEnd, h_cols.begin());
        for (index_t i = 0; i <= localRows; ++i) {
            h_rowDelimiters[i] = h_rowDelimiters_full_bc[startRow + i] - localStart;
        }
        
        // Send to other ranks
        for (int r = 1; r < size; ++r) {
            index_t rStartRow = r * rowsPerRank + std::min(static_cast<index_t>(r), remainder);
            index_t rEndRow = rStartRow + rowsPerRank + (r < remainder ? 1 : 0);
            index_t rLocalRows = rEndRow - rStartRow;
            index_t rLocalStart = h_rowDelimiters_full_bc[rStartRow];
            index_t rLocalEnd = h_rowDelimiters_full_bc[rEndRow];
            index_t rLocalNnz = rLocalEnd - rLocalStart;
            
            MPI_Send(&rLocalNnz, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
            MPI_Send(h_val_full.data() + rLocalStart, rLocalNnz, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
            MPI_Send(h_cols_full.data() + rLocalStart, rLocalNnz, MPI_UNSIGNED, r, 2, MPI_COMM_WORLD);
            
            std::vector<index_t> rRowDelimiters(rLocalRows + 1);
            for (index_t i = 0; i <= rLocalRows; ++i) {
                rRowDelimiters[i] = h_rowDelimiters_full_bc[rStartRow + i] - rLocalStart;
            }
            MPI_Send(rRowDelimiters.data(), rLocalRows + 1, MPI_UNSIGNED, r, 3, MPI_COMM_WORLD);
        }
    } else {
        // Receive from rank 0
        index_t recvLocalNnz;
        MPI_Recv(&recvLocalNnz, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_val.data(), recvLocalNnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_cols.data(), recvLocalNnz, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_rowDelimiters.data(), localRows + 1, MPI_UNSIGNED, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // For validation, compute reference solution (distributed)
    std::vector<double> h_reference_local;
    std::vector<double> h_reference_full;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference_local.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), localRows, h_reference_local.data());
        
        // Gather reference solution to rank 0
        if (rank == 0) {
            h_reference_full.resize(numRows);
        }
        
        std::vector<int> recvCounts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            index_t rStartRow = r * rowsPerRank + std::min(static_cast<index_t>(r), remainder);
            index_t rEndRow = rStartRow + rowsPerRank + (r < remainder ? 1 : 0);
            recvCounts[r] = rEndRow - rStartRow;
            displs[r] = rStartRow;
        }
        
        MPI_Gatherv(h_reference_local.data(), localRows, MPI_DOUBLE,
                    h_reference_full.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", globalDurationMs);
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results to rank 0
    std::vector<double> h_out_full;
    if (rank == 0) {
        h_out_full.resize(numRows);
    }
    
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        index_t rStartRow = r * rowsPerRank + std::min(static_cast<index_t>(r), remainder);
        index_t rEndRow = rStartRow + rowsPerRank + (r < remainder ? 1 : 0);
        recvCounts[r] = rEndRow - rStartRow;
        displs[r] = rStartRow;
    }
    
    MPI_Gatherv(h_out.data(), localRows, MPI_DOUBLE,
                h_out_full.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
        print_results(h_out_full, "OutputVector");
    }

    // Validation (rank 0 only)
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference_full.data(), h_out_full.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Finalize();
    return result;
}
