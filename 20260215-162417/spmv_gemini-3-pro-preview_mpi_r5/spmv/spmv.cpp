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
//   skip: number of random values to skip before filling
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal, const index_t skip = 0) {
    for (index_t i = 0; i < skip; ++i) {
        rand();
    }
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
//   cols:          vector to fill with column indexes of elements
//   rowDelimiters: vector to fill with row delimiters
//   n:             number of nonzero elements (global)
//   dim:           number of rows/columns in the matrix
//   startRow:      first row index for this process
//   endRow:        last row index (exclusive) for this process
//   nnzOffset:     output parameter for number of nonzeros before startRow
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters, const index_t n, const index_t dim, const index_t startRow, const index_t endRow, index_t& nnzOffset) {
    index_t nnzAssigned = 0;
    index_t localNnz = 0;
    nnzOffset = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    
    // We must simulate the RNG for all rows up to endRow to maintain consistency
    for (index_t i = 0; i < endRow; ++i) {
        if (i >= startRow) {
            rowDelimiters[i - startRow] = localNnz;
        }
        
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (i >= startRow) {
                    cols.push_back(j);
                    localNnz++;
                } else {
                    nnzOffset++;
                }
                nnzAssigned++;
            }
        }
    }
    // Set the final delimiter
    rowDelimiters[endRow - startRow] = localNnz;
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
            printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Processes: %d\n", size);
    }

    // Determine local row range
    index_t rowsPerProc = numRows / size;
    index_t startRow = rank * rowsPerProc;
    index_t endRow = (rank == size - 1) ? numRows : startRow + rowsPerProc;
    index_t localNumRows = endRow - startRow;

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;
    
    // Allocate and initialize data structures
    // h_val and h_cols will be sized dynamically based on local NNZ
    std::vector<double> h_val;                  // Non-zero values
    std::vector<index_t> h_cols;                // Column indices
    std::vector<index_t> h_rowDelimiters(localNumRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);         // Dense vector (replicated)
    std::vector<double> h_out(localNumRows);    // Output vector (local part)

    if (rank == 0) printf("Initializing data structures...\n");
    
    // Initialize vector (replicated on all ranks with same seed)
    srand(42); // Seed for vector fill
    fill(h_vec.data(), numRows, maxVal);

    // Initialize matrix (distributed)
    // We need to know how many nonzeros to skip for values
    index_t nnzOffset = 0;
    initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows, startRow, endRow, nnzOffset);
    
    index_t localNnz = h_cols.size();
    h_val.resize(localNnz);
    
    // Initialize values with correct offset in RNG sequence
    // Note: initRandomMatrix re-seeded with 8675309. 
    // We should probably use a different seed for values or reset.
    // The original code:
    // fill(h_vec, ...); // uses rand()
    // fill(h_val, ...); // uses rand() continue from above
    // initRandomMatrix(...); // resets srand(8675309)
    
    // Wait, original code:
    // fill(h_vec...);
    // fill(h_val...);
    // initRandomMatrix(...);
    
    // So h_vec and h_val used the initial seed (whatever it was, 1 usually, or not set). 
    // initRandomMatrix explicitly sets srand(8675309).
    // To match original behavior for h_vec and h_val, we should maintain that.
    // However, for parallel consistency, let's explicit seed for vector and values.
    
    // The original code didn't seed for fill(), so it used default seed 1.
    // I'll use srand(1) for h_vec and h_val generation to match default behavior if I can.
    // But h_val generation needs to be distributed.
    
    // For h_vec:
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    
    // For h_val:
    // We need to skip values that belong to previous ranks.
    // But `fill` just calls rand(). 
    // We need to know how many times rand() was called for h_vec. -> numRows times.
    // Then we need to skip `nnzOffset` times.
    fill(h_val.data(), localNnz, maxVal, nnzOffset);

    // For validation, compute reference solution (on rank 0 only)
    std::vector<double> h_reference;
    std::vector<double> h_val_ref;
    std::vector<index_t> h_cols_ref;
    std::vector<index_t> h_rowDelimiters_ref;
    
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        
        // We need the full matrix for reference.
        // Re-generate it. 
        // Note: this is expensive but safe.
        // Or we can gather the distributed matrix. 
        // Re-generating is easier to implement right now.
        
        index_t dummyOffset;
        h_cols_ref.reserve(nItems); // approximate
        h_rowDelimiters_ref.resize(numRows + 1);
        
        initRandomMatrix(h_cols_ref, h_rowDelimiters_ref, nItems, numRows, 0, numRows, dummyOffset);
        
        h_val_ref.resize(h_cols_ref.size());
        srand(1);
        fill(h_vec.data(), numRows, maxVal); // Re-fill vec to reset RNG state correctly or just skip
        fill(h_val_ref.data(), h_cols_ref.size(), maxVal); // Fill all values
        
        spmvCpu(h_val_ref.data(), h_cols_ref.data(), h_rowDelimiters_ref.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localNumRows, h_out.data());
        
        // In a real application, we might need to Allgather the result vector 
        // if the next iteration depends on it (e.g., in iterative solvers).
        // The benchmark loop just repeats SpMV with the same input vector.
        // So we don't strictly need communication *inside* the loop for this specific benchmark code
        // because h_vec is not updated.
        // However, if we want to measure "distributed SpMV" time, usually that includes 
        // communication if it's part of a solver step.
        // But here, h_vec is const. So no communication needed per iteration.
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results for validation/printing
    std::vector<double> global_out;
    if (rank == 0) {
        global_out.resize(numRows);
    }
    
    // Gather the partial results
    // We need to use MPI_Gatherv because counts might differ if numRows % size != 0
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int local_count = static_cast<int>(localNumRows);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(h_out.data(), local_count, MPI_DOUBLE, 
                global_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(global_out, "OutputVector");
    }

    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
