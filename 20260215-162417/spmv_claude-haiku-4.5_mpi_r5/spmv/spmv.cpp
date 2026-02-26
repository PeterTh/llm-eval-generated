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
// Function: spmvCpuLocal
//
// Purpose:
//   Computes sparse matrix-vector multiplication for a local subset of rows
//
// Arguments:
//   val: array holding the non-zero values for the local matrix rows
//   cols: array of column indices for each element
//   rowDelimiters: array of size (numLocalRows+1) holding indices to rows
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of columns in the matrix (global)
//   numLocalRows: number of rows assigned to this process
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpuLocal(const double* val, const index_t* cols, const index_t* rowDelimiters,
                  const double* vec, const index_t, const index_t numLocalRows, double* out) {
    for (index_t i = 0; i < numLocalRows; ++i) {
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

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpi_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark [MPI]\n");
        printf("MPI processes: %d\n", mpi_size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute rows evenly across processes
    index_t localNumRows = numRows / mpi_size;
    index_t remainder = numRows % mpi_size;
    if (mpi_rank < (int)remainder) {
        localNumRows++;
    }
    
    index_t startRow = 0;
    for (int i = 0; i < mpi_rank; ++i) {
        index_t rows = numRows / mpi_size;
        if (i < (int)remainder) rows++;
        startRow += rows;
    }
    index_t endRow = startRow + localNumRows;

    // Master process allocates and initializes full matrix
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(localNumRows);
    std::vector<double> h_reference;

    if (mpi_rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast input vector to all processes
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast row delimiters to all processes so they can determine local nonzeros
    std::vector<index_t> global_rowDelimiters(numRows + 1);
    if (mpi_rank == 0) {
        std::copy(h_rowDelimiters.begin(), h_rowDelimiters.end(), global_rowDelimiters.begin());
    }
    MPI_Bcast(global_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Calculate local nonzeros on all processes
    index_t localNonzeros = global_rowDelimiters[endRow] - global_rowDelimiters[startRow];

    std::vector<double> local_val(localNonzeros);
    std::vector<index_t> local_cols(localNonzeros);
    std::vector<index_t> local_rowDelimiters(localNumRows + 1);

    // Each process extracts its own local matrix data
    if (localNonzeros > 0 && mpi_rank == 0) {
        std::copy(h_val.begin() + global_rowDelimiters[startRow],
                 h_val.begin() + global_rowDelimiters[endRow],
                 local_val.begin());
        std::copy(h_cols.begin() + global_rowDelimiters[startRow],
                 h_cols.begin() + global_rowDelimiters[endRow],
                 local_cols.begin());
    }
    for (index_t i = 0; i <= localNumRows; ++i) {
        local_rowDelimiters[i] = global_rowDelimiters[startRow + i] - global_rowDelimiters[startRow];
    }

    // Broadcast full sparse matrix to all processes (only rank 0 has it)
    if (mpi_rank == 0) {
        MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    } else {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    // Each process extracts its own local matrix data
    if (localNonzeros > 0) {
        std::copy(h_val.begin() + global_rowDelimiters[startRow],
                 h_val.begin() + global_rowDelimiters[endRow],
                 local_val.begin());
        std::copy(h_cols.begin() + global_rowDelimiters[startRow],
                 h_cols.begin() + global_rowDelimiters[endRow],
                 local_cols.begin());
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (mpi_rank == 0) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpuLocal(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                     h_vec.data(), numRows, localNumRows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather results on master process for validation and printing
    std::vector<double> h_out_global;
    std::vector<int> recvCounts(mpi_size);
    std::vector<int> displs(mpi_size);

    if (mpi_rank == 0) {
        h_out_global.resize(numRows);
        int offset = 0;
        for (int i = 0; i < mpi_size; ++i) {
            index_t rows = numRows / mpi_size;
            if (i < (int)remainder) rows++;
            recvCounts[i] = rows;
            displs[i] = offset;
            offset += rows;
        }
    }

    MPI_Gatherv(h_out.data(), localNumRows, MPI_DOUBLE,
                h_out_global.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(h_out_global, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out_global.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
