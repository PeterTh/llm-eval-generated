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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0)
    if (mpi_rank == 0) {
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

    // Broadcast configuration parameters to all processes
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpi_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI processes: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Calculate row distribution
    index_t local_rows = numRows / mpi_size;
    index_t row_start = mpi_rank * local_rows;
    index_t row_end = (mpi_rank == mpi_size - 1) ? numRows : row_start + local_rows;
    local_rows = row_end - row_start;

    // Rank 0 initializes the full matrix, then distributes
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;

    if (mpi_rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast dense vector to all processes
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute sparse matrix structure to all processes
    // Each process needs its portion of the CSR data

    std::vector<index_t> local_rowDelimiters;
    std::vector<index_t> local_cols;
    std::vector<double> local_val;

    if (mpi_rank == 0) {
        // Calculate how many non-zeros each process gets
        std::vector<index_t> nnz_per_process(mpi_size, 0);
        std::vector<index_t> row_starts(mpi_size);
        
        row_starts[0] = 0;
        for (int p = 1; p < mpi_size; ++p) {
            row_starts[p] = (p < mpi_size - 1) ? (p * numRows / mpi_size) : (numRows - ((mpi_size - 1) * numRows / mpi_size));
        }
        // Actually calculate proper boundaries
        row_starts.clear();
        row_starts.resize(mpi_size);
        for (int p = 0; p < mpi_size; ++p) {
            row_starts[p] = p * (numRows / mpi_size);
        }

        for (int p = 0; p < mpi_size; ++p) {
            index_t p_row_start = row_starts[p];
            index_t p_row_end = (p == mpi_size - 1) ? numRows : row_starts[p] + (numRows / mpi_size);
            nnz_per_process[p] = h_rowDelimiters[p_row_end] - h_rowDelimiters[p_row_start];
        }

        // Send to each process
        for (int p = 0; p < mpi_size; ++p) {
            index_t p_row_start = row_starts[p];
            index_t p_row_end = (p == mpi_size - 1) ? numRows : row_starts[p] + (numRows / mpi_size);
            index_t p_local_rows = p_row_end - p_row_start;
            index_t p_nnz = nnz_per_process[p];

            if (p == 0) {
                // Keep data on rank 0
                local_rowDelimiters.resize(p_local_rows + 1);
                local_cols.resize(p_nnz);
                local_val.resize(p_nnz);

                for (index_t i = 0; i <= p_local_rows; ++i) {
                    local_rowDelimiters[i] = h_rowDelimiters[p_row_start + i] - h_rowDelimiters[p_row_start];
                }
                std::copy(h_cols.begin() + h_rowDelimiters[p_row_start],
                         h_cols.begin() + h_rowDelimiters[p_row_end],
                         local_cols.begin());
                std::copy(h_val.begin() + h_rowDelimiters[p_row_start],
                         h_val.begin() + h_rowDelimiters[p_row_end],
                         local_val.begin());
            } else {
                // Send to other ranks
                index_t* send_cols = &h_cols[h_rowDelimiters[p_row_start]];
                double* send_val = &h_val[h_rowDelimiters[p_row_start]];

                // Send row count and nnz count first
                MPI_Send(&p_local_rows, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
                MPI_Send(&p_nnz, 1, MPI_UINT32_T, p, 1, MPI_COMM_WORLD);
                
                // Send arrays
                std::vector<index_t> adjusted_rowDel(p_local_rows + 1);
                for (index_t i = 0; i <= p_local_rows; ++i) {
                    adjusted_rowDel[i] = h_rowDelimiters[p_row_start + i] - h_rowDelimiters[p_row_start];
                }
                MPI_Send(adjusted_rowDel.data(), p_local_rows + 1, MPI_UINT32_T, p, 2, MPI_COMM_WORLD);
                MPI_Send(send_cols, p_nnz, MPI_UINT32_T, p, 3, MPI_COMM_WORLD);
                MPI_Send(send_val, p_nnz, MPI_DOUBLE, p, 4, MPI_COMM_WORLD);
            }
        }
    } else {
        // Receive data
        index_t p_local_rows, p_nnz;
        MPI_Recv(&p_local_rows, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&p_nnz, 1, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_rowDelimiters.resize(p_local_rows + 1);
        local_cols.resize(p_nnz);
        local_val.resize(p_nnz);

        MPI_Recv(local_rowDelimiters.data(), p_local_rows + 1, MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_cols.data(), p_nnz, MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_val.data(), p_nnz, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_rows = p_local_rows;
    }

    // Allocate output vectors
    std::vector<double> h_out(local_rows);
    std::vector<double> global_out;
    if (mpi_rank == 0) {
        global_out.resize(numRows);
    }

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (mpi_rank == 0) {
        printf("Computing SpMV...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), local_rows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results
    std::vector<index_t> local_rows_per_process(mpi_size);
    MPI_Gather(&local_rows, 1, MPI_UINT32_T, 
               local_rows_per_process.data(), 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        // Gather results from all processes
        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size, 0);
        for (int p = 0; p < mpi_size; ++p) {
            recvcounts[p] = local_rows_per_process[p];
            if (p > 0) {
                displs[p] = displs[p - 1] + recvcounts[p - 1];
            }
        }

        MPI_Gatherv(h_out.data(), local_rows, MPI_DOUBLE,
                   global_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(global_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    } else {
        MPI_Gatherv(h_out.data(), local_rows, MPI_DOUBLE,
                   nullptr, nullptr, nullptr, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();
    
    if (mpi_rank == 0) {
        if (validate) {
            const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);
            return valid ? 0 : 1;
        }
    }

    return 0;
}
