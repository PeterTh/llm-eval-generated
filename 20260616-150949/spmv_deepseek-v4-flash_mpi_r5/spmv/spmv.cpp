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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int parseError = 0;

    // Parse command line arguments on rank 0
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
                parseError = -1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseError = 1;
            }
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseError, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (parseError != 0) {
        MPI_Finalize();
        return (parseError < 0) ? 0 : parseError;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Only rank 0 prints status messages
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark (MPI)\n");
        printf("Processes: %d\n", nprocs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates all data
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;
    std::vector<double> h_reference;

    if (rank == 0) {
        printf("Initializing data structures...\n");

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

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

    // Determine row distribution across processes
    const index_t rows_per_proc_base = numRows / nprocs;
    const index_t rows_remainder = numRows % nprocs;

    // Compute row counts and offsets for all ranks
    std::vector<int> proc_row_counts(nprocs);
    std::vector<int> proc_row_offsets(nprocs);
    {
        int accum = 0;
        for (int r = 0; r < nprocs; ++r) {
            int cnt = static_cast<int>(rows_per_proc_base) + (r < static_cast<int>(rows_remainder) ? 1 : 0);
            proc_row_counts[r] = cnt;
            proc_row_offsets[r] = accum;
            accum += cnt;
        }
    }

    const index_t local_num_rows = static_cast<index_t>(proc_row_counts[rank]);

    // Broadcast h_vec to all ranks (every rank needs the full vector)
    std::vector<double> local_vec(numRows);
    if (rank == 0) {
        std::copy(h_vec.begin(), h_vec.end(), local_vec.begin());
    }
    MPI_Bcast(local_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prepare scatter parameters on rank 0 and broadcast to all ranks
    std::vector<int> sendcounts_val(nprocs, 0);
    std::vector<int> displs_val(nprocs, 0);
    std::vector<int> sendcounts_delim(nprocs, 0);
    std::vector<int> displs_delim(nprocs, 0);

    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            index_t r_offset = static_cast<index_t>(proc_row_offsets[r]);
            index_t r_rows = static_cast<index_t>(proc_row_counts[r]);
            index_t start_nnz = h_rowDelimiters[r_offset];
            index_t end_nnz = h_rowDelimiters[r_offset + r_rows];
            index_t local_nnz = end_nnz - start_nnz;

            sendcounts_val[r] = static_cast<int>(local_nnz);
            displs_val[r] = static_cast<int>(start_nnz);
            sendcounts_delim[r] = static_cast<int>(r_rows + 1);
            displs_delim[r] = static_cast<int>(r_offset);
        }
    }

    // Broadcast scatter parameters so all ranks know their local nnz
    MPI_Bcast(sendcounts_val.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs_val.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendcounts_delim.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs_delim.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t local_nnz = static_cast<index_t>(sendcounts_val[rank]);

    // Scatter matrix values
    std::vector<double> local_val(local_nnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 sendcounts_val.data(), displs_val.data(),
                 MPI_DOUBLE, local_val.data(), static_cast<int>(local_nnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Scatter column indices
    std::vector<index_t> local_cols(local_nnz);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 sendcounts_val.data(), displs_val.data(),
                 MPI_UINT32_T, local_cols.data(), static_cast<int>(local_nnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Scatter row delimiters (global indices for each rank's chunk)
    std::vector<index_t> local_rowDelimiters(local_num_rows + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 sendcounts_delim.data(), displs_delim.data(),
                 MPI_UINT32_T, local_rowDelimiters.data(), static_cast<int>(local_num_rows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Adjust local row delimiters to be relative to local data
    if (local_num_rows > 0) {
        index_t base = local_rowDelimiters[0];
        for (index_t i = 0; i <= local_num_rows; ++i) {
            local_rowDelimiters[i] -= base;
        }
    }

    // Local output buffer
    std::vector<double> local_out(local_num_rows);

    // Barrier before timing to synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    const double start_time = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                local_vec.data(), local_num_rows, local_out.data());
    }

    const double end_time = MPI_Wtime();
    const double local_duration_ms = (end_time - start_time) * 1000.0;

    // Get max wall time across all ranks (true parallel runtime)
    double global_duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(local_num_rows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                proc_row_counts.data(), proc_row_offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 prints results and performs validation
    if (rank == 0) {
        const long long duration_ms = static_cast<long long>(global_duration_ms);

        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (global_duration_ms / 1000.0) / 1e9;
        const double avgTime = global_duration_ms / static_cast<double>(iterations);

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
