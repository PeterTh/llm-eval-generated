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

    // Compute row distribution across processes
    int* recvcounts = new int[nprocs];
    int* displs = new int[nprocs];
    {
        int offset = 0;
        const int rem = static_cast<int>(numRows % nprocs);
        const int base = static_cast<int>(numRows / nprocs);
        for (int i = 0; i < nprocs; ++i) {
            recvcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }
    const int my_rows = recvcounts[rank];
    const int my_offset = displs[rank];

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
    }

    // Rank 0 generates full matrix data; others allocate for receive
    std::vector<double> h_val_full;
    std::vector<index_t> h_cols_full;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        h_val_full.resize(nItems);
        h_cols_full.resize(nItems);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val_full.data(), nItems, maxVal);
        initRandomMatrix(h_cols_full.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast dense vector and row delimiters to all processes
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T,
              0, MPI_COMM_WORLD);

    // Compute scatter counts/displacements for val and cols based on row distribution
    int* val_counts = new int[nprocs];
    int* val_displs = new int[nprocs];
    for (int i = 0; i < nprocs; ++i) {
        const int end_row = (i < nprocs - 1) ? displs[i + 1] : static_cast<int>(numRows);
        val_counts[i] = static_cast<int>(h_rowDelimiters[static_cast<index_t>(end_row)]
                                       - h_rowDelimiters[static_cast<index_t>(displs[i])]);
        val_displs[i] = static_cast<int>(h_rowDelimiters[static_cast<index_t>(displs[i])]);
    }

    // Each process receives only its local portion of val and cols
    std::vector<double> local_val(static_cast<size_t>(val_counts[rank]));
    std::vector<index_t> local_cols(static_cast<size_t>(val_counts[rank]));

    MPI_Scatterv(rank == 0 ? h_val_full.data() : nullptr,
                 val_counts, val_displs, MPI_DOUBLE,
                 local_val.data(), val_counts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols_full.data() : nullptr,
                 val_counts, val_displs, MPI_UINT32_T,
                 local_cols.data(), val_counts[rank], MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Build local zero-based row delimiters
    std::vector<index_t> local_rowDelimiters(static_cast<size_t>(my_rows) + 1);
    for (int i = 0; i < my_rows; ++i) {
        const index_t gi = static_cast<index_t>(my_offset + i);
        local_rowDelimiters[static_cast<size_t>(i)] =
            h_rowDelimiters[gi] - h_rowDelimiters[static_cast<index_t>(my_offset)];
    }
    local_rowDelimiters[static_cast<size_t>(my_rows)] =
        static_cast<index_t>(val_counts[rank]);

    // Local output
    std::vector<double> h_out_local(static_cast<size_t>(my_rows));

    // Reference solution computed sequentially on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val_full.data(), h_cols_full.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Timed parallel SpMV computation
    if (rank == 0) {
        printf("Computing SpMV using %d MPI processes...\n", nprocs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < my_rows; ++i) {
            double t = 0.0;
            const index_t row_start = local_rowDelimiters[static_cast<size_t>(i)];
            const index_t row_end   = local_rowDelimiters[static_cast<size_t>(i) + 1];
            for (index_t j = row_start; j < row_end; ++j) {
                t += local_val[j] * h_vec[local_cols[j]];
            }
            h_out_local[static_cast<size_t>(i)] = t;
        }
    }

    const double t_end = MPI_Wtime();
    const double local_time = t_end - t_start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print performance metrics from rank 0
    if (rank == 0) {
        const double duration_ms = max_time * 1000.0;
        printf("Computation time: %.0f ms\n", duration_ms);
        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations))
                               / max_time / 1e9;
        const double avgTime = duration_ms / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather results for validation or output
    std::vector<double> h_out_full;
    bool valid = true;
    if (validate || printResults) {
        if (rank == 0) h_out_full.resize(numRows);
        MPI_Gatherv(h_out_local.data(), my_rows, MPI_DOUBLE,
                    rank == 0 ? h_out_full.data() : nullptr,
                    recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(h_out_full, "OutputVector");
            }
            if (validate) {
                printf("Validating result...\n");
                valid = verifyResults(h_reference.data(), h_out_full.data(), numRows);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }

    int exit_code = (validate && rank == 0 && !valid) ? 1 : 0;

    delete[] recvcounts;
    delete[] displs;
    delete[] val_counts;
    delete[] val_displs;

    MPI_Finalize();
    return exit_code;
}
