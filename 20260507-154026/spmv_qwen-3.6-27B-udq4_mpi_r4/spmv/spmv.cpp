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
// Function: spmvLocal
//
// Purpose:
//   Computes sparse matrix-vector multiplication using local CSR format.
//   Each MPI rank computes its assigned rows independently.
//
// Arguments:
//   val: array holding the non-zero values for the local rows
//   cols: array of column indices for each element
//   rowDelimiters: local row delimiters (0-based, size = localRows+1)
//   vec: dense vector of size dim (broadcast to all ranks)
//   localRows: number of local rows
//   out: output - result from the spmv calculation (local rows only)
//
// ****************************************************************************
void spmvLocal(const double* val, const index_t* cols, const index_t* rowDelimiters,
               const double* vec, const index_t localRows, double* out) {
    for (index_t i = 0; i < localRows; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
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

// Helper: compute row start/end for a given rank with block distribution
static inline void compute_row_bounds(int rank, int world_size, index_t numRows,
                                       index_t& row_start, index_t& row_end) {
    const index_t baseRows = numRows / world_size;
    const index_t remainder = numRows % world_size;
    if (rank < static_cast<int>(remainder)) {
        row_start = static_cast<index_t>(rank * (baseRows + 1));
        row_end = row_start + baseRows + 1;
    } else {
        row_start = static_cast<index_t>(remainder * (baseRows + 1) + (rank - remainder) * baseRows);
        row_end = row_start + baseRows;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Block distribution of rows across MPI ranks
    index_t myRowStart, myRowEnd;
    compute_row_bounds(world_rank, world_size, numRows, myRowStart, myRowEnd);
    const index_t myNumRows = myRowEnd - myRowStart;

    // Print benchmark info (rank 0 only)
    if (world_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0: allocate and initialize the full matrix + vector
    std::vector<double> full_val;
    std::vector<index_t> full_cols;
    std::vector<index_t> full_rowDelimiters;
    std::vector<double> full_vec;

    if (world_rank == 0) {
        full_val.resize(nItems);
        full_cols.resize(nItems);
        full_rowDelimiters.resize(numRows + 1);
        full_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(full_vec.data(), numRows, maxVal);
        fill(full_val.data(), nItems, maxVal);
        initRandomMatrix(full_cols.data(), full_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast rowDelimiters to all ranks (needed for local delimiter computation)
    std::vector<index_t> global_rowDelimiters(numRows + 1);
    if (world_rank == 0) {
        global_rowDelimiters = full_rowDelimiters;
    }
    MPI_Bcast(global_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Compute per-rank nnz counts and displacements for Scatterv (all ranks)
    std::vector<int> recvcounts(world_size, 0);
    std::vector<int> displs(world_size, 0);
    for (int r = 0; r < world_size; ++r) {
        index_t rStart, rEnd;
        compute_row_bounds(r, world_size, numRows, rStart, rEnd);
        recvcounts[r] = static_cast<int>(global_rowDelimiters[rEnd] - global_rowDelimiters[rStart]);
        displs[r] = (r > 0) ? displs[r - 1] + recvcounts[r - 1] : 0;
    }

    // Allocate local CSR storage
    const index_t myNnz = static_cast<index_t>(recvcounts[world_rank]);
    std::vector<double> local_val(myNnz);
    std::vector<index_t> local_cols(myNnz);

    // Scatter val and cols from rank 0 to all ranks
    if (world_rank == 0) {
        MPI_Scatterv(full_val.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     local_val.data(), static_cast<int>(myNnz), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(full_cols.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                     local_cols.data(), static_cast<int>(myNnz), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                     local_val.data(), static_cast<int>(myNnz), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, recvcounts.data(), displs.data(), MPI_UNSIGNED,
                     local_cols.data(), static_cast<int>(myNnz), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Build local row delimiters (0-based, relative to scattered data)
    std::vector<index_t> local_rowDelimiters(myNumRows + 1);
    {
        const index_t global_base = global_rowDelimiters[myRowStart];
        for (index_t i = 0; i <= myNumRows; ++i) {
            local_rowDelimiters[i] = global_rowDelimiters[myRowStart + i] - global_base;
        }
    }

    // Broadcast input vector to all ranks (all ranks need it for column lookups)
    std::vector<double> local_vec(numRows);
    if (world_rank == 0) {
        MPI_Bcast(full_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(local_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (world_rank == 0) {
        local_vec = std::move(full_vec);
    }

    // Allocate local output
    std::vector<double> local_out(myNumRows);

    // For validation: compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && world_rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        std::vector<double> ref_out(numRows);
        // Use full CSR data for reference
        for (index_t i = 0; i < numRows; ++i) {
            double t = 0.0;
            for (index_t j = full_rowDelimiters[i]; j < full_rowDelimiters[i + 1]; ++j) {
                t += full_val[j] * local_vec[full_cols[j]];
            }
            ref_out[i] = t;
        }
        h_reference = std::move(ref_out);
    }

    // Perform SpMV computation
    if (world_rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocal(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                  local_vec.data(), myNumRows, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results back to rank 0
    std::vector<double> h_out(numRows);
    std::vector<int> out_recvcounts(world_size);
    std::vector<int> out_displs(world_size);
    if (world_rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            index_t rStart, rEnd;
            compute_row_bounds(r, world_size, numRows, rStart, rEnd);
            out_recvcounts[r] = static_cast<int>(rEnd - rStart);
            out_displs[r] = static_cast<int>(rStart);
        }
        MPI_Gatherv(local_out.data(), static_cast<int>(myNumRows), MPI_DOUBLE,
                    h_out.data(), out_recvcounts.data(), out_displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local_out.data(), static_cast<int>(myNumRows), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Report results (rank 0 only)
    if (world_rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        const double gflops = (2.0 * nItems * iterations) / (max_duration_ms / 1000.0) / 1e9;
        const double avgTime = max_duration_ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

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
