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
//   Computes sparse matrix-vector multiplication using CSR format.
//   Each rank computes its local row block.
//
// Arguments:
//   val:           array holding the non-zero values for the local rows
//   cols:          array of column indices for each element
//   rowDelimiters: array of size local_dim+1 holding local indices to rows
//   vec:           dense vector of size global_dim (broadcast to all ranks)
//   local_dim:     number of local rows on this rank
//   out:           output - result for the local rows
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t local_dim, double* out) {
    for (index_t i = 0; i < local_dim; ++i) {
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
    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Compute block distribution of rows across ranks
    index_t base_rows = numRows / static_cast<index_t>(numRanks);
    index_t rem = numRows % static_cast<index_t>(numRanks);
    index_t local_num_rows = base_rows + (static_cast<index_t>(rank) < rem ? 1 : 0);

    // All ranks allocate the full input vector (needed for random column access)
    std::vector<double> vec(numRows);
    std::vector<double> local_out(local_num_rows);

    // Rank 0 generates and initializes all data
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI Ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast input vector to all ranks
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute scatter parameters for CSR data distribution
    std::vector<int> val_sendcounts(numRanks);
    std::vector<int> val_sdispls(numRanks);
    std::vector<int> rowDel_sendcounts(numRanks);
    std::vector<int> rowDel_sdispls(numRanks);

    if (rank == 0) {
        for (index_t r = 0; r < static_cast<index_t>(numRanks); ++r) {
            index_t r_start = r * base_rows + (r < rem ? r : rem);
            index_t r_end = r_start + base_rows + (r < rem ? 1 : 0);

            val_sendcounts[r] = static_cast<int>(h_rowDelimiters[r_end] - h_rowDelimiters[r_start]);
            val_sdispls[r] = static_cast<int>(h_rowDelimiters[r_start]);
            rowDel_sendcounts[r] = static_cast<int>(r_end - r_start + 1);
            rowDel_sdispls[r] = static_cast<int>(r_start);
        }
    }

    // Broadcast scatter parameters so every rank knows its allocation size
    MPI_Bcast(val_sendcounts.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(val_sdispls.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDel_sendcounts.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDel_sdispls.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);

    int local_nnz = val_sendcounts[rank];

    // Allocate local CSR data
    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);
    std::vector<index_t> local_rowDelimiters(local_num_rows + 1);

    // Scatter matrix values
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? val_sendcounts.data() : nullptr,
                 rank == 0 ? val_sdispls.data() : nullptr,
                 MPI_DOUBLE,
                 local_val.data(), local_nnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Scatter column indices
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? val_sendcounts.data() : nullptr,
                 rank == 0 ? val_sdispls.data() : nullptr,
                 MPI_UNSIGNED,
                 local_cols.data(), local_nnz, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Scatter row delimiters
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowDel_sendcounts.data() : nullptr,
                 rank == 0 ? rowDel_sdispls.data() : nullptr,
                 MPI_UNSIGNED,
                 local_rowDelimiters.data(), rowDel_sendcounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Adjust local row delimiters to 0-based local indexing
    if (local_num_rows > 0) {
        index_t base_idx = local_rowDelimiters[0];
        for (index_t i = 0; i <= local_num_rows; ++i) {
            local_rowDelimiters[i] -= base_idx;
        }
    }

    // Reference computation on rank 0 (for validation)
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                vec.data(), numRows, h_reference.data());
    }

    // Benchmark: parallel SpMV
    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                vec.data(), local_num_rows, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    const long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather output vector on rank 0
    if (rank == 0) {
        std::vector<int> recvcounts(numRanks);
        std::vector<int> recvdispls(numRanks);
        std::vector<double> h_out(numRows);

        for (index_t r = 0; r < static_cast<index_t>(numRanks); ++r) {
            index_t r_start = r * base_rows + (r < rem ? r : rem);
            index_t r_end = r_start + base_rows + (r < rem ? 1 : 0);
            recvcounts[r] = static_cast<int>(r_end - r_start);
            recvdispls[r] = static_cast<int>(r_start);
        }

        MPI_Gatherv(local_out.data(), static_cast<int>(local_num_rows), MPI_DOUBLE,
                    h_out.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        printf("Computation time: %ld ms\n", global_duration_ms);

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
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    } else {
        MPI_Gatherv(local_out.data(), static_cast<int>(local_num_rows), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
