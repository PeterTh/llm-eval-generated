#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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
//   Computes sparse matrix-vector multiplication using CSR format for a
//   local row partition. Each rank processes its assigned rows.
//
// Arguments:
//   val:       array holding the non-zero values for the local partition
//   cols:      array of column indices for each element (local partition)
//   localRD:   array of size local_nrows+1 with local row delimiters
//              (offsets into local val/cols arrays)
//   vec:       dense vector of size dim (full vector, broadcast to all ranks)
//   local_nrows: number of local rows on this rank
//   out:       output - result for the local rows
//
// ****************************************************************************
void spmvLocal(const double* val, const index_t* cols, const index_t* localRD,
               const double* vec, const index_t local_nrows, double* out) {
    for (index_t i = 0; i < local_nrows; ++i) {
        double t = 0.0;
        for (index_t j = localRD[i]; j < localRD[i + 1]; ++j) {
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0, then broadcast
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Print benchmark info only on rank 0
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark (MPI: %d ranks)\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Data generation on rank 0
    // =========================================================================
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // =========================================================================
    // Compute row distribution: block distribution of rows across ranks
    // =========================================================================
    // rowsPerRank[i] = number of rows assigned to rank i
    // rowStart[i]    = global start row index for rank i
    std::vector<index_t> rowsPerRank(numRanks);
    std::vector<index_t> rowStart(numRanks);
    {
        index_t base = numRows / numRanks;
        index_t remainder = numRows % numRanks;
        index_t cumulative = 0;
        for (int r = 0; r < numRanks; ++r) {
            rowsPerRank[r] = base + (static_cast<index_t>(r) < remainder ? 1 : 0);
            rowStart[r] = cumulative;
            cumulative += rowsPerRank[r];
        }
    }

    // Compute per-rank nnz counts from the CSR structure
    std::vector<index_t> local_nnz(numRanks, 0);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            local_nnz[r] = h_rowDelimiters[rowStart[r] + rowsPerRank[r]] - h_rowDelimiters[rowStart[r]];
        }
    }

    // Broadcast local_nnz so every rank knows how much to receive
    MPI_Bcast(local_nnz.data(), numRanks, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Distribute data: val, cols, and row delimiters via MPI_Scatterv
    // =========================================================================
    const index_t my_nnz = local_nnz[rank];
    const index_t my_nrows = rowsPerRank[rank];

    // Allocate local data
    std::vector<double> local_val(my_nnz);
    std::vector<index_t> local_cols(my_nnz);
    // local_rowDelimiters has size my_nrows+1, but we store global offsets
    // then convert to local offsets later
    std::vector<index_t> local_rowDelimiters(my_nrows + 1);

    // Build scatterv counts and displacements for val and cols
    std::vector<int> sendcounts_val(numRanks);
    std::vector<int> displs_val(numRanks);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            sendcounts_val[r] = local_nnz[r];
            displs_val[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
        }
    }

    // Scatter val
    MPI_Scatterv(h_val.data(), sendcounts_val.data(), displs_val.data(), MPI_DOUBLE,
                 local_val.data(), my_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter cols
    MPI_Scatterv(h_cols.data(), sendcounts_val.data(), displs_val.data(), MPI_UINT32_T,
                 local_cols.data(), my_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // For row delimiters: each rank needs (my_nrows+1) delimiters.
    // Rank r needs delimiters [rowStart[r], rowStart[r]+my_nrows] from global array.
    {
        std::vector<int> rd_counts(numRanks);
        std::vector<int> rd_displs(numRanks);
        if (rank == 0) {
            for (int r = 0; r < numRanks; ++r) {
                rd_counts[r] = rowsPerRank[r] + 1;
                rd_displs[r] = static_cast<int>(rowStart[r]);
            }
        }
        MPI_Scatterv(h_rowDelimiters.data(), rd_counts.data(), rd_displs.data(), MPI_UINT32_T,
                     local_rowDelimiters.data(), my_nrows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        // Convert global offsets to local offsets (relative to this rank's data)
        const index_t base_offset = local_rowDelimiters[0];
        for (index_t i = 0; i <= my_nrows; ++i) {
            local_rowDelimiters[i] -= base_offset;
        }
    }

    // =========================================================================
    // Broadcast the full input vector to all ranks (needed for column lookups)
    // =========================================================================
    std::vector<double> local_vec(numRows);
    if (rank == 0) {
        MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(local_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        local_vec = h_vec;
    }

    // =========================================================================
    // Compute reference solution on rank 0 if validation is enabled
    // =========================================================================
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        // Inline reference SpMV (sequential, on full data)
        for (index_t i = 0; i < numRows; ++i) {
            double t = 0.0;
            for (index_t j = h_rowDelimiters[i]; j < h_rowDelimiters[i + 1]; ++j) {
                t += h_val[j] * h_vec[h_cols[j]];
            }
            h_reference[i] = t;
        }
    }

    // =========================================================================
    // Parallel SpMV benchmark
    // =========================================================================
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> local_out(my_nrows);
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocal(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                  local_vec.data(), my_nrows, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long localDuration = static_cast<long long>(duration.count());
    long long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Gather results to rank 0
    // =========================================================================
    std::vector<double> h_out(numRows);
    {
        std::vector<int> recvcounts(numRanks);
        std::vector<int> recvdispls(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvcounts[r] = rowsPerRank[r];
            recvdispls[r] = static_cast<int>(rowStart[r]);
        }
        MPI_Gatherv(local_out.data(), my_nrows, MPI_DOUBLE,
                    h_out.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // =========================================================================
    // Report results on rank 0
    // =========================================================================
    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxDuration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDuration / 1000.0) / 1e9;
        const double avgTime = maxDuration / static_cast<double>(iterations);

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
