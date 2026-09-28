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

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits the rows of the matrix into contiguous, near-equal blocks, one
//   per MPI rank. Computed identically (and independently) on every rank
//   so no communication is required to agree on the partition.
//
// Arguments:
//   numRows:     total number of matrix rows
//   numRanks:    number of MPI ranks
//   rowsPerRank: output - number of rows assigned to each rank
//   rowStart:    output - global index of the first row assigned to each rank
//
// ****************************************************************************
void computeRowPartition(const index_t numRows, const int numRanks,
                          std::vector<int>& rowsPerRank, std::vector<int>& rowStart) {
    rowsPerRank.assign(numRanks, 0);
    rowStart.assign(numRanks, 0);
    const index_t base = numRows / static_cast<index_t>(numRanks);
    const index_t remainder = numRows % static_cast<index_t>(numRanks);
    index_t offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        const index_t count = base + (static_cast<index_t>(r) < remainder ? 1 : 0);
        rowsPerRank[r] = static_cast<int>(count);
        rowStart[r] = static_cast<int>(offset);
        offset += count;
    }
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank receives the same argv from
    // the MPI launcher, so this can be done independently and without
    // communication).
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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-based partitioning: each rank owns a contiguous block of matrix
    // rows. Computed identically on every rank, so no communication is
    // needed to agree on it.
    std::vector<int> rowsPerRank, rowStart;
    computeRowPartition(numRows, numRanks, rowsPerRank, rowStart);
    const index_t numRowsLocal = static_cast<index_t>(rowsPerRank[rank]);

    // The dense vector is small (O(dim)) and read by every row regardless
    // of which rank owns it, so it is replicated on all ranks.
    std::vector<double> h_vec(numRows);

    // Non-zero counts (and byte offsets) per rank, needed to scatter the
    // CSR value/column arrays.
    std::vector<int> nnzPerRank(numRanks, 0);
    std::vector<int> nnzStart(numRanks, 0);

    // Full matrix data. Only rank 0 needs to hold it (it is generated
    // sequentially there and then distributed); other ranks leave these
    // empty to avoid replicating O(nnz) memory across the cluster.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_reference;

    if (isRoot) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        for (int r = 0; r < numRanks; ++r) {
            nnzStart[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
            nnzPerRank[r] = static_cast<int>(h_rowDelimiters[rowStart[r] + rowsPerRank[r]]) - nnzStart[r];
        }

        // For validation, compute reference solution on the full (still
        // undistributed) matrix, matching the original serial semantics.
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Distribute the dense vector and the per-rank non-zero counts.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzPerRank.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzStart.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t nItemsLocal = static_cast<index_t>(nnzPerRank[rank]);

    // Scatter the CSR value/column arrays: each rank receives exactly the
    // non-zeros belonging to the rows it owns.
    std::vector<double> local_val(nItemsLocal);
    std::vector<index_t> local_cols(nItemsLocal);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzPerRank.data(), nnzStart.data(), MPI_DOUBLE,
                 local_val.data(), static_cast<int>(nItemsLocal), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzPerRank.data(), nnzStart.data(), MPI_UINT32_T,
                 local_cols.data(), static_cast<int>(nItemsLocal), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Scatter row-start offsets (global, absolute nnz indices) for each row
    // owned by this rank, then rebuild a local, zero-based rowDelimiters
    // array of size numRowsLocal + 1.
    std::vector<index_t> local_rowDelimiters(numRowsLocal + 1);
    MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, rowsPerRank.data(), rowStart.data(), MPI_UINT32_T,
                 local_rowDelimiters.data(), static_cast<int>(numRowsLocal), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    local_rowDelimiters[numRowsLocal] = static_cast<index_t>(nnzPerRank[rank]);
    for (index_t i = 0; i < numRowsLocal; ++i) {
        local_rowDelimiters[i] -= static_cast<index_t>(nnzStart[rank]);
    }

    std::vector<double> local_out(numRowsLocal);
    std::vector<double> h_out;
    if (isRoot) h_out.resize(numRows);

    // Perform SpMV computation. Since the dense vector never changes
    // between iterations, no inter-rank communication is required inside
    // the timed loop, only local computation on each rank's row block.
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), numRowsLocal, local_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the per-rank partial outputs into the full result vector on
    // rank 0, only once after the timed loop (the intermediate values of
    // earlier iterations are never observed, matching the original code).
    MPI_Gatherv(local_out.data(), static_cast<int>(numRowsLocal), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowsPerRank.data(), rowStart.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    bool valid = true;
    if (isRoot) {
        printf("Computation time: %ld ms\n", maxDuration);

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
            valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    int exitCode = 0;
    if (validate) {
        MPI_Bcast(&valid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
