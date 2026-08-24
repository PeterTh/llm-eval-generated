#include <mpi.h>

#include <algorithm>
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
//   Computes sparse matrix-vector multiplication using CSR format
//   (used for reference computation on rank 0)
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
// Function: spmvLocal
//
// Purpose:
//   Computes sparse matrix-vector multiplication for a local row partition
//   using CSR format. Each MPI rank owns a contiguous block of rows.
//
// Arguments:
//   val: local array of non-zero values
//   cols: local array of column indices
//   rowDelimiters: local row delimiters (adjusted to start from 0)
//   vec: full dense vector (broadcast to all ranks)
//   localRows: number of rows owned by this rank
//   out: local output vector
//
// ****************************************************************************
void spmvLocal(const double* val, const index_t* cols, const index_t* rowDelimiters,
               const double* vec, const index_t localRows, double* out) {
    for (index_t i = 0; i < localRows; ++i) {
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, same values)
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

    // Print benchmark info (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Data initialization on rank 0
    // =========================================================================
    std::vector<double> h_val(nItems);                  // Non-zero values (rank 0)
    std::vector<index_t> h_cols(nItems);                // Column indices (rank 0)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (rank 0)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // =========================================================================
    // Broadcast input vector to all ranks (needed for all column lookups)
    // =========================================================================
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Broadcast row delimiters to all ranks (small array, needed for partitioning)
    // =========================================================================
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Compute row distribution across ranks (balanced block distribution)
    // =========================================================================
    index_t rowsPerRank = numRows / numRanks;
    index_t extraRows = numRows % numRanks;

    // Rank r owns rows [localRowStart, localRowStart + rowsPerThisRank)
    index_t localRowStart = rank * rowsPerRank + std::min(static_cast<index_t>(rank), extraRows);
    index_t rowsPerThisRank = rowsPerRank + (static_cast<index_t>(rank) < extraRows ? 1 : 0);
    index_t localRowEnd = localRowStart + rowsPerThisRank;

    // =========================================================================
    // Compute scatter/gather counts and displacements (all ranks, same logic)
    // =========================================================================
    std::vector<int> sendCounts(numRanks);
    std::vector<int> displs(numRanks);

    for (int r = 0; r < numRanks; ++r) {
        index_t rStart = r * rowsPerRank + std::min(static_cast<index_t>(r), extraRows);
        index_t rCount = rowsPerRank + (static_cast<index_t>(r) < extraRows ? 1 : 0);
        index_t rEnd = rStart + rCount;

        index_t nnzStart = h_rowDelimiters[rStart];
        index_t nnzEnd = h_rowDelimiters[rEnd];

        sendCounts[r] = static_cast<int>(nnzEnd - nnzStart);
        displs[r] = static_cast<int>(nnzStart);
    }

    // =========================================================================
    // Scatter matrix data (val, cols) to each rank
    // =========================================================================
    index_t localNnzStart = h_rowDelimiters[localRowStart];
    index_t localNnzEnd = h_rowDelimiters[localRowEnd];
    index_t localNnz = localNnzEnd - localNnzStart;

    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);

    MPI_Scatterv(h_val.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 local_val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(h_cols.data(), sendCounts.data(), displs.data(), MPI_UNSIGNED,
                 local_cols.data(), static_cast<int>(localNnz), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Build local row delimiters (adjusted to start from 0)
    // =========================================================================
    std::vector<index_t> local_rowDelimiters(rowsPerThisRank + 1);
    for (index_t i = 0; i <= rowsPerThisRank; ++i) {
        local_rowDelimiters[i] = h_rowDelimiters[localRowStart + i] - localNnzStart;
    }

    // =========================================================================
    // Local output buffer
    // =========================================================================
    std::vector<double> local_out(rowsPerThisRank);

    // =========================================================================
    // Reference solution on rank 0 (for validation)
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // =========================================================================
    // Timing barrier + SpMV computation
    // =========================================================================
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocal(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                  h_vec.data(), rowsPerThisRank, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long localDurationMs = static_cast<long long>(duration.count());
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Gather results back to rank 0
    // =========================================================================
    std::vector<double> h_out(numRows);

    std::vector<int> recvCounts(numRanks);
    std::vector<int> recvDispls(numRanks);

    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart = r * rowsPerRank + std::min(static_cast<index_t>(r), extraRows);
            index_t rCount = rowsPerRank + (static_cast<index_t>(r) < extraRows ? 1 : 0);
            recvCounts[r] = static_cast<int>(rCount);
            recvDispls[r] = static_cast<int>(rStart);
        }
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(rowsPerThisRank), MPI_DOUBLE,
                h_out.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Performance reporting (rank 0 only)
    // =========================================================================
    if (rank == 0) {
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

        printf("Computation time: %lld ms\n", globalDurationMs);
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
            }
        }
    }

    MPI_Finalize();
    return 0;
}