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

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse for consistency)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // Print benchmark parameters (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI Ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Data initialization on rank 0
    // =========================================================================
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // =========================================================================
    // Compute row partitioning across MPI ranks
    // =========================================================================
    const index_t rowsPerRank = numRows / numRanks;
    const index_t remainder = numRows % numRanks;

    // Local row range for this rank
    const index_t localRowStart = static_cast<index_t>(rank) * rowsPerRank +
                                  (rank < static_cast<int>(remainder) ? rank : remainder);
    const index_t localRowEnd = static_cast<index_t>(rank + 1) * rowsPerRank +
                                ((rank + 1) < static_cast<int>(remainder) ? rank + 1 : remainder);
    const index_t localNumRows = localRowEnd - localRowStart;

    // =========================================================================
    // Broadcast row delimiters so all ranks can compute their local partition
    // =========================================================================
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Local non-zero count from global row delimiters
    const index_t localNnz = h_rowDelimiters[localRowEnd] - h_rowDelimiters[localRowStart];

    // Build local row delimiters (shifted to local indices)
    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[localRowStart + i] - h_rowDelimiters[localRowStart];
    }

    // =========================================================================
    // Scatter CSR data (values and column indices) from rank 0
    // =========================================================================
    std::vector<int> localValCounts(numRanks);
    std::vector<int> localValDispls(numRanks);

    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            const index_t rStart = static_cast<index_t>(r) * rowsPerRank +
                                   (r < static_cast<int>(remainder) ? r : remainder);
            const index_t rEnd = static_cast<index_t>(r + 1) * rowsPerRank +
                                 ((r + 1) < static_cast<int>(remainder) ? r + 1 : remainder);

            localValCounts[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
            localValDispls[r] = static_cast<int>(h_rowDelimiters[rStart]);
        }
    }

    // Allocate local storage
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    // Scatter values
    MPI_Scatterv(h_val.data(), localValCounts.data(), localValDispls.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Scatter column indices
    MPI_Scatterv(h_cols.data(), localValCounts.data(), localValDispls.data(), MPI_UNSIGNED,
                 localCols.data(), static_cast<int>(localNnz), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // =========================================================================
    // Broadcast input vector to all ranks
    // =========================================================================
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================================================================
    // SpMV benchmark loop
    // =========================================================================
    std::vector<double> localOut(localNumRows);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localNumRows, localOut.data());
    }

    // Synchronize all ranks after timing
    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    // =========================================================================
    // Gather results to rank 0
    // =========================================================================
    std::vector<double> h_out(numRows);

    std::vector<int> recvCounts(numRanks);
    std::vector<int> recvDispls(numRanks);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            const index_t rStart = static_cast<index_t>(r) * rowsPerRank +
                                   (r < static_cast<int>(remainder) ? r : remainder);
            const index_t rEnd = static_cast<index_t>(r + 1) * rowsPerRank +
                                 ((r + 1) < static_cast<int>(remainder) ? r + 1 : remainder);
            recvCounts[r] = static_cast<int>(rEnd - rStart);
            recvDispls[r] = static_cast<int>(rStart);
        }
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                h_out.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // =========================================================================
    // Performance reporting (rank 0)
    // =========================================================================
    if (rank == 0) {
        const double elapsed = endTime - startTime;
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = elapsed / static_cast<double>(iterations) * 1000.0;

        printf("Computation time: %.3f s\n", elapsed);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> h_reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());

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
