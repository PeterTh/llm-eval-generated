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
//                  last element is the index one past the the last element
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
//   Computes sparse matrix-vector multiplication using CSR format for
//   a local row partition. The rowDelimiters are shifted so that
//   rowDelimiters[0] is the global starting index into the val/cols arrays.
//
// Arguments:
//   val: array holding the non-zero values for the matrix (full global)
//   cols: array of column indices for each element (full global)
//   localRowDelimiters: local row delimiters for this rank's rows;
//                       first entry is the global start index into val/cols
//                       last entry is the global end index into val/cols
//   vec: dense vector of size dim (full global)
//   localDim: number of rows in this rank's partition
//   out: output buffer for this rank's local rows
//
// ****************************************************************************
void spmvCpuLocal(const double* val, const index_t* cols,
                  const index_t* localRowDelimiters,
                  const double* vec, const index_t localDim, double* out) {
    for (index_t i = 0; i < localDim; ++i) {
        double t = 0.0;
        for (index_t j = localRowDelimiters[i]; j < localRowDelimiters[i + 1]; ++j) {
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Print configuration (rank 0 only)
    if (rank == 0) {
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

    // =========================================================================
    // Phase 1: Data initialization on rank 0, then distribute
    // =========================================================================

    // Each rank computes its local row partition
    // Use block distribution: rank r gets rows [r*blockSize, (r+1)*blockSize)
    // with possible remainder on last ranks
    const index_t baseRows = numRows / static_cast<index_t>(numRanks);
    const index_t remainder = numRows % static_cast<index_t>(numRanks);
    index_t localDim = baseRows + (static_cast<index_t>(rank) < remainder ? 1 : 0);

    // Rank 0 initializes the full matrix, then scatters to all ranks
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
    // Phase 2: Distribute data to all ranks
    // =========================================================================

    // Broadcast the vector (all ranks need it for SpMV)
    h_vec.resize(numRows);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute send counts and displacements for rowDelimiters, val, cols
    // Each rank needs: its local portion of rowDelimiters (localDim+1 entries)
    //                  and the corresponding val/cols entries

    // Build send counts for val/cols per rank
    std::vector<int> sendCountsVal(numRanks);
    std::vector<int> sendCountsRowDel(numRanks);
    std::vector<int> displsVal(numRanks);
    std::vector<int> displsRowDel(numRanks);

    if (rank == 0) {
        // Compute per-rank sizes
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart = r * baseRows + std::min(static_cast<index_t>(r), remainder);
            index_t rEnd = (r + 1) * baseRows + std::min(static_cast<index_t>(r + 1), remainder);
            index_t rLocalDim = rEnd - rStart;

            // Row delimiters: need rLocalDim+1 entries (start..end inclusive)
            sendCountsRowDel[r] = static_cast<int>(rLocalDim + 1);
            // Displacement in rowDelimiters array = starting row index
            displsRowDel[r] = static_cast<int>(rStart);

            // Val/cols: entries from rowDelimiters[rStart] to rowDelimiters[rEnd]
            sendCountsVal[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
        }

        // Compute displacements for val/cols (cumulative)
        displsVal[0] = 0;
        for (int r = 1; r < numRanks; ++r) {
            displsVal[r] = displsVal[r - 1] + sendCountsVal[r - 1];
        }
    }

    // Broadcast send counts to all ranks
    MPI_Bcast(sendCountsVal.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendCountsRowDel.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);

    int localNnz = sendCountsVal[rank];

    // Allocate local storage
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(localDim + 1);

    // Scatterv the data from rank 0
    if (rank == 0) {
        MPI_Scatterv(h_rowDelimiters.data(), sendCountsRowDel.data(), displsRowDel.data(),
                     MPI_UNSIGNED,
                     localRowDelimiters.data(), sendCountsRowDel[rank], MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(h_val.data(), sendCountsVal.data(), displsVal.data(),
                     MPI_DOUBLE,
                     localVal.data(), sendCountsVal[rank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(h_cols.data(), sendCountsVal.data(), displsVal.data(),
                     MPI_UNSIGNED,
                     localCols.data(), sendCountsVal[rank], MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_UNSIGNED,
                     localRowDelimiters.data(), sendCountsRowDel[rank], MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_DOUBLE,
                     localVal.data(), sendCountsVal[rank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_UNSIGNED,
                     localCols.data(), sendCountsVal[rank], MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Offset local row delimiters to be relative to local val/cols arrays
    // localRowDelimiters[0] is the global start index; subtract it from all entries
    const index_t rowDelOffset = localRowDelimiters[0];
    for (index_t i = 0; i <= localDim; ++i) {
        localRowDelimiters[i] -= rowDelOffset;
    }

    // =========================================================================
    // Phase 3: Compute reference solution on rank 0 (if validation requested)
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // =========================================================================
    // Phase 4: Perform SpMV computation (timed)
    // =========================================================================

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Allocate local output buffer
    std::vector<double> localOut(localDim);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Use MPI_Wtime for consistent timing across ranks
    double startTime = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpuLocal(localVal.data(), localCols.data(), localRowDelimiters.data(),
                     h_vec.data(), localDim, localOut.data());
    }

    double endTime = MPI_Wtime();

    // Synchronize all ranks after timing
    MPI_Barrier(MPI_COMM_WORLD);

    double elapsed = endTime - startTime;

    // =========================================================================
    // Phase 5: Gather results and compute timing
    // =========================================================================

    // Gather local outputs to rank 0
    std::vector<double> h_out(numRows);

    // Build recv counts for output gather
    std::vector<int> recvCountsOut(numRanks);
    std::vector<int> recvDisplsOut(numRanks);

    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart = r * baseRows + std::min(static_cast<index_t>(r), remainder);
            index_t rEnd = (r + 1) * baseRows + std::min(static_cast<index_t>(r + 1), remainder);
            recvCountsOut[r] = static_cast<int>(rEnd - rStart);
        }
        recvDisplsOut[0] = 0;
        for (int r = 1; r < numRanks; ++r) {
            recvDisplsOut[r] = recvDisplsOut[r - 1] + recvCountsOut[r - 1];
        }
    }

    MPI_Gatherv(localOut.data(), localDim, MPI_DOUBLE,
                h_out.data(), recvCountsOut.data(), recvDisplsOut.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // =========================================================================
    // Phase 6: Print results (rank 0 only)
    // =========================================================================

    if (rank == 0) {
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = (elapsed / static_cast<double>(iterations)) * 1000.0;

        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
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

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
