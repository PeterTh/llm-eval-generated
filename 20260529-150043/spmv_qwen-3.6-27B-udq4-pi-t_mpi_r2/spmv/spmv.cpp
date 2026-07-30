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

// Compute row range [start, end) for a given rank in a block distribution
static void computeRowRange(int rank, int numRanks, index_t numRows,
                            index_t& rowStart, index_t& rowEnd) {
    if (numRanks <= static_cast<int>(numRows)) {
        index_t block = numRows / numRanks;
        index_t remainder = numRows % numRanks;
        rowStart = rank * block + (rank < static_cast<int>(remainder) ? rank : remainder);
        rowEnd = rowStart + block + (rank < static_cast<int>(remainder) ? 1 : 0);
    } else {
        rowStart = static_cast<index_t>(rank);
        rowEnd = rank < static_cast<int>(numRows) ? static_cast<index_t>(rank + 1) : static_cast<index_t>(rank);
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int mpiErr = MPI_Init(&argc, &argv);
    if (mpiErr != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed with error %d\n", mpiErr);
        return 1;
    }

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

    // =========================================================================
    // Compute row partitioning: block distribution of rows across ranks
    // =========================================================================
    index_t localRowStart, localRowEnd;
    computeRowRange(rank, numRanks, numRows, localRowStart, localRowEnd);
    index_t localNumRows = localRowEnd - localRowStart;

    // =========================================================================
    // Shared data structures (declared outside if/else for both branches)
    // =========================================================================
    std::vector<double> h_vec(numRows);
    std::vector<index_t> globalRowDelimiters(numRows + 1, 0);

    // =========================================================================
    // Build partitioning info: send counts and displacements for Scatterv
    // Every rank computes this since it depends only on global parameters.
    // =========================================================================
    std::vector<int> sendcounts(numRanks, 0);
    std::vector<int> displs(numRanks, 0);
    for (int r = 0; r < numRanks; ++r) {
        index_t rStart, rEnd;
        computeRowRange(r, numRanks, numRows, rStart, rEnd);
        // Will be filled after globalRowDelimiters is known
    }

    // =========================================================================
    // Rank 0: initialize all data, then distribute to all ranks
    // =========================================================================
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI parallel: %d rank(s)\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");

        // Allocate and initialize data structures on rank 0
        std::vector<double> h_val(nItems);
        std::vector<index_t> h_cols(nItems);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), globalRowDelimiters.data(), nItems, numRows);

        // Recompute sendcounts/displs now that globalRowDelimiters is populated
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rEnd;
            computeRowRange(r, numRanks, numRows, rStart, rEnd);
            sendcounts[r] = static_cast<int>(globalRowDelimiters[rEnd] - globalRowDelimiters[rStart]);
            displs[r] = static_cast<int>(globalRowDelimiters[rStart]);
        }

        // =========================================================================
        // Compute reference solution on rank 0 if validation is enabled
        // =========================================================================
        std::vector<double> h_reference;
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), globalRowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // =========================================================================
        // Broadcast the vector (all ranks need it for column lookups)
        // =========================================================================
        MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // =========================================================================
        // Broadcast global rowDelimiters so every rank knows the full layout
        // =========================================================================
        MPI_Bcast(globalRowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        // =========================================================================
        // Scatter val and cols to all ranks using Scatterv
        // =========================================================================
        index_t localNnz = globalRowDelimiters[localRowEnd] - globalRowDelimiters[localRowStart];
        std::vector<double> localVal(localNnz);
        std::vector<index_t> localCols(localNnz);

        MPI_Scatterv(h_val.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        MPI_Scatterv(h_cols.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                     0, MPI_COMM_WORLD);

        // =========================================================================
        // Build flat buffer for all local rowDelimiters and scatter
        // =========================================================================
        std::vector<index_t> allLocalRdFlat;
        std::vector<int> rdCounts(numRanks);
        std::vector<int> rdDispls(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rEnd;
            computeRowRange(r, numRanks, numRows, rStart, rEnd);
            rdDispls[r] = static_cast<int>(allLocalRdFlat.size());
            index_t rBase = globalRowDelimiters[rStart];
            for (index_t i = 0; i <= rEnd - rStart; ++i) {
                allLocalRdFlat.push_back(globalRowDelimiters[rStart + i] - rBase);
            }
            rdCounts[r] = static_cast<int>(rEnd - rStart + 1);
        }

        int localRdSize = static_cast<int>(localRowEnd - localRowStart + 1);
        std::vector<index_t> localRowDelimiters(localRdSize);
        MPI_Scatterv(allLocalRdFlat.data(), rdCounts.data(), rdDispls.data(), MPI_UINT32_T,
                     localRowDelimiters.data(), localRdSize, MPI_UINT32_T,
                     0, MPI_COMM_WORLD);

        // =========================================================================
        // Perform SpMV computation on local data
        // =========================================================================
        printf("Computing SpMV...\n");

        std::vector<double> localOut(localNumRows);

        // Synchronize all ranks before timing
        MPI_Barrier(MPI_COMM_WORLD);
        double startTime = MPI_Wtime();

        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                    h_vec.data(), localNumRows, localOut.data());
        }

        double endTime = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        double duration = endTime - startTime;

        // =========================================================================
        // Gather results back to rank 0
        // =========================================================================
        std::vector<double> h_out(numRows);
        std::vector<int> recvcounts(numRanks);
        std::vector<int> recvDispls(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rEnd;
            computeRowRange(r, numRanks, numRows, rStart, rEnd);
            recvcounts[r] = static_cast<int>(rEnd - rStart);
            recvDispls[r] = static_cast<int>(rStart);
        }

        MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                    h_out.data(), recvcounts.data(), recvDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // =========================================================================
        // Print timing and performance (only rank 0)
        // =========================================================================
        const double gflops = (2.0 * nItems * iterations) / duration / 1e9;
        const double avgTime = (duration / static_cast<double>(iterations)) * 1000.0;

        printf("Computation time: %.3f ms\n", duration * 1000.0);
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
    } else {
        // =========================================================================
        // Non-root ranks: receive broadcasted data
        // =========================================================================
        MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(globalRowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        // Recompute sendcounts/displs (same logic, now that globalRowDelimiters is known)
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rEnd;
            computeRowRange(r, numRanks, numRows, rStart, rEnd);
            sendcounts[r] = static_cast<int>(globalRowDelimiters[rEnd] - globalRowDelimiters[rStart]);
            displs[r] = static_cast<int>(globalRowDelimiters[rStart]);
        }

        // Receive local val and cols
        index_t localNnz = globalRowDelimiters[localRowEnd] - globalRowDelimiters[localRowStart];
        std::vector<double> localVal(localNnz);
        std::vector<index_t> localCols(localNnz);

        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                     localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                     0, MPI_COMM_WORLD);

        // Receive local row delimiters
        {
            std::vector<int> rdCounts(numRanks);
            std::vector<int> rdDispls(numRanks);
            // Build counts and displs for the flat rowDelimiters buffer
            {
                size_t offset = 0;
                for (int r = 0; r < numRanks; ++r) {
                    index_t rStart, rEnd;
                    computeRowRange(r, numRanks, numRows, rStart, rEnd);
                    rdDispls[r] = static_cast<int>(offset);
                    rdCounts[r] = static_cast<int>(rEnd - rStart + 1);
                    offset += rdCounts[r];
                }
            }

            int localRdSize = static_cast<int>(localRowEnd - localRowStart + 1);
            std::vector<index_t> localRowDelimiters(localRdSize);
            MPI_Scatterv(nullptr, rdCounts.data(), rdDispls.data(), MPI_UINT32_T,
                         localRowDelimiters.data(), localRdSize, MPI_UINT32_T,
                         0, MPI_COMM_WORLD);

            // =========================================================================
            // Perform SpMV computation on local data
            // =========================================================================
            std::vector<double> localOut(localNumRows);

            // Synchronize all ranks before timing
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Wtime(); // start timing

            for (index_t iter = 0; iter < iterations; ++iter) {
                spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                        h_vec.data(), localNumRows, localOut.data());
            }

            MPI_Wtime(); // end timing
            MPI_Barrier(MPI_COMM_WORLD);

            // =========================================================================
            // Gather results back to rank 0
            // =========================================================================
            std::vector<int> recvcounts(numRanks);
            std::vector<int> recvDispls(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                index_t rStart, rEnd;
                computeRowRange(r, numRanks, numRows, rStart, rEnd);
                recvcounts[r] = static_cast<int>(rEnd - rStart);
                recvDispls[r] = static_cast<int>(rStart);
            }

            MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                        nullptr, recvcounts.data(), recvDispls.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
