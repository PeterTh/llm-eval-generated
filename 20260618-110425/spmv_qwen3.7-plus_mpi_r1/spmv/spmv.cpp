#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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

    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

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
            printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nRanks);
    }

    // Compute 1D row decomposition: distribute rows across ranks
    const index_t baseRows = numRows / nRanks;
    const index_t extraRows = numRows % nRanks;

    std::vector<index_t> rowsPerRank(nRanks);
    std::vector<index_t> startRowPerRank(nRanks + 1);
    startRowPerRank[0] = 0;
    for (int r = 0; r < nRanks; ++r) {
        rowsPerRank[r] = baseRows + (r < static_cast<int>(extraRows) ? 1 : 0);
        startRowPerRank[r + 1] = startRowPerRank[r] + rowsPerRank[r];
    }

    const index_t localNumRows = rowsPerRank[rank];

    // Rank 0 generates all data (preserves identical RNG sequence)
    std::vector<double> g_val;
    std::vector<index_t> g_cols;
    std::vector<index_t> g_rowDelimiters;
    std::vector<double> g_vec(numRows);  // All ranks need this allocated for Bcast

    if (rank == 0) {
        g_val.resize(nItems);
        g_cols.resize(nItems);
        g_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(g_vec.data(), numRows, maxVal);
        fill(g_val.data(), nItems, maxVal);
        initRandomMatrix(g_cols.data(), g_rowDelimiters.data(), nItems, numRows);
    }

    // Compute nnz counts and displacements for Scatterv of val/cols
    std::vector<int> nnzCounts(nRanks);
    std::vector<int> nnzDispls(nRanks);
    if (rank == 0) {
        for (int r = 0; r < nRanks; ++r) {
            nnzDispls[r] = static_cast<int>(g_rowDelimiters[startRowPerRank[r]]);
            nnzCounts[r] = static_cast<int>(g_rowDelimiters[startRowPerRank[r + 1]] - g_rowDelimiters[startRowPerRank[r]]);
        }
    }

    // Broadcast nnz distribution so all ranks know sizes
    MPI_Bcast(nnzCounts.data(), nRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), nRanks, MPI_INT, 0, MPI_COMM_WORLD);

    const int localNnz = nnzCounts[rank];

    // Allocate local CSR data
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rd(localNumRows + 1);
    std::vector<double> local_out(localNumRows);

    // Scatter val and cols arrays
    MPI_Scatterv(g_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 local_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UNSIGNED,
                 local_cols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Scatter rowDelimiters: each rank needs localNumRows+1 entries
    // Build expanded send buffer on rank 0 with boundary entries duplicated
    {
        std::vector<int> rdCounts(nRanks);
        std::vector<int> rdDispls(nRanks);
        int totalRd = 0;
        for (int r = 0; r < nRanks; ++r) {
            rdCounts[r] = static_cast<int>(rowsPerRank[r]) + 1;
            rdDispls[r] = totalRd;
            totalRd += rdCounts[r];
        }

        if (rank == 0) {
            std::vector<index_t> rdSendBuf(totalRd);
            int offset = 0;
            for (int r = 0; r < nRanks; ++r) {
                for (index_t j = 0; j <= rowsPerRank[r]; ++j) {
                    rdSendBuf[offset + j] = g_rowDelimiters[startRowPerRank[r] + j];
                }
                offset += rdCounts[r];
            }
            MPI_Scatterv(rdSendBuf.data(), rdCounts.data(), rdDispls.data(), MPI_UNSIGNED,
                         local_rd.data(), static_cast<int>(localNumRows) + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(nullptr, rdCounts.data(), rdDispls.data(), MPI_UNSIGNED,
                         local_rd.data(), static_cast<int>(localNumRows) + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
    }

    // Rebase local row delimiters so local_rd[0] = 0
    const index_t baseNnz = local_rd[0];
    for (index_t i = 0; i <= localNumRows; ++i) {
        local_rd[i] -= baseNnz;
    }

    // Broadcast the full dense vector to all ranks (needed for column lookups)
    MPI_Bcast(g_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double* vecPtr = g_vec.data();

    // Free rank-0-only global data to save memory
    if (rank == 0) {
        g_val.clear();
        g_val.shrink_to_fit();
        g_cols.clear();
        g_cols.shrink_to_fit();
        g_rowDelimiters.clear();
        g_rowDelimiters.shrink_to_fit();
    }

    // For validation, compute reference solution locally
    std::vector<double> local_ref;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        local_ref.resize(localNumRows);
        spmvCpu(local_val.data(), local_cols.data(), local_rd.data(),
                vecPtr, localNumRows, local_ref.data());
    }

    // Perform SpMV computation with MPI timing
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rd.data(),
                vecPtr, localNumRows, local_out.data());
    }

    double endTime = MPI_Wtime();

    // Get the maximum time across all ranks (wall-clock parallel time)
    double maxTime;
    MPI_Reduce(&endTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Also get min start time for accurate wall-clock duration
    double minStart;
    MPI_Reduce(&startTime, &minStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    double elapsedSec = 0.0;
    double gflops = 0.0;
    double avgTime = 0.0;

    if (rank == 0) {
        elapsedSec = maxTime - minStart;
        const long elapsedMs = static_cast<long>(elapsedSec * 1000.0);
        printf("Computation time: %ld ms\n", elapsedMs);

        gflops = (2.0 * nItems * iterations) / elapsedSec / 1e9;
        avgTime = elapsedSec / static_cast<double>(iterations) * 1000.0; // in ms

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Prepare counts/displs for Gatherv operations
    std::vector<int> outCounts(nRanks);
    std::vector<int> outDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        outCounts[r] = static_cast<int>(rowsPerRank[r]);
        outDispls[r] = static_cast<int>(startRowPerRank[r]);
    }

    // For print_results and validation, gather output to rank 0
    std::vector<double> gathered_out;
    if (printResults || validate) {
        if (rank == 0) {
            gathered_out.resize(numRows);
        }

        MPI_Gatherv(local_out.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                    gathered_out.data(), outCounts.data(), outDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            print_results(gathered_out, "OutputVector");
        }
    }

    // Validation
    int validationPassed = 0;  // 0 = pass
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        // Gather reference to rank 0 as well
        std::vector<double> gathered_ref;
        if (rank == 0) {
            gathered_ref.resize(numRows);
        }

        MPI_Gatherv(local_ref.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                    gathered_ref.data(), outCounts.data(), outDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            const bool valid = verifyResults(gathered_ref.data(), gathered_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                validationPassed = 0;
            } else {
                printf("Validation: FAILED\n");
                validationPassed = 1;
            }
        }

        // Broadcast validation result so all ranks return consistent exit code
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return validationPassed;
}
