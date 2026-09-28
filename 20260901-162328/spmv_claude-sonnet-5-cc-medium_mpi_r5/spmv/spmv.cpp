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

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits `dim` matrix rows as evenly as possible across `numRanks` MPI
//   ranks using contiguous row blocks (any leftover rows go to the first
//   ranks). Returns the row counts and starting row offsets per rank.
//
// ****************************************************************************
void computeRowPartition(index_t dim, int numRanks, std::vector<index_t>& rowCounts,
                          std::vector<index_t>& rowStarts) {
    rowCounts.assign(numRanks, 0);
    rowStarts.assign(numRanks, 0);
    const index_t base = dim / static_cast<index_t>(numRanks);
    const index_t extra = dim % static_cast<index_t>(numRanks);
    index_t offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        rowCounts[r] = base + (static_cast<index_t>(r) < extra ? 1 : 0);
        rowStarts[r] = offset;
        offset += rowCounts[r];
    }
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
            if (isRoot) {
                printUsage(argv[0]);
            }
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
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Row partition: contiguous row blocks, distributed as evenly as possible
    std::vector<index_t> rowCounts;
    std::vector<index_t> rowStarts;
    computeRowPartition(numRows, numRanks, rowCounts, rowStarts);
    const index_t localNumRows = rowCounts[rank];

    // Full data structures only allocated (and populated) on the root rank;
    // the dense vector is broadcast to all ranks since every row may
    // reference any column.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_out;
    std::vector<double> h_reference;
    std::vector<double> h_vec(numRows);

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast the dense vector to every rank
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter each rank's slice of the row delimiters (rebased to start at 0)
    std::vector<int> rdCounts(numRanks);
    std::vector<int> rdDispls(numRanks);
    std::vector<int> nnzCounts(numRanks);
    std::vector<int> nnzDispls(numRanks);
    if (isRoot) {
        for (int r = 0; r < numRanks; ++r) {
            rdCounts[r] = static_cast<int>(rowCounts[r] + 1);
            rdDispls[r] = static_cast<int>(rowStarts[r]);
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowStarts[r] + rowCounts[r]] -
                                             h_rowDelimiters[rowStarts[r]]);
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStarts[r]]);
        }
    }

    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, rdCounts.data(), rdDispls.data(),
                 MPI_UINT32_T, localRowDelimiters.data(), static_cast<int>(localNumRows + 1),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Broadcast the per-rank nnz counts/displacements so every rank knows how
    // many non-zeros it will receive.
    int localNnz = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Rebase local row delimiters so they start at 0
    const index_t rdOffset = localRowDelimiters[0];
    for (index_t i = 0; i <= localNumRows; ++i) {
        localRowDelimiters[i] -= rdOffset;
    }

    std::vector<double> localOut(localNumRows);

    // Perform SpMV computation (each rank operates on its own row block; the
    // dense vector never changes between iterations, so no communication is
    // needed inside the timed loop).
    if (isRoot) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localNumRows, localOut.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);
    long globalDurationMs = 0;
    MPI_Reduce(&durationMs, &globalDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed output vector back onto the root rank
    std::vector<int> outCounts(numRanks);
    std::vector<int> outDispls(numRanks);
    if (isRoot) {
        for (int r = 0; r < numRanks; ++r) {
            outCounts[r] = static_cast<int>(rowCounts[r]);
            outDispls[r] = static_cast<int>(rowStarts[r]);
        }
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, outCounts.data(), outDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", globalDurationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int validationResult = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                validationResult = 0;
            } else {
                printf("Validation: FAILED\n");
                validationResult = 1;
            }
        }
        MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validationResult;
    }

    MPI_Finalize();
    return 0;
}
