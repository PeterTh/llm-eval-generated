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
//   Initialize array with random values. The full random sequence of n values
//   is always generated (so that every rank observes the identical stream of
//   pseudo random numbers), but only the values belonging to the half open
//   range [start, end) are stored, at A[i - start].
//
// Arguments:
//   A: pointer to the (partial) array to initialize, size end - start
//   n: number of elements in the global array
//   maxVal: specifies range of random values [0, maxVal]
//   start, end: range of global indices owned by this rank
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal, const index_t start, const index_t end) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= start && i < end) {
            A[i - start] = v;
        }
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
//   Every rank walks the identical random sequence and therefore reconstructs
//   the identical (global) row delimiter array, but only the column indices of
//   the rows it owns are stored.
//
// Arguments:
//   localCols:     receives the column indices of the locally owned rows
//   rowDelimiters: array of size dim+1 holding (global) indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowStart, rowEnd: range of rows owned by this rank
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& localCols, index_t* rowDelimiters, const index_t n,
                      const index_t dim, const index_t rowStart, const index_t rowEnd) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const bool owned = (i >= rowStart && i < rowEnd);
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (owned) {
                    localCols.push_back(j);
                }
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
//   Computes the locally owned rows of a sparse matrix-vector multiplication
//   using CSR format
//
// Arguments:
//   val: array holding the non-zero values of the locally owned rows
//   cols: array of column indices of the locally owned rows
//   rowDelimiters: array of size dim+1 holding global indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   rowStart, rowEnd: range of rows owned by this rank
//   out: output - result from the spmv calculation, size rowEnd - rowStart
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t rowStart, const index_t rowEnd, double* out) {
    // Local val/cols arrays start at the first non-zero of the first owned row
    const index_t nnzBase = rowDelimiters[rowStart];
    for (index_t i = rowStart; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i] - nnzBase; j < rowDelimiters[i + 1] - nnzBase; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i - rowStart] = t;
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute the rows of the matrix (and thus of the output vector) evenly
    // across all ranks. As the non-zeros are distributed uniformly at random,
    // an even row decomposition also balances the non-zeros very well.
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowOffsets(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t s = (static_cast<size_t>(r) * numRows) / numRanks;
        const size_t e = (static_cast<size_t>(r + 1) * numRows) / numRanks;
        rowOffsets[r] = static_cast<int>(s);
        rowCounts[r] = static_cast<int>(e - s);
    }
    const index_t rowStart = static_cast<index_t>(rowOffsets[rank]);
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t rowEnd = rowStart + localRows;

    // Allocate and initialize data structures
    std::vector<index_t> h_cols;                        // Column indices (local rows)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (global)
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_out(localRows);               // Output vector (local rows)

    if (rank == 0) {
        printf("Initializing data structures...\n");
    }

    // The matrix structure is generated first because it re-seeds the random
    // number generator anyway; knowing the row delimiters up front allows the
    // values of the locally owned rows to be picked out of the value stream in
    // a single pass.
    h_cols.reserve(static_cast<size_t>(nItems) / numRanks + numRows / numRanks + 64);
    initRandomMatrix(h_cols, h_rowDelimiters.data(), nItems, numRows, rowStart, rowEnd);

    const index_t nnzStart = h_rowDelimiters[rowStart];
    const index_t localNnz = h_rowDelimiters[rowEnd] - nnzStart;
    std::vector<double> h_val(localNnz);  // Non-zero values (local rows)

    // Reproduce the original random sequence: the default generator state is
    // equivalent to srand(1), the vector is drawn first, the matrix values second.
    srand(1);
    fill(h_vec.data(), numRows, maxVal, 0, numRows);
    fill(h_val.data(), nItems, maxVal, nnzStart, nnzStart + localNnz);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    std::vector<double> h_localReference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_localReference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), rowStart, rowEnd, h_localReference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // The full result vector is only assembled on rank 0
    std::vector<double> h_outGlobal(rank == 0 ? numRows : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), rowStart, rowEnd, h_out.data());
    }

    MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, h_outGlobal.data(),
                rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMs = 0.0;
    MPI_Allreduce(&localMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(elapsedMs);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_outGlobal, "OutputVector");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            h_reference.resize(numRows);
        }
        MPI_Gatherv(h_localReference.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_reference.data(), rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_outGlobal.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
