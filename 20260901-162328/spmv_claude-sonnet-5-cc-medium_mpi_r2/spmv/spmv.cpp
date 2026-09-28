#include <mpi.h>

#include <algorithm>
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

// ****************************************************************************
// Function: computeRowRange
//
// Purpose:
//   Computes the contiguous, load-balanced block of matrix rows owned by a
//   given MPI rank. Rows are split as evenly as possible, with the first
//   (numRows % numRanks) ranks receiving one extra row.
//
// ****************************************************************************
void computeRowRange(const index_t numRows, const int numRanks, const int rank,
                      index_t& rowStart, index_t& rowEnd) {
    const index_t base = numRows / static_cast<index_t>(numRanks);
    const index_t remainder = numRows % static_cast<index_t>(numRanks);
    const index_t r = static_cast<index_t>(rank);
    rowStart = r * base + std::min(r, remainder);
    rowEnd = rowStart + base + (r < remainder ? 1 : 0);
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

    // Full matrix/vector data is only materialized on the root rank so that
    // the generated matrix is bit-identical to the sequential version
    // regardless of the number of ranks (rand() is inherently sequential).
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<double> h_vec(numRows);                 // Dense vector (needed in full by every rank)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (broadcast in full)
    std::vector<double> h_out;                          // Full output vector (root only)

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast the dense vector and row delimiters to every rank
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Determine this rank's contiguous block of rows
    index_t rowStart = 0, rowEnd = 0;
    computeRowRange(numRows, numRanks, rank, rowStart, rowEnd);
    const index_t localRows = rowEnd - rowStart;
    const index_t localNnz = h_rowDelimiters[rowEnd] - h_rowDelimiters[rowStart];

    // Compute per-rank send counts/displacements for scattering the non-zero
    // values and column indices (every rank can compute this locally since
    // the full row delimiters array has already been broadcast).
    std::vector<int> sendCounts(numRanks), sendDispls(numRanks);
    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        index_t rs, re;
        computeRowRange(numRows, numRanks, r, rs, re);
        rowCounts[r] = static_cast<int>(re - rs);
        rowDispls[r] = static_cast<int>(rs);
        sendCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
        sendDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
    }

    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                 local_val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, sendCounts.data(), sendDispls.data(), MPI_UINT32_T,
                 local_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Build local row delimiters, relative to this rank's local arrays
    std::vector<index_t> local_rowDelimiters(localRows + 1);
    const index_t baseOffset = h_rowDelimiters[rowStart];
    for (index_t i = 0; i < localRows; ++i) {
        local_rowDelimiters[i] = h_rowDelimiters[rowStart + i] - baseOffset;
    }
    local_rowDelimiters[localRows] = localNnz;

    std::vector<double> local_out(localRows);

    // For validation, compute reference solution sequentially on the root
    // rank, exactly as in the non-distributed version.
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), localRows, local_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    double elapsedSeconds = end - start;
    double maxElapsedSeconds = 0.0;
    MPI_Reduce(&elapsedSeconds, &maxElapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed result back into the full output vector on root
    MPI_Gatherv(local_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    bool valid = true;

    if (isRoot) {
        const long durationMs = static_cast<long>(maxElapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / maxElapsedSeconds / 1e9;
        const double avgTime = (maxElapsedSeconds * 1000.0) / static_cast<double>(iterations);

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

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    int exitCode = 0;
    if (validate) {
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
