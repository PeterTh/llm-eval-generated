#include <chrono>
#include <cmath>
#include <cstdint>
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
//   Splits the rows into contiguous ranges whose non-zero counts are as
//   balanced as possible across the available processes.
//
// Arguments:
//   rowDelimiters: global CSR row delimiters of size dim+1
//   dim:           number of rows in the matrix
//   nprocs:        number of MPI processes
//   bounds:        output array of size nprocs+1 holding the row range bounds
//
// ****************************************************************************
void computeRowPartition(const index_t* rowDelimiters, const index_t dim,
                         const int nprocs, index_t* bounds) {
    const double totalNnz = static_cast<double>(rowDelimiters[dim]);

    bounds[0] = 0;
    bounds[nprocs] = dim;

    // If there are fewer rows than processes, fall back to an even row split;
    // some processes will legitimately own an empty range.
    if (dim < static_cast<index_t>(nprocs)) {
        for (int r = 1; r < nprocs; ++r) {
            bounds[r] = static_cast<index_t>(
                (static_cast<uint64_t>(r) * static_cast<uint64_t>(dim)) /
                static_cast<uint64_t>(nprocs));
        }
        return;
    }

    for (int r = 1; r < nprocs; ++r) {
        const double target = totalNnz * static_cast<double>(r) / static_cast<double>(nprocs);
        // Leave at least one row for every process that has not been served yet.
        const index_t maxRow = dim - static_cast<index_t>(nprocs - r);
        index_t b = bounds[r - 1];

        // Advance while moving past the next row brings the cumulative non-zero
        // count closer to this process' target share.
        while (b < maxRow) {
            const double cur = static_cast<double>(rowDelimiters[b]);
            const double next = static_cast<double>(rowDelimiters[b + 1]);
            if (std::abs(next - target) <= std::abs(cur - target)) {
                ++b;
            } else {
                break;
            }
        }
        bounds[r] = b;
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("Initializing data structures...\n");
    }

    // Global data structures. The full matrix is generated on rank 0 and then
    // distributed row-wise; only the process that owns the reference needs the
    // complete copy.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_reference;                    // Reference solution (rank 0)

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // The row delimiters and the dense vector are required by every process.
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution (serial semantics)
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Determine contiguous, non-zero-balanced row ranges for all processes.
    std::vector<index_t> bounds(static_cast<size_t>(nprocs) + 1);
    if (rank == 0) {
        computeRowPartition(h_rowDelimiters.data(), numRows, nprocs, bounds.data());
    }
    MPI_Bcast(bounds.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t localStart = bounds[rank];
    const index_t localEnd = bounds[rank + 1];
    const index_t localRows = localEnd - localStart;
    const index_t localNnz = h_rowDelimiters[localEnd] - h_rowDelimiters[localStart];

    // Build the scatter/gather metadata from the global row delimiters.
    std::vector<int> nnzCounts(static_cast<size_t>(nprocs));
    std::vector<int> nnzDispls(static_cast<size_t>(nprocs));
    std::vector<int> rowCounts(static_cast<size_t>(nprocs));
    std::vector<int> rowDispls(static_cast<size_t>(nprocs));
    for (int r = 0; r < nprocs; ++r) {
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[bounds[r]]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[bounds[r + 1]] - h_rowDelimiters[bounds[r]]);
        rowDispls[r] = static_cast<int>(bounds[r]);
        rowCounts[r] = static_cast<int>(bounds[r + 1] - bounds[r]);
    }

    // Local CSR data for the rows owned by this process.
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Rebase the row delimiter array onto the local row/column numbering.
    const index_t base = h_rowDelimiters[localStart];
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[localStart + i] - base;
    }

    // The full matrix copies are no longer needed once distribution is done.
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    std::vector<double> localOut(localRows);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Perform SpMV computation. Every process works on its local row block using
    // the replicated dense vector, so the iteration loop needs no communication.
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // The wall-clock time of the parallel computation is the slowest process.
    long long durationMs = 0;
    MPI_Reduce(&localMs, &durationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the distributed output into the global vector on rank 0.
    std::vector<double> globalOut;
    if (rank == 0) {
        globalOut.resize(numRows);
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? globalOut.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalOut, "OutputVector");
    }

    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), globalOut.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
