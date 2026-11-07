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
//   Computes sparse matrix-vector multiplication using CSR format for a
//   block of rows. rowDelimiters must be local (rowDelimiters[0] indexes
//   the first entry of val/cols for this block).
//
// Arguments:
//   val: array holding the non-zero values for the matrix (block)
//   cols: array of column indices for each element (block)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector (full length) to be used for multiplication
//   dim: number of rows in this block
//   out: output - result from the spmv calculation (block)
//
// ****************************************************************************
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < rowEnd; ++j) {
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks receive identical argv)
    int parseStatus = -1;  // -1: continue, otherwise exit code
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
            parseStatus = 0;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_Finalize();
        return parseStatus;
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
        printf("MPI ranks: %d\n", nprocs);
    }

    // ------------------------------------------------------------------
    // Root generates the full problem exactly as the sequential code does
    // (the rand()-driven generation is inherently sequential), then the
    // matrix is partitioned by rows and scattered. The dense input vector
    // is broadcast because column indices are random.
    // ------------------------------------------------------------------
    std::vector<double> h_val;              // Non-zero values (root only)
    std::vector<index_t> h_cols;            // Column indices (root only)
    std::vector<index_t> h_rowDelimiters;   // Row delimiters (root only)
    std::vector<double> h_vec(numRows);     // Dense vector (all ranks)
    std::vector<double> h_out;              // Output vector (root only)

    // Partitioning metadata (computed on root, broadcast to all)
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    std::vector<int> nnzCounts(nprocs), nnzDispls(nprocs);

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Partition rows so that each rank gets a near-equal share of the
        // non-zeros (load balance), keeping contiguous row blocks.
        index_t row = 0;
        for (int p = 0; p < nprocs; ++p) {
            const index_t rowStart = row;
            // Target cumulative nnz for the end of partition p
            const unsigned long long target =
                (static_cast<unsigned long long>(nItems) * (p + 1)) / nprocs;
            if (p == nprocs - 1) {
                row = numRows;
            } else {
                while (row < numRows && h_rowDelimiters[row + 1] <= target) {
                    ++row;
                }
                // Make sure remaining ranks can still get at least one row each
                const index_t maxRow = numRows - static_cast<index_t>(nprocs - 1 - p);
                if (row > maxRow) row = maxRow;
                if (row < rowStart) row = rowStart;
            }
            rowDispls[p] = static_cast<int>(rowStart);
            rowCounts[p] = static_cast<int>(row - rowStart);
            nnzDispls[p] = static_cast<int>(h_rowDelimiters[rowStart]);
            nnzCounts[p] = static_cast<int>(h_rowDelimiters[row] - h_rowDelimiters[rowStart]);
        }
    }

    MPI_Bcast(rowCounts.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDispls.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCounts.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);

    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    std::vector<index_t> l_rowDelimiters(localRows + 1);
    std::vector<double> l_out(localRows);

    // Scatter the CSR blocks. Row delimiters: send localRows+1 entries per
    // rank (blocks overlap by one element), then rebase them to local offsets.
    {
        std::vector<int> rdCounts(nprocs), rdDispls(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            rdCounts[p] = rowCounts[p] + 1;
            rdDispls[p] = rowDispls[p];
        }
        MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                     MPI_DOUBLE, l_val.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                     MPI_UINT32_T, l_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, rdCounts.data(), rdDispls.data(),
                     MPI_UINT32_T, l_rowDelimiters.data(), static_cast<int>(localRows + 1),
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
        const index_t base = l_rowDelimiters[0];
        for (index_t i = 0; i <= localRows; ++i) {
            l_rowDelimiters[i] -= base;
        }
    }

    // For validation, compute reference solution (sequentially on root,
    // exactly as the original code does)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation. Each rank owns a block of rows and the full
    // input vector, so no communication is needed inside the timed loop.
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(),
                h_vec.data(), localRows, l_out.data());
    }

    double localElapsed = MPI_Wtime() - tStart;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed result on root
    MPI_Gatherv(l_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (isRoot) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = (elapsed * 1000.0) / static_cast<double>(iterations);

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
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
