#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
#if defined(__GNUC__) || defined(__clang__)
#define SPMV_RESTRICT __restrict__
#else
#define SPMV_RESTRICT
#endif

void spmvCpu(const double* SPMV_RESTRICT val, const index_t* SPMV_RESTRICT cols,
             const index_t* SPMV_RESTRICT rowDelimiters,
             const double* SPMV_RESTRICT vec, const index_t dim,
             double* SPMV_RESTRICT out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

#undef SPMV_RESTRICT

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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    // MPI_Scatterv takes int counts and displacements.  The benchmark's
    // original 32-bit CSR format is retained, with sizes additionally
    // constrained to the range portable across MPI implementations.
    const std::uint64_t totalEntries = static_cast<std::uint64_t>(numRows) * numRows;
    const std::uint64_t nItems64 = sparsity == 0 ? 0 : totalEntries / sparsity;
    const bool validConfiguration = numRows > 0 && sparsity > 0 && iterations > 0 &&
        nItems64 <= static_cast<std::uint64_t>(std::numeric_limits<index_t>::max()) &&
        nItems64 <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
        numRows <= static_cast<index_t>(std::numeric_limits<int>::max());
    if (!validConfiguration) {
        if (rank == 0) {
            printf("Invalid problem size: rows, sparsity, and iterations must be positive; "
                   "the CSR data must fit MPI int counts.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI ranks: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Build the canonical matrix on rank zero, then distribute a contiguous
    // CSR row range to each rank.  Rows are partitioned by accumulated NNZ,
    // which balances the actual SpMV work despite uneven random row lengths.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;
    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    std::vector<int> nnzCounts(worldSize);
    std::vector<int> nnzDisplacements(worldSize);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        std::vector<int> rowBoundaries(worldSize + 1);
        rowBoundaries.front() = 0;
        rowBoundaries.back() = static_cast<int>(numRows);
        for (int process = 1; process < worldSize; ++process) {
            if (nItems == 0) {
                rowBoundaries[process] = static_cast<int>(
                    (static_cast<std::uint64_t>(numRows) * process) / worldSize);
            } else {
                const index_t target = static_cast<index_t>(
                    (static_cast<std::uint64_t>(nItems) * process) / worldSize);
                rowBoundaries[process] = static_cast<int>(std::lower_bound(
                    h_rowDelimiters.begin(), h_rowDelimiters.end(), target) -
                    h_rowDelimiters.begin());
            }
        }

        for (int process = 0; process < worldSize; ++process) {
            const int firstRow = rowBoundaries[process];
            const int lastRow = rowBoundaries[process + 1];
            rowDisplacements[process] = firstRow;
            rowCounts[process] = lastRow - firstRow;
            nnzDisplacements[process] = static_cast<int>(h_rowDelimiters[firstRow]);
            nnzCounts[process] = static_cast<int>(
                h_rowDelimiters[lastRow] - h_rowDelimiters[firstRow]);
        }

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localRows = 0;
    int localNnz = 0;
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Send each row's starting offset.  The final local offset is derived
    // from localNnz, avoiding an overlapping delimiter transfer at boundaries.
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, rowCounts.data(),
                 rowDisplacements.data(), MPI_UINT32_T, localRowDelimiters.data(), localRows,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (localRows > 0) {
        const index_t firstOffset = localRowDelimiters[0];
        for (int row = 0; row < localRows; ++row) {
            localRowDelimiters[row] -= firstOffset;
        }
    }
    localRowDelimiters[localRows] = static_cast<index_t>(localNnz);

    // The root no longer needs the global matrix.  The timed computation has
    // only local CSR storage on every rank, while the immutable dense vector
    // is replicated once as required by this row-wise decomposition.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), h_vec.data(),
                static_cast<index_t>(localRows), localOut.data());
    }
    const double localElapsedSeconds = MPI_Wtime() - start;

    // Report the slowest rank's time: it is the distributed iteration's
    // completion time and therefore the meaningful scalable throughput.
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<double> h_out;
    if (printResults || validate) {
        if (rank == 0) {
            h_out.resize(numRows);
        }
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        const double durationMilliseconds = elapsedSeconds * 1000.0;
        const double gflops = elapsedSeconds > 0.0
            ? (2.0 * static_cast<double>(nItems) * iterations) / elapsedSeconds / 1.0e9
            : 0.0;
        const double avgTime = durationMilliseconds / iterations;

        printf("Computation time: %.3f ms\n", durationMilliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            if (verifyResults(h_reference.data(), h_out.data(), numRows)) {
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
