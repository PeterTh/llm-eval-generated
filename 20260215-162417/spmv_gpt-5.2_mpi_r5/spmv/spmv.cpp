#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <climits>
#include <cstdint>

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

static inline void computeRowRange(const index_t nRows, const int rank, const int size, index_t& startRow,
                                   index_t& endRow) {
    const index_t base = nRows / static_cast<index_t>(size);
    const index_t rem = nRows % static_cast<index_t>(size);
    const index_t r = static_cast<index_t>(rank);
    startRow = r * base + (r < rem ? r : rem);
    endRow = startRow + base + (r < rem ? 1u : 0u);
}

static void spmvCpuLocal(const double* val, const index_t* cols, const index_t* rowDelimiters,
                         const double* vec, const index_t numLocalRows, double* out) {
    for (index_t i = 0; i < numLocalRows; ++i) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[i];
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;

    int exitEarly = 0;
    int exitCode = 0;

    if (rank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitEarly = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitEarly = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitEarly, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (numRows > static_cast<index_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: numRows too large for MPI counts (%u)\n", numRows);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate number of non-zero elements (same expression as original code)
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(static_cast<size_t>(numRows));

    std::vector<double> h_reference;

    std::vector<int> sendcounts_rows;
    std::vector<int> displs_rows;
    std::vector<int> sendcounts_rowdelims;
    std::vector<int> displs_rowdelims;
    std::vector<int> sendcounts_nnz;
    std::vector<int> displs_nnz;

    if (rank == 0) {
        if (nItems > static_cast<index_t>(INT_MAX)) {
            printf("Error: nnz too large for MPI_Scatterv/Gatherv counts (%u)\n", nItems);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate and initialize global data structures on rank 0
        h_val.resize(static_cast<size_t>(nItems));
        h_cols.resize(static_cast<size_t>(nItems));
        h_rowDelimiters.resize(static_cast<size_t>(numRows + 1));

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution on rank 0
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(static_cast<size_t>(numRows));
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }

        sendcounts_rows.resize(static_cast<size_t>(size));
        displs_rows.resize(static_cast<size_t>(size));
        sendcounts_rowdelims.resize(static_cast<size_t>(size));
        displs_rowdelims.resize(static_cast<size_t>(size));
        sendcounts_nnz.resize(static_cast<size_t>(size));
        displs_nnz.resize(static_cast<size_t>(size));

        for (int r = 0; r < size; ++r) {
            index_t s = 0;
            index_t e = 0;
            computeRowRange(numRows, r, size, s, e);
            const index_t rows = e - s;
            if (rows > static_cast<index_t>(INT_MAX)) {
                printf("Error: local row count too large for MPI (%u)\n", rows);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            sendcounts_rows[static_cast<size_t>(r)] = static_cast<int>(rows);
            displs_rows[static_cast<size_t>(r)] = static_cast<int>(s);
            sendcounts_rowdelims[static_cast<size_t>(r)] = static_cast<int>(rows + 1);
            displs_rowdelims[static_cast<size_t>(r)] = static_cast<int>(s);

            const index_t nnzStart = h_rowDelimiters[static_cast<size_t>(s)];
            const index_t nnzEnd = h_rowDelimiters[static_cast<size_t>(e)];
            const index_t nnzCount = nnzEnd - nnzStart;
            if (nnzCount > static_cast<index_t>(INT_MAX)) {
                printf("Error: local nnz count too large for MPI (%u)\n", nnzCount);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            sendcounts_nnz[static_cast<size_t>(r)] = static_cast<int>(nnzCount);
            displs_nnz[static_cast<size_t>(r)] = static_cast<int>(nnzStart);
        }
    }

    index_t startRow = 0;
    index_t endRow = 0;
    computeRowRange(numRows, rank, size, startRow, endRow);
    const index_t localRows = endRow - startRow;
    const int localRowsInt = static_cast<int>(localRows);

    const int localRowDelimCount = localRowsInt + 1;

    int localNnzInt = 0;
    MPI_Scatter(rank == 0 ? sendcounts_nnz.data() : nullptr, 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    std::vector<index_t> local_rowDelimiters(static_cast<size_t>(localRowDelimCount));
    std::vector<index_t> local_cols(static_cast<size_t>(localNnzInt));
    std::vector<double> local_val(static_cast<size_t>(localNnzInt));
    std::vector<double> local_out(static_cast<size_t>(localRowsInt));

    // Distribute row delimiters slice (global offsets)
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? sendcounts_rowdelims.data() : nullptr,
                 rank == 0 ? displs_rowdelims.data() : nullptr, MPI_UINT32_T, local_rowDelimiters.data(),
                 localRowDelimCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Convert row delimiters to local (0-based) offsets
    const index_t baseNnz = local_rowDelimiters.empty() ? 0 : local_rowDelimiters[0];
    for (int i = 0; i < localRowDelimCount; ++i) {
        local_rowDelimiters[static_cast<size_t>(i)] -= baseNnz;
    }

    // Distribute CSR data for the owned rows
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? sendcounts_nnz.data() : nullptr,
                 rank == 0 ? displs_nnz.data() : nullptr, MPI_UINT32_T, local_cols.data(), localNnzInt,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? sendcounts_nnz.data() : nullptr,
                 rank == 0 ? displs_nnz.data() : nullptr, MPI_DOUBLE, local_val.data(), localNnzInt,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast dense vector to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpuLocal(local_val.data(), local_cols.data(), local_rowDelimiters.data(), h_vec.data(), localRows,
                     local_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localSeconds = t1 - t0;
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxSeconds * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);

        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / maxSeconds /
                              1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> h_out;
    if (validate || printResults) {
        if (rank == 0) {
            h_out.resize(static_cast<size_t>(numRows));
        }

        MPI_Gatherv(local_out.data(), localRowsInt, MPI_DOUBLE, rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? sendcounts_rows.data() : nullptr, rank == 0 ? displs_rows.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(h_out, "OutputVector");
        }

        int finalCode = 0;
        if (rank == 0 && validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                finalCode = 0;
            } else {
                printf("Validation: FAILED\n");
                finalCode = 1;
            }
        }

        MPI_Bcast(&finalCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return finalCode;
    }

    MPI_Finalize();
    return 0;
}
