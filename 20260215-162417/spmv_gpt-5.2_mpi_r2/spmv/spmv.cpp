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

static inline void block_decompose_1d(const index_t n, const int commSize, const int rank,
                                     index_t& start, index_t& count) {
    const index_t base = n / static_cast<index_t>(commSize);
    const index_t rem = n % static_cast<index_t>(commSize);
    count = base + (static_cast<index_t>(rank) < rem ? 1u : 0u);
    start = static_cast<index_t>(rank) * base + (static_cast<index_t>(rank) < rem ? static_cast<index_t>(rank) : rem);
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
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate_i = 0;
    int printResults_i = 0;

    int parseOk = 1;
    int showHelp = 0;

    // Parse command line arguments on rank 0
    if (rank == 0) {
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
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = 0;
            }
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!parseOk) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    // Calculate number of non-zero elements
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows);
    const index_t nItems = static_cast<index_t>(totalEntries / static_cast<uint64_t>(sparsity));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", commSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / static_cast<double>(totalEntries)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize global data on rank 0 (exact same RNG usage/order as serial)
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector and row delimiters to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Row-wise decomposition
    index_t rowStart = 0;
    index_t rowCount = 0;
    block_decompose_1d(numRows, commSize, rank, rowStart, rowCount);

    const index_t nnzStart = h_rowDelimiters[rowStart];
    const index_t nnzEnd = h_rowDelimiters[rowStart + rowCount];
    const index_t localNnz = nnzEnd - nnzStart;

    // Build local CSR row delimiters with local nnz offsets
    std::vector<index_t> localRowDelimiters(rowCount + 1);
    for (index_t i = 0; i <= rowCount; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[rowStart + i] - nnzStart;
    }

    // Scatter CSR (val/cols) by nnz
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);

    std::vector<int> sendcounts;
    std::vector<int> displs;
    if (rank == 0) {
        sendcounts.resize(commSize);
        displs.resize(commSize);
        for (int r = 0; r < commSize; ++r) {
            index_t rs = 0, rc = 0;
            block_decompose_1d(numRows, commSize, r, rs, rc);
            const index_t s = h_rowDelimiters[rs];
            const index_t e = h_rowDelimiters[rs + rc];
            const uint64_t c = static_cast<uint64_t>(e - s);
            if (c > static_cast<uint64_t>(INT32_MAX) || static_cast<uint64_t>(s) > static_cast<uint64_t>(INT32_MAX)) {
                printf("Problem size too large for MPI_Scatterv counts/displs (int)\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            sendcounts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(s);
        }
    }

    double* local_val_ptr = local_val.empty() ? nullptr : local_val.data();
    index_t* local_cols_ptr = local_cols.empty() ? nullptr : local_cols.data();

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE,
                 local_val_ptr,
                 static_cast<int>(localNnz),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UINT32_T,
                 local_cols_ptr,
                 static_cast<int>(localNnz),
                 MPI_UINT32_T,
                 0,
                 MPI_COMM_WORLD);

    // Perform SpMV computation (distributed)
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    std::vector<double> local_out(rowCount);
    double* local_out_ptr = local_out.empty() ? nullptr : local_out.data();

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val_ptr, local_cols_ptr, localRowDelimiters.data(), h_vec.data(), rowCount, local_out_ptr);
    }

    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather output vector for optional printing/validation
    std::vector<double> h_out;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;

    if (rank == 0) {
        h_out.resize(numRows);
        recvcounts.resize(commSize);
        rdispls.resize(commSize);
        for (int r = 0; r < commSize; ++r) {
            index_t rs = 0, rc = 0;
            block_decompose_1d(numRows, commSize, r, rs, rc);
            if (static_cast<uint64_t>(rc) > static_cast<uint64_t>(INT32_MAX) || static_cast<uint64_t>(rs) > static_cast<uint64_t>(INT32_MAX)) {
                printf("Problem size too large for MPI_Gatherv counts/displs (int)\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            recvcounts[r] = static_cast<int>(rc);
            rdispls[r] = static_cast<int>(rs);
        }
    }

    MPI_Gatherv(local_out_ptr,
                static_cast<int>(rowCount),
                MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? rdispls.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        const double totalMs = maxTime * 1000.0;
        const double avgMs = totalMs / static_cast<double>(iterations);
        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / maxTime / 1e9;

        printf("Computation time: %.3f ms\n", totalMs);
        printf("Average time per iteration: %.3f ms\n", avgMs);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> h_reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());

            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
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
