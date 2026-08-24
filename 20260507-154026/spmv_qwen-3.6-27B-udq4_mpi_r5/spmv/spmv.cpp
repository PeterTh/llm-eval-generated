#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

int main(int argc, char** argv) {
    int mpiArgc = argc;
    char** mpiArgv = argv;
    MPI_Init(&mpiArgc, &mpiArgv);

    int rank, numRanks;
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
            if (rank == 0) printUsage(argv[0]);
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
        printf("MPI Ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row distribution: contiguous blocks across ranks
    index_t baseRows = numRows / numRanks;
    index_t extraRows = numRows % numRanks;

    auto getRowRange = [&](int r) -> std::pair<index_t, index_t> {
        index_t start = r * baseRows + std::min(static_cast<index_t>(r), extraRows);
        index_t end = (r + 1) * baseRows + std::min(static_cast<index_t>(r + 1), extraRows);
        return {start, end};
    };

    auto [localStartRow, localEndRow] = getRowRange(rank);
    index_t localNumRows = localEndRow - localStartRow;

    // Rank 0: allocate and initialize all data
    std::vector<double> full_val, full_vec;
    std::vector<index_t> full_cols, full_rowDelimiters;
    std::vector<double> reference;

    if (rank == 0) {
        full_val.resize(nItems);
        full_cols.resize(nItems);
        full_rowDelimiters.resize(numRows + 1);
        full_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(full_vec.data(), numRows, maxVal);
        fill(full_val.data(), nItems, maxVal);
        initRandomMatrix(full_cols.data(), full_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(full_val.data(), full_cols.data(), full_rowDelimiters.data(),
                    full_vec.data(), numRows, reference.data());
        }
    }

    // Broadcast rowDelimiters to all ranks so each rank knows the global structure
    std::vector<index_t> rowDelimiters(numRows + 1);
    if (rank == 0) {
        rowDelimiters = full_rowDelimiters;
    }
    MPI_Bcast(rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Each rank computes its local number of non-zeros
    index_t localNnz = rowDelimiters[localEndRow] - rowDelimiters[localStartRow];

    // Build scatter parameters (computed on all ranks for correctness)
    std::vector<int> sendcounts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        auto [rStart, rEnd] = getRowRange(r);
        sendcounts[r] = static_cast<int>(rowDelimiters[rEnd] - rowDelimiters[rStart]);
        displs[r] = static_cast<int>(rowDelimiters[rStart]);
    }

    // Scatter val and cols to all ranks
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);

    MPI_Scatterv(rank == 0 ? full_val.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_val.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? full_cols.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_cols.data(), static_cast<int>(localNnz), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Broadcast the input vector to all ranks
    std::vector<double> local_vec(numRows);
    if (rank == 0) {
        MPI_Bcast(full_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_vec = full_vec;
    } else {
        MPI_Bcast(local_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Adjust rowDelimiters for local use (convert global indices to local offsets)
    std::vector<index_t> local_rowDelimiters(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; ++i) {
        local_rowDelimiters[i] = rowDelimiters[localStartRow + i] - rowDelimiters[localStartRow];
    }

    // Local output vector
    std::vector<double> local_out(localNumRows);

    // Benchmark: parallel SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                local_vec.data(), localNumRows, local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather output from all ranks to rank 0
    std::vector<double> full_out;
    std::vector<int> recvcounts(numRanks), recvdispls(numRanks);

    if (rank == 0) {
        full_out.resize(numRows);
        for (int r = 0; r < numRanks; ++r) {
            auto [rStart, rEnd] = getRowRange(r);
            recvcounts[r] = static_cast<int>(rEnd - rStart);
            recvdispls[r] = static_cast<int>(rStart);
        }
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                rank == 0 ? full_out.data() : nullptr,
                recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0: report results, validate, print
    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration / 1000.0) / 1e9;
        const double avgTime = duration / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(full_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), full_out.data(), numRows);

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
