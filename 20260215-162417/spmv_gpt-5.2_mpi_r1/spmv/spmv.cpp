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

static inline void rowRange(const index_t nRows, const int rank, const int nRanks, index_t& rowStart,
                            index_t& rowEnd) {
    const index_t base = nRows / static_cast<index_t>(nRanks);
    const index_t rem = nRows % static_cast<index_t>(nRanks);
    const index_t r = static_cast<index_t>(rank);
    rowStart = r * base + (r < rem ? r : rem);
    rowEnd = rowStart + base + (r < rem ? 1u : 0u);
}

static inline void spmvLocalCsr(const double* val, const index_t* cols, const index_t* rowDelimitersLocal,
                               const double* vec, const index_t localRows, double* outLocal) {
    for (index_t i = 0; i < localRows; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimitersLocal[i]; j < rowDelimitersLocal[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        outLocal[i] = t;
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
    int nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate_i = 0;
    int printResults_i = 0;

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
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", nRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    index_t rowStart = 0, rowEnd = 0;
    rowRange(numRows, rank, nRanks, rowStart, rowEnd);
    const index_t localRows = rowEnd - rowStart;

    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;

    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        if (rank == 0) {
            printf("Initializing data structures...\n");
        }
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> sendCountsRows;
    std::vector<int> displsRows;
    std::vector<int> sendCountsNnz;
    std::vector<int> displsNnz;

    if (rank == 0) {
        sendCountsRows.resize(nRanks);
        displsRows.resize(nRanks);
        sendCountsNnz.resize(nRanks);
        displsNnz.resize(nRanks);

        for (int r = 0; r < nRanks; ++r) {
            index_t rs = 0, re = 0;
            rowRange(numRows, r, nRanks, rs, re);
            const index_t lr = re - rs;
            displsRows[r] = static_cast<int>(rs);
            sendCountsRows[r] = static_cast<int>(lr + 1);

            const index_t nzStart = h_rowDelimiters[rs];
            const index_t nzEnd = h_rowDelimiters[re];
            displsNnz[r] = static_cast<int>(nzStart);
            sendCountsNnz[r] = static_cast<int>(nzEnd - nzStart);
        }
    }

    std::vector<index_t> localRowDelimsGlobal(localRows + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? sendCountsRows.data() : nullptr,
                 rank == 0 ? displsRows.data() : nullptr, MPI_UINT32_T, localRowDelimsGlobal.data(),
                 static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t nzStartLocal = localRowDelimsGlobal[0];
    const index_t nzEndLocal = localRowDelimsGlobal[localRows];
    const index_t localNnz = nzEndLocal - nzStartLocal;

    std::vector<index_t> localRowDelims(localRows + 1);
    for (index_t i = 0; i < localRows + 1; ++i) {
        localRowDelims[i] = localRowDelimsGlobal[i] - nzStartLocal;
    }

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? sendCountsNnz.data() : nullptr,
                 rank == 0 ? displsNnz.data() : nullptr, MPI_DOUBLE, localVal.data(),
                 static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? sendCountsNnz.data() : nullptr,
                 rank == 0 ? displsNnz.data() : nullptr, MPI_UINT32_T, localCols.data(),
                 static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<double> localOut(localRows);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocalCsr(localVal.data(), localCols.data(), localRowDelims.data(), h_vec.data(), localRows,
                     localOut.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> recvCountsOut;
    std::vector<int> displsOut;
    std::vector<double> h_out;

    if (rank == 0) {
        recvCountsOut.resize(nRanks);
        displsOut.resize(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            index_t rs = 0, re = 0;
            rowRange(numRows, r, nRanks, rs, re);
            recvCountsOut[r] = static_cast<int>(re - rs);
            displsOut[r] = static_cast<int>(rs);
        }
        h_out.resize(numRows);
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? h_out.data() : nullptr,
               rank == 0 ? recvCountsOut.data() : nullptr, rank == 0 ? displsOut.data() : nullptr,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double totalMs = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", totalMs);

        const double gflops = (2.0 * nItems * iterations) / (maxTime) / 1e9;
        const double avgTime = totalMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
