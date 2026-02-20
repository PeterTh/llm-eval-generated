#include <chrono>
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype mpi_index_t = MPI_DATATYPE_NULL;
    if (sizeof(index_t) == sizeof(unsigned int)) {
        mpi_index_t = MPI_UNSIGNED;
    } else if (sizeof(index_t) == sizeof(unsigned long)) {
        mpi_index_t = MPI_UNSIGNED_LONG;
    } else if (sizeof(index_t) == sizeof(unsigned long long)) {
        mpi_index_t = MPI_UNSIGNED_LONG_LONG;
    } else {
        if (rank == 0) {
            printf("Unsupported index_t size for MPI\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    int parseStatus = 0;  // 0=ok, 1=error, 2=help
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
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseStatus = 1;
                break;
            }
        }

        if (parseStatus == 2) {
            printUsage(argv[0]);
        } else if (parseStatus == 1) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return (parseStatus == 1) ? 1 : 0;
    }

    int validate_i = validate ? 1 : 0;
    int printResults_i = printResults ? 1 : 0;

    MPI_Bcast(&numRows, 1, mpi_index_t, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, mpi_index_t, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, mpi_index_t, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    validate = (validate_i != 0);
    printResults = (printResults_i != 0);

    const uint64_t nItems64 = (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows)) /
                             static_cast<uint64_t>(sparsity);
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            printf("Problem size too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        const double totalEntries = static_cast<double>(static_cast<uint64_t>(numRows) *
                                                        static_cast<uint64_t>(numRows));
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;

    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;

    std::vector<int> rowsPerRank;
    std::vector<int> rowDispls;
    std::vector<int> nnzPerRank;
    std::vector<int> nnzDispls;
    std::vector<int> rowDelimSendCounts;
    std::vector<int> rowDelimDispls;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        if (size == 1) {
            h_out.resize(numRows);
        }

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        rowsPerRank.resize(size);
        rowDispls.resize(size);
        nnzPerRank.resize(size);
        nnzDispls.resize(size);
        rowDelimSendCounts.resize(size);
        rowDelimDispls.resize(size);

        for (int r = 0; r < size; ++r) {
            const auto rs = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / size);
            const auto re = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / size);
            const auto lr = static_cast<index_t>(re - rs);

            const uint64_t startNnz = h_rowDelimiters[rs];
            const uint64_t endNnz = h_rowDelimiters[re];
            const uint64_t lnnz = endNnz - startNnz;

            if (lr > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                lnnz > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                rs > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                startNnz > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                printf("Counts/displacements exceed MPI int limits\n");
                MPI_Abort(MPI_COMM_WORLD, 4);
            }

            rowsPerRank[r] = static_cast<int>(lr);
            rowDispls[r] = static_cast<int>(rs);
            nnzPerRank[r] = static_cast<int>(lnnz);
            nnzDispls[r] = static_cast<int>(startNnz);
            rowDelimSendCounts[r] = static_cast<int>(lr + 1);
            rowDelimDispls[r] = static_cast<int>(rs);
        }

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localRows_i = 0;
    int localNnz_i = 0;
    if (rank == 0) {
        localRows_i = rowsPerRank[0];
        localNnz_i = nnzPerRank[0];
    }
    MPI_Scatter(rank == 0 ? rowsPerRank.data() : nullptr, 1, MPI_INT, &localRows_i, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzPerRank.data() : nullptr, 1, MPI_INT, &localNnz_i, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    const auto localRows = static_cast<index_t>(localRows_i);
    const auto localNnz = static_cast<index_t>(localNnz_i);

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<double> localOut(localRows);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowDelimSendCounts.data() : nullptr,
                 rank == 0 ? rowDelimDispls.data() : nullptr, mpi_index_t, localRowDelimiters.data(),
                 static_cast<int>(localRows + 1), mpi_index_t, 0, MPI_COMM_WORLD);

    const index_t base = localRowDelimiters[0];
    for (index_t i = 0; i < localRows + 1; ++i) {
        localRowDelimiters[i] -= base;
    }

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? nnzPerRank.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE, localVal.data(),
                 static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? nnzPerRank.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, mpi_index_t, localCols.data(),
                 static_cast<int>(localNnz), mpi_index_t, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        h_out.resize(numRows);
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), h_vec.data(), localRows,
                localOut.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localElapsed = t1 - t0;

    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rank == 0 ? rowsPerRank.data() : nullptr,
                rank == 0 ? rowDispls.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double totalMs = maxElapsed * 1000.0;
        const double avgMs = totalMs / static_cast<double>(iterations);
        const double gflops = (maxElapsed > 0.0)
                                  ? (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
                                        maxElapsed / 1e9
                                  : 0.0;

        printf("Computation time: %.3f ms\n", totalMs);
        printf("Average time per iteration: %.3f ms\n", avgMs);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
