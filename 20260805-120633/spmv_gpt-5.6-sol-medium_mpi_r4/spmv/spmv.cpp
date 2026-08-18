#include <algorithm>
#include <climits>
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
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * dim;
    double prob = static_cast<double>(n) / static_cast<double>(totalEntries);

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = totalEntries -
                (static_cast<uint64_t>(i) * dim + j);
            const uint64_t needToAssign = n - nnzAssigned;
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

    // Calculate number of non-zero elements
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = sparsity == 0 ? 0 : totalEntries / sparsity;
    const bool validArguments = numRows > 0 && sparsity > 0 && iterations > 0 &&
        numRows <= static_cast<index_t>(INT_MAX) &&
        nItems64 <= static_cast<uint64_t>(std::numeric_limits<index_t>::max()) &&
        nItems64 <= static_cast<uint64_t>(INT_MAX);
    if (!validArguments) {
        if (rank == 0) {
            fprintf(stderr, "Invalid or unsupported problem size: -n, -s, and -i must be positive, "
                    "and MPI counts must fit in a 32-bit integer.\n");
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
               nItems, 100.0 * (1.0 - static_cast<double>(nItems64) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("MPI processes: %d\n", worldSize);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank zero creates the deterministic input once. The CSR rows are then
    // distributed so that each process retains only its local matrix slice.
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRows;
    std::vector<double> h_vec(numRows);
    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRows.resize(numRows + 1);
        // C and C++ define the default rand() sequence as if seeded with 1.
        // Make that original benchmark behavior independent of MPI startup.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(globalVal.data(), globalCols.data(), globalRows.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Select contiguous row boundaries by cumulative nonzeros, rather than by
    // row count. This keeps work balanced even for irregular matrices.
    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    std::vector<int> nnzCounts(worldSize);
    std::vector<int> nnzDisplacements(worldSize);
    if (rank == 0) {
        std::vector<index_t> boundary(worldSize + 1);
        boundary[0] = 0;
        boundary[worldSize] = numRows;
        for (int p = 1; p < worldSize; ++p) {
            const index_t target = static_cast<index_t>((nItems64 * p) / worldSize);
            boundary[p] = static_cast<index_t>(std::lower_bound(
                globalRows.begin(), globalRows.end(), target) - globalRows.begin());
            boundary[p] = std::min(boundary[p], numRows);
        }
        for (int p = 0; p < worldSize; ++p) {
            rowDisplacements[p] = static_cast<int>(boundary[p]);
            rowCounts[p] = static_cast<int>(boundary[p + 1] - boundary[p]);
            nnzDisplacements[p] = static_cast<int>(globalRows[boundary[p]]);
            nnzCounts[p] = static_cast<int>(globalRows[boundary[p + 1]] -
                                            globalRows[boundary[p]]);
        }
    }

    int localRowCount = 0;
    int localNnzCount = 0;
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRowCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnzCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(localNnzCount);
    std::vector<index_t> localCols(localNnzCount);
    std::vector<index_t> localRows(static_cast<size_t>(localRowCount) + 1);
    MPI_Scatterv(globalVal.data(), nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 localVal.data(), localNnzCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalCols.data(), nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T,
                 localCols.data(), localNnzCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // Scatter each row's global start offset and turn it into a local CSR offset.
    MPI_Scatterv(globalRows.data(), rowCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                 localRows.data(), localRowCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localBase = localRowCount == 0 ? 0 : localRows[0];
    for (int i = 0; i < localRowCount; ++i) {
        localRows[i] -= localBase;
    }
    localRows[localRowCount] = static_cast<index_t>(localNnzCount);

    // The full matrix is not part of rank zero's steady-state working set.
    globalVal.clear();
    globalCols.clear();
    globalRows.clear();
    globalVal.shrink_to_fit();
    globalCols.shrink_to_fit();
    globalRows.shrink_to_fit();

    std::vector<double> localOut(localRowCount);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRows.data(),
                h_vec.data(), static_cast<index_t>(localRowCount), localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops = (2.0 * nItems64 * iterations) / duration / 1e9;
        const double durationMs = duration * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);
        printf("Average time per iteration: %.3f ms\n", durationMs / iterations);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> h_out;
    if (rank == 0 && (printResults || validate)) {
        h_out.resize(numRows);
    }
    if (printResults || validate) {
        MPI_Gatherv(localOut.data(), localRowCount, MPI_DOUBLE, h_out.data(),
                    rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int valid = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        valid = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
        if (valid != 0) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return valid != 0 ? 0 : 1;
}
