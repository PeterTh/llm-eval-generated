#include <cmath>
#include <cstdint>
#include <climits>
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
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    const index_t nItems = static_cast<index_t>(totalEntries / sparsity);

    // Give consecutive, almost equally sized row blocks to the ranks.
    const index_t rowStart = static_cast<index_t>(
        (static_cast<uint64_t>(rank) * numRows) / worldSize);
    const index_t rowEnd = static_cast<index_t>(
        (static_cast<uint64_t>(rank + 1) * numRows) / worldSize);
    const index_t localRows = rowEnd - rowStart;

    std::vector<int> rowCounts(worldSize), rowDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / worldSize);
        const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / worldSize);
        rowCounts[r] = static_cast<int>(end - begin);
        rowDisplacements[r] = static_cast<int>(begin);
    }

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank zero creates the same global data as the original benchmark.  CSR
    // rows and values are then scattered, so the working set is distributed.
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_reference;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<index_t> localRowDelimiters(localRows + 1);
    std::vector<index_t> globalNnzCounts(worldSize);
    std::vector<int> nnzCounts(worldSize), nnzDisplacements(worldSize);
    if (rank == 0) {
        for (int r = 0; r < worldSize; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / worldSize);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / worldSize);
            globalNnzCounts[r] = globalRowDelimiters[end] - globalRowDelimiters[begin];
            if (globalNnzCounts[r] > static_cast<index_t>(INT_MAX)) MPI_Abort(MPI_COMM_WORLD, 2);
            nnzCounts[r] = static_cast<int>(globalNnzCounts[r]);
            nnzDisplacements[r] = static_cast<int>(globalRowDelimiters[begin]);
        }
    }
    MPI_Bcast(globalNnzCounts.data(), worldSize, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localNnz = globalNnzCounts[rank];
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rowCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    // The row-pointer scatter above sends row starts; send the final endpoint
    // separately because neighboring rank blocks do not overlap at that item.
    index_t localEnd = 0;
    std::vector<index_t> rowEnds(worldSize);
    if (rank == 0) for (int r = 0; r < worldSize; ++r) {
        const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / worldSize);
        rowEnds[r] = globalRowDelimiters[end];
    }
    MPI_Scatter(rowEnds.data(), 1, MPI_UINT32_T, &localEnd, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    localRowDelimiters[localRows] = localEnd;
    index_t localBase = 0;
    // Each rank needs its own CSR base.
    std::vector<index_t> rowBases(worldSize);
    if (rank == 0) for (int r = 0; r < worldSize; ++r) {
        const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / worldSize);
        rowBases[r] = globalRowDelimiters[begin];
    }
    MPI_Scatter(rowBases.data(), 1, MPI_UINT32_T, &localBase, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (index_t& offset : localRowDelimiters) offset -= localBase;
    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(globalVal.data(), globalCols.data(), globalRowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }
    // The distributed computation no longer needs the root's full CSR copy.
    if (rank == 0) {
        globalVal.clear();
        globalVal.shrink_to_fit();
        globalCols.clear();
        globalCols.shrink_to_fit();
        globalRowDelimiters.clear();
        globalRowDelimiters.shrink_to_fit();
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    std::vector<double> localOut(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_out;
    if (rank == 0) h_out.resize(numRows);
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = seconds * 1000.0;
        printf("Computation time: %.0f ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = seconds > 0.0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double avgTime = iterations > 0 ? durationMs / static_cast<double>(iterations) : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) print_results(h_out, "OutputVector");
    }

    // Validation
    if (validate && rank == 0) {
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

    MPI_Finalize();
    return 0;
}
