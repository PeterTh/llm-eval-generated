#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
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
            const uint64_t numEntriesLeft =
                (static_cast<uint64_t>(dim) * dim) - (static_cast<uint64_t>(i) * dim + j);
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

// Each rank owns one contiguous range of rows.  The associated CSR values and
// column indices are also contiguous, so they can be sent without packing.
// Point-to-point transfers are used here rather than MPI_Scatterv so that the
// matrix size is not constrained by MPI's int displacement arguments.
template <typename T>
void sendLarge(const T* data, size_t count, MPI_Datatype datatype, const int destination,
               const int tag) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, maxCount));
        MPI_Send(data, chunk, datatype, destination, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

template <typename T>
void receiveLarge(T* data, size_t count, MPI_Datatype datatype, const int source, const int tag) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, maxCount));
        MPI_Recv(data, chunk, datatype, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

template <typename T>
void distributeCsrArray(const std::vector<T>& global, std::vector<T>& local,
                        const index_t localNnzStart, const int rank, const int ranks,
                        MPI_Datatype datatype, const int tag,
                        const std::vector<index_t>& nnzStarts,
                        const std::vector<index_t>& nnzCounts) {
    if (rank == 0) {
        if (!local.empty()) {
            std::copy_n(global.data() + localNnzStart, local.size(), local.data());
        }
        for (int target = 1; target < ranks; ++target) {
            const size_t count = nnzCounts[target];
            if (count != 0) {
                sendLarge(global.data() + nnzStarts[target], count, datatype, target, tag);
            }
        }
    } else if (!local.empty()) {
        receiveLarge(local.data(), local.size(), datatype, 0, tag);
    }
}

void gatherOutput(const std::vector<double>& local, std::vector<double>& global,
                  const index_t firstRow, const int rank, const int ranks,
                  const std::vector<index_t>& rowStarts) {
    constexpr int outputTag = 103;
    if (rank == 0) {
        if (!local.empty()) {
            std::copy_n(local.data(), local.size(), global.data() + firstRow);
        }
        for (int source = 1; source < ranks; ++source) {
            const index_t sourceFirstRow = rowStarts[source];
            const index_t sourceLastRow = rowStarts[source + 1];
            const size_t sourceRows = static_cast<size_t>(sourceLastRow - sourceFirstRow);
            if (sourceRows != 0) {
                receiveLarge(global.data() + sourceFirstRow, sourceRows, MPI_DOUBLE, source, outputTag);
            }
        }
    } else if (!local.empty()) {
        sendLarge(local.data(), local.size(), MPI_DOUBLE, 0, outputTag);
    }
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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalStart) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       globalStart + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       globalStart + i, ref, res, relError);
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        matrixEntries > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            printf("Matrix dimensions and sparsity must produce at most %u non-zero entries; iterations must be positive.\n",
                   std::numeric_limits<index_t>::max());
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixEntries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", ranks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Root initializes the exact same random matrix and vector as the serial
    // program.  After distribution, every rank keeps only its local CSR rows.
    std::vector<double> h_vec(numRows);
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Assign contiguous row ranges by nonzero count rather than by the raw
    // number of rows.  This preserves CSR locality while balancing the actual
    // work, including for sparse matrices and high rank counts.
    std::vector<index_t> rowStarts;
    std::vector<index_t> nnzStarts(ranks);
    std::vector<index_t> nnzCounts(ranks);
    if (rank == 0) {
        rowStarts.resize(static_cast<size_t>(ranks) + 1);
        for (int target = 0; target < ranks; ++target) {
            const index_t targetNnz =
                static_cast<index_t>((static_cast<uint64_t>(nItems) * target) / ranks);
            const index_t targetFirstRow = static_cast<index_t>(
                std::lower_bound(globalRowDelimiters.begin(), globalRowDelimiters.end(), targetNnz) -
                globalRowDelimiters.begin());
            rowStarts[target] = targetFirstRow;
        }
        rowStarts[ranks] = numRows;
        for (int target = 0; target < ranks; ++target) {
            const index_t targetFirstRow = rowStarts[target];
            const index_t targetLastRow = rowStarts[target + 1];
            nnzStarts[target] = globalRowDelimiters[targetFirstRow];
            nnzCounts[target] = globalRowDelimiters[targetLastRow] - nnzStarts[target];
        }
    }

    index_t firstRow = 0;
    index_t lastRow = 0;
    MPI_Scatter(rank == 0 ? rowStarts.data() : nullptr, 1, MPI_UINT32_T,
                &firstRow, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? rowStarts.data() + 1 : nullptr, 1, MPI_UINT32_T,
                &lastRow, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = lastRow - firstRow;

    index_t localNnzInfo[2] = {0, 0};
    MPI_Scatter(rank == 0 ? nnzStarts.data() : nullptr, 1, MPI_UINT32_T,
                &localNnzInfo[0], 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_UINT32_T,
                &localNnzInfo[1], 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localNnzStart = localNnzInfo[0];
    const index_t localNnz = localNnzInfo[1];

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    if (rank == 0) {
        if (localRows != 0) {
            std::copy_n(globalRowDelimiters.data() + firstRow, localRows,
                        localRowDelimiters.data());
        }
        for (int target = 1; target < ranks; ++target) {
            const index_t targetFirstRow = rowStarts[target];
            const index_t targetLastRow = rowStarts[target + 1];
            const size_t targetRows = static_cast<size_t>(targetLastRow - targetFirstRow);
            if (targetRows != 0) {
                sendLarge(globalRowDelimiters.data() + targetFirstRow, targetRows,
                          MPI_UINT32_T, target, 101);
            }
        }
    } else if (localRows != 0) {
        receiveLarge(localRowDelimiters.data(), localRows, MPI_UINT32_T, 0, 101);
    }
    for (index_t row = 0; row < localRows; ++row) {
        localRowDelimiters[row] -= localNnzStart;
    }
    localRowDelimiters[localRows] = localNnz;

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    distributeCsrArray(globalVal, localVal, localNnzStart, rank, ranks, MPI_DOUBLE, 102,
                       nnzStarts, nnzCounts);
    distributeCsrArray(globalCols, localCols, localNnzStart, rank, ranks, MPI_UINT32_T, 103,
                       nnzStarts, nnzCounts);

    // Release the root's temporary global CSR representation before the
    // benchmark.  Matrix storage is now distributed across the ranks.
    if (rank == 0) {
        std::vector<double>().swap(globalVal);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRowDelimiters);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    std::vector<double> h_out(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Calculate performance metrics
    if (rank == 0) {
        const double durationMs = duration * 1000.0;
        const double gflops = duration > 0.0
            ? (2.0 * nItems * iterations) / duration / 1e9
            : 0.0;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Computation time: %.3f ms\n", durationMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> globalOut;
        if (rank == 0) {
            globalOut.resize(numRows);
        }
        gatherOutput(h_out, globalOut, firstRow, rank, ranks, rowStarts);
        if (rank == 0) {
            print_results(globalOut, "OutputVector");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResults(h_reference.data(), h_out.data(), localRows, firstRow) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
