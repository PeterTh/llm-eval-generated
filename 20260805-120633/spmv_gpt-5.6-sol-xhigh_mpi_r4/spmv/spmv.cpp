#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
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
                static_cast<uint64_t>(dim) * dim -
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
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters,
             const double* __restrict vec, const index_t dim,
             double* __restrict out) {
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

// Select contiguous row ranges whose nonzero counts are as even as possible.
// Contiguous ranges preserve CSR streaming behavior and allow a single Gatherv
// to reconstruct the original row order.
std::vector<index_t> makeRowPartitions(const std::vector<index_t>& rowDelimiters,
                                       const index_t numRows,
                                       const index_t nItems,
                                       const int numRanks) {
    std::vector<index_t> boundaries(static_cast<size_t>(numRanks) + 1);
    boundaries.front() = 0;
    boundaries.back() = numRows;

    for (int rank = 1; rank < numRanks; ++rank) {
        const uint64_t target =
            static_cast<uint64_t>(nItems) * static_cast<uint64_t>(rank) /
            static_cast<uint64_t>(numRanks);
        auto next = std::lower_bound(rowDelimiters.begin(), rowDelimiters.end(),
                                     static_cast<index_t>(target));
        index_t row = static_cast<index_t>(next - rowDelimiters.begin());

        // Pick the closer of the two adjacent CSR row boundaries.
        if (next != rowDelimiters.begin() && next != rowDelimiters.end()) {
            const uint64_t nextDistance = static_cast<uint64_t>(*next) - target;
            const uint64_t prevDistance =
                target - static_cast<uint64_t>(*(next - 1));
            if (prevDistance <= nextDistance) {
                --row;
            }
        }

        // Give every rank a row when possible. This also keeps the boundaries
        // monotonic for matrices containing runs of empty rows.
        if (static_cast<uint64_t>(numRanks) <= numRows) {
            const index_t minRow = boundaries[rank - 1] + 1;
            const index_t maxRow =
                numRows - static_cast<index_t>(numRanks - rank);
            row = std::clamp(row, minRow, maxRow);
        } else {
            row = std::clamp(row, boundaries[rank - 1], numRows);
        }
        boundaries[rank] = row;
    }

    return boundaries;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    int parseStatus = 0;  // 0: run, 1: help, 2: invalid command line

    // Rank zero owns argument parsing and all user-visible output.
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
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            }
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    int validConfiguration =
        numRows > 0 && sparsity > 0 && iterations > 0 &&
        totalEntries / sparsity <= std::numeric_limits<index_t>::max() &&
        numRows <= static_cast<index_t>(INT_MAX) &&
        totalEntries / sparsity <= static_cast<uint64_t>(INT_MAX);
    if (!validConfiguration) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Invalid problem size: -n, -s, and -i must be positive; "
                         "rows and nonzeros must fit MPI collective counts\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const index_t nItems = static_cast<index_t>(totalEntries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) /
                                  static_cast<double>(totalEntries)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Rank zero constructs the same deterministic problem as the serial
    // benchmark. The dense input is replicated once; CSR rows are distributed.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // The C standard defines the implicit initial rand() seed as 1. Set it
        // explicitly so MPI/runtime internals cannot perturb benchmark data.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    std::vector<int> offsetCounts;
    std::vector<int> offsetDisplacements;
    std::vector<index_t> packedRowOffsets;

    if (rank == 0) {
        const std::vector<index_t> boundaries =
            makeRowPartitions(h_rowDelimiters, numRows, nItems, numRanks);
        rowCounts.resize(numRanks);
        rowDisplacements.resize(numRanks);
        nnzCounts.resize(numRanks);
        nnzDisplacements.resize(numRanks);
        offsetCounts.resize(numRanks);
        offsetDisplacements.resize(numRanks);
        packedRowOffsets.reserve(static_cast<size_t>(numRows) + numRanks);

        for (int worker = 0; worker < numRanks; ++worker) {
            const index_t firstRow = boundaries[worker];
            const index_t lastRow = boundaries[worker + 1];
            const index_t firstNnz = h_rowDelimiters[firstRow];
            const index_t lastNnz = h_rowDelimiters[lastRow];

            rowCounts[worker] = static_cast<int>(lastRow - firstRow);
            rowDisplacements[worker] = static_cast<int>(firstRow);
            nnzCounts[worker] = static_cast<int>(lastNnz - firstNnz);
            nnzDisplacements[worker] = static_cast<int>(firstNnz);
            offsetCounts[worker] = rowCounts[worker] + 1;
            offsetDisplacements[worker] =
                static_cast<int>(packedRowOffsets.size());
            for (index_t row = firstRow; row <= lastRow; ++row) {
                packedRowOffsets.push_back(h_rowDelimiters[row] - firstNnz);
            }
        }
    }

    int localRows = 0;
    int localNnz = 0;
    MPI_Scatter(rank == 0 ? rowCounts.data() : nullptr, 1, MPI_INT,
                &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));
    std::vector<index_t> localRowOffsets(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(static_cast<size_t>(localRows));

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedRowOffsets.data() : nullptr,
                 rank == 0 ? offsetCounts.data() : nullptr,
                 rank == 0 ? offsetDisplacements.data() : nullptr,
                 MPI_UINT32_T, localRowOffsets.data(), localRows + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // The full CSR arrays are not retained by rank zero during computation.
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
        std::vector<index_t>().swap(packedRowOffsets);
        printf("Computing SpMV...\n");
    }

    // Synchronize the start and report the slowest rank, avoiding optimistic
    // timings when rank workloads or cluster nodes differ.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowOffsets.data(),
                h_vec.data(), static_cast<index_t>(localRows), localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const bool gatherOutput = validate || printResults;
    std::vector<double> h_out;
    if (rank == 0 && gatherOutput) {
        h_out.resize(numRows);
    }
    if (gatherOutput) {
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? rowCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        const double gflops =
            (2.0 * static_cast<double>(nItems) * iterations) / duration / 1e9;
        const double avgTime =
            duration * 1000.0 / static_cast<double>(iterations);

        printf("Computation time: %ld ms\n",
               static_cast<long>(duration * 1000.0));
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid =
                verifyResults(h_reference.data(), h_out.data(), numRows);

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
