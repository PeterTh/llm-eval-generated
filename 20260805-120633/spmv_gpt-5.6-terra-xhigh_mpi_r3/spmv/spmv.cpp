#include <algorithm>
#include <cstddef>
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
    const std::uint64_t totalEntries = static_cast<std::uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const std::uint64_t numEntriesLeft = totalEntries -
                (static_cast<std::uint64_t>(i) * dim + j);
            const std::uint64_t needToAssign = n - nnzAssigned;
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

// Returns zero to continue, one for a successful immediate exit (-h), and
// minus one for invalid arguments.  Only rank zero parses and prints usage so
// an MPI launch has the same user-facing output as the original program.
int parseCommandLine(int argc, char** argv, index_t& numRows, index_t& sparsity,
                     index_t& iterations, double& maxVal, bool& validate,
                     bool& printResults) {
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
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

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

    int parseStatus = 0;
    if (rank == 0) {
        parseStatus = parseCommandLine(argc, argv, numRows, sparsity, iterations,
                                       maxVal, validate, printResults);
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus < 0 ? 1 : 0;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    // MPI collective counts are ints.  The original representation uses
    // 32-bit indexes, so reject instances that cannot be represented safely
    // by the MPI collectives used for CSR distribution.
    int inputValid = sparsity != 0 &&
                     numRows <= static_cast<index_t>(std::numeric_limits<int>::max());
    const std::uint64_t requestedNnz =
        sparsity == 0 ? 0 : (static_cast<std::uint64_t>(numRows) * numRows) / sparsity;
    inputValid = inputValid &&
                 requestedNnz <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (!inputValid) {
        if (rank == 0) {
            fprintf(stderr, "Matrix dimensions exceed MPI collective limits or sparsity is zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const index_t nItems = static_cast<index_t>(requestedNnz);

    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;

    // Generate the benchmark input on one rank to retain the original random
    // sequence exactly, then distribute row-owned CSR segments below.
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // The reference remains a serial calculation on rank zero and is
        // intentionally outside the timed MPI computation.
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // NNZ-balanced contiguous row blocks keep each local CSR segment compact,
    // while avoiding the load imbalance that equal-sized row blocks can have
    // for an irregular sparse matrix.  They also let the output be gathered
    // directly into global row order.
    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    std::vector<int> nnzCounts(worldSize);
    std::vector<int> nnzDisplacements(worldSize);
    if (rank == 0) {
        std::vector<index_t> rowBoundaries(static_cast<size_t>(worldSize) + 1);
        rowBoundaries.front() = 0;
        size_t previousBoundary = 0;
        const std::uint64_t totalNnz = h_rowDelimiters[numRows];

        for (int process = 1; process < worldSize; ++process) {
            const std::uint64_t targetNnz = (totalNnz * process) / worldSize;
            auto boundary = std::lower_bound(
                h_rowDelimiters.begin() + static_cast<std::ptrdiff_t>(previousBoundary),
                h_rowDelimiters.end(), static_cast<index_t>(targetNnz));
            size_t boundaryRow = static_cast<size_t>(boundary - h_rowDelimiters.begin());

            // Pick the closest row boundary to the ideal NNZ split.  The
            // lower_bound result is never before the previous boundary, so
            // partitions remain ordered even when rows have no nonzeros.
            if (boundaryRow > previousBoundary) {
                const std::uint64_t beforeDistance = targetNnz -
                    h_rowDelimiters[boundaryRow - 1];
                const std::uint64_t afterDistance =
                    h_rowDelimiters[boundaryRow] - targetNnz;
                if (beforeDistance <= afterDistance) {
                    --boundaryRow;
                }
            }
            rowBoundaries[process] = static_cast<index_t>(boundaryRow);
            previousBoundary = boundaryRow;
        }
        rowBoundaries.back() = numRows;

        for (int process = 0; process < worldSize; ++process) {
            const index_t firstRow = rowBoundaries[process];
            const index_t lastRow = rowBoundaries[process + 1];
            rowCounts[process] = static_cast<int>(lastRow - firstRow);
            rowDisplacements[process] = static_cast<int>(firstRow);
            nnzDisplacements[process] = static_cast<int>(h_rowDelimiters[firstRow]);
            nnzCounts[process] = static_cast<int>(
                h_rowDelimiters[lastRow] - h_rowDelimiters[firstRow]);
        }
    }

    int localRows = 0;
    int localNnz = 0;
    MPI_Scatter(rank == 0 ? rowCounts.data() : nullptr, 1, MPI_INT,
                &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<double> localOut(localRows);

    // Scatter the first delimiter for each row.  The final delimiter is the
    // local NNZ count; changing the received global offsets to local offsets
    // makes the existing CSR kernel directly usable on every rank.
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRows == 0 ? nullptr : localRowDelimiters.data(), localRows,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (localRows > 0) {
        const index_t localOffset = localRowDelimiters[0];
        for (int row = 0; row < localRows; ++row) {
            localRowDelimiters[row] -= localOffset;
        }
    }
    localRowDelimiters[localRows] = static_cast<index_t>(localNnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localNnz == 0 ? nullptr : localVal.data(), localNnz,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localNnz == 0 ? nullptr : localCols.data(), localNnz,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // The global CSR copies are no longer needed after distribution.  Keeping
    // only local matrix storage is important for distributed-memory scaling.
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), static_cast<index_t>(localRows), localOut.data());
    }
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(localRows == 0 ? nullptr : localOut.data(), localRows, MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr,
                rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        const double durationMilliseconds = duration * 1000.0;
        printf("Computation time: %.3f ms\n", durationMilliseconds);

        // Report the global work divided by the slowest rank's elapsed time.
        const double gflops = duration > 0.0
            ? (2.0 * nItems * iterations) / duration / 1e9
            : 0.0;
        const double avgTime = iterations > 0
            ? durationMilliseconds / static_cast<double>(iterations)
            : 0.0;
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
