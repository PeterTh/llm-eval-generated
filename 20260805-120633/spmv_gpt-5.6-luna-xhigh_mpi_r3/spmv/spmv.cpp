#include <mpi.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

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
//   cols: array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n: number of nonzero elements
//   dim: number of rows/columns in the matrix
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
//   indexOffset: global index of the first local element
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t indexOffset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       i + indexOffset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + indexOffset, ref, res, relError);
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

    // Parse command line arguments. Every rank parses the same arguments so
    // that all ranks take the same control-flow decisions.
    bool parseError = false;
    bool showHelp = false;
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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parseError || sparsity == 0) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or incomplete command line option\n");
            } else {
                printf("Sparsity must be greater than zero\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // MPI_Scatterv uses int counts and displacements. This limit is much
    // larger than practical allocations for this benchmark and avoids silent
    // truncation of CSR offsets in the communication setup.
    const std::uint64_t matrixElementCount =
        static_cast<std::uint64_t>(numRows) * static_cast<std::uint64_t>(numRows);
    const std::uint64_t nItems64 = matrixElementCount / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max() || nItems64 > INT_MAX ||
        static_cast<std::uint64_t>(numRows) > INT_MAX) {
        if (rank == 0) {
            printf("Matrix is too large for the MPI communication count limits\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems64) /
                                static_cast<double>(matrixElementCount)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing distributed data structures...\n");
    }

    // Rank zero reproduces the original initialization stream exactly, then
    // the CSR rows are partitioned and scattered. The matrix is distributed;
    // only the dense input vector is replicated because every local row may
    // reference any column.
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> h_vec(numRows);                  // Dense vector

    std::vector<index_t> rowStarts(static_cast<size_t>(worldSize) + 1, 0);
    std::vector<index_t> localNnzByRank;
    std::vector<int> rowCounts;
    std::vector<int> rowDelimiterCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;

    if (rank == 0) {
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);

        // rand() starts in the same state as srand(1) in the original
        // single-process program. Make that state explicit after MPI_Init.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);

        // Use contiguous row ranges and choose boundaries by cumulative NNZ so
        // that irregular random rows do not create avoidable load imbalance.
        index_t nextRow = 0;
        for (int r = 1; r < worldSize; ++r) {
            const std::uint64_t targetNnz =
                (nItems64 * static_cast<std::uint64_t>(r)) /
                static_cast<std::uint64_t>(worldSize);
            while (nextRow < numRows &&
                   static_cast<std::uint64_t>(globalRowDelimiters[nextRow]) < targetNnz) {
                ++nextRow;
            }
            rowStarts[r] = nextRow;
        }
        rowStarts[worldSize] = numRows;

        localNnzByRank.resize(worldSize);
        rowCounts.resize(worldSize);
        rowDelimiterCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        nnzCounts.resize(worldSize);
        nnzDisplacements.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const index_t beginRow = rowStarts[r];
            const index_t endRow = rowStarts[r + 1];
            const index_t beginNnz = globalRowDelimiters[beginRow];
            const index_t endNnz = globalRowDelimiters[endRow];
            localNnzByRank[r] = endNnz - beginNnz;
            rowCounts[r] = static_cast<int>(endRow - beginRow);
            rowDelimiterCounts[r] = rowCounts[r] + 1;
            rowDisplacements[r] = static_cast<int>(beginRow);
            nnzCounts[r] = static_cast<int>(localNnzByRank[r]);
            nnzDisplacements[r] = static_cast<int>(beginNnz);
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowStarts.data(), worldSize + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t localRowEnd = rowStarts[rank + 1];
    const index_t localRows = localRowEnd - rowStarts[rank];
    index_t localNnz = 0;
    MPI_Scatter(rank == 0 ? localNnzByRank.data() : nullptr, 1, MPI_UINT32_T,
                &localNnz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<double> h_val(localNnz);                 // Local non-zero values
    std::vector<index_t> h_cols(localNnz);               // Local column indices
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> h_out(localRows);                // Local output rows

    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_DOUBLE, h_val.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_UINT32_T, h_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rank == 0 ? rowDelimiterCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr,
                 MPI_UINT32_T, h_rowDelimiters.data(), static_cast<int>(localRows) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Local CSR offsets must start at zero after the global CSR slice is
    // received. The first offset is present even for an empty row partition.
    const index_t localNnzOffset = h_rowDelimiters[0];
    for (index_t& offset : h_rowDelimiters) {
        offset -= localNnzOffset;
    }

    // The root no longer needs the global CSR representation after the
    // one-time distribution, which keeps its steady-state memory footprint
    // comparable to the other ranks.
    if (rank == 0) {
        std::vector<double>().swap(globalVal);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRowDelimiters);
    }

    // For validation, compute a local reference using the same row ownership
    // and the same serial accumulation order as the original implementation.
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation. Since the input vector is invariant, all
    // communication is deliberately outside this timed loop.
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics using the unrounded wall-clock time;
        // millisecond rounding makes short runs report misleading GFLOPS.
        const double gflops = elapsed > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * iterations) /
                                        elapsed / 1e9
                                  : 0.0;
        const double avgTime = iterations > 0
                                   ? elapsed * 1000.0 / static_cast<double>(iterations)
                                   : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather only when requested. The timed path remains distributed and has
    // no per-iteration communication.
    std::vector<double> globalOut;
    std::vector<double> globalReference;
    if (printResults || validate) {
        if (rank == 0) {
            globalOut.resize(numRows);
        }
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOut.data() : nullptr,
                    rank == 0 ? rowCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (validate) {
            if (rank == 0) {
                globalReference.resize(numRows);
            }
            MPI_Gatherv(h_reference.data(), static_cast<int>(localRows), MPI_DOUBLE,
                        rank == 0 ? globalReference.data() : nullptr,
                        rank == 0 ? rowCounts.data() : nullptr,
                        rank == 0 ? rowDisplacements.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // Print results for external validation in global row order.
    if (printResults && rank == 0) {
        print_results(globalOut, "OutputVector");
    }

    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(globalReference.data(), globalOut.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
