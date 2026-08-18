#include <algorithm>
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
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                            (static_cast<uint64_t>(i) * dim + j);
            const uint64_t needToAssign = static_cast<uint64_t>(n) - nnzAssigned;
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
[[gnu::noinline]] void spmvCpu(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t dim,
                              double* __restrict__ out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// Divide contiguous rows so that each rank receives approximately the same
// number of nonzeros plus rows. Including rows in the cost also balances empty
// and very sparse matrices.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowDelimiters,
                                   const index_t dim, const index_t nnz,
                                   const int processCount) {
    std::vector<index_t> boundaries(static_cast<size_t>(processCount) + 1);
    boundaries.front() = 0;
    boundaries.back() = dim;

    const uint64_t totalWork = static_cast<uint64_t>(nnz) + dim;
    for (int rank = 1; rank < processCount; ++rank) {
        // unsigned __int128 prevents overflow when forming the fractional target.
        const uint64_t target = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(totalWork) * rank) / processCount);

        index_t low = boundaries[static_cast<size_t>(rank) - 1];
        index_t high = dim;
        while (low < high) {
            const index_t mid = low + (high - low) / 2;
            const uint64_t workThroughMid = static_cast<uint64_t>(rowDelimiters[mid]) + mid;
            if (workThroughMid < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        boundaries[rank] = low;
    }
    return boundaries;
}

template <typename T>
void mpiBcastLarge(T* data, size_t count, MPI_Datatype datatype, int root,
                   MPI_Comm communicator) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        MPI_Bcast(data + offset, chunk, datatype, root, communicator);
    }
}

template <typename T>
void mpiSendLarge(const T* data, size_t count, MPI_Datatype datatype, int destination,
                  int tag, MPI_Comm communicator) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        MPI_Send(data + offset, chunk, datatype, destination, tag, communicator);
    }
}

template <typename T>
void mpiRecvLarge(T* data, size_t count, MPI_Datatype datatype, int source,
                  int tag, MPI_Comm communicator) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        MPI_Recv(data + offset, chunk, datatype, source, tag, communicator,
                 MPI_STATUS_IGNORE);
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
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

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

    const uint64_t matrixElements = static_cast<uint64_t>(numRows) * numRows;
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        matrixElements / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            fprintf(stderr,
                    "Matrix size, sparsity, and iterations must be positive, and the "
                    "number of nonzeros must fit in a 32-bit CSR index.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements without overflowing the 32-bit index type.
    const index_t nItems = static_cast<index_t>(matrixElements / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / matrixElements));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", processCount);
    }

    // The input vector is read-only during the benchmark and is replicated once.
    // Matrix rows, values, columns, and output remain distributed.
    std::vector<double> h_vec(numRows);
    std::vector<double> globalValues;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalValues.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);

        // The C library specifies that the default state is equivalent to srand(1).
        // Seed explicitly so MPI-library internals cannot perturb the serial data set.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);
    }
    mpiBcastLarge(h_vec.data(), h_vec.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(globalValues.data(), globalCols.data(), globalRowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Compute a contiguous, work-balanced row partition on rank zero.
    std::vector<index_t> rowBoundaries(static_cast<size_t>(processCount) + 1);
    std::vector<uint64_t> rankNnzCounts;
    std::vector<uint64_t> rankNnzDisplacements;
    if (rank == 0) {
        rowBoundaries = partitionRows(globalRowDelimiters, numRows, nItems, processCount);
        rankNnzCounts.resize(processCount);
        rankNnzDisplacements.resize(processCount);
        for (int target = 0; target < processCount; ++target) {
            const index_t firstRow = rowBoundaries[target];
            const index_t endRow = rowBoundaries[target + 1];
            rankNnzDisplacements[target] = globalRowDelimiters[firstRow];
            rankNnzCounts[target] = static_cast<uint64_t>(globalRowDelimiters[endRow]) -
                                    globalRowDelimiters[firstRow];
        }
    }
    MPI_Bcast(rowBoundaries.data(), processCount + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t firstLocalRow = rowBoundaries[rank];
    const index_t localRows = rowBoundaries[rank + 1] - firstLocalRow;
    uint64_t localNnz64 = 0;
    uint64_t firstLocalNnz64 = 0;
    MPI_Scatter(rank == 0 ? rankNnzCounts.data() : nullptr, 1, MPI_UINT64_T,
                &localNnz64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? rankNnzDisplacements.data() : nullptr, 1, MPI_UINT64_T,
                &firstLocalNnz64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localValues(static_cast<size_t>(localNnz64));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz64));

    // Use optimized MPI collectives whenever their standard int count/displacement
    // interface can represent the data. The point-to-point fallback handles larger CSR.
    int collectiveDistribution = 1;
    std::vector<int> rowSendCounts;
    std::vector<int> rowSendDisplacements;
    std::vector<int> nnzSendCounts;
    std::vector<int> nnzSendDisplacements;
    if (rank == 0) {
        rowSendCounts.resize(processCount);
        rowSendDisplacements.resize(processCount);
        nnzSendCounts.resize(processCount);
        nnzSendDisplacements.resize(processCount);
        for (int target = 0; target < processCount; ++target) {
            const uint64_t rows = static_cast<uint64_t>(rowBoundaries[target + 1]) -
                                  rowBoundaries[target];
            if (rows + 1 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                rowBoundaries[target] > static_cast<index_t>(std::numeric_limits<int>::max()) ||
                rankNnzCounts[target] > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                rankNnzDisplacements[target] >
                    static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                collectiveDistribution = 0;
                break;
            }
            rowSendCounts[target] = static_cast<int>(rows + 1);
            rowSendDisplacements[target] = static_cast<int>(rowBoundaries[target]);
            nnzSendCounts[target] = static_cast<int>(rankNnzCounts[target]);
            nnzSendDisplacements[target] = static_cast<int>(rankNnzDisplacements[target]);
        }
    }
    MPI_Bcast(&collectiveDistribution, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (collectiveDistribution) {
        MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                     rank == 0 ? rowSendCounts.data() : nullptr,
                     rank == 0 ? rowSendDisplacements.data() : nullptr, MPI_UINT32_T,
                     localRowDelimiters.data(), static_cast<int>(localRows) + 1,
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                     rank == 0 ? nnzSendCounts.data() : nullptr,
                     rank == 0 ? nnzSendDisplacements.data() : nullptr, MPI_DOUBLE,
                     localValues.data(), static_cast<int>(localNnz64), MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                     rank == 0 ? nnzSendCounts.data() : nullptr,
                     rank == 0 ? nnzSendDisplacements.data() : nullptr, MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNnz64), MPI_UINT32_T, 0,
                     MPI_COMM_WORLD);
    } else if (rank == 0) {
        for (int target = 0; target < processCount; ++target) {
            const size_t rows = static_cast<size_t>(rowBoundaries[target + 1]) -
                                rowBoundaries[target];
            const size_t nnz = static_cast<size_t>(rankNnzCounts[target]);
            const size_t nnzOffset = static_cast<size_t>(rankNnzDisplacements[target]);
            if (target == 0) {
                std::copy_n(globalRowDelimiters.data() + rowBoundaries[target], rows + 1,
                            localRowDelimiters.data());
                std::copy_n(globalValues.data() + nnzOffset, nnz, localValues.data());
                std::copy_n(globalCols.data() + nnzOffset, nnz, localCols.data());
            } else {
                mpiSendLarge(globalRowDelimiters.data() + rowBoundaries[target], rows + 1,
                             MPI_UINT32_T, target, 100, MPI_COMM_WORLD);
                mpiSendLarge(globalValues.data() + nnzOffset, nnz, MPI_DOUBLE, target, 101,
                             MPI_COMM_WORLD);
                mpiSendLarge(globalCols.data() + nnzOffset, nnz, MPI_UINT32_T, target, 102,
                             MPI_COMM_WORLD);
            }
        }
    } else {
        mpiRecvLarge(localRowDelimiters.data(), localRowDelimiters.size(), MPI_UINT32_T, 0,
                     100, MPI_COMM_WORLD);
        mpiRecvLarge(localValues.data(), localValues.size(), MPI_DOUBLE, 0, 101,
                     MPI_COMM_WORLD);
        mpiRecvLarge(localCols.data(), localCols.size(), MPI_UINT32_T, 0, 102,
                     MPI_COMM_WORLD);
    }

    const index_t firstLocalNnz = static_cast<index_t>(firstLocalNnz64);
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= firstLocalNnz;
    }

    // Release the root-only copy before the benchmark. Rank zero now has the same
    // distributed memory footprint as every other rank (apart from validation data).
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRowDelimiters);
    }

    std::vector<double> localOut(localRows);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops = (2.0 * nItems * iterations) / duration / 1e9;
        const double avgTimeMs = duration * 1000.0 / iterations;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTimeMs);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the result only for features that require the complete vector.
    std::vector<double> h_out;
    if (rank == 0 && (validate || printResults)) {
        h_out.resize(numRows);
    }
    if (validate || printResults) {
        bool collectiveGather = true;
        for (int target = 0; target < processCount; ++target) {
            const uint64_t rows = static_cast<uint64_t>(rowBoundaries[target + 1]) -
                                  rowBoundaries[target];
            if (rows > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                rowBoundaries[target] > static_cast<index_t>(std::numeric_limits<int>::max())) {
                collectiveGather = false;
                break;
            }
        }

        if (collectiveGather) {
            std::vector<int> outputCounts;
            std::vector<int> outputDisplacements;
            if (rank == 0) {
                outputCounts.resize(processCount);
                outputDisplacements.resize(processCount);
                for (int target = 0; target < processCount; ++target) {
                    outputCounts[target] = static_cast<int>(rowBoundaries[target + 1] -
                                                            rowBoundaries[target]);
                    outputDisplacements[target] = static_cast<int>(rowBoundaries[target]);
                }
            }
            MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                        rank == 0 ? h_out.data() : nullptr,
                        rank == 0 ? outputCounts.data() : nullptr,
                        rank == 0 ? outputDisplacements.data() : nullptr, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
        } else if (rank == 0) {
            std::copy(localOut.begin(), localOut.end(), h_out.begin() + firstLocalRow);
            for (int source = 1; source < processCount; ++source) {
                const size_t rows = static_cast<size_t>(rowBoundaries[source + 1]) -
                                    rowBoundaries[source];
                mpiRecvLarge(h_out.data() + rowBoundaries[source], rows, MPI_DOUBLE, source,
                             200, MPI_COMM_WORLD);
            }
        } else {
            mpiSendLarge(localOut.data(), localOut.size(), MPI_DOUBLE, 0, 200,
                         MPI_COMM_WORLD);
        }
    }

    int exitCode = 0;
    if (rank == 0 && printResults) {
        print_results(h_out, "OutputVector");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
