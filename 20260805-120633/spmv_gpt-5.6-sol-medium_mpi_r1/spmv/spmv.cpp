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
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                            (static_cast<uint64_t>(i) * dim + j);
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

// MPI collectives use int counts.  Windowing keeps the implementation correct
// for CSR arrays whose global offsets exceed INT_MAX while retaining optimized
// collective communication algorithms.
constexpr uint64_t MPI_WINDOW_ELEMENTS = uint64_t{1} << 30;

template <typename T>
void broadcastLarge(T* data, const uint64_t count, const MPI_Datatype datatype,
                    const int root, MPI_Comm comm) {
    for (uint64_t offset = 0; offset < count; offset += MPI_WINDOW_ELEMENTS) {
        const int chunk = static_cast<int>(std::min(MPI_WINDOW_ELEMENTS, count - offset));
        MPI_Bcast(data + offset, chunk, datatype, root, comm);
    }
}

template <typename T>
void scatterContiguous(const T* rootData, const uint64_t totalCount,
                       const std::vector<uint64_t>& rootStarts,
                       const std::vector<uint64_t>& rootLengths,
                       T* localData, const uint64_t localStart,
                       const uint64_t localCount, const MPI_Datatype datatype,
                       const int root, const int rank, const int ranks,
                       MPI_Comm comm) {
    std::vector<int> counts(rank == root ? ranks : 0);
    std::vector<int> displacements(rank == root ? ranks : 0);
    const uint64_t localEnd = localStart + localCount;

    for (uint64_t window = 0; window < totalCount; window += MPI_WINDOW_ELEMENTS) {
        const uint64_t windowEnd = std::min(totalCount, window + MPI_WINDOW_ELEMENTS);
        if (rank == root) {
            for (int r = 0; r < ranks; ++r) {
                const uint64_t begin = std::max(rootStarts[r], window);
                const uint64_t end = std::min(rootStarts[r] + rootLengths[r], windowEnd);
                counts[r] = static_cast<int>(end > begin ? end - begin : 0);
                displacements[r] = static_cast<int>(begin - window);
            }
        }
        const uint64_t recvBegin = std::max(localStart, window);
        const uint64_t recvEnd = std::min(localEnd, windowEnd);
        const int recvCount = static_cast<int>(recvEnd > recvBegin ? recvEnd - recvBegin : 0);
        T* recvBuffer = recvCount == 0 ? localData : localData + (recvBegin - localStart);
        MPI_Scatterv(rank == root ? rootData + window : nullptr,
                     rank == root ? counts.data() : nullptr,
                     rank == root ? displacements.data() : nullptr, datatype,
                     recvBuffer, recvCount, datatype, root, comm);
    }
}

template <typename T>
void gatherContiguous(const T* localData, const uint64_t localStart,
                      const uint64_t localCount, T* rootData,
                      const uint64_t totalCount,
                      const std::vector<uint64_t>& rootStarts,
                      const std::vector<uint64_t>& rootLengths,
                      const MPI_Datatype datatype, const int root,
                      const int rank, const int ranks, MPI_Comm comm) {
    std::vector<int> counts(rank == root ? ranks : 0);
    std::vector<int> displacements(rank == root ? ranks : 0);
    const uint64_t localEnd = localStart + localCount;

    for (uint64_t window = 0; window < totalCount; window += MPI_WINDOW_ELEMENTS) {
        const uint64_t windowEnd = std::min(totalCount, window + MPI_WINDOW_ELEMENTS);
        if (rank == root) {
            for (int r = 0; r < ranks; ++r) {
                const uint64_t begin = std::max(rootStarts[r], window);
                const uint64_t end = std::min(rootStarts[r] + rootLengths[r], windowEnd);
                counts[r] = static_cast<int>(end > begin ? end - begin : 0);
                displacements[r] = static_cast<int>(begin - window);
            }
        }
        const uint64_t sendBegin = std::max(localStart, window);
        const uint64_t sendEnd = std::min(localEnd, windowEnd);
        const int sendCount = static_cast<int>(sendEnd > sendBegin ? sendEnd - sendBegin : 0);
        const T* sendBuffer = sendCount == 0 ? localData : localData + (sendBegin - localStart);
        MPI_Gatherv(sendBuffer, sendCount, datatype,
                    rank == root ? rootData + window : nullptr,
                    rank == root ? counts.data() : nullptr,
                    rank == root ? displacements.data() : nullptr, datatype,
                    root, comm);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse once so all ranks use an identical configuration and only rank 0
    // writes benchmark output.
    int parseStatus = 0;  // 0: run, 1: error, 2: help
    for (int i = 1; rank == 0 && i < argc; ++i) {
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
            parseStatus = 2;
            break;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            parseStatus = 1;
            break;
        }
    }

    if (rank == 0 && parseStatus == 0 &&
        (numRows == 0 || sparsity == 0 || iterations == 0)) {
        fprintf(stderr, "Error: -n, -s, and -i must all be greater than zero\n");
        parseStatus = 1;
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, comm);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 1 : 0;
    }

    index_t integerOptions[3] = {numRows, sparsity, iterations};
    int flagOptions[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(integerOptions, 3, MPI_UINT32_T, 0, comm);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(flagOptions, 2, MPI_INT, 0, comm);
    numRows = integerOptions[0];
    sparsity = integerOptions[1];
    iterations = integerOptions[2];
    validate = flagOptions[0] != 0;
    printResults = flagOptions[1] != 0;

    // Calculate number of non-zero elements
    const uint64_t requestedItems = static_cast<uint64_t>(numRows) * numRows / sparsity;
    if (requestedItems > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of non-zero elements exceeds CSR index capacity\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(requestedItems);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) /
                                  (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
    }

    // Rank 0 preserves the original generator's exact random-number ordering.
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(rank == 0 ? numRows + 1 : 0);
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    broadcastLarge(h_vec.data(), numRows, MPI_DOUBLE, 0, comm);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Choose contiguous row boundaries nearest equal-NNZ targets.  Contiguous
    // ownership allows direct Gatherv output while balancing the dominant work.
    std::vector<uint64_t> rowStarts(rank == 0 ? ranks : 0);
    std::vector<uint64_t> rowCounts(rank == 0 ? ranks : 0);
    std::vector<uint64_t> nnzStarts(rank == 0 ? ranks : 0);
    std::vector<uint64_t> nnzCounts(rank == 0 ? ranks : 0);
    if (rank == 0) {
        std::vector<uint64_t> boundaries(ranks + 1);
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        for (int r = 1; r < ranks; ++r) {
            const uint64_t target = static_cast<uint64_t>(nItems) * r / ranks;
            const bool oneRowPerRank = static_cast<uint64_t>(ranks) <= numRows;
            const uint64_t minRow = oneRowPerRank ? boundaries[r - 1] + 1
                                                  : boundaries[r - 1];
            const uint64_t maxRow = oneRowPerRank
                                        ? numRows - static_cast<uint64_t>(ranks - r)
                                        : numRows;
            auto first = h_rowDelimiters.begin() + minRow;
            auto last = h_rowDelimiters.begin() + maxRow + 1;
            auto it = std::lower_bound(first, last, target);
            uint64_t chosen = std::min<uint64_t>(it - h_rowDelimiters.begin(), maxRow);
            if (chosen > minRow) {
                const uint64_t highError = h_rowDelimiters[chosen] > target
                                               ? h_rowDelimiters[chosen] - target
                                               : target - h_rowDelimiters[chosen];
                const uint64_t lowError = h_rowDelimiters[chosen - 1] > target
                                              ? h_rowDelimiters[chosen - 1] - target
                                              : target - h_rowDelimiters[chosen - 1];
                if (lowError <= highError) --chosen;
            }
            boundaries[r] = chosen;
        }
        for (int r = 0; r < ranks; ++r) {
            rowStarts[r] = boundaries[r];
            rowCounts[r] = boundaries[r + 1] - boundaries[r];
            nnzStarts[r] = h_rowDelimiters[boundaries[r]];
            nnzCounts[r] = h_rowDelimiters[boundaries[r + 1]] - nnzStarts[r];
        }
    }

    uint64_t localRowStart = 0, localRows = 0, localNnzStart = 0, localNnz = 0;
    MPI_Scatter(rank == 0 ? rowStarts.data() : nullptr, 1, MPI_UINT64_T,
                &localRowStart, 1, MPI_UINT64_T, 0, comm);
    MPI_Scatter(rank == 0 ? rowCounts.data() : nullptr, 1, MPI_UINT64_T,
                &localRows, 1, MPI_UINT64_T, 0, comm);
    MPI_Scatter(rank == 0 ? nnzStarts.data() : nullptr, 1, MPI_UINT64_T,
                &localNnzStart, 1, MPI_UINT64_T, 0, comm);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_UINT64_T,
                &localNnz, 1, MPI_UINT64_T, 0, comm);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(localRows + 1);
    scatterContiguous(h_val.data(), nItems, nnzStarts, nnzCounts, localVal.data(),
                      localNnzStart, localNnz, MPI_DOUBLE, 0, rank, ranks, comm);
    scatterContiguous(h_cols.data(), nItems, nnzStarts, nnzCounts, localCols.data(),
                      localNnzStart, localNnz, MPI_UINT32_T, 0, rank, ranks, comm);
    scatterContiguous(h_rowDelimiters.data(), numRows, rowStarts, rowCounts,
                      localRowDelimiters.data(), localRowStart, localRows,
                      MPI_UINT32_T, 0, rank, ranks, comm);
    for (uint64_t i = 0; i < localRows; ++i) {
        localRowDelimiters[i] -= static_cast<index_t>(localNnzStart);
    }
    localRowDelimiters[localRows] = static_cast<index_t>(localNnz);

    // Release the root-only global CSR before the measured region.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);
    std::vector<double> localOut(localRows);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), static_cast<index_t>(localRows), localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const double gflops = (2.0 * nItems * iterations) / duration / 1e9;
        const double avgTime = duration * 1000.0 / iterations;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> h_out(rank == 0 && (printResults || validate) ? numRows : 0);
    if (printResults || validate) {
        gatherContiguous(localOut.data(), localRowStart, localRows, h_out.data(),
                         numRows, rowStarts, rowCounts, MPI_DOUBLE, 0, rank, ranks, comm);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return exitCode;
}
