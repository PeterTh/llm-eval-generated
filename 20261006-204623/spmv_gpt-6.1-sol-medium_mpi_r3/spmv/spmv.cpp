#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            uint64_t numEntriesLeft = uint64_t{dim} * dim - (uint64_t{i} * dim + j);
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

// Transfer contiguous segments with MPI-3 collectives. Windowing keeps counts
// and displacements representable as int even for CSR arrays larger than INT_MAX.
template <typename T>
void transferSegments(T* global, T* local, MPI_Datatype type,
                      const std::vector<uint64_t>& counts,
                      const std::vector<uint64_t>& offsets, uint64_t total,
                      int rank, bool scatter) {
    std::vector<int> windowCounts(counts.size());
    std::vector<int> windowOffsets(counts.size());
    for (uint64_t begin = 0; begin < total; begin += INT_MAX) {
        const uint64_t end = std::min(total, begin + INT_MAX);
        for (size_t r = 0; r < counts.size(); ++r) {
            const uint64_t first = std::max(begin, offsets[r]);
            const uint64_t last = std::min(end, offsets[r] + counts[r]);
            windowCounts[r] = last > first ? static_cast<int>(last - first) : 0;
            windowOffsets[r] = windowCounts[r] ? static_cast<int>(first - begin) : 0;
        }
        const uint64_t localOffset = windowCounts[rank]
            ? std::max(begin, offsets[rank]) - offsets[rank] : 0;
        // Use a valid dummy buffer for empty segments (including zero-row ranks).
        T dummy{};
        T* localBuffer = windowCounts[rank] ? local + localOffset : &dummy;
        T* globalBuffer = rank == 0 ? global + begin : nullptr;
        if (scatter) {
            MPI_Scatterv(globalBuffer, windowCounts.data(), windowOffsets.data(), type,
                         localBuffer, windowCounts[rank], type, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(localBuffer, windowCounts[rank], type, globalBuffer,
                        windowCounts.data(), windowOffsets.data(), type, 0, MPI_COMM_WORLD);
        }
    }
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

    // Every rank parses the same command line; only rank zero writes output.
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

    if (sparsity == 0) {
        if (rank == 0) fprintf(stderr, "Sparsity must be greater than zero.\n");
        MPI_Finalize();
        return 1;
    }
    const uint64_t matrixEntries = uint64_t{numRows} * numRows;
    const uint64_t nonzeros = matrixEntries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds the CSR index range.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               matrixEntries ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> h_vec(numRows);
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_reference;
    std::vector<uint64_t> rowOffsets(static_cast<size_t>(ranks) + 1);
    std::vector<uint64_t> nnzOffsets(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // Preserve the serial random stream and the order of initialization.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // Balance nonzeros plus row overhead, keeping every row on one rank.
        // This also balances empty matrices and allows ranks with no rows.
        const uint64_t totalWork = nonzeros + numRows;
        for (int r = 1; r < ranks; ++r) {
            const uint64_t target = totalWork * static_cast<uint64_t>(r) / ranks;
            uint64_t first = 0;
            uint64_t last = numRows;
            while (first < last) {
                const uint64_t mid = first + (last - first) / 2;
                if (h_rowDelimiters[mid] + mid < target) first = mid + 1;
                else last = mid;
            }
            rowOffsets[r] = first;
        }
        rowOffsets[ranks] = numRows;
        for (int r = 0; r <= ranks; ++r) {
            nnzOffsets[r] = h_rowDelimiters[rowOffsets[r]];
        }
    }
    MPI_Bcast(rowOffsets.data(), ranks + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzOffsets.data(), ranks + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    for (uint64_t begin = 0; begin < numRows; begin += INT_MAX) {
        const int count = static_cast<int>(std::min(uint64_t{numRows} - begin,
                                                    uint64_t{INT_MAX}));
        MPI_Bcast(h_vec.data() + begin, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    std::vector<uint64_t> rowCounts(ranks), nnzCounts(ranks);
    for (int r = 0; r < ranks; ++r) {
        rowCounts[r] = rowOffsets[r + 1] - rowOffsets[r];
        nnzCounts[r] = nnzOffsets[r + 1] - nnzOffsets[r];
    }
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    std::vector<double> localVal(nnzCounts[rank]);
    std::vector<index_t> localCols(nnzCounts[rank]);
    std::vector<index_t> localPointers(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);
    transferSegments(h_val.data(), localVal.data(), MPI_DOUBLE, nnzCounts,
                     nnzOffsets, nonzeros, rank, true);
    transferSegments(h_cols.data(), localCols.data(), MPI_UINT32_T, nnzCounts,
                     nnzOffsets, nonzeros, rank, true);
    // Scatter row starts without overlapping send segments; the terminal
    // pointer follows directly from the local nonzero count.
    transferSegments(h_rowDelimiters.data(), localPointers.data(), MPI_UINT32_T,
                     rowCounts, rowOffsets, numRows, rank, true);
    for (index_t row = 0; row < localRows; ++row) {
        localPointers[row] -= static_cast<index_t>(nnzOffsets[rank]);
    }
    localPointers[localRows] = static_cast<index_t>(nnzCounts[rank]);

    // Only local CSR storage and the fixed input vector are needed from here on.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localPointers.data(),
                h_vec.data(), localRows, localOut.data());
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_out;
    if (validate || printResults) {
        if (rank == 0) h_out.resize(numRows);
        transferSegments(h_out.data(), localOut.data(), MPI_DOUBLE, rowCounts,
                         rowOffsets, numRows, rank, false);
    }
    int result = 0;
    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));
        const double gflops = seconds > 0.0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double avgTime = iterations ? milliseconds / iterations : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
