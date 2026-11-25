#include <mpi.h>

#include <algorithm>
#include <limits>
#include <ranges>
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
    const uint64_t entries = uint64_t{dim} * dim;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            uint64_t numEntriesLeft = entries - (uint64_t{i} * dim + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft == needToAssign || static_cast<double>(rand()) / RAND_MAX <= prob) {
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

// Scatter contiguous slices in windows so MPI int counts and displacements
// can also describe CSR arrays with more than INT_MAX entries.
template <typename T>
void scatterSlices(const std::vector<T>& source, std::vector<T>& local,
                   const std::vector<index_t>& offsets, MPI_Datatype type, int rank) {
    const size_t ranks = offsets.size() - 1;
    std::vector<int> counts(rank == 0 ? ranks : 0), displs(counts.size());
    for (uint64_t base = 0; base < offsets.back();) {
        const uint64_t end = std::min<uint64_t>(offsets.back(), base + std::numeric_limits<int>::max());
        if (rank == 0) {
            for (size_t r = 0; r < ranks; ++r) {
                const auto first = std::clamp<uint64_t>(offsets[r], base, end);
                const auto last = std::clamp<uint64_t>(offsets[r + 1], base, end);
                counts[r] = static_cast<int>(last - first);
                displs[r] = static_cast<int>(first - base);
            }
        }
        const auto first = std::clamp<uint64_t>(base, offsets[rank], offsets[rank + 1]);
        const auto last = std::clamp<uint64_t>(end, offsets[rank], offsets[rank + 1]);
        MPI_Scatterv(rank == 0 ? source.data() + base : nullptr, counts.data(), displs.data(),
                     type, last > first ? local.data() + (first - offsets[rank]) : nullptr,
                     static_cast<int>(last - first), type, 0, MPI_COMM_WORLD);
        base = end;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
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

    const uint64_t entries = uint64_t{numRows} * numRows;
    if (sparsity == 0 || numRows > std::numeric_limits<int>::max() ||
        entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Unsupported matrix size or sparsity.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);
    std::vector<double> vec(numRows), reference;
    std::vector<index_t> rowStarts(size_t(ranks) + 1), itemStarts(rowStarts.size());
    std::vector<index_t> localRows, localCols;
    std::vector<double> localValues;
    {
        std::vector<double> values;
        std::vector<index_t> cols, rows;
        if (rank == 0) {
            printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
            printf("Matrix size: %u x %u\n", numRows, numRows);
            printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
            printf("Non-zero elements: %u (%.2f%% sparse)\n",
                   nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
            printf("Iterations: %u\n", iterations);
            printf("Max value: %.2f\n", maxVal);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing data structures...\n");
            values.resize(nItems);
            cols.resize(nItems);
            rows.resize(size_t(numRows) + 1);
            srand(1);  // Original default rand() stream, independent of MPI startup.
            fill(vec.data(), numRows, maxVal);
            fill(values.data(), nItems, maxVal);
            initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
            if (validate) {
                printf("Computing reference solution...\n");
                reference.resize(numRows);
                spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
            }

            // Balance both per-row overhead and nonzeros while keeping each row intact.
            const uint64_t work = uint64_t{numRows} + nItems;
            const auto rowIndices = std::views::iota(index_t{0}, numRows + 1);
            auto workBefore = [&](index_t row) { return uint64_t{row} + rows[row]; };
            for (int r = 1; r < ranks; ++r) {
                const uint64_t target = work * r / ranks;
                rowStarts[r] = *std::ranges::lower_bound(rowIndices, target, {}, workBefore);
                itemStarts[r] = rows[rowStarts[r]];
            }
        }
        MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(rowStarts.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(itemStarts.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        rowStarts.back() = numRows;
        itemStarts.back() = nItems;
        if (ranks == 1) {
            localValues.swap(values);
            localCols.swap(cols);
            localRows.swap(rows);
        } else {
            localRows.resize(size_t(rowStarts[rank + 1] - rowStarts[rank]) + 1);
            localValues.resize(itemStarts[rank + 1] - itemStarts[rank]);
            localCols.resize(localValues.size());
            scatterSlices(rows, localRows, rowStarts, MPI_UINT32_T, rank);
            scatterSlices(values, localValues, itemStarts, MPI_DOUBLE, rank);
            scatterSlices(cols, localCols, itemStarts, MPI_UINT32_T, rank);
            localRows.back() = itemStarts[rank + 1];
            for (auto& offset : localRows) offset -= itemStarts[rank];
        }
    }

    const index_t localDim = rowStarts[rank + 1] - rowStarts[rank];
    std::vector<double> out(localDim);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localRows.data(), vec.data(), localDim, out.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<double> result;
        std::vector<int> counts, displs;
        if (rank == 0) {
            result.resize(numRows);
            counts.resize(ranks);
            displs.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = rowStarts[r + 1] - rowStarts[r];
                displs[r] = rowStarts[r];
            }
        }
        MPI_Gatherv(out.data(), localDim, MPI_DOUBLE, result.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(result, "OutputVector");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = verifyResults(reference.data(), result.data(), numRows);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
