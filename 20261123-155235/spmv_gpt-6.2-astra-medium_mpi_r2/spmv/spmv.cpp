#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
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
    const uint64_t entries = static_cast<uint64_t>(dim) * dim;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const uint64_t remaining = entries - static_cast<uint64_t>(i) * dim;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            if (remaining - j == n - nnzAssigned ||
                static_cast<double>(rand()) / RAND_MAX <= prob) {
                cols[nnzAssigned++] = j;
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

// MPI_Scatterv uses signed int counts and displacements. Slice very large
// arrays into windows while keeping each rank's CSR indices local.
template <typename T>
void scatterEntries(const std::vector<T>& global, std::vector<T>& local,
                    MPI_Datatype type, const std::vector<index_t>& offsets,
                    int rank, int ranks, index_t total, index_t localStart) {
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(counts.size());
    for (uint64_t start = 0; start < total; start += INT_MAX) {
        const uint64_t end = std::min<uint64_t>(start + INT_MAX, total);
        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const auto first = std::clamp<uint64_t>(offsets[r], start, end);
                counts[r] = std::clamp<uint64_t>(offsets[r + 1], start, end) - first;
                displacements[r] = first - start;
            }
        }
        const auto first = std::clamp<uint64_t>(localStart, start, end);
        const int count = std::clamp<uint64_t>(localStart + uint64_t(local.size()), start, end) - first;
        MPI_Scatterv(rank == 0 ? global.data() + start : nullptr, counts.data(),
                     displacements.data(), type, count ? local.data() + (first - localStart) : nullptr,
                     count, type, 0, MPI_COMM_WORLD);
    }
}

int run(int argc, char** argv, int rank, int ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    const uint64_t cells = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || numRows > INT_MAX ||
        cells / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity for CSR storage.\n");
        return 1;
    }
    const index_t nItems = cells / sparsity;
    std::vector<double> vec(numRows), values, reference;
    std::vector<index_t> columns, rows, entryOffsets;
    std::vector<int> rowCounts, firstRows;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / cells));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        printf("Initializing data structures...\n");
        values.resize(nItems);
        columns.resize(nItems);
        rows.resize(size_t(numRows) + 1);
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), columns.data(), rows.data(), vec.data(),
                    numRows, reference.data());
        }

        // Charge one unit per row as well as one per nonzero so empty rows
        // also contribute to the partitioning cost.
        rowCounts.resize(ranks);
        firstRows.resize(ranks);
        entryOffsets.resize(size_t(ranks) + 1);
        const uint64_t work = uint64_t(nItems) + numRows;
        int row = 0;
        for (int r = 0; r < ranks; ++r) {
            firstRows[r] = row;
            entryOffsets[r] = rows[row];
            const uint64_t target = (work / ranks) * (r + 1) +
                                    (work % ranks) * (r + 1) / ranks;
            while (row < static_cast<int>(numRows) && uint64_t(rows[row]) + row < target) ++row;
            rowCounts[r] = row - firstRows[r];
        }
        entryOffsets[ranks] = nItems;
    }

    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localRowCount;
    index_t firstEntry, lastEntry;
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRowCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(entryOffsets.data(), 1, MPI_UINT32_T, &firstEntry, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? entryOffsets.data() + 1 : nullptr, 1, MPI_UINT32_T,
                &lastEntry, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<index_t> localRows(size_t(localRowCount) + 1);
    std::vector<double> localValues;
    std::vector<index_t> localColumns;
    if (ranks == 1) {
        localRows.swap(rows);
        localColumns.swap(columns);
        localValues.swap(values);
    } else {
        MPI_Scatterv(rows.data(), rowCounts.data(), firstRows.data(), MPI_UINT32_T,
                     localRows.data(), localRowCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        localRows.back() = lastEntry;
        for (auto& offset : localRows) offset -= firstEntry;
        localValues.resize(lastEntry - firstEntry);
        scatterEntries(values, localValues, MPI_DOUBLE, entryOffsets, rank, ranks, nItems, firstEntry);
        std::vector<double>().swap(values);
        localColumns.resize(lastEntry - firstEntry);
        scatterEntries(columns, localColumns, MPI_UINT32_T, entryOffsets, rank, ranks, nItems, firstEntry);
        std::vector<index_t>().swap(columns);
        std::vector<index_t>().swap(rows);
    }

    std::vector<double> localOut(localRowCount);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localColumns.data(), localRows.data(),
                vec.data(), localRowCount, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<double> out(rank == 0 ? numRows : 0);
        MPI_Gatherv(localOut.data(), localRowCount, MPI_DOUBLE, out.data(), rowCounts.data(),
                    firstRows.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(out, "OutputVector");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = verifyResults(reference.data(), out.data(), numRows);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
