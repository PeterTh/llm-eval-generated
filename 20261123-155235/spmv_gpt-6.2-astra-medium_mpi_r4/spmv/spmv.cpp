#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            uint64_t numEntriesLeft = uint64_t(dim - i) * dim - j;
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

// Scatter contiguous slices. Window the global offsets for MPI implementations
// whose Scatterv counts and displacements are limited to signed int.
template <class T>
void scatterSlices(const std::vector<T>& source, std::vector<T>& local,
                   const std::vector<index_t>& bounds, int rank, MPI_Datatype type) {
    const int ranks = static_cast<int>(bounds.size() - 1);
    std::vector<int> counts(rank == 0 ? ranks : 0), offsets(counts.size());
    for (uint64_t start = 0; start < bounds.back(); start += INT_MAX) {
        const uint64_t end = std::min<uint64_t>(start + INT_MAX, bounds.back());
        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const auto first = std::clamp<uint64_t>(bounds[r], start, end);
                offsets[r] = static_cast<int>(first - start);
                counts[r] = static_cast<int>(std::clamp<uint64_t>(bounds[r + 1], start, end) - first);
            }
        }
        const auto first = std::clamp<uint64_t>(bounds[rank], start, end);
        const int count = static_cast<int>(std::clamp<uint64_t>(bounds[rank + 1], start, end) - first);
        MPI_Scatterv(rank == 0 ? source.data() + start : nullptr, counts.data(), offsets.data(), type,
                     count ? local.data() + (first - bounds[rank]) : nullptr, count, type, 0, MPI_COMM_WORLD);
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

    const uint64_t cells = uint64_t(numRows) * numRows;
    if (!sparsity || numRows > INT_MAX || cells / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Unsupported matrix size or sparsity.\n");
        return 1;
    }
    const index_t nItems = cells / sparsity;
    std::vector<double> vec(numRows), values, reference;
    std::vector<index_t> columns, rows;
    std::vector<index_t> rowBounds(size_t(ranks) + 1), entryBounds(rowBounds.size());

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
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), columns.data(), rows.data(), vec.data(), numRows, reference.data());
        }

        // Prefix work at row i is i + rows[i], including the cost of empty rows.
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int r = 1; r < ranks; ++r) {
            const uint64_t target = work * r / ranks;
            index_t lo = rowBounds[r - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(mid) + rows[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            rowBounds[r] = lo;
            entryBounds[r] = rows[lo];
        }
        rowBounds.back() = numRows;
        entryBounds.back() = nItems;
    }

    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Boundary zero is implicit. Broadcast the remaining endpoints.
    MPI_Bcast(rowBounds.data() + 1, ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(entryBounds.data() + 1, ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = rowBounds[rank + 1] - rowBounds[rank];
    const index_t localEntries = entryBounds[rank + 1] - entryBounds[rank];
    if (ranks > 1) {
        std::vector<double> localValues(localEntries);
        scatterSlices(values, localValues, entryBounds, rank, MPI_DOUBLE);
        values = std::move(localValues);
        std::vector<index_t> localColumns(localEntries);
        scatterSlices(columns, localColumns, entryBounds, rank, MPI_UINT32_T);
        columns = std::move(localColumns);
        std::vector<index_t> localOffsets(size_t(localRows) + 1);
        scatterSlices(rows, localOffsets, rowBounds, rank, MPI_UINT32_T);
        for (index_t i = 0; i < localRows; ++i) localOffsets[i] -= entryBounds[rank];
        localOffsets.back() = localEntries;
        rows = std::move(localOffsets);
    }
    std::vector<double> localOut(localRows);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(values.data(), columns.data(), rows.data(), vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }

    int status = 0;
    if (validate || printResults) {
        std::vector<int> counts, offsets;
        std::vector<double> output;
        if (rank == 0) {
            output.resize(numRows);
            offsets.assign(rowBounds.begin(), rowBounds.end() - 1);
            counts.resize(ranks);
            for (int r = 0; r < ranks; ++r) counts[r] = rowBounds[r + 1] - rowBounds[r];
        }
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, output.data(), counts.data(),
                    offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(output, "OutputVector");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = verifyResults(reference.data(), output.data(), numRows);
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
