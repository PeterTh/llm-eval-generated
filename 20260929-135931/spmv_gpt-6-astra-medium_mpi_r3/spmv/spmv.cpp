#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <exception>
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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

// Transfer contiguous partitions using collectives. Windows keep both counts
// and displacements within MPI's int range, even for large CSR arrays.
template <typename T>
void transferPartitions(T* global, T* local, const std::vector<uint64_t>& bounds,
                        MPI_Datatype type, int rank, bool gather = false) {
    const int ranks = static_cast<int>(bounds.size() - 1);
    std::vector<int> counts(ranks), offsets(ranks);
    T dummy{};
    for (uint64_t base = 0; base < bounds.back(); base += INT_MAX) {
        const uint64_t end = std::min(bounds.back(), base + INT_MAX);
        for (int r = 0; r < ranks; ++r) {
            const uint64_t first = std::max(base, bounds[r]);
            const uint64_t last = std::min(end, bounds[r + 1]);
            counts[r] = last > first ? static_cast<int>(last - first) : 0;
            offsets[r] = counts[r] ? static_cast<int>(first - base) : 0;
        }
        T* part = counts[rank] ? local + (std::max(base, bounds[rank]) - bounds[rank]) : &dummy;
        T* whole = rank == 0 ? global + base : &dummy;
        if (gather) {
            MPI_Gatherv(part, counts[rank], type, whole, counts.data(), offsets.data(),
                        type, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(whole, counts.data(), offsets.data(), type, part, counts[rank],
                         type, 0, MPI_COMM_WORLD);
        }
    }
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool validArguments = true;
    auto parseIndex = [&](const char* text) {
        char* end = nullptr;
        errno = 0;
        const auto value = strtoull(text, &end, 10);
        if (errno || text[0] == '-' || end == text || *end ||
            value > std::numeric_limits<index_t>::max()) {
            validArguments = false;
            return index_t(0);
        }
        return static_cast<index_t>(value);
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIndex(argv[++i]);
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
    if (!validArguments || !sparsity) {
        if (rank == 0) fprintf(stderr, "Invalid integer argument or zero sparsity.\n");
        return 1;
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Nonzero count exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    std::vector<double> h_val, h_out, h_reference;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<uint64_t> rowBounds(size_t(ranks) + 1), itemBounds(size_t(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        // Keep the original RNG order and seed for identical input data.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        // Include per-row overhead so empty/sparse rows are also balanced.
        // Rows stay intact, preserving the order of every floating-point sum.
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int r = 1; r < ranks; ++r) {
            const uint64_t target = work * r / ranks;
            uint64_t lo = rowBounds[r - 1], hi = numRows;
            while (lo < hi) {
                const uint64_t mid = lo + (hi - lo) / 2;
                if (uint64_t(h_rowDelimiters[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            rowBounds[r] = lo;
        }
        rowBounds[ranks] = numRows;
        for (int r = 0; r <= ranks; ++r) itemBounds[r] = h_rowDelimiters[rowBounds[r]];
    }
    MPI_Bcast(rowBounds.data(), ranks + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(itemBounds.data(), ranks + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    for (uint64_t base = 0; base < numRows; base += INT_MAX) {
        MPI_Bcast(h_vec.data() + base, static_cast<int>(std::min<uint64_t>(INT_MAX, numRows - base)),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    const index_t localRows = static_cast<index_t>(rowBounds[rank + 1] - rowBounds[rank]);
    const index_t localItems = static_cast<index_t>(itemBounds[rank + 1] - itemBounds[rank]);
    std::vector<double> val(localItems), out(localRows);
    std::vector<index_t> cols(localItems), rows(size_t(localRows) + 1);
    transferPartitions(h_val.data(), val.data(), itemBounds, MPI_DOUBLE, rank);
    transferPartitions(h_cols.data(), cols.data(), itemBounds, MPI_UINT32_T, rank);
    transferPartitions(h_rowDelimiters.data(), rows.data(), rowBounds, MPI_UINT32_T, rank);
    for (index_t i = 0; i < localRows; ++i) rows[i] -= static_cast<index_t>(itemBounds[rank]);
    rows[localRows] = localItems;
    // Release the global CSR before benchmarking; only local matrix data remains.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(val.data(), cols.data(), rows.data(), h_vec.data(), localRows, out.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        if (rank == 0) h_out.resize(numRows);
        transferPartitions(h_out.data(), out.data(), rowBounds, MPI_DOUBLE, rank, true);
    }
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = duration > 0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
