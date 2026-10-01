#include <algorithm>
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
            const uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

// Use the optimized collective for ordinary matrices, with a chunked fallback
// for CSR arrays exceeding MPI's signed-int count/displacement range.
template <typename T>
void scatterEntries(const std::vector<T>& source, std::vector<T>& local,
                    const std::vector<index_t>& offsets, MPI_Datatype type,
                    int rank, int ranks) {
    if (offsets.back() <= INT_MAX) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = offsets[p + 1] - offsets[p];
            displacements[p] = offsets[p];
        }
        MPI_Scatterv(source.data(), counts.data(), displacements.data(), type,
                     local.data(), static_cast<int>(local.size()), type, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        std::copy_n(source.begin(), local.size(), local.begin());
        for (int p = 1; p < ranks; ++p) {
            uint64_t pos = offsets[p];
            while (pos < offsets[p + 1]) {
                const int count = static_cast<int>(std::min<uint64_t>(
                    offsets[p + 1] - pos, INT_MAX));
                MPI_Send(source.data() + pos, count, type, p, 0, MPI_COMM_WORLD);
                pos += count;
            }
        }
    } else {
        size_t pos = 0;
        while (pos < local.size()) {
            const int count = static_cast<int>(std::min<size_t>(local.size() - pos, INT_MAX));
            MPI_Recv(local.data() + pos, count, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pos += count;
        }
    }
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
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

    const uint64_t entries = uint64_t(numRows) * numRows;
    if (sparsity == 0 || numRows > INT_MAX ||
        entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid dimensions or sparsity: matrix exceeds CSR index limits.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // Generate exactly the original data on rank zero. Only the dense input
    // vector is replicated; CSR storage is distributed before the timed loop.
    std::vector<double> h_val, h_vec(numRows), h_out, h_reference;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<index_t> rowOffsets(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> itemOffsets(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\n", iterations, maxVal);
        printf("Validation: %s\nMPI ranks: %d\n", validate ? "enabled" : "disabled", ranks);
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        // Balance nonzero work plus per-row overhead, including empty rows.
        // Keeping whole rows preserves the serial floating-point addition order.
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = work * static_cast<uint64_t>(p) / ranks;
            index_t lo = rowOffsets[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(h_rowDelimiters[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            rowOffsets[p] = lo;
            itemOffsets[p] = h_rowDelimiters[lo];
        }
        rowOffsets[ranks] = numRows;
        itemOffsets[ranks] = nItems;
    }
    MPI_Bcast(rowOffsets.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(itemOffsets.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t localRows = rowOffsets[rank + 1] - rowOffsets[rank];
    const index_t localItems = itemOffsets[rank + 1] - itemOffsets[rank];
    std::vector<double> localVal(localItems), localOut(localRows);
    std::vector<index_t> localCols(localItems), localDelimiters(localRows);
    scatterEntries(h_val, localVal, itemOffsets, MPI_DOUBLE, rank, ranks);
    scatterEntries(h_cols, localCols, itemOffsets, MPI_UINT32_T, rank, ranks);
    scatterEntries(h_rowDelimiters, localDelimiters, rowOffsets, MPI_UINT32_T, rank, ranks);
    for (auto& offset : localDelimiters) offset -= itemOffsets[rank];
    localDelimiters.push_back(localItems);
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // The input is invariant across iterations: no halo exchange or output
    // all-gather is needed inside the hot loop.
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Assemble the output only when it will be consumed, outside kernel timing.
    if (printResults || validate) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = rowOffsets[p + 1] - rowOffsets[p];
            displacements[p] = rowOffsets[p];
        }
        if (rank == 0) h_out.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_out.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
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
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
