#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <limits>
#include <exception>
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

// MPI's traditional interfaces use int counts. Chunk large transfers so CSR
// arrays can still use the full range of index_t without narrowing counts.
template <typename T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    auto parseIndex = [](const char* text, index_t& value, bool allowZero = true) {
        char* end = nullptr;
        errno = 0;
        const auto parsed = strtoull(text, &end, 10);
        if (errno || end == text || *end || *text == '-' || (!allowZero && parsed == 0) ||
            parsed > std::numeric_limits<index_t>::max()) return false;
        value = static_cast<index_t>(parsed);
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        bool ok = true;
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            ok = parseIndex(argv[++i], numRows);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            ok = parseIndex(argv[++i], sparsity, false);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            ok = parseIndex(argv[++i], iterations);
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
            ok = false;
        }
        if (!ok) {
            if (rank == 0) {
                printf("Invalid option or value: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);
    std::vector<double> h_val, h_vec(numRows), h_reference;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\n", iterations, maxVal);
        printf("Validation: %s\nMPI ranks: %d\n", validate ? "enabled" : "disabled", ranks);
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        // Retain the original order of calls to rand(), including its seed.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        // Balance both nonzeros and row overhead; empty matrices and more
        // ranks than rows are handled without special cases.
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = work * uint64_t(p) / ranks;
            index_t lo = boundaries[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(h_rowDelimiters[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
    }
    MPI_Bcast(boundaries.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    boundaries[ranks] = numRows;
    for (size_t offset = 0; offset < numRows;) {
        const int count = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        MPI_Bcast(h_vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> rows(size_t(localRows) + 1), cols;
    std::vector<double> values, out(localRows);
    if (rank == 0) {
        // Send each CSR slice once. Only rank 0 ever stores the global matrix.
        for (int p = 1; p < ranks; ++p) {
            const index_t begin = boundaries[p], end = boundaries[p + 1];
            transfer(h_rowDelimiters.data() + begin, size_t(end) - begin + 1,
                     MPI_UINT32_T, p, true);
            const index_t first = h_rowDelimiters[begin];
            const size_t count = h_rowDelimiters[end] - first;
            if (count) {
                transfer(h_cols.data() + first, count, MPI_UINT32_T, p, true);
                transfer(h_val.data() + first, count, MPI_DOUBLE, p, true);
            }
        }
        std::copy_n(h_rowDelimiters.begin(), rows.size(), rows.begin());
        // Reuse rank 0's allocation, releasing the distributed portion.
        h_cols.resize(rows.back());
        h_val.resize(rows.back());
        h_cols.shrink_to_fit();
        h_val.shrink_to_fit();
        cols.swap(h_cols);
        values.swap(h_val);
        std::vector<index_t>().swap(h_rowDelimiters);
    } else {
        transfer(rows.data(), rows.size(), MPI_UINT32_T, 0, false);
        const index_t first = rows.front(), count = rows.back() - first;
        for (auto& row : rows) row -= first;
        cols.resize(count);
        values.resize(count);
        transfer(cols.data(), count, MPI_UINT32_T, 0, false);
        transfer(values.data(), count, MPI_DOUBLE, 0, false);
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // The vector is immutable across iterations, so no halo exchange or
    // collective is necessary in this loop. Row summation order is unchanged.
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(values.data(), cols.data(), rows.data(), h_vec.data(), localRows, out.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> h_out;
    if (validate || printResults) {
        if (rank == 0) {
            h_out.resize(numRows);
            std::copy(out.begin(), out.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p) {
                const size_t count = boundaries[p + 1] - boundaries[p];
                if (count) transfer(h_out.data() + boundaries[p], count, MPI_DOUBLE, p, false);
            }
        } else {
            transfer(out.data(), out.size(), MPI_DOUBLE, 0, true);
        }
    }
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ?
               (2.0 * nItems * iterations) / duration / 1e9 : 0.0);
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
