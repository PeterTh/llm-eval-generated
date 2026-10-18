#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// MPI's portable count type is int. Chunk large arrays without truncating
// the CSR index range; matching messages are ordered by source and tag.
template <typename T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) {
            MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        } else {
            MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        data += chunk;
        count -= chunk;
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

    // Calculate number of non-zero elements
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (sparsity == 0 || entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, entries == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", ranks);
    }

    std::vector<double> h_val;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;
    std::vector<index_t> rowStarts;
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        printf("Initializing data structures...\n");
        // Preserve the original rand() sequence and matrix exactly.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // Whole rows preserve the original summation order. Choose boundaries
        // by nonzero count rather than row count to balance sparse workloads.
        rowStarts.resize(size_t(ranks) + 1);
        for (int p = 1; p < ranks; ++p) {
            if (nItems == 0) {
                rowStarts[p] = uint64_t(numRows) * p / ranks;
            } else {
                const index_t target = uint64_t(nItems) * p / ranks;
                rowStarts[p] = std::lower_bound(h_rowDelimiters.begin(),
                                               h_rowDelimiters.end(), target)
                              - h_rowDelimiters.begin();
            }
        }
        rowStarts[ranks] = numRows;
    }

    // The input vector does not change between iterations. Replicate it once,
    // avoiding any communication or synchronization in the timed kernel.
    for (size_t offset = 0; offset < numRows;) {
        const int count = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        MPI_Bcast(h_vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }

    index_t localRows = 0;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t firstRow = rowStarts[p];
            const index_t rows = rowStarts[p + 1] - firstRow;
            const index_t firstItem = h_rowDelimiters[firstRow];
            const index_t items = h_rowDelimiters[rowStarts[p + 1]] - firstItem;
            index_t sizes[2] = {rows, items};
            MPI_Send(sizes, 2, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            transfer(h_rowDelimiters.data() + firstRow, size_t(rows) + 1,
                     MPI_UINT32_T, p, 1, true);
            if (items != 0) {
                transfer(h_val.data() + firstItem, items, MPI_DOUBLE, p, 2, true);
                transfer(h_cols.data() + firstItem, items, MPI_UINT32_T, p, 3, true);
            }
        }
        localRows = rowStarts[1];
        if (ranks > 1) {
            const index_t localItems = h_rowDelimiters[localRows];
            // Release the global matrix, including excess vector capacity.
            std::vector<double>(h_val.begin(), h_val.begin() + localItems).swap(h_val);
            std::vector<index_t>(h_cols.begin(), h_cols.begin() + localItems).swap(h_cols);
            std::vector<index_t>(h_rowDelimiters.begin(),
                                 h_rowDelimiters.begin() + size_t(localRows) + 1).swap(h_rowDelimiters);
        }
    } else {
        index_t sizes[2];
        MPI_Recv(sizes, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localRows = sizes[0];
        h_val.resize(sizes[1]);
        h_cols.resize(sizes[1]);
        h_rowDelimiters.resize(size_t(localRows) + 1);
        transfer(h_rowDelimiters.data(), h_rowDelimiters.size(), MPI_UINT32_T, 0, 1, false);
        transfer(h_val.data(), h_val.size(), MPI_DOUBLE, 0, 2, false);
        transfer(h_cols.data(), h_cols.size(), MPI_UINT32_T, 0, 3, false);
        const index_t base = h_rowDelimiters[0];
        for (auto& offset : h_rowDelimiters) offset -= base;
    }
    std::vector<double> localOut(localRows);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);

        // Calculate global throughput using the slowest rank's kernel time.
        const double gflops = seconds > 0.0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double avgTime = iterations != 0 ? seconds * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather once, and only when a consumer needs the full result.
    std::vector<double> h_out;
    if (printResults || validate) {
        if (rank == 0) {
            h_out.resize(numRows);
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p) {
                const size_t count = rowStarts[p + 1] - rowStarts[p];
                if (count != 0) {
                    transfer(h_out.data() + rowStarts[p], count, MPI_DOUBLE, p, 4, false);
                }
            }
        } else {
            transfer(localOut.data(), localOut.size(), MPI_DOUBLE, 0, 4, true);
        }
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
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
