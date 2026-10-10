#include <algorithm>
#include <cerrno>
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
    const uint64_t entries = static_cast<uint64_t>(dim) * dim;
    const double prob = entries == 0 ? 0.0 : static_cast<double>(n) / entries;

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    index_t i = 0;
    for (; i < dim && nnzAssigned < n; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            const uint64_t numEntriesLeft = entries - (static_cast<uint64_t>(i) * dim + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            // Once the remaining positions are forced, further random draws
            // cannot affect the matrix. The RNG is not used after initialization.
            if (fillRemaining || static_cast<double>(rand()) / RAND_MAX <= prob) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Finish any empty trailing rows and the final delimiter.
    std::fill(rowDelimiters + i, rowDelimiters + static_cast<size_t>(dim) + 1, n);
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

// Assign contiguous whole rows by nonzero count plus per-row overhead. Keeping
// each row intact preserves the original floating-point accumulation order.
void partitionRows(const std::vector<index_t>& delimiters, std::vector<index_t>& rows,
                   std::vector<index_t>& items, const index_t numRows, const int ranks) {
    const uint64_t work = static_cast<uint64_t>(delimiters[numRows]) + numRows;
    rows.front() = 0;
    rows.back() = numRows;
    for (int rank = 1; rank < ranks; ++rank) {
        // Give every rank a row when there are enough rows to do so.
        const bool enoughRows = numRows >= static_cast<index_t>(ranks);
        const index_t first = rows[rank - 1] + (enoughRows ? 1 : 0);
        const index_t last = enoughRows ? numRows - (ranks - rank) : numRows;
        const uint64_t target = work * static_cast<uint64_t>(rank) / ranks;
        index_t low = first;
        index_t high = last;
        while (low < high) {
            const index_t mid = low + (high - low) / 2;
            if (static_cast<uint64_t>(delimiters[mid]) + mid < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > first) {
            const uint64_t before = static_cast<uint64_t>(delimiters[low - 1]) + low - 1;
            const uint64_t after = static_cast<uint64_t>(delimiters[low]) + low;
            if (after >= target && target - before <= after - target) {
                --low;
            }
        }
        rows[rank] = low;
    }
    for (int rank = 0; rank <= ranks; ++rank) {
        items[rank] = delimiters[rows[rank]];
    }
}

// MPI_Scatterv is the fast path. Chunked transfers also support CSR arrays whose
// 32-bit unsigned offsets exceed the signed counts/displacements of MPI 3.
template <typename T>
void scatterArray(const T* source, T* local, const std::vector<index_t>& offsets,
                  const MPI_Datatype type, const int rank, const int ranks) {
    const index_t count = offsets[rank + 1] - offsets[rank];
    if (offsets.back() <= static_cast<index_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int peer = 0; peer < ranks; ++peer) {
                counts[peer] = static_cast<int>(offsets[peer + 1] - offsets[peer]);
                displacements[peer] = static_cast<int>(offsets[peer]);
            }
        }
        MPI_Scatterv(source, counts.data(), displacements.data(), type, local,
                     static_cast<int>(count), type, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        if (count != 0) {
            std::copy_n(source, count, local);
        }
        for (int peer = 1; peer < ranks; ++peer) {
            index_t position = offsets[peer];
            while (position < offsets[peer + 1]) {
                const int chunk =
                    static_cast<int>(std::min<index_t>(offsets[peer + 1] - position, INT_MAX));
                MPI_Send(source + position, chunk, type, peer, 0, MPI_COMM_WORLD);
                position += static_cast<index_t>(chunk);
            }
        }
    } else {
        index_t position = 0;
        while (position < count) {
            const int chunk = static_cast<int>(std::min<index_t>(count - position, INT_MAX));
            MPI_Recv(local + position, chunk, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            position += static_cast<index_t>(chunk);
        }
    }
}

int runBenchmark(int argc, char** argv, const int rank, const int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse once and communicate the outcome so help and errors also finalize
    // collectively, without duplicate output or leaving ranks in a collective.
    int argumentStatus = 0;
    if (rank == 0) {
        const auto parseIndex = [&](const char* text, index_t& value) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(text, &end, 10);
            if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<index_t>::max()) {
                printf("Invalid integer: %s\n", text);
                argumentStatus = 1;
            } else {
                value = static_cast<index_t>(parsed);
            }
        };
        for (int i = 1; i < argc && argumentStatus == 0; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                parseIndex(argv[++i], numRows);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                parseIndex(argv[++i], sparsity);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                parseIndex(argv[++i], iterations);
            } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                maxVal = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                argumentStatus = -1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                argumentStatus = 1;
            }
        }
        if (argumentStatus == 0 && sparsity == 0) {
            printf("Sparsity must be greater than zero.\n");
            argumentStatus = 1;
        }
        if (argumentStatus == 0 && numRows > static_cast<index_t>(INT_MAX - 1)) {
            printf("Matrix dimension exceeds the MPI row-count limit.\n");
            argumentStatus = 1;
        }
        if (argumentStatus == 0 && static_cast<uint64_t>(numRows) * numRows / sparsity >
                                       std::numeric_limits<index_t>::max()) {
            printf("Non-zero count exceeds the CSR index limit.\n");
            argumentStatus = 1;
        }
        if (argumentStatus > 0) {
            printUsage(argv[0]);
        }
    }
    MPI_Bcast(&argumentStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (argumentStatus != 0) {
        return argumentStatus < 0 ? 0 : 1;
    }
    index_t options[] = {numRows, sparsity, iterations, static_cast<index_t>(validate),
                         static_cast<index_t>(printResults)};
    MPI_Bcast(options, 5, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    numRows = options[0];
    sparsity = options[1];
    iterations = options[2];
    validate = options[3] != 0;
    printResults = options[4] != 0;

    // Calculate number of non-zero elements
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank 0 generates the global CSR matrix, preserving the original RNG
    // sequence. Other ranks allocate only their own rows and nonzero entries.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;
    std::vector<index_t> rowOffsets(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> itemOffsets(static_cast<size_t>(ranks) + 1);

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        srand(1); // The implicit seed in the original program, before any MPI calls.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
        partitionRows(h_rowDelimiters, rowOffsets, itemOffsets, numRows, ranks);
    }

    MPI_Bcast(rowOffsets.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(itemOffsets.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t localRows = rowOffsets[rank + 1] - rowOffsets[rank];
    const index_t localItems = itemOffsets[rank + 1] - itemOffsets[rank];
    std::vector<double> localVal(localItems);
    std::vector<index_t> localCols(localItems);
    std::vector<index_t> localDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);
    scatterArray(h_val.data(), localVal.data(), itemOffsets, MPI_DOUBLE, rank, ranks);
    scatterArray(h_cols.data(), localCols.data(), itemOffsets, MPI_UINT32_T, rank, ranks);
    // Scatter row starts only, avoiding overlapping reads of boundary delimiters.
    scatterArray(h_rowDelimiters.data(), localDelimiters.data(), rowOffsets, MPI_UINT32_T, rank,
                 ranks);
    for (index_t row = 0; row < localRows; ++row) {
        localDelimiters[row] -= itemOffsets[rank];
    }
    localDelimiters[localRows] = localItems;

    // Release the global matrix before timing; rank 0 keeps only an optional
    // reference vector in addition to the same local storage as all other ranks.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // The input vector is invariant across iterations. No communication or
    // synchronization is needed inside the timed loop.
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localDelimiters.data(), h_vec.data(), localRows,
                localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);

        // Calculate performance metrics
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations == 0 ? 0.0 : duration * 1000.0 / iterations;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Materialize the global output only if it will actually be consumed.
    std::vector<double> h_out;
    if (validate || printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            h_out.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int peer = 0; peer < ranks; ++peer) {
                counts[peer] = static_cast<int>(rowOffsets[peer + 1] - rowOffsets[peer]);
                displacements[peer] = static_cast<int>(rowOffsets[peer]);
            }
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int resultStatus = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            resultStatus = 1;
        }
    }

    MPI_Bcast(&resultStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return resultStatus;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
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
