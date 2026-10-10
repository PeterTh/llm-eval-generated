#include <algorithm>
#include <cerrno>
#include <cmath>
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

    if (n == 0) {
        std::fill_n(rowDelimiters, static_cast<size_t>(dim) + 1, index_t{0});
        return;
    }

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim
                                         - (static_cast<uint64_t>(i) * dim + j);
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

// Balance contiguous row ranges by nonzeros plus a small per-row cost. This
// also distributes empty rows, and never splits a row's ordered accumulation.
void partitionRows(const std::vector<index_t>& rowDelimiters,
                   std::vector<index_t>& rowOffsets,
                   std::vector<index_t>& itemOffsets, const index_t numRows,
                   const int ranks) {
    const uint64_t totalWork = static_cast<uint64_t>(rowDelimiters[numRows]) + numRows;
    rowOffsets.front() = 0;
    rowOffsets.back() = numRows;
    for (int p = 1; p < ranks; ++p) {
        const uint64_t target = totalWork * static_cast<uint64_t>(p) / ranks;
        index_t low = rowOffsets[p - 1];
        index_t high = numRows;
        while (low < high) {
            const index_t mid = low + (high - low) / 2;
            if (static_cast<uint64_t>(rowDelimiters[mid]) + mid < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > rowOffsets[p - 1]) {
            const uint64_t below = static_cast<uint64_t>(rowDelimiters[low - 1]) + low - 1;
            const uint64_t above = static_cast<uint64_t>(rowDelimiters[low]) + low;
            if (target - below <= above - target) {
                --low;
            }
        }
        rowOffsets[p] = low;
    }
    for (int p = 0; p <= ranks; ++p) {
        itemOffsets[p] = rowDelimiters[rowOffsets[p]];
    }
}

// MPI's portable collectives use int counts/displacements. Chunk global
// arrays so the CSR index range is not restricted to INT_MAX elements.
template <typename T>
void broadcastArray(std::vector<T>& data, const MPI_Datatype type) {
    constexpr size_t maxCount = std::numeric_limits<int>::max();
    for (size_t first = 0; first < data.size(); first += maxCount) {
        const int count = static_cast<int>(std::min(maxCount, data.size() - first));
        MPI_Bcast(data.data() + first, count, type, 0, MPI_COMM_WORLD);
    }
}

template <typename T>
void transferSegments(std::vector<T>& global, std::vector<T>& local,
                      const std::vector<index_t>& offsets,
                      const MPI_Datatype type, const int rank, const bool gather) {
    const size_t ranks = offsets.size() - 1;
    const size_t total = offsets.back();
    const size_t localFirst = offsets[rank];
    const size_t localLast = offsets[rank + 1];
    constexpr size_t maxCount = std::numeric_limits<int>::max();
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(rank == 0 ? ranks : 0);

    for (size_t first = 0; first < total; first += maxCount) {
        const size_t last = first + std::min(maxCount, total - first);
        if (rank == 0) {
            for (size_t p = 0; p < ranks; ++p) {
                const size_t begin = std::max(first, static_cast<size_t>(offsets[p]));
                const size_t end = std::min(last, static_cast<size_t>(offsets[p + 1]));
                counts[p] = begin < end ? static_cast<int>(end - begin) : 0;
                displacements[p] = counts[p] != 0 ? static_cast<int>(begin - first) : 0;
            }
        }
        const size_t begin = std::max(first, localFirst);
        const size_t end = std::min(last, localLast);
        const int count = begin < end ? static_cast<int>(end - begin) : 0;
        T* localData = count != 0 ? local.data() + (begin - localFirst) : nullptr;
        T* globalData = rank == 0 ? global.data() + first : nullptr;
        if (gather) {
            MPI_Gatherv(localData, count, type, globalData, counts.data(),
                        displacements.data(), type, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(globalData, counts.data(), displacements.data(), type,
                         localData, count, type, 0, MPI_COMM_WORLD);
        }
    }
}

bool parseIndex(const char* text, index_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

int runBenchmark(int argc, char** argv, const int rank, const int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Only rank zero handles arguments and user-visible output.
    int status = 0;
    for (int i = 1; rank == 0 && i < argc && status == 0; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], numRows)) status = 1;
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], sparsity) || sparsity == 0) status = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], iterations)) status = 1;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || errno == ERANGE || !std::isfinite(maxVal)) {
                status = 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            status = 2;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            status = 1;
        }
    }

    if (rank == 0 && status == 1) {
        fprintf(stderr, "Invalid command line arguments.\n");
        printUsage(argv[0]);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        return status == 2 ? 0 : 1;
    }
    index_t options[] = {numRows, sparsity, iterations,
                         static_cast<index_t>(validate), static_cast<index_t>(printResults)};
    MPI_Bcast(options, 5, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    numRows = options[0];
    sparsity = options[1];
    iterations = options[2];
    validate = options[3] != 0;
    printResults = options[4] != 0;

    // Calculate number of non-zero elements
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t itemCount = entries / sparsity;
    if (itemCount > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(itemCount);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, entries != 0 ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", ranks);
    }

    // Preserve the original random stream on rank zero. Other ranks allocate
    // only their CSR partition and the read-only input vector.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;
    std::vector<index_t> rowOffsets(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> itemOffsets(static_cast<size_t>(ranks) + 1);

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        // MPI initialization must not affect the original implicit rand seed.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        partitionRows(h_rowDelimiters, rowOffsets, itemOffsets, numRows, ranks);
    }

    broadcastArray(rowOffsets, MPI_UINT32_T);
    broadcastArray(itemOffsets, MPI_UINT32_T);
    broadcastArray(h_vec, MPI_DOUBLE);

    const index_t localRows = rowOffsets[rank + 1] - rowOffsets[rank];
    const index_t localItems = itemOffsets[rank + 1] - itemOffsets[rank];
    std::vector<double> localVal(localItems);
    std::vector<index_t> localCols(localItems);
    std::vector<index_t> localDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);
    transferSegments(h_val, localVal, itemOffsets, MPI_DOUBLE, rank, false);
    transferSegments(h_cols, localCols, itemOffsets, MPI_UINT32_T, rank, false);
    // Scatter only each row's start; the final endpoint is known locally.
    // Thus adjacent partitions do not overlap in the Scatterv send buffer.
    transferSegments(h_rowDelimiters, localDelimiters, rowOffsets, MPI_UINT32_T, rank, false);
    for (index_t i = 0; i < localRows; ++i) {
        localDelimiters[i] -= itemOffsets[rank];
    }
    localDelimiters[localRows] = localItems;

    // Release the full matrix before timing; steady-state storage is distributed.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);

        // Calculate performance metrics
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations != 0 ? duration * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // The input vector is invariant, so no communication is needed between
    // iterations. Collect the final output only when the caller needs it.
    if (printResults || validate) {
        if (rank == 0) h_out.resize(numRows);
        transferSegments(h_out, localOut, rowOffsets, MPI_DOUBLE, rank, true);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
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
