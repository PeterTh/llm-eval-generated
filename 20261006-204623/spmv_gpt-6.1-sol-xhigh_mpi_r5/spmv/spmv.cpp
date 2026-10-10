#include <algorithm>
#include <cerrno>
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

struct Options {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
};

bool parseIndex(const char* text, index_t& value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        parsed > std::numeric_limits<index_t>::max()) {
        printf("Invalid integer argument: %s\n", text);
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

// Only rank zero parses and prints; broadcast the outcome before any data transfer.
int parseOptions(int argc, char** argv, Options& options) {
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.numRows)) return 1;
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.sparsity)) return 1;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.iterations)) return 1;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            options.maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 2;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    if (options.sparsity == 0) {
        printf("Sparsity must be greater than zero.\n");
        return 1;
    }
    const uint64_t nItems = static_cast<uint64_t>(options.numRows) * options.numRows
                          / options.sparsity;
    if (nItems > std::numeric_limits<index_t>::max()) {
        printf("Too many non-zero elements for 32-bit CSR indices.\n");
        return 1;
    }
    return 0;
}

// Partition complete rows by CSR work, including overhead for empty rows.
// Never split a row: each dot product retains the original addition order.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowDelimiters,
                                   const index_t numRows, const int ranks) {
    std::vector<index_t> cuts(static_cast<size_t>(ranks) + 1);
    cuts.back() = numRows;
    const uint64_t work = static_cast<uint64_t>(rowDelimiters.back()) + numRows;
    for (int rank = 1; rank < ranks; ++rank) {
        const uint64_t target = work * static_cast<uint64_t>(rank) / ranks;
        index_t lo = cuts[rank - 1];
        index_t hi = numRows;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (static_cast<uint64_t>(rowDelimiters[mid]) + mid < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo > cuts[rank - 1]) {
            const uint64_t before = static_cast<uint64_t>(rowDelimiters[lo - 1]) + lo - 1;
            const uint64_t after = static_cast<uint64_t>(rowDelimiters[lo]) + lo;
            if (target - before <= after - target) --lo;
        }
        cuts[rank] = lo;
    }
    return cuts;
}

// Scatter/gather contiguous slices in collective windows so that MPI's int
// counts/displacements do not limit the total matrix to INT_MAX nonzeros.
template <typename T>
void transferSlices(T* global, T* local, const std::vector<index_t>& starts,
                    const std::vector<uint64_t>& sizes, MPI_Datatype datatype,
                    const int rank, const bool scatter) {
    const size_t ranks = sizes.size();
    uint64_t extent = 0;
    for (size_t p = 0; p < ranks; ++p) {
        extent = std::max(extent, static_cast<uint64_t>(starts[p]) + sizes[p]);
    }
    std::vector<int> counts(ranks), displacements(ranks);
    const uint64_t windowSize = std::numeric_limits<int>::max();
    for (uint64_t window = 0; window < extent; window += windowSize) {
        const uint64_t windowEnd = std::min(extent, window + windowSize);
        for (size_t p = 0; p < ranks; ++p) {
            const uint64_t begin = std::max(window, static_cast<uint64_t>(starts[p]));
            const uint64_t end = std::min(windowEnd, static_cast<uint64_t>(starts[p]) + sizes[p]);
            counts[p] = end > begin ? static_cast<int>(end - begin) : 0;
            displacements[p] = counts[p] ? static_cast<int>(begin - window) : 0;
        }
        T* localWindow = counts[rank]
            ? local + (std::max(window, static_cast<uint64_t>(starts[rank])) - starts[rank])
            : local;
        T* globalWindow = rank == 0 ? global + window : nullptr;
        if (scatter) {
            MPI_Scatterv(globalWindow, counts.data(), displacements.data(), datatype,
                         localWindow, counts[rank], datatype, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(localWindow, counts[rank], datatype, globalWindow,
                        counts.data(), displacements.data(), datatype, 0, MPI_COMM_WORLD);
        }
    }
}

int runBenchmark(int argc, char** argv, const int rank, const int ranks) {
    Options options;
    int parseStatus = rank == 0 ? parseOptions(argc, argv, options) : 0;
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) return parseStatus == 2 ? 0 : 1;

    index_t parameters[3] = {options.numRows, options.sparsity, options.iterations};
    int flags[2] = {options.validate, options.printResults};
    MPI_Bcast(parameters, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t numRows = parameters[0];
    const index_t sparsity = parameters[1];
    const index_t iterations = parameters[2];
    const double maxVal = options.maxVal;
    const bool validate = flags[0];
    const bool printResults = flags[1];
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // The original random sequence is generated once, on rank zero only.
    // Worker ranks never allocate the global matrix or reference solution.
    std::vector<double> h_val, h_reference, h_out;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<index_t> rowCuts(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> nnzCuts(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", ranks);
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // MPI initialization may use rand(); reproduce the C library's default seed.
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
        if (printResults || validate) h_out.resize(numRows);
        rowCuts = partitionRows(h_rowDelimiters, numRows, ranks);
        for (int p = 0; p <= ranks; ++p) nnzCuts[p] = h_rowDelimiters[rowCuts[p]];
    }

    // The input vector is immutable across iterations: broadcast it once.
    // Use chunks for vectors larger than MPI's int count range as well.
    const uint64_t maxCount = std::numeric_limits<int>::max();
    for (uint64_t offset = 0; offset < numRows; offset += maxCount) {
        MPI_Bcast(h_vec.data() + offset,
                  static_cast<int>(std::min(maxCount, static_cast<uint64_t>(numRows) - offset)),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(rowCuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t localRows = rowCuts[rank + 1] - rowCuts[rank];
    const index_t localNnz = nnzCuts[rank + 1] - nnzCuts[rank];
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<uint64_t> rowSizes(ranks), nnzSizes(ranks);
    for (int p = 0; p < ranks; ++p) {
        rowSizes[p] = static_cast<uint64_t>(rowCuts[p + 1]) - rowCuts[p];
        nnzSizes[p] = static_cast<uint64_t>(nnzCuts[p + 1]) - nnzCuts[p];
    }
    transferSlices(h_val.data(), localVal.data(), nnzCuts, nnzSizes, MPI_DOUBLE, rank, true);
    transferSlices(h_cols.data(), localCols.data(), nnzCuts, nnzSizes, MPI_UINT32_T, rank, true);
    transferSlices(h_rowDelimiters.data(), localRowDelimiters.data(), rowCuts, rowSizes,
                   MPI_UINT32_T, rank, true);
    for (index_t row = 0; row < localRows; ++row) {
        localRowDelimiters[row] -= nnzCuts[rank];
    }
    localRowDelimiters.back() = localNnz;

    // Release root's global CSR before timing; every rank now owns only its slice.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Assemble the final vector only when it is observable. There is no
    // communication in the repeated SpMV loop, including with validation enabled.
    if (printResults || validate) {
        transferSlices(h_out.data(), localOut.data(), rowCuts, rowSizes, MPI_DOUBLE, rank, false);
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
    int status = 0;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
