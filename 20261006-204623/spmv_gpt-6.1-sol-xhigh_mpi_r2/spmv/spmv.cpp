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
#include <utility>
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
    const uint64_t entries = static_cast<uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = entries - (static_cast<uint64_t>(i) * dim + j);
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

// Include row overhead in the partition weight, so empty rows are also balanced.
// Whole rows retain the original floating-point summation order.
void partitionRows(const std::vector<index_t>& rowDelimiters, const index_t numRows,
                   std::vector<index_t>& rowCuts, std::vector<index_t>& itemCuts) {
    const size_t ranks = rowCuts.size() - 1;
    const uint64_t totalWork = static_cast<uint64_t>(rowDelimiters[numRows]) + numRows;
    for (size_t p = 1; p < ranks; ++p) {
        const uint64_t target = totalWork * p / ranks;
        // If there are enough rows, give every rank at least one.
        const index_t first = rowCuts[p - 1] + (numRows >= ranks ? 1 : 0);
        const index_t last = numRows >= ranks ? numRows - (ranks - p) : numRows;
        index_t low = first;
        index_t high = last;
        while (low < high) {
            const index_t mid = low + (high - low) / 2;
            const uint64_t work = static_cast<uint64_t>(rowDelimiters[mid]) + mid;
            if (work < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > first) {
            const uint64_t before = static_cast<uint64_t>(rowDelimiters[low - 1]) + low - 1;
            const uint64_t after = static_cast<uint64_t>(rowDelimiters[low]) + low;
            if (after >= target && target - before <= after - target) {
                --low;
            }
        }
        rowCuts[p] = low;
    }
    rowCuts[ranks] = numRows;
    for (size_t p = 0; p <= ranks; ++p) {
        itemCuts[p] = rowDelimiters[rowCuts[p]];
    }
}

// Chunk transfers at MPI's int count/displacement limit. The normal case uses
// one collective, while large CSR arrays do not overflow MPI counts.
template <typename T>
void broadcastVector(std::vector<T>& data, MPI_Datatype datatype) {
    for (size_t offset = 0; offset < data.size();) {
        const int count = static_cast<int>(std::min(data.size() - offset, size_t{INT_MAX}));
        MPI_Bcast(data.data() + offset, count, datatype, 0, MPI_COMM_WORLD);
        offset += count;
    }
}

void sliceWindow(const std::vector<index_t>& cuts, const uint64_t offset,
                 const uint64_t end, std::vector<int>& counts, std::vector<int>& displacements) {
    for (size_t p = 0; p < counts.size(); ++p) {
        const uint64_t first = std::max<uint64_t>(cuts[p], offset);
        const uint64_t last = std::min<uint64_t>(cuts[p + 1], end);
        counts[p] = last > first ? static_cast<int>(last - first) : 0;
        displacements[p] = last > first ? static_cast<int>(first - offset) : 0;
    }
}

template <typename T>
void scatterSlices(std::vector<T>& global, std::vector<T>& local,
                   const std::vector<index_t>& cuts, const int rank, MPI_Datatype datatype) {
    const size_t ranks = cuts.size() - 1;
    if (ranks == 1) {
        local = std::move(global);
        return;
    }
    local.resize(static_cast<size_t>(cuts[rank + 1]) - cuts[rank]);
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(counts.size());
    T dummy{};
    for (uint64_t offset = 0; offset < cuts.back();) {
        const uint64_t end = std::min<uint64_t>(cuts.back(), offset + INT_MAX);
        if (rank == 0) {
            sliceWindow(cuts, offset, end, counts, displacements);
        }
        const uint64_t first = std::max<uint64_t>(cuts[rank], offset);
        const uint64_t last = std::min<uint64_t>(cuts[rank + 1], end);
        const int count = last > first ? static_cast<int>(last - first) : 0;
        T* receive = count ? local.data() + (first - cuts[rank]) : &dummy;
        MPI_Scatterv(rank == 0 ? global.data() + offset : nullptr, counts.data(),
                     displacements.data(), datatype, receive, count, datatype, 0, MPI_COMM_WORLD);
        offset = end;
    }
    // Keep only the rank's CSR block after distribution, including on rank 0.
    std::vector<T>().swap(global);
}

void gatherOutput(std::vector<double>& local, std::vector<double>& global,
                  const std::vector<index_t>& cuts, const int rank) {
    const size_t ranks = cuts.size() - 1;
    if (ranks == 1) {
        global = std::move(local);
        return;
    }
    if (rank == 0) {
        global.resize(cuts.back());
    }
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(counts.size());
    double dummy = 0.0;
    for (uint64_t offset = 0; offset < cuts.back();) {
        const uint64_t end = std::min<uint64_t>(cuts.back(), offset + INT_MAX);
        if (rank == 0) {
            sliceWindow(cuts, offset, end, counts, displacements);
        }
        const uint64_t first = std::max<uint64_t>(cuts[rank], offset);
        const uint64_t last = std::min<uint64_t>(cuts[rank + 1], end);
        const int count = last > first ? static_cast<int>(last - first) : 0;
        const double* send = count ? local.data() + (first - cuts[rank]) : &dummy;
        MPI_Gatherv(send, count, MPI_DOUBLE, rank == 0 ? global.data() + offset : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset = end;
    }
}

bool parseIndex(const char* text, index_t& value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
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

// Interleave independent row accumulations to hide addition latency. Each row
// still visits its entries in CSR order, with the same rounding as spmvCpu.
void spmvLocal(const double* val, const index_t* cols, const index_t* rowDelimiters,
               const double* vec, const index_t dim, double* out) {
    // Short rows do not amortize the extra bookkeeping of interleaving.
    if (dim < 4 || static_cast<uint64_t>(rowDelimiters[dim] - rowDelimiters[0]) < 16ULL * dim) {
        spmvCpu(val, cols, rowDelimiters, vec, dim, out);
        return;
    }
    index_t i = 0;
    for (; dim - i >= 4; i += 4) {
        index_t j0 = rowDelimiters[i];
        index_t j1 = rowDelimiters[i + 1];
        index_t j2 = rowDelimiters[i + 2];
        index_t j3 = rowDelimiters[i + 3];
        const index_t end0 = j1;
        const index_t end1 = j2;
        const index_t end2 = j3;
        const index_t end3 = rowDelimiters[i + 4];
        const index_t common = std::min(std::min(end0 - j0, end1 - j1),
                                        std::min(end2 - j2, end3 - j3));
        const index_t commonEnd = j0 + common;
        double t0 = 0.0;
        double t1 = 0.0;
        double t2 = 0.0;
        double t3 = 0.0;
        for (; j0 < commonEnd; ++j0, ++j1, ++j2, ++j3) {
            t0 += val[j0] * vec[cols[j0]];
            t1 += val[j1] * vec[cols[j1]];
            t2 += val[j2] * vec[cols[j2]];
            t3 += val[j3] * vec[cols[j3]];
        }
        for (; j0 < end0; ++j0) t0 += val[j0] * vec[cols[j0]];
        for (; j1 < end1; ++j1) t1 += val[j1] * vec[cols[j1]];
        for (; j2 < end2; ++j2) t2 += val[j2] * vec[cols[j2]];
        for (; j3 < end3; ++j3) t3 += val[j3] * vec[cols[j3]];
        out[i] = t0;
        out[i + 1] = t1;
        out[i + 2] = t2;
        out[i + 3] = t3;
    }
    if (i < dim) {
        spmvCpu(val, cols, rowDelimiters + i, vec, dim - i, out + i);
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

int runBenchmark(int argc, char** argv, const int rank, const int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], numRows)) {
                if (rank == 0) printf("Invalid matrix dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], sparsity) || sparsity == 0) {
                if (rank == 0) printf("Invalid sparsity: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], iterations)) {
                if (rank == 0) printf("Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
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
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) printf("Too many non-zero elements for the CSR index type\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
    }

    // Generate once on rank 0 to preserve the original random stream.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;
    std::vector<index_t> rowCuts(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> itemCuts(rowCuts.size());

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        // rand() initially has the srand(1) state in the serial program. MPI's
        // initialization must not be allowed to perturb that state.
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
        partitionRows(h_rowDelimiters, numRows, rowCuts, itemCuts);
    }

    broadcastVector(rowCuts, MPI_UINT32_T);
    broadcastVector(itemCuts, MPI_UINT32_T);
    broadcastVector(h_vec, MPI_DOUBLE);

    const index_t localRows = rowCuts[rank + 1] - rowCuts[rank];
    const index_t localItems = itemCuts[rank + 1] - itemCuts[rank];
    std::vector<double> localVal;
    std::vector<index_t> localCols;
    std::vector<index_t> localRowDelimiters;
    scatterSlices(h_val, localVal, itemCuts, rank, MPI_DOUBLE);
    scatterSlices(h_cols, localCols, itemCuts, rank, MPI_UINT32_T);
    // Scatter row starts without overlapping shared end delimiters.
    scatterSlices(h_rowDelimiters, localRowDelimiters, rowCuts, rank, MPI_UINT32_T);
    localRowDelimiters.resize(static_cast<size_t>(localRows) + 1);
    for (index_t i = 0; i < localRows; ++i) {
        localRowDelimiters[i] -= itemCuts[rank];
    }
    localRowDelimiters[localRows] = localItems;
    std::vector<double> localOut(localRows);

    // The vector is invariant: no communication is needed inside this loop.
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localRows != 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvLocal(localVal.data(), localCols.data(), localRowDelimiters.data(),
                      h_vec.data(), localRows, localOut.data());
        }
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (printResults || validate) {
        gatherOutput(localOut, h_out, rowCuts, rank);
    }

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);

        // Calculate performance metrics
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    // Every rank returns the same validation status and finalizes normally.
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
        // A failure on one rank must not leave its peers blocked in collectives.
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
