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

// Scatter CSR arrays using optimized collectives for ordinary sizes. For arrays
// beyond MPI's int count/displacement range, transfer bounded chunks instead.
template <typename T>
void scatterArray(const std::vector<T>& source, T* destination, index_t localCount,
                  const std::vector<index_t>& counts,
                  const std::vector<index_t>& offsets, MPI_Datatype type,
                  int rank, int ranks) {
    int collective = 1;
    std::vector<int> mpiCounts, mpiOffsets;
    if (rank == 0) {
        mpiCounts.resize(ranks);
        mpiOffsets.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            if (counts[p] > INT_MAX || offsets[p] > INT_MAX) collective = 0;
            mpiCounts[p] = static_cast<int>(counts[p]);
            mpiOffsets[p] = static_cast<int>(offsets[p]);
        }
    }
    MPI_Bcast(&collective, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (collective) {
        MPI_Scatterv(source.data(), mpiCounts.data(), mpiOffsets.data(), type,
                     destination, static_cast<int>(localCount), type, 0, MPI_COMM_WORLD);
        return;
    }
    if (rank == 0) {
        if (localCount) std::copy_n(source.data() + offsets[0], localCount, destination);
        for (int p = 1; p < ranks; ++p) {
            for (uint64_t pos = 0; pos < counts[p];) {
                int count = static_cast<int>(std::min<uint64_t>(counts[p] - pos, INT_MAX));
                MPI_Send(source.data() + offsets[p] + pos, count, type, p, 0, MPI_COMM_WORLD);
                pos += count;
            }
        }
    } else {
        for (uint64_t pos = 0; pos < localCount;) {
            int count = static_cast<int>(std::min<uint64_t>(localCount - pos, INT_MAX));
            MPI_Recv(destination + pos, count, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    if (numRows == std::numeric_limits<index_t>::max() ||
        sparsity == 0 || entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    std::vector<double> values, vec(numRows), reference, out;
    std::vector<index_t> cols, rows;
    std::vector<index_t> rowCounts, rowOffsets, nnzCounts, nnzOffsets;
    // Each rank receives its row count and its nonzero count/offset.
    std::vector<index_t> partitions;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        printf("Initializing data structures...\n");
        values.resize(nItems);
        cols.resize(nItems);
        rows.resize(size_t(numRows) + 1);
        // Restore the C default seed even if MPI startup used rand().
        // Preserve the serial program's exact random-number sequence.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
        }

        rowCounts.resize(ranks);
        rowOffsets.resize(ranks);
        nnzCounts.resize(ranks);
        nnzOffsets.resize(ranks);
        partitions.resize(size_t(ranks) * 3);
        index_t first = 0;
        for (int p = 0; p < ranks; ++p) {
            index_t last = numRows;
            if (p + 1 < ranks) {
                if (nItems == 0) {
                    last = static_cast<index_t>(uint64_t(numRows) * (p + 1) / ranks);
                } else {
                    const index_t target = static_cast<index_t>(uint64_t(nItems) * (p + 1) / ranks);
                    last = static_cast<index_t>(std::lower_bound(rows.begin(), rows.end(), target) - rows.begin());
                    // Choose the closer row boundary without splitting a row's sum.
                    if (last > first && target - rows[last - 1] < rows[last] - target) --last;
                    last = std::max(first, last);
                }
            }
            rowOffsets[p] = first;
            rowCounts[p] = last - first;
            nnzOffsets[p] = rows[first];
            nnzCounts[p] = rows[last] - rows[first];
            partitions[3 * size_t(p)] = rowCounts[p];
            partitions[3 * size_t(p) + 1] = nnzCounts[p];
            partitions[3 * size_t(p) + 2] = nnzOffsets[p];
            first = last;
        }
    }

    index_t localInfo[3];
    MPI_Scatter(partitions.data(), 3, MPI_UINT32_T, localInfo, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = localInfo[0], localNnz = localInfo[1], nnzOffset = localInfo[2];
    std::vector<double> localValues(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localDelimiters(size_t(localRows) + 1);
    scatterArray(values, localValues.data(), localNnz, nnzCounts, nnzOffsets, MPI_DOUBLE, rank, ranks);
    scatterArray(cols, localCols.data(), localNnz, nnzCounts, nnzOffsets, MPI_UINT32_T, rank, ranks);
    // Send each row start once; reconstruct the terminal delimiter locally.
    scatterArray(rows, localDelimiters.data(), localRows, rowCounts, rowOffsets,
                 MPI_UINT32_T, rank, ranks);
    for (index_t i = 0; i < localRows; ++i) localDelimiters[i] -= nnzOffset;
    localDelimiters[localRows] = localNnz;
    for (uint64_t pos = 0; pos < numRows;) {
        int count = static_cast<int>(std::min<uint64_t>(numRows - pos, INT_MAX));
        MPI_Bcast(vec.data() + pos, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        pos += count;
    }
    // The full CSR matrix is no longer needed, including on the root.
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(cols);
    std::vector<index_t>().swap(rows);
    std::vector<index_t>().swap(partitions);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localDelimiters.data(),
                vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (validate || printResults) {
        if (rank == 0) out.resize(numRows);
        if (numRows <= INT_MAX) {
            std::vector<int> counts, offsets;
            if (rank == 0) {
                counts.assign(rowCounts.begin(), rowCounts.end());
                offsets.assign(rowOffsets.begin(), rowOffsets.end());
            }
            MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                        out.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else if (rank == 0) {
            // Preserve row order even beyond MPI's int count range.
            std::copy(localOut.begin(), localOut.end(), out.begin());
            for (int p = 1; p < ranks; ++p) {
                for (uint64_t pos = 0; pos < rowCounts[p];) {
                    int count = static_cast<int>(std::min<uint64_t>(rowCounts[p] - pos, INT_MAX));
                    MPI_Recv(out.data() + rowOffsets[p] + pos, count, MPI_DOUBLE,
                             p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += count;
                }
            }
        } else {
            for (uint64_t pos = 0; pos < localRows;) {
                int count = static_cast<int>(std::min<uint64_t>(localRows - pos, INT_MAX));
                MPI_Send(localOut.data() + pos, count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                pos += count;
            }
        }
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results(out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), out.data(), numRows);
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
