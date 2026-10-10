#include <algorithm>
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

// Use collectives for ordinary sizes, and chunk transfers when a CSR array
// exceeds the int count/displacement range supported by MPI-3 collectives.
template <typename T>
void scatterArray(const std::vector<T>& source, std::vector<T>& local,
                  const std::vector<uint64_t>& offsets,
                  const std::vector<uint64_t>& counts, MPI_Datatype type,
                  int rank, int ranks) {
    int large = 0;
    std::vector<int> mpiCounts, mpiOffsets;
    if (rank == 0) {
        mpiCounts.resize(ranks);
        mpiOffsets.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            if (counts[p] > INT32_MAX || offsets[p] > INT32_MAX) large = 1;
            else {
                mpiCounts[p] = static_cast<int>(counts[p]);
                mpiOffsets[p] = static_cast<int>(offsets[p]);
            }
        }
    }
    MPI_Bcast(&large, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!large) {
        MPI_Scatterv(source.data(), mpiCounts.data(), mpiOffsets.data(), type,
                     local.data(), static_cast<int>(local.size()), type, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        if (!local.empty())
            std::copy_n(source.data() + offsets[0], local.size(), local.data());
        for (int p = 1; p < ranks; ++p) {
            for (uint64_t done = 0; done < counts[p];) {
                const int chunk = static_cast<int>(std::min<uint64_t>(counts[p] - done, INT32_MAX));
                MPI_Send(source.data() + offsets[p] + done, chunk, type, p, 0, MPI_COMM_WORLD);
                done += chunk;
            }
        }
    } else {
        for (uint64_t done = 0; done < local.size();) {
            const int chunk = static_cast<int>(std::min<uint64_t>(local.size() - done, INT32_MAX));
            MPI_Recv(local.data() + done, chunk, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done += chunk;
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
    if (sparsity == 0 || (sparsity != 0 && cells / sparsity > std::numeric_limits<index_t>::max())) {
        if (rank == 0) fprintf(stderr, "Sparsity must be positive and the nonzero count must fit in a 32-bit CSR index.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(cells / sparsity);
    std::vector<double> h_val, h_vec(numRows), h_out, h_reference;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<uint64_t> rowOffsets(ranks), rowCounts(ranks), nnzOffsets(ranks), nnzCounts(ranks);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               cells ? 100.0 * (1.0 - static_cast<double>(nItems) / cells) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // Balance nonzeros plus row overhead, without splitting a row. Keeping
        // each row on one rank preserves the original accumulation order.
        uint64_t first = 0;
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int p = 0; p < ranks; ++p) {
            uint64_t last = numRows;
            if (p + 1 < ranks) {
                const uint64_t target = work * (p + 1) / ranks;
                uint64_t lo = first, hi = numRows;
                while (lo < hi) {
                    const uint64_t mid = lo + (hi - lo) / 2;
                    if (uint64_t(h_rowDelimiters[mid]) + mid < target) lo = mid + 1;
                    else hi = mid;
                }
                last = lo;
            }
            rowOffsets[p] = first;
            rowCounts[p] = last - first;
            nnzOffsets[p] = h_rowDelimiters[first];
            nnzCounts[p] = uint64_t(h_rowDelimiters[last]) - nnzOffsets[p];
            first = last;
        }
    }

    uint64_t metadata[3];
    std::vector<uint64_t> allMetadata;
    if (rank == 0) {
        allMetadata.resize(size_t(ranks) * 3);
        for (int p = 0; p < ranks; ++p) {
            allMetadata[3 * size_t(p)] = rowCounts[p];
            allMetadata[3 * size_t(p) + 1] = nnzCounts[p];
            allMetadata[3 * size_t(p) + 2] = nnzOffsets[p];
        }
    }
    MPI_Scatter(allMetadata.data(), 3, MPI_UINT64_T, metadata, 3, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const index_t localRows = static_cast<index_t>(metadata[0]);
    std::vector<double> localValues(metadata[1]), localOut(localRows);
    std::vector<index_t> localCols(metadata[1]), localRowsCSR(localRows);
    scatterArray(h_val, localValues, nnzOffsets, nnzCounts, MPI_DOUBLE, rank, ranks);
    scatterArray(h_cols, localCols, nnzOffsets, nnzCounts, MPI_UINT32_T, rank, ranks);
    scatterArray(h_rowDelimiters, localRowsCSR, rowOffsets, rowCounts, MPI_UINT32_T, rank, ranks);
    for (auto& offset : localRowsCSR) offset -= static_cast<index_t>(metadata[2]);
    localRowsCSR.push_back(static_cast<index_t>(metadata[1]));
    for (uint64_t done = 0; done < numRows;) {
        const int chunk = static_cast<int>(std::min<uint64_t>(uint64_t(numRows) - done, INT32_MAX));
        MPI_Bcast(h_vec.data() + done, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        done += chunk;
    }
    // Only the distributed CSR is needed during the benchmark.
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localRowsCSR.data(),
                h_vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather once, and only when the ordered global result is requested.
    if (validate || printResults) {
        if (rank == 0) h_out.resize(numRows);
        if (numRows <= INT32_MAX) {
            std::vector<int> counts(ranks), offsets(ranks);
            if (rank == 0) {
                for (int p = 0; p < ranks; ++p) {
                    counts[p] = static_cast<int>(rowCounts[p]);
                    offsets[p] = static_cast<int>(rowOffsets[p]);
                }
            }
            MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                        h_out.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else if (rank == 0) {
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p) {
                for (uint64_t done = 0; done < rowCounts[p];) {
                    const int chunk = static_cast<int>(std::min<uint64_t>(rowCounts[p] - done, INT32_MAX));
                    MPI_Recv(h_out.data() + rowOffsets[p] + done, chunk, MPI_DOUBLE,
                             p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    done += chunk;
                }
            }
        } else {
            for (uint64_t done = 0; done < localRows;) {
                const int chunk = static_cast<int>(std::min<uint64_t>(uint64_t(localRows) - done, INT32_MAX));
                MPI_Send(localOut.data() + done, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                done += chunk;
            }
        }
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
