#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
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

static inline void computeRowRange(const index_t nRows, const int nRanks, const int rank,
                                  index_t& rowStart, index_t& rowEnd) {
    const index_t base = nRows / static_cast<index_t>(nRanks);
    const index_t rem = nRows % static_cast<index_t>(nRanks);
    rowStart = static_cast<index_t>(rank) * base + (static_cast<index_t>(rank) < rem ? static_cast<index_t>(rank) : rem);
    rowEnd = rowStart + base + (static_cast<index_t>(rank) < rem ? 1u : 0u);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int argError = 0;

    if (rank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                showHelp = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                argError = 1;
                break;
            }
        }
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&argError, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (showHelp) {
        MPI_Finalize();
        return 0;
    }
    if (argError) {
        MPI_Finalize();
        return 1;
    }

    const uint64_t nItems64 = (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows)) /
                              static_cast<uint64_t>(sparsity);
    if (nItems64 > static_cast<uint64_t>(std::numeric_limits<index_t>::max())) {
        if (rank == 0) {
            printf("Problem size too large (nnz=%llu exceeds 32-bit index range)\n",
                   static_cast<unsigned long long>(nItems64));
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", nRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Root initializes full matrix/vector to preserve original semantics.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast dense vector to all ranks.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute this rank's row block.
    index_t rowStart = 0, rowEnd = 0;
    computeRowRange(numRows, nRanks, rank, rowStart, rowEnd);
    const index_t localRows = rowEnd - rowStart;

    // Scatter row delimiters slice (global nnz offsets per row), then locally rebase.
    std::vector<int> rdCounts;
    std::vector<int> rdDispls;
    if (rank == 0) {
        rdCounts.resize(nRanks);
        rdDispls.resize(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            index_t rs = 0, re = 0;
            computeRowRange(numRows, nRanks, r, rs, re);
            rdCounts[r] = static_cast<int>((re - rs) + 1u);
            rdDispls[r] = static_cast<int>(rs);
        }
    }

    std::vector<index_t> localRowDelimiters(localRows + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rdCounts.data() : nullptr,
                 rank == 0 ? rdDispls.data() : nullptr,
                 MPI_UINT32_T,
                 localRowDelimiters.data(),
                 static_cast<int>(localRows + 1),
                 MPI_UINT32_T,
                 0,
                 MPI_COMM_WORLD);

    const index_t nnzStart = localRowDelimiters.empty() ? 0 : localRowDelimiters.front();
    const index_t nnzEnd = localRowDelimiters.empty() ? 0 : localRowDelimiters.back();
    const index_t localNnz = nnzEnd - nnzStart;
    for (index_t i = 0; i < localRowDelimiters.size(); ++i) {
        localRowDelimiters[i] -= nnzStart;
    }

    // Scatter CSR value/column segments corresponding to this row block.
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;
    if (rank == 0) {
        nnzCounts.resize(nRanks);
        nnzDispls.resize(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            index_t rs = 0, re = 0;
            computeRowRange(numRows, nRanks, r, rs, re);
            const index_t s = h_rowDelimiters[rs];
            const index_t e = h_rowDelimiters[re];
            nnzCounts[r] = static_cast<int>(e - s);
            nnzDispls[r] = static_cast<int>(s);
        }
    }

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_DOUBLE,
                 localVal.data(),
                 static_cast<int>(localNnz),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_UINT32_T,
                 localCols.data(),
                 static_cast<int>(localNnz),
                 MPI_UINT32_T,
                 0,
                 MPI_COMM_WORLD);

    // Free root-only full CSR if no longer needed.
    if (rank == 0) {
        h_val.clear();
        h_cols.clear();
        h_rowDelimiters.clear();
        h_val.shrink_to_fit();
        h_cols.shrink_to_fit();
        h_rowDelimiters.shrink_to_fit();
    }

    std::vector<double> localOut(localRows);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const double* __restrict vec = h_vec.data();
    const double* __restrict val = localVal.data();
    const index_t* __restrict cols = localCols.data();
    const index_t* __restrict rowDelim = localRowDelimiters.data();
    double* __restrict out = localOut.data();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (index_t i = 0; i < localRows; ++i) {
            double t = 0.0;
            const index_t start = rowDelim[i];
            const index_t end = rowDelim[i + 1];
            for (index_t j = start; j < end; ++j) {
                t += val[j] * vec[cols[j]];
            }
            out[i] = t;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localElapsed = t1 - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxElapsed * 1000.0;
        printf("Computation time: %ld ms\n", static_cast<long>(durationMs));

        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / maxElapsed / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather output on root only when needed.
    std::vector<double> h_out;
    std::vector<int> outCounts;
    std::vector<int> outDispls;
    if (printResults || validate) {
        if (rank == 0) {
            h_out.resize(numRows);
            outCounts.resize(nRanks);
            outDispls.resize(nRanks);
            for (int r = 0; r < nRanks; ++r) {
                index_t rs = 0, re = 0;
                computeRowRange(numRows, nRanks, r, rs, re);
                outCounts[r] = static_cast<int>(re - rs);
                outDispls[r] = static_cast<int>(rs);
            }
        }

        MPI_Gatherv(localOut.data(),
                    static_cast<int>(localRows),
                    MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? outCounts.data() : nullptr,
                    rank == 0 ? outDispls.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int ok = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            ok = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return ok ? 0 : 1;
}
