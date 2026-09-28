#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fillRange
//
// Purpose:
//   Initialize array with random values. The random number generator is
//   advanced exactly n times (as in the original serial code), but only the
//   values with global index in [from, to) are actually stored, starting at
//   A[0]. This keeps the generated values bit-identical to the serial version
//   while allowing every rank to materialize only its own slice.
//
// Arguments:
//   A: pointer to the (local) array to initialize
//   n: number of elements in the global array
//   from, to: half-open range of global indices to keep
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fillRange(double* A, const index_t n, const index_t from, const index_t to,
               const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= from && i < to) {
            A[i - from] = v;
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrixLocal
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
//   Every rank replays the exact same random sequence as the original serial
//   generator (so the resulting global matrix is identical), but only the rows
//   in [rowStart, rowEnd) are stored locally. Column indices remain global
//   since the dense vector is replicated.
//
// Arguments:
//   cols:          output array for column indexes of the local rows
//   rowDelimiters: output array of size (rowEnd-rowStart)+1 holding local
//                  (zero-based) indices into cols
//   n:             number of nonzero elements in the global matrix
//   dim:           number of rows/columns in the matrix
//   rowStart/rowEnd: half-open range of rows owned by this rank
//   nnzStart:      output - global index of the first local nonzero
//
// ****************************************************************************
void initRandomMatrixLocal(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters,
                           const index_t n, const index_t dim, const index_t rowStart,
                           const index_t rowEnd, index_t& nnzStart) {
    index_t nnzAssigned = 0;
    nnzStart = n;

    cols.clear();
    rowDelimiters.assign(static_cast<size_t>(rowEnd - rowStart) + 1, 0);

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        const bool mine = (i >= rowStart && i < rowEnd);
        if (mine) {
            if (i == rowStart) {
                nnzStart = nnzAssigned;
            }
            rowDelimiters[i - rowStart] = nnzAssigned - nnzStart;
        }
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (mine) {
                    cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[rowEnd - rowStart] = static_cast<index_t>(cols.size());
    if (rowStart >= rowEnd) {
        nnzStart = nnzAssigned;
    }
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
//   dim: number of rows to compute
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
//   offset: global index of the first local element (for reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset,
                       ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + offset, ref, res, relError);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numProcs);
    }

    // Block-distribute the rows of the matrix; nonzeros per row are statistically
    // uniform, so an even row split also balances the nonzeros.
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowOffsets(numProcs + 1);
    {
        const index_t base = numRows / static_cast<index_t>(numProcs);
        const index_t rem = numRows % static_cast<index_t>(numProcs);
        rowOffsets[0] = 0;
        for (int p = 0; p < numProcs; ++p) {
            rowCounts[p] = static_cast<int>(base + (static_cast<index_t>(p) < rem ? 1u : 0u));
            rowOffsets[p + 1] = rowOffsets[p] + rowCounts[p];
        }
    }
    const index_t rowStart = static_cast<index_t>(rowOffsets[rank]);
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t rowEnd = rowStart + localRows;

    // Allocate and initialize data structures
    std::vector<index_t> h_cols;           // Column indices (local rows only)
    std::vector<index_t> h_rowDelimiters;  // Row delimiters (local, zero-based)
    std::vector<double> h_vec(numRows);    // Dense vector (replicated)
    std::vector<double> h_out(localRows);  // Output vector (local rows)

    if (rank == 0) {
        printf("Initializing data structures...\n");
    }

    // The matrix structure is generated first: it re-seeds the RNG anyway, and its
    // result tells us which slice of the value array belongs to this rank.
    index_t nnzStart = 0;
    initRandomMatrixLocal(h_cols, h_rowDelimiters, nItems, numRows, rowStart, rowEnd, nnzStart);
    const index_t localNnz = static_cast<index_t>(h_cols.size());

    // Reproduce the original RNG stream for the value arrays: the program starts
    // with the generator in its default state, which is equivalent to srand(1).
    std::vector<double> h_val(localNnz);  // Non-zero values (local rows only)
    srand(1);
    fillRange(h_vec.data(), numRows, 0, numRows, maxVal);
    fillRange(h_val.data(), nItems, nnzStart, nnzStart + localNnz, maxVal);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), localRows,
                h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), localRows,
                h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The benchmark time is determined by the slowest rank
    long long localMs = static_cast<long long>(duration.count());
    long long elapsedMs = 0;
    MPI_Allreduce(&localMs, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", elapsedMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> gathered;
        if (rank == 0) {
            gathered.resize(numRows);
        }
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, gathered.data(),
                    rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(gathered, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        int localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowStart) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
