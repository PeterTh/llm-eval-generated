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
// Function: rowBlockStart
//
// Purpose:
//   First row owned by a rank in a block distribution of numRows rows.
//   Rank r owns rows [rowBlockStart(r), rowBlockStart(r+1)).
//
// Arguments:
//   numRows: number of rows/columns in the matrix
//   rank:    rank index in [0, size]
//   size:    number of ranks
//
// ****************************************************************************
index_t rowBlockStart(const index_t numRows, const int rank, const int size) {
    return static_cast<index_t>(static_cast<uint64_t>(numRows) * rank / size);
}

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
// Function: fillRange
//
// Purpose:
//   Initialize a slice of a logically global array of random values. The
//   random stream is consumed in the same order as fill(), so the values
//   match the serial program; values before 'begin' are discarded.
//
// Arguments:
//   A: pointer to the local slice to initialize
//   begin: global index of the first element of the slice
//   count: number of elements in the slice
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fillRange(double* A, const index_t begin, const index_t count, const double maxVal) {
    const index_t end = begin + count;
    for (index_t i = 0; i < end; ++i) {
        const double val = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= begin) {
            A[i - begin] = val;
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. Every rank walks the full matrix so the random
//   sequence matches the serial program, but only the column indexes of
//   rows in [rowBegin, rowEnd) are stored.
//
// Arguments:
//   cols:          output, column indexes of the non-zeroes in rows [rowBegin, rowEnd)
//   rowDelimiters: array of size dim+1 holding global indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row stored by this rank
//   rowEnd:        one past the last row stored by this rank
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim, const index_t rowBegin, const index_t rowEnd) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const bool owned = i >= rowBegin && i < rowEnd;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (owned) {
                    cols.push_back(j);
                }
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
//   indexOffset: global index of the first element (for reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t indexOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", indexOffset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       indexOffset + i, ref, res, relError);
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool isRoot = (rank == 0);

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
            if (isRoot) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Rows owned by this rank
    const index_t rowBegin = rowBlockStart(numRows, rank, size);
    const index_t rowEnd = rowBlockStart(numRows, rank + 1, size);
    const index_t localRows = rowEnd - rowBegin;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Allocate and initialize data structures.
    // The matrix is generated identically on every rank; each rank keeps only its rows.
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Global row delimiters
    std::vector<index_t> h_cols;                        // Column indices of local rows
    initRandomMatrix(h_cols, h_rowDelimiters.data(), nItems, numRows, rowBegin, rowEnd);

    const index_t nnzBegin = h_rowDelimiters[rowBegin];
    const index_t nnzEnd = h_rowDelimiters[rowEnd];

    std::vector<index_t> h_localRowDelimiters(localRows + 1);  // Row delimiters relative to nnzBegin
    for (index_t i = 0; i <= localRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[rowBegin + i] - nnzBegin;
    }

    // Values and vector come from the same random stream as the serial program
    // (default seed, dense vector first, then non-zero values).
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated on all ranks)
    std::vector<double> h_val(nnzEnd - nnzBegin);       // Non-zero values of local rows
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    fillRange(h_val.data(), nnzBegin, nnzEnd - nnzBegin, maxVal);

    std::vector<double> h_out(localRows);               // Local output rows

    // For validation, compute reference solution for local rows
    std::vector<double> h_reference;
    if (validate) {
        if (isRoot) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localElapsed = MPI_Wtime() - start;

    // The slowest rank determines the wall-clock time
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        const double elapsedMs = elapsed * 1000.0;

        printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            displs[r] = static_cast<int>(rowBlockStart(numRows, r, size));
            counts[r] = static_cast<int>(rowBlockStart(numRows, r + 1, size)) - displs[r];
        }

        std::vector<double> h_full;
        if (isRoot) {
            h_full.resize(numRows);
        }
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (isRoot) {
            print_results(h_full, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (isRoot) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
