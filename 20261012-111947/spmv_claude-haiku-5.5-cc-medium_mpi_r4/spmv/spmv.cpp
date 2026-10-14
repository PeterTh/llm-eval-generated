#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: rowStart
//
// Purpose:
//   Block row partitioning: returns the first row owned by a rank. Rank r owns
//   rows [rowStart(r), rowStart(r+1)). The first (dim % nprocs) ranks own one
//   extra row, so rowStart(nprocs) == dim.
//
// Arguments:
//   rank: rank whose first row is requested (may equal nprocs)
//   nprocs: number of MPI ranks
//   dim: number of rows in the matrix
//
// ****************************************************************************
index_t rowStart(const int rank, const int nprocs, const index_t dim) {
    const index_t r = static_cast<index_t>(rank);
    const index_t p = static_cast<index_t>(nprocs);
    return r * (dim / p) + std::min(r, dim % p);
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
// Function: fillLocal
//
// Purpose:
//   Draws n random values from the same stream as fill(), but stores only the
//   values with indices in [keepBegin, keepEnd). Lets each rank keep just its
//   share of the non-zero values without storing the whole array.
//
// Arguments:
//   out: receives the kept values, in index order
//   n: number of values drawn
//   maxVal: specifies range of random values [0, maxVal]
//   keepBegin: first index to keep
//   keepEnd: one past the last index to keep
//
// ****************************************************************************
void fillLocal(std::vector<double>& out, const index_t n, const double maxVal,
               const index_t keepBegin, const index_t keepEnd) {
    out.reserve(keepEnd - keepBegin);
    for (index_t i = 0; i < n; ++i) {
        const double val = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= keepBegin && i < keepEnd) {
            out.push_back(val);
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. Every rank runs the full pass so that the row
//   delimiters match the serial program, but only the column indices of rows
//   [rowBegin, rowEnd) are kept.
//
// Arguments:
//   cols:          array for column indexes of this rank's rows; receives the
//                  entries of rows [rowBegin, rowEnd) in order
//   rowDelimiters: array of size dim+1 holding indices to rows (global);
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row owned by this rank
//   rowEnd:        one past the last row owned by this rank
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
        const bool ownsRow = (i >= rowBegin && i < rowEnd);
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (ownsRow) {
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
//   rowOffset: global index of the first element (used when reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t rowOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + rowOffset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + rowOffset, ref, res, relError);
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // This rank owns rows [rowBegin, rowEnd) and the matching nonzeros
    const index_t rowBegin = rowStart(rank, nprocs, numRows);
    const index_t rowEnd = rowStart(rank + 1, nprocs, numRows);
    const index_t localRows = rowEnd - rowBegin;

    if (rank == 0) {
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

    // Generate the sparsity pattern exactly as the serial program does, keeping
    // only the column indices of this rank's rows
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (global)
    std::vector<index_t> h_cols;                        // Column indices (local)
    initRandomMatrix(h_cols, h_rowDelimiters.data(), nItems, numRows, rowBegin, rowEnd);

    // Local nonzeros are the contiguous range [nnzBegin, nnzEnd) of the global ordering
    const index_t nnzBegin = h_rowDelimiters[rowBegin];
    const index_t nnzEnd = h_rowDelimiters[rowEnd];
    std::vector<index_t> h_localRowDelimiters(localRows + 1);  // Row delimiters (local)
    for (index_t i = 0; i <= localRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[rowBegin + i] - nnzBegin;
    }

    // Values and the dense vector come from the default-seeded stream, which the
    // serial program uses first (rand() starts as if srand(1) was called)
    srand(1);
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_val;                          // Non-zero values (local)
    fill(h_vec.data(), numRows, maxVal);
    fillLocal(h_val, nItems, maxVal, nnzBegin, nnzEnd);
    std::vector<double> h_out(localRows);               // Output vector (local rows)

    // For validation, compute reference solution for the local rows
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation. Each rank works on its own rows and the
    // dense vector is already replicated, so no communication is needed here.
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The computation is only done when the slowest rank finishes
    long localMs = duration.count();
    long maxMs = 0;
    MPI_Allreduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(nprocs);
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const index_t begin = rowStart(r, nprocs, numRows);
            counts[r] = static_cast<int>(rowStart(r + 1, nprocs, numRows) - begin);
            displs[r] = static_cast<int>(begin);
        }
        std::vector<double> h_fullOut(rank == 0 ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_fullOut.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(h_fullOut, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
