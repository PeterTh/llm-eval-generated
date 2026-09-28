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
//   Draws n random values from the current RNG stream (exactly as the serial
//   reference implementation does), but only stores the values belonging to
//   the half-open index range [first, last) into A (A holds last-first items).
//   This keeps the random sequence identical to the serial version while
//   allowing every rank to materialize only its own slice of the data.
//
// Arguments:
//   A: pointer to the array receiving the values of the local slice
//   n: total number of elements drawn
//   first, last: half-open range of global indices to store
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fillRange(double* A, const index_t n, const index_t first, const index_t last,
               const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= first && i < last) {
            A[i - first] = v;
        }
    }
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
    fillRange(A, n, 0, n, maxVal);
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
//   The random decisions are replayed in full on every rank (the RNG stream is
//   inherently sequential), but only the column indices of the locally owned
//   rows [rowFirst, rowLast) are stored, so that no rank ever holds the
//   complete matrix. The row delimiters (dim+1 entries) are always computed
//   completely, since they are needed to derive the distribution.
//
// Arguments:
//   cols:          array for column indexes of the locally owned elements;
//                  may be nullptr to only compute the row delimiters
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowFirst, rowLast: half-open range of locally owned rows
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim,
                      const index_t rowFirst, const index_t rowLast) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Offset of the first locally owned nonzero (known once rowDelimiters exist)
    const index_t nnzOffset = (cols != nullptr) ? rowDelimiters[rowFirst] : 0;

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const bool store = (cols != nullptr) && (i >= rowFirst) && (i < rowLast);
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (store) {
                    cols[nnzAssigned - nnzOffset] = j;
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
//   indexOffset: global index of the first local element (for reporting)
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
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       i + indexOffset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + indexOffset, ref, res, relError);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // ---------------------------------------------------------------------
    // Data distribution
    //
    // The matrix is partitioned by rows. Since every rank owns a contiguous
    // block of rows and multiplies with the full (replicated, read-only) input
    // vector, the per-iteration SpMV needs no communication at all; only the
    // distributed result vector is collected once at the end.
    //
    // The row blocks are chosen such that the number of nonzeros per rank is
    // as even as possible (load balancing), which requires knowing the row
    // delimiters first. They are therefore computed in a first pass that
    // stores no column indices; the second pass then materializes only the
    // locally owned column indices.
    // ---------------------------------------------------------------------
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (global, small)
    initRandomMatrix(nullptr, h_rowDelimiters.data(), nItems, numRows, 0, 0);

    // Determine the row block of every rank by balancing the nonzero counts
    std::vector<index_t> rowStart(numRanks + 1);
    {
        index_t row = 0;
        for (int r = 0; r < numRanks; ++r) {
            rowStart[r] = row;
            // Target for the last nonzero of this rank's block
            const index_t target =
                static_cast<index_t>((static_cast<uint64_t>(nItems) * (r + 1)) / numRanks);
            // Advance while the block stays below its nonzero target, leaving at
            // least one row for every remaining rank (if there are enough rows).
            const int64_t maxRow = static_cast<int64_t>(numRows) - (numRanks - r - 1);
            while (static_cast<int64_t>(row) < maxRow && h_rowDelimiters[row + 1] <= target) {
                ++row;
            }
        }
        rowStart[numRanks] = numRows;
    }

    const index_t myFirstRow = rowStart[rank];
    const index_t myLastRow = rowStart[rank + 1] > myFirstRow ? rowStart[rank + 1] : myFirstRow;
    const index_t myRows = myLastRow - myFirstRow;
    const index_t myFirstNnz = h_rowDelimiters[myFirstRow];
    const index_t myNnz = h_rowDelimiters[myLastRow] - myFirstNnz;

    // Allocate and initialize the local data structures
    std::vector<double> h_val(myNnz);      // Non-zero values (local rows)
    std::vector<index_t> h_cols(myNnz);    // Column indices (local rows)
    std::vector<double> h_vec(numRows);    // Dense vector (replicated)
    std::vector<double> h_out(myRows);     // Output vector (local rows)

    // Reproduce the serial RNG stream: the vector is drawn first, then the
    // matrix values (srand(1) is the implicit default seed).
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    fillRange(h_val.data(), nItems, myFirstNnz, myFirstNnz + myNnz, maxVal);

    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows, myFirstRow, myLastRow);

    // Local CSR row delimiters, rebased to the local nonzero array
    std::vector<index_t> h_localRowDelimiters(myRows + 1);
    for (index_t i = 0; i <= myRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[myFirstRow + i] - myFirstNnz;
    }

    // Displacements/counts for collecting the distributed result vector
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        recvCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        displs[r] = static_cast<int>(rowStart[r]);
    }

    // For validation, compute the reference solution for the local rows
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
            fflush(stdout);
        }
        h_reference.resize(myRows);
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), myRows, h_reference.data());
    }

    // Full result vector (only needed on rank 0 for output/validation printing)
    std::vector<double> h_fullOut(rank == 0 ? numRows : 0);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(),
                h_vec.data(), myRows, h_out.data());
    }

    // Collect the distributed result vector on rank 0
    MPI_Gatherv(h_out.data(), static_cast<int>(myRows), MPI_DOUBLE, h_fullOut.data(),
                recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // The benchmark time is the slowest rank's time
    long long duration = localDuration;
    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration / 1000.0) / 1e9;
        const double avgTime = duration / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_fullOut, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            fflush(stdout);
        }
        int localValid = verifyResults(h_reference.data(), h_out.data(), myRows, myFirstRow) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
