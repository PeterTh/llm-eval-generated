#include <algorithm>
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
//   Computes sparse matrix-vector multiplication using CSR format for a
//   contiguous block of rows. rowDelimiters is expected to be re-based so
//   that rowDelimiters[0] == 0, i.e. it indexes into a local val/cols block.
//
// Arguments:
//   val: array holding the non-zero values for the local row block
//   cols: array of column indices for each element in the local row block
//   rowDelimiters: array of size numLocalRows+1 holding indices into val/cols
//   vec: dense vector of size dim to be used for multiplication
//   numLocalRows: number of rows in the local block
//   out: output - result from the spmv calculation for the local block
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t numLocalRows, double* out) {
    for (index_t i = 0; i < numLocalRows; ++i) {
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

// ****************************************************************************
// Function: computeRowBlock
//
// Purpose:
//   Computes a contiguous, near-equal block distribution of `numRows` rows
//   across `numRanks` MPI ranks. Every rank computes this independently and
//   deterministically, so no communication is required to agree on it.
//
// ****************************************************************************
void computeRowBlock(const index_t numRows, const int numRanks, const int rank,
                      index_t& rowStart, index_t& rowCount) {
    const index_t base = numRows / static_cast<index_t>(numRanks);
    const index_t rem = numRows % static_cast<index_t>(numRanks);
    const index_t r = static_cast<index_t>(rank);
    rowCount = base + (r < rem ? 1 : 0);
    rowStart = r * base + std::min(r, rem);
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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank 0 generates the full matrix and vector so that the random
    // number sequence (and thus the resulting matrix) is bit-identical to
    // the original single-process program.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_reference;
    std::vector<double> h_out;

    std::vector<double> h_vec(numRows);  // Dense vector, replicated on every rank

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        h_out.resize(numRows);
    }

    // Broadcast the dense vector to every rank (every row may reference any column)
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Every rank independently computes the same contiguous row-block distribution
    index_t rowStart = 0;
    index_t rowCount = 0;
    computeRowBlock(numRows, numRanks, rank, rowStart, rowCount);

    // Root builds per-rank send counts/displacements for the row-delimiter scatter.
    // Adjacent blocks intentionally overlap by one element (the shared boundary),
    // which MPI_Scatterv supports since it only reads from the send buffer.
    std::vector<int> delimCounts, delimDispls, nnzCounts, nnzDispls;
    if (rank == 0) {
        delimCounts.resize(numRanks);
        delimDispls.resize(numRanks);
        nnzCounts.resize(numRanks);
        nnzDispls.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rCount;
            computeRowBlock(numRows, numRanks, r, rStart, rCount);
            delimCounts[r] = static_cast<int>(rCount + 1);
            delimDispls[r] = static_cast<int>(rStart);
            const index_t nnzStart = h_rowDelimiters[rStart];
            const index_t nnzEnd = h_rowDelimiters[rStart + rCount];
            nnzCounts[r] = static_cast<int>(nnzEnd - nnzStart);
            nnzDispls[r] = static_cast<int>(nnzStart);
        }
    }

    std::vector<index_t> local_rowDelimiters(rowCount + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? delimCounts.data() : nullptr,
                 rank == 0 ? delimDispls.data() : nullptr,
                 MPI_UINT32_T, local_rowDelimiters.data(), static_cast<int>(rowCount + 1),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Re-base the local row delimiters so they index into the local val/cols arrays
    const index_t localOffset = local_rowDelimiters[0];
    for (index_t i = 0; i < rowCount + 1; ++i) {
        local_rowDelimiters[i] -= localOffset;
    }
    const index_t localNnz = local_rowDelimiters[rowCount];

    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_DOUBLE, local_val.data(), static_cast<int>(localNnz),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_UINT32_T, local_cols.data(), static_cast<int>(localNnz),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<double> local_out(rowCount);

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), rowCount, local_out.data());
    }

    const double t1 = MPI_Wtime();
    const double localElapsed = t1 - t0;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the per-rank row blocks back into the full output vector on rank 0
    std::vector<int> outCounts, outDispls;
    if (rank == 0) {
        outCounts.resize(numRanks);
        outDispls.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            index_t rStart, rCount;
            computeRowBlock(numRows, numRanks, r, rStart, rCount);
            outCounts[r] = static_cast<int>(rCount);
            outDispls[r] = static_cast<int>(rStart);
        }
    }
    MPI_Gatherv(local_out.data(), static_cast<int>(rowCount), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? outCounts.data() : nullptr,
                rank == 0 ? outDispls.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = duration_ms / static_cast<double>(iterations);

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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
