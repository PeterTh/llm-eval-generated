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
// Function: spmvLocal
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format for
//   a local subset of rows.
//
// Arguments:
//   val: array holding the non-zero values for the local matrix rows
//   cols: array of column indices for each element
//   localRowStart: global index of the first row this rank owns
//   localRowEnd: global index one past the last row this rank owns
//   rowDelimiters: global row delimiters array (needed for nnz indexing)
//   vec: dense vector of size dim to be used for multiplication
//   out: output - result from the spmv calculation for local rows only
//
// ****************************************************************************
void spmvLocal(const double* val, const index_t* cols,
               const index_t localRowStart, const index_t localRowEnd,
               const index_t* rowDelimiters,
               const double* vec, double* out) {
    const index_t base = rowDelimiters[localRowStart];
    for (index_t i = localRowStart; i < localRowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i] - base; j < rowDelimiters[i + 1] - base; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i - localRowStart] = t;
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    // Rank 0 prints configuration
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Data initialization on rank 0
    // =========================================================================
    std::vector<double> full_val;
    std::vector<index_t> full_cols;
    std::vector<index_t> full_rowDelimiters;
    std::vector<double> full_vec;

    if (rank == 0) {
        full_val.resize(nItems);
        full_cols.resize(nItems);
        full_rowDelimiters.resize(numRows + 1);
        full_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(full_vec.data(), numRows, maxVal);
        fill(full_val.data(), nItems, maxVal);
        initRandomMatrix(full_cols.data(), full_rowDelimiters.data(), nItems, numRows);
    }

    // =========================================================================
    // Compute row distribution for each rank
    // =========================================================================
    // Distribute rows as evenly as possible: base_rows per rank, first extra_rows
    // ranks get one additional row.
    const index_t base_rows = numRows / numRanks;
    const index_t extra_rows = numRows % numRanks;
    const index_t localRowStart = rank * base_rows + std::min(static_cast<index_t>(rank), extra_rows);
    const index_t localRowEnd = (rank + 1) * base_rows + std::min(static_cast<index_t>(rank + 1), extra_rows);
    const index_t localRows = localRowEnd - localRowStart;

    // =========================================================================
    // Broadcast rowDelimiters (needed by all ranks for indexing)
    // =========================================================================
    // Rank 0 uses full_rowDelimiters as the broadcast source
    std::vector<index_t> rowDelimiters(numRows + 1);
    if (rank == 0) {
        rowDelimiters = full_rowDelimiters;
    }
    MPI_Bcast(rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Each rank computes its own nnz count from the broadcast rowDelimiters
    const index_t local_nnz = rowDelimiters[localRowEnd] - rowDelimiters[localRowStart];

    // =========================================================================
    // Compute scatter send counts and displacements (rank 0)
    // =========================================================================
    std::vector<int> nnz_counts(numRanks);
    std::vector<int> nnz_displs(numRanks);
    if (rank == 0) {
        nnz_displs[0] = rowDelimiters[0];  // always 0
        for (int r = 0; r < numRanks; ++r) {
            const index_t rs = r * base_rows + std::min(static_cast<index_t>(r), extra_rows);
            const index_t re = (r + 1) * base_rows + std::min(static_cast<index_t>(r + 1), extra_rows);
            nnz_counts[r] = rowDelimiters[re] - rowDelimiters[rs];
            nnz_displs[r] = rowDelimiters[rs];
        }
    }

    // =========================================================================
    // Scatter matrix data to all ranks
    // =========================================================================
    // Allocate local storage
    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);

    // Scatter val and cols using MPI_Scatterv from rank 0
    MPI_Scatterv(rank == 0 ? full_val.data() : nullptr,
                 rank == 0 ? nnz_counts.data() : nullptr,
                 rank == 0 ? nnz_displs.data() : nullptr,
                 MPI_DOUBLE,
                 local_val.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? full_cols.data() : nullptr,
                 rank == 0 ? nnz_counts.data() : nullptr,
                 rank == 0 ? nnz_displs.data() : nullptr,
                 MPI_UNSIGNED,
                 local_cols.data(), local_nnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Broadcast vector
    std::vector<double> vec(numRows);
    if (rank == 0) {
        vec = full_vec;
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Reference solution (for validation) - computed on rank 0 only
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        // Inline reference SpMV on rank 0
        for (index_t i = 0; i < numRows; ++i) {
            double t = 0.0;
            for (index_t j = full_rowDelimiters[i]; j < full_rowDelimiters[i + 1]; ++j) {
                const auto col = full_cols[j];
                t += full_val[j] * full_vec[col];
            }
            h_reference[i] = t;
        }
    }

    // =========================================================================
    // SpMV computation loop
    // =========================================================================
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    std::vector<double> local_out(localRows);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocal(local_val.data(), local_cols.data(),
                  localRowStart, localRowEnd,
                  rowDelimiters.data(),
                  vec.data(), local_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Gather results to rank 0 and compute timing
    // =========================================================================
    std::vector<double> h_out(numRows);
    if (rank == 0) {
        // Compute row-based displacements for gathering output
        std::vector<int> row_counts(numRanks);
        std::vector<int> row_displs(numRanks);
        row_displs[0] = 0;
        for (int r = 0; r < numRanks; ++r) {
            const index_t rs = r * base_rows + std::min(static_cast<index_t>(r), extra_rows);
            const index_t re = (r + 1) * base_rows + std::min(static_cast<index_t>(r + 1), extra_rows);
            row_counts[r] = re - rs;
            if (r > 0) {
                row_displs[r] = row_displs[r - 1] + row_counts[r - 1];
            }
        }
        MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                    h_out.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDuration / 1000.0) / 1e9;
        const double avgTime = maxDuration / static_cast<double>(iterations);

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
            }
        }
    }

    // Free full data on rank 0
    full_val.clear();
    full_cols.clear();
    full_rowDelimiters.clear();
    full_vec.clear();

    MPI_Finalize();

    return 0;
}
