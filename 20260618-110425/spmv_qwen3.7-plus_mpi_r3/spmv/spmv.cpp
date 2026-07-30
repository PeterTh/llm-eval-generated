#include <mpi.h>

#include <chrono>
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

// Compute the row range [start, start+count) for a given MPI rank
static void getRowRange(int r, int nprocs, index_t numRows, index_t& start, index_t& count) {
    index_t base = numRows / nprocs;
    index_t rem = numRows % nprocs;
    if (static_cast<index_t>(r) < rem) {
        count = base + 1;
        start = static_cast<index_t>(r) * (base + 1);
    } else {
        count = base;
        start = rem * (base + 1) + (static_cast<index_t>(r) - rem) * base;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (same argv on all ranks via mpirun)
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

    // Compute local row distribution (deterministic, same on all ranks)
    index_t localRowStart, localRows;
    getRowRange(rank, nprocs, numRows, localRowStart, localRows);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI Processes: %d\n", nprocs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // *****************************************************************
    // Data generation (rank 0 only, preserves original RNG sequence)
    // *****************************************************************
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // *****************************************************************
    // Compute Scatterv parameters on rank 0
    // *****************************************************************
    std::vector<int> nnzCounts(nprocs);
    std::vector<int> nnzDisps(nprocs);

    if (rank == 0) {
        for (int p = 0; p < nprocs; ++p) {
            index_t pStart, pCount;
            getRowRange(p, nprocs, numRows, pStart, pCount);
            nnzCounts[p] = static_cast<int>(h_rowDelimiters[pStart + pCount] - h_rowDelimiters[pStart]);
            nnzDisps[p] = static_cast<int>(h_rowDelimiters[pStart]);
        }
    }

    // Distribute nnz count to each rank for buffer allocation
    int localNnz = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // *****************************************************************
    // Allocate local buffers
    // *****************************************************************
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);
    std::vector<double> local_out(localRows);

    // *****************************************************************
    // Distribute CSR matrix data (val and cols) via Scatterv
    // *****************************************************************
    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDisps.data(), MPI_DOUBLE,
                 local_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDisps.data(), MPI_UNSIGNED,
                 local_cols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // *****************************************************************
    // Broadcast rowDelimiters so each rank can extract its local portion
    // *****************************************************************
    if (rank != 0) {
        h_rowDelimiters.resize(numRows + 1);
    }
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Extract local row delimiters with offset adjustment for local indexing
    {
        const index_t nnzOffset = h_rowDelimiters[localRowStart];
        for (index_t i = 0; i <= localRows; ++i) {
            local_rowDelimiters[i] = h_rowDelimiters[localRowStart + i] - nnzOffset;
        }
    }

    // Free global rowDelimiters on non-root ranks to save memory
    if (rank != 0) {
        h_rowDelimiters.clear();
        h_rowDelimiters.shrink_to_fit();
    }

    // *****************************************************************
    // Broadcast dense vector (all ranks need full vector for SpMV)
    // *****************************************************************
    if (rank != 0) {
        h_vec.resize(numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // *****************************************************************
    // Parallel SpMV computation with timing
    // *****************************************************************
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    double localStart = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (index_t i = 0; i < localRows; ++i) {
            double t = 0.0;
            const index_t rowBegin = local_rowDelimiters[i];
            const index_t rowEnd = local_rowDelimiters[i + 1];
            for (index_t j = rowBegin; j < rowEnd; ++j) {
                t += local_val[j] * h_vec[local_cols[j]];
            }
            local_out[i] = t;
        }
    }

    double localEnd = MPI_Wtime();
    double localElapsed = localEnd - localStart;

    // Gather maximum elapsed time across all ranks
    double maxElapsed;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // *****************************************************************
    // Gather output vector on rank 0
    // *****************************************************************
    std::vector<int> outCounts(nprocs);
    std::vector<int> outDisps(nprocs);
    if (rank == 0) {
        for (int p = 0; p < nprocs; ++p) {
            index_t pStart, pCount;
            getRowRange(p, nprocs, numRows, pStart, pCount);
            outCounts[p] = static_cast<int>(pCount);
            outDisps[p] = static_cast<int>(pStart);
        }
    }

    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                h_out.data(), outCounts.data(), outDisps.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // *****************************************************************
    // Report results (rank 0 only)
    // *****************************************************************
    if (rank == 0) {
        long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        const double gflops = (2.0 * nItems * iterations) / maxElapsed / 1e9;
        const double avgTime = (maxElapsed * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
