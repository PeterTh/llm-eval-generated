#include <chrono>
#include <cmath>
#include <cstdint>
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
//   local, contiguous block of rows. Column indices refer to the global
//   (replicated) input vector.
//
// Arguments:
//   val: array holding the non-zero values for the local rows
//   cols: array of column indices for each local element
//   rowDelimiters: local row delimiters (size numLocalRows+1)
//   vec: dense vector of size dim to be used for multiplication
//   numLocalRows: number of rows owned by this rank
//   out: output - local result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t numLocalRows, double* __restrict out) {
    for (index_t i = 0; i < numLocalRows; ++i) {
        double t = 0.0;
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
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
//   offset: global index of the first element (for diagnostics)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       offset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       offset + i, ref, res, relError);
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
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine this rank's contiguous block of rows (balanced partitioning).
    const uint64_t rows64 = numRows;
    const uint64_t start64 = (rows64 * static_cast<uint64_t>(rank)) / static_cast<uint64_t>(numProcs);
    const uint64_t end64 = (rows64 * static_cast<uint64_t>(rank + 1)) / static_cast<uint64_t>(numProcs);
    const index_t rowStart = static_cast<index_t>(start64);
    const index_t rowEnd = static_cast<index_t>(end64);
    const index_t numLocalRows = rowEnd - rowStart;

    // Root owns the full problem during initialization; it generates the same
    // deterministic data as the serial reference before distributing it.
    std::vector<double> fullVal;
    std::vector<index_t> fullCols;
    std::vector<index_t> fullRowDelimiters;
    std::vector<double> vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fullVal.resize(nItems);
        fullCols.resize(nItems);
        fullRowDelimiters.resize(numRows + 1);

        fill(vec.data(), numRows, maxVal);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), fullRowDelimiters.data(), nItems, numRows);
    } else {
        fullRowDelimiters.resize(numRows + 1);
    }

    // Replicate the input vector and the global row offsets on every rank.
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(fullRowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

    // Number of nonzeros owned by this rank.
    const index_t localNnz = fullRowDelimiters[rowEnd] - fullRowDelimiters[rowStart];

    // Local CSR storage (columns remain global indices into the replicated vec).
    std::vector<double> val(localNnz);
    std::vector<index_t> cols(localNnz);
    std::vector<index_t> rowDelimiters(numLocalRows + 1);

    const index_t localBase = fullRowDelimiters[rowStart];
    for (index_t i = 0; i < numLocalRows; ++i) {
        rowDelimiters[i] = fullRowDelimiters[rowStart + i] - localBase;
    }
    rowDelimiters[numLocalRows] = localNnz;

    // Scatter the CSR values and column indices according to row ownership.
    std::vector<int> sendCounts;
    std::vector<int> sendDispls;
    if (rank == 0) {
        sendCounts.resize(numProcs);
        sendDispls.resize(numProcs);
        for (int r = 0; r < numProcs; ++r) {
            const uint64_t rs = (rows64 * static_cast<uint64_t>(r)) / static_cast<uint64_t>(numProcs);
            const uint64_t re = (rows64 * static_cast<uint64_t>(r + 1)) / static_cast<uint64_t>(numProcs);
            sendDispls[r] = static_cast<int>(fullRowDelimiters[rs]);
            sendCounts[r] = static_cast<int>(fullRowDelimiters[re] - fullRowDelimiters[rs]);
        }
    }

    MPI_Scatterv(rank == 0 ? fullVal.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_DOUBLE, val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullCols.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UINT32_T, cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Release the root's full problem storage; the working set is now distributed.
    if (rank == 0) {
        std::vector<double>().swap(fullVal);
        std::vector<index_t>().swap(fullCols);
    }
    std::vector<index_t>().swap(fullRowDelimiters);

    std::vector<double> out(numLocalRows);

    // For validation, compute the local reference solution. Each row is
    // accumulated in exactly the same order as the original serial code.
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        reference.resize(numLocalRows);
        spmvCpu(val.data(), cols.data(), rowDelimiters.data(), vec.data(), numLocalRows,
                reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(val.data(), cols.data(), rowDelimiters.data(), vec.data(), numLocalRows,
                out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const long long duration = maxDuration;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration);

        // Calculate performance metrics
        const double timeSeconds = (duration > 0) ? (static_cast<double>(duration) / 1000.0)
                                                  : (1e-9);
        const double gflops = (2.0 * nItems * iterations) / timeSeconds / 1e9;
        const double avgTime = (iterations > 0) ? (static_cast<double>(duration) / iterations) : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> fullOut;
        std::vector<int> recvCounts;
        std::vector<int> recvDispls;

        if (rank == 0) {
            fullOut.resize(numRows);
            recvCounts.resize(numProcs);
            recvDispls.resize(numProcs);
            for (int r = 0; r < numProcs; ++r) {
                const uint64_t rs = (rows64 * static_cast<uint64_t>(r)) / static_cast<uint64_t>(numProcs);
                const uint64_t re = (rows64 * static_cast<uint64_t>(r + 1)) / static_cast<uint64_t>(numProcs);
                recvDispls[r] = static_cast<int>(rs);
                recvCounts[r] = static_cast<int>(re - rs);
            }
        }

        MPI_Gatherv(out.data(), static_cast<int>(numLocalRows), MPI_DOUBLE,
                    rank == 0 ? fullOut.data() : nullptr, recvCounts.data(), recvDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(fullOut, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        const bool localValid = verifyResults(reference.data(), out.data(), numLocalRows, rowStart);
        int localValidInt = localValid ? 1 : 0;
        int globalValidInt = 1;
        MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        const bool valid = (globalValidInt != 0);

        if (rank == 0) {
            printf("Validating result...\n");
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
