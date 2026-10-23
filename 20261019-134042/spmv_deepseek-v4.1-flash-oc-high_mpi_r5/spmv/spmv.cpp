#include <mpi.h>

#include <algorithm>
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
// Structure: LocalMatrix
//
// Purpose:
//   Holds the locally owned, contiguous block of rows of the matrix in CSR
//   format. Column indices remain global so that only the distributed rows of
//   the matrix need to be stored.
//
// ****************************************************************************
struct LocalMatrix {
    std::vector<double> val;               // Non-zero values (local rows)
    std::vector<index_t> cols;             // Column indices (local rows)
    std::vector<index_t> rowDelimiters;    // Local row delimiters
    index_t firstNnz = 0;                  // Global index of first local nnz
    index_t localNnz = 0;                  // Number of local non-zeros
};

// ****************************************************************************
// Function: generateLocalMatrix
//
// Purpose:
//   Reproduces the exact matrix that the serial initRandomMatrix generates,
//   but only stores the rows owned by this rank. The global rand() stream is
//   replayed up to the local row range, so the resulting matrix is identical
//   for any number of MPI processes.
//
// Arguments:
//   lm:       local matrix to fill
//   n:        total number of non-zero elements
//   dim:      number of rows/columns in the matrix
//   rowStart: first locally owned row (inclusive)
//   rowEnd:   last locally owned row (exclusive)
//
// ****************************************************************************
void generateLocalMatrix(LocalMatrix& lm, const index_t n, const index_t dim,
                         const index_t rowStart, const index_t rowEnd) {
    const index_t localRows = rowEnd - rowStart;

    lm.rowDelimiters.assign(static_cast<size_t>(localRows) + 1, 0);
    lm.cols.clear();

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));

    // Match the serial matrix generation exactly
    srand(8675309);

    index_t nnzAssigned = 0;
    index_t firstNnz = n;
    bool fillRemaining = false;

    // Approximate local non-zero count for a single allocation
    const size_t estimatedNnz =
        (dim > 0) ? static_cast<size_t>(static_cast<double>(n) *
                                        static_cast<double>(localRows) /
                                        static_cast<double>(dim)) + 64
                  : 64;
    lm.cols.reserve(estimatedNnz);

    // Only replay the rows up to the end of this rank's block. Earlier ranks
    // do proportionally less work, so initialization remains scalable.
    for (index_t i = 0; i < rowEnd; ++i) {
        if (i == rowStart) {
            firstNnz = nnzAssigned;
        }

        const bool localRow = (i >= rowStart);
        if (localRow) {
            lm.rowDelimiters[i - rowStart] = static_cast<index_t>(lm.cols.size());
        }

        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                if (localRow) {
                    lm.cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }

    if (localRows == 0) {
        firstNnz = n;
    }

    lm.rowDelimiters[localRows] = static_cast<index_t>(lm.cols.size());
    lm.firstNnz = firstNnz;
    lm.localNnz = static_cast<index_t>(lm.cols.size());
}

// ****************************************************************************
// Function: generateLocalValues
//
// Purpose:
//   Reproduces the exact value vector and the locally owned values that the
//   serial fill() calls would produce. The dense input vector is replicated on
//   every rank because local rows may reference any column.
//
// ****************************************************************************
void generateLocalValues(LocalMatrix& lm, const index_t numRows,
                         const double maxVal, std::vector<double>& vec) {
    // The standard guarantees that rand() before any srand() behaves exactly
    // like srand(1), which matches the serial program's fill() sequence.
    srand(1);
    vec.resize(numRows);
    fill(vec.data(), numRows, maxVal);

    lm.val.resize(lm.localNnz);
    if (lm.localNnz == 0) {
        return;
    }

    // Advance the stream past the values owned by lower-numbered ranks
    for (index_t k = 0; k < lm.firstNnz; ++k) {
        (void)rand();
    }
    fill(lm.val.data(), lm.localNnz, maxVal);
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

    // Partition the rows contiguously among MPI ranks
    const index_t baseRows = numRows / static_cast<index_t>(numProcs);
    const index_t remainder = numRows % static_cast<index_t>(numProcs);
    const index_t rowStart = static_cast<index_t>(rank) * baseRows +
                             std::min(static_cast<index_t>(rank), remainder);
    const index_t localRows = baseRows +
                              (static_cast<index_t>(rank) < remainder ? 1 : 0);
    const index_t rowEnd = rowStart + localRows;

    if (rank == 0) {
        printf("Initializing data structures...\n");
    }

    // Build the locally owned part of the matrix, plus the replicated vector
    LocalMatrix localMatrix;
    std::vector<double> h_vec;  // Dense vector, replicated (needed for all columns)
    generateLocalMatrix(localMatrix, nItems, numRows, rowStart, rowEnd);
    generateLocalValues(localMatrix, numRows, maxVal, h_vec);

    std::vector<double> localOut(localRows);

    // For validation, compute reference solution on the same local rows
    std::vector<double> localReference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        localReference.resize(localRows);
        spmvCpu(localMatrix.val.data(), localMatrix.cols.data(),
                localMatrix.rowDelimiters.data(), h_vec.data(), localRows,
                localReference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localMatrix.val.data(), localMatrix.cols.data(),
                localMatrix.rowDelimiters.data(), h_vec.data(), localRows,
                localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(std::llround(duration * 1000.0));
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics using the global problem size
        const double seconds = (duration > 0.0) ? duration : 1e-9;
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = (seconds * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather the distributed vector)
    if (printResults) {
        std::vector<int> counts(numProcs);
        std::vector<int> displs(numProcs);
        for (int r = 0; r < numProcs; ++r) {
            const index_t rStart = static_cast<index_t>(r) * baseRows +
                                   std::min(static_cast<index_t>(r), remainder);
            const index_t rRows = baseRows +
                                  (static_cast<index_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(rRows);
            displs[r] = static_cast<int>(rStart);
        }

        std::vector<double> h_out;
        if (rank == 0) {
            h_out.resize(numRows);
        }

        MPI_Gatherv((localRows > 0) ? localOut.data() : nullptr,
                    static_cast<int>(localRows), MPI_DOUBLE,
                    (rank == 0) ? h_out.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        const int localValid = verifyResults(localReference.data(), localOut.data(),
                                             localRows)
                                   ? 1
                                   : 0;
        int globalValid = 0;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return globalValid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
