#include <mpi.h>

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
// Function: fillLocal
//
// Purpose:
//   Initialize the local block of an array with the same random values that
//   fill() would produce, continuing the current random stream. Values that
//   belong to other ranks are drawn and discarded.
//
// Arguments:
//   A: pointer to the local block (size should be = count)
//   begin: global index of the first local element
//   count: number of local elements
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fillLocal(double* A, const index_t begin, const index_t count, const double maxVal) {
    for (index_t i = 0; i < begin; ++i) {
        rand();
    }
    for (index_t i = 0; i < count; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. Every rank walks the same global random sequence, so
//   the matrix is identical regardless of rank count, but only the rows
//   [rowBegin, rowEnd) are stored locally.
//
// Arguments:
//   cols:          column indexes of the local elements
//   rowDelimiters: array of size (rowEnd - rowBegin) + 1 holding indices to
//                  local rows; last element is the index one past the last
//                  local element
//   nnzBegin:      global index of the first local element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row owned by this rank
//   rowEnd:        one past the last row owned by this rank
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters, index_t& nnzBegin,
                      const index_t n, const index_t dim, const index_t rowBegin, const index_t rowEnd) {
    index_t nnzAssigned = 0;
    cols.clear();
    rowDelimiters.assign(rowEnd - rowBegin + 1, 0);
    nnzBegin = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        const bool ownsRow = i >= rowBegin && i < rowEnd;
        if (i == rowBegin) {
            nnzBegin = nnzAssigned;
        }
        if (ownsRow) {
            rowDelimiters[i - rowBegin] = static_cast<index_t>(cols.size());
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
                if (ownsRow) {
                    cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of local non-zeroes at the end of the row delimiters array
    rowDelimiters[rowEnd - rowBegin] = static_cast<index_t>(cols.size());
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rows are split into contiguous blocks; rank r owns [rowBound(r), rowBound(r + 1))
    auto rowBound = [numRows, numRanks](int r) {
        return static_cast<index_t>(static_cast<uint64_t>(numRows) * r / numRanks);
    };
    const index_t rowBegin = rowBound(rank);
    const index_t rowEnd = rowBound(rank + 1);
    const index_t localRows = rowEnd - rowBegin;

    // Allocate and initialize data structures
    if (root) {
        printf("Initializing data structures...\n");
    }
    std::vector<index_t> h_cols;                        // Column indices of local elements
    std::vector<index_t> h_rowDelimiters;               // Row delimiters of local rows
    index_t nnzBegin = 0;                               // Global index of first local element
    initRandomMatrix(h_cols, h_rowDelimiters, nnzBegin, nItems, numRows, rowBegin, rowEnd);
    const index_t localNnz = static_cast<index_t>(h_cols.size());

    // Same random stream as the original unseeded program (the C standard defines it as seed 1)
    srand(1);
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated on every rank)
    fill(h_vec.data(), numRows, maxVal);
    std::vector<double> h_val(localNnz);                // Non-zero values of local elements
    fillLocal(h_val.data(), nnzBegin, localNnz, maxVal);

    std::vector<double> h_out(localRows);               // Local part of the output vector

    // For validation, compute reference solution for the local rows
    std::vector<double> h_reference;
    if (validate) {
        if (root) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation; each rank works on its rows without communication
    if (root) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(elapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gathers the row blocks of a distributed vector onto the root (outside the timed region)
    auto gatherRows = [numRows, numRanks, root, &rowBound](const std::vector<double>& local) {
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            displs[r] = static_cast<int>(rowBound(r));
            counts[r] = static_cast<int>(rowBound(r + 1) - rowBound(r));
        }
        std::vector<double> full(root ? numRows : 0);
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return full;
    };

    // Print results for external validation
    std::vector<double> h_fullOut;
    if (printResults || validate) {
        h_fullOut = gatherRows(h_out);
    }
    if (printResults && root) {
        print_results(h_fullOut, "OutputVector");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        const std::vector<double> h_fullReference = gatherRows(h_reference);
        if (root) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_fullReference.data(), h_fullOut.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
