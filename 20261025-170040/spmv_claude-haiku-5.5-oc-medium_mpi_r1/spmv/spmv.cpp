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
// Function: rowBegin
//
// Purpose:
//   First row owned by a rank in a contiguous block partition of dim rows
//   across nRanks ranks. rowBegin(nRanks) == dim.
//
// ****************************************************************************
index_t rowBegin(int rank, int nRanks, index_t dim) {
    return static_cast<index_t>((static_cast<uint64_t>(dim) * static_cast<uint64_t>(rank)) /
                                static_cast<uint64_t>(nRanks));
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
// Function: initLocalMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix, exactly as the serial reference does (same RNG stream, same
//   decisions), but only retains the rows in [rowLo, rowHi) owned by this
//   rank. The local part is encoded in CSR format with row delimiters
//   relative to the first local nonzero.
//
// Arguments:
//   n:             number of nonzero elements (global)
//   dim:           number of rows/columns in the matrix (global)
//   rowLo, rowHi:  half-open range of rows owned by this rank
//   localCols:     output, column indexes of local nonzeros
//   localDelim:    output, size (rowHi-rowLo+1) row delimiters (local offsets)
//   globalBase:    output, global index of the first local nonzero
//
// ****************************************************************************
void initLocalMatrix(const index_t n, const index_t dim, const index_t rowLo, const index_t rowHi,
                     std::vector<index_t>& localCols, std::vector<index_t>& localDelim,
                     index_t& globalBase) {
    index_t nnzAssigned = 0;
    index_t base = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    localDelim.assign(rowHi - rowLo + 1, 0);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i <= dim; ++i) {
        if (i == rowLo) {
            base = nnzAssigned;
        }
        if (i >= rowLo && i <= rowHi) {
            localDelim[i - rowLo] = nnzAssigned - base;
        }
        if (i == dim) {
            break;
        }
        const bool localRow = (i >= rowLo && i < rowHi);
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (localRow) {
                    localCols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    globalBase = base;
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
//   dim: number of rows in the (local) matrix
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
//   offset: global row index of element 0 (for diagnostics)
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
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset, ref, res);
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
    int nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
        printf("MPI ranks: %d\n", nRanks);
        printf("Initializing data structures...\n");
    }

    // Row-block partition of the matrix and of the output vector
    const index_t rowLo = rowBegin(rank, nRanks, numRows);
    const index_t rowHi = rowBegin(rank + 1, nRanks, numRows);
    const index_t localRows = rowHi - rowLo;

    // Local matrix: only rows [rowLo, rowHi) are stored on this rank
    std::vector<index_t> localCols;
    std::vector<index_t> localDelim;
    index_t globalBase = 0;
    initLocalMatrix(nItems, numRows, rowLo, rowHi, localCols, localDelim, globalBase);
    const index_t localNnz = localDelim[localRows];

    // Dense vector (replicated on every rank) and local non-zero values.
    // The serial code draws the vector and then the values from the default
    // RNG stream, which is the same as srand(1) (the C-standard initial seed),
    // before reseeding for the matrix pattern. Replay that stream here and keep
    // only the values this rank needs.
    std::vector<double> h_vec(numRows);
    std::vector<double> localVal(localNnz);
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    for (index_t k = 0; k < nItems; ++k) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (k >= globalBase && k - globalBase < localNnz) {
            localVal[k - globalBase] = v;
        }
    }

    // For validation, compute the reference solution for the local rows
    std::vector<double> localRef;
    if (validate) {
        if (root) {
            printf("Computing reference solution...\n");
        }
        localRef.resize(localRows);
        spmvCpu(localVal.data(), localCols.data(), localDelim.data(),
                h_vec.data(), localRows, localRef.data());
    }

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
    }
    std::vector<double> localOut(localRows);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localDelim.data(),
                h_vec.data(), localRows, localOut.data());
    }
    const double localElapsed = MPI_Wtime() - t0;

    // The benchmark time is the slowest rank
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
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
        std::vector<int> counts(nRanks);
        std::vector<int> displs(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            const index_t lo = rowBegin(r, nRanks, numRows);
            const index_t hi = rowBegin(r + 1, nRanks, numRows);
            counts[r] = static_cast<int>(hi - lo);
            displs[r] = static_cast<int>(lo);
        }
        std::vector<double> h_out(root ? numRows : 0);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        const int localOk = verifyResults(localRef.data(), localOut.data(), localRows, rowLo) ? 1 : 0;
        int globalOk = 0;
        MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (root) {
            if (globalOk) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Finalize();
        return globalOk ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
