#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
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

// Partition contiguous CSR rows so that each rank owns roughly the same
// number of nonzeros.  Keeping rows intact makes the inner SpMV loop exactly
// the same as the serial implementation while avoiding communication during
// the timed region.
std::vector<index_t> makeRowPartition(const std::vector<index_t>& rowDelimiters,
                                      const index_t numRows, const int ranks) {
    std::vector<index_t> firstRow(static_cast<size_t>(ranks) + 1);
    firstRow[0] = 0;
    firstRow[ranks] = numRows;
    const index_t totalNnz = rowDelimiters[numRows];
    for (int rank = 1; rank < ranks; ++rank) {
        if (totalNnz == 0) {
            firstRow[rank] = static_cast<index_t>(
                (static_cast<uint64_t>(numRows) * rank) / ranks);
            continue;
        }
        const uint64_t target = (static_cast<uint64_t>(totalNnz) * rank) / ranks;
        index_t row = firstRow[rank - 1];
        while (row < numRows && rowDelimiters[row] < target) {
            ++row;
        }
        firstRow[rank] = row;
    }
    return firstRow;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;
    std::vector<index_t> firstRow;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        firstRow = makeRowPartition(h_rowDelimiters, numRows, ranks);
    }

    // All ranks need the dense vector, but each receives only its own CSR rows.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        firstRow.resize(static_cast<size_t>(ranks) + 1);
    }
    MPI_Bcast(firstRow.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t localFirstRow = firstRow[rank];
    const index_t localEndRow = firstRow[rank + 1];
    const index_t localRows = localEndRow - localFirstRow;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> outputCounts;
    if (rank == 0) {
        nnzCounts.resize(ranks);
        nnzDisplacements.resize(ranks);
        rowCounts.resize(ranks);
        rowDisplacements.resize(ranks);
        outputCounts.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = h_rowDelimiters[firstRow[r]];
            const index_t end = h_rowDelimiters[firstRow[r + 1]];
            if (end - begin > static_cast<index_t>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "Local MPI message exceeds MPI count limit\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            nnzCounts[r] = static_cast<int>(end - begin);
            nnzDisplacements[r] = static_cast<int>(begin);
            rowCounts[r] = static_cast<int>(firstRow[r + 1] - firstRow[r] + 1);
            rowDisplacements[r] = static_cast<int>(firstRow[r]);
            outputCounts[r] = static_cast<int>(firstRow[r + 1] - firstRow[r]);
        }
    }
    const int localNnz = rank == 0 ? nnzCounts[0] : 0;
    int receivedNnz = localNnz;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &receivedNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(receivedNnz);
    std::vector<index_t> localCols(receivedNnz);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_DOUBLE, localVal.data(), receivedNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_UINT32_T, localCols.data(), receivedNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr, rank == 0 ? rowDisplacements.data() : nullptr,
                 MPI_UINT32_T, localRowDelimiters.data(), static_cast<int>(localRows) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localNnzOffset = localRowDelimiters[0];
    for (index_t& offset : localRowDelimiters) {
        offset -= localNnzOffset;
    }

    // For validation, compute reference solution
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // The root no longer needs the global matrix after setup.  Releasing it
    // before the timed loop keeps its memory footprint close to every other
    // rank's local CSR footprint.
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    std::vector<double> localOut(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_out;
    if (validate || printResults) {
        if (rank == 0) {
            h_out.resize(numRows);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr,
                    rank == 0 ? outputCounts.data() : nullptr, rank == 0 ? rowDisplacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double durationMs = duration * 1000.0;
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = durationMs / static_cast<double>(iterations);
        printf("Computation time: %.3f ms\n", durationMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
