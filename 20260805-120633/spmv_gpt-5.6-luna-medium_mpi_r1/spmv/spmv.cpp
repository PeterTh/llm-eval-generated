#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    if (sparsity == 0) {
        if (rank == 0) printf("Sparsity must be non-zero\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const uint64_t nItems64 = (static_cast<uint64_t>(numRows) * numRows) / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max() || nItems64 > std::numeric_limits<int>::max()) {
        if (rank == 0) printf("Matrix is too large for the benchmark's 32-bit CSR format\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    // Keep MPI counts representable by the MPI C++ bindings used here.
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix dimension is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    // Divide contiguous rows as evenly as possible.  Contiguous ownership
    // preserves CSR locality and requires no communication during SpMV.
    const index_t baseRows = numRows / static_cast<index_t>(worldSize);
    const index_t extraRows = numRows % static_cast<index_t>(worldSize);
    const index_t localRows = baseRows + (static_cast<index_t>(rank) < extraRows ? 1 : 0);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (MPI SpMV Benchmark, %d ranks)\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) /
                        (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank zero creates the global input.  Each rank retains its rows.
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(rank == 0 ? numRows + 1 : 0);
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(localRows);               // Local output vector

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    std::vector<int> rowCounts(worldSize), rowDisplacements(worldSize);
    std::vector<int> delimCounts(worldSize), delimDisplacements(worldSize);
    std::vector<int> nnzCounts(worldSize), nnzDisplacements(worldSize);
    int nnzOffset = 0;
    for (int r = 0; r < worldSize; ++r) {
        const index_t rows = baseRows + (static_cast<index_t>(r) < extraRows ? 1 : 0);
        const index_t begin = static_cast<index_t>(r) * baseRows +
                              std::min<index_t>(static_cast<index_t>(r), extraRows);
        const index_t end = begin + rows;
        rowCounts[r] = static_cast<int>(rows);
        rowDisplacements[r] = static_cast<int>(begin);
        delimCounts[r] = static_cast<int>(rows + 1);
        delimDisplacements[r] = static_cast<int>(begin);
        if (rank == 0) {
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[end] - h_rowDelimiters[begin]);
            nnzDisplacements[r] = nnzOffset;
            nnzOffset += nnzCounts[r];
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<index_t> localCols(nnzCounts[rank]);
    std::vector<double> localVal(nnzCounts[rank]);
    std::vector<index_t> localRowsDelimiters(localRows + 1);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_UINT32_T, localCols.data(), nnzCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_DOUBLE, localVal.data(), nnzCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, delimCounts.data(), delimDisplacements.data(),
                 MPI_UINT32_T, localRowsDelimiters.data(), delimCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t globalFirstNnz = localRowsDelimiters[0];
    for (index_t i = 0; i <= localRows; ++i) {
        localRowsDelimiters[i] -= globalFirstNnz;
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowsDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
    
    // Calculate performance metrics
    const double gflops = elapsedSeconds > 0.0 ? (2.0 * nItems * iterations) / elapsedSeconds / 1e9 : 0.0;
    const double avgTime = iterations != 0 ? elapsedSeconds * 1000.0 / iterations : 0.0;
    
    if (rank == 0) {
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOut(rank == 0 ? numRows : 0);
    MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? globalOut.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalOut, "OutputVector");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = true;
        if (rank == 0) valid = verifyResults(h_reference.data(), globalOut.data(), numRows);
        int validInt = valid ? 1 : 0;
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (validInt) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
