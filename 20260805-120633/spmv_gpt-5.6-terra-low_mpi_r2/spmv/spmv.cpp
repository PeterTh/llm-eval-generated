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

    int parseStatus = 0;
    // Parse on rank zero, then distribute one consistent configuration.
    if (rank == 0) for (int i = 1; i < argc; ++i) {
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
            printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            parseStatus = 1;
            break;
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (sparsity == 0) {
        if (rank == 0) printf("Sparsity must be non-zero\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nItems > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix dimensions exceed MPI count limits\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    if (rank == 0) printf("Matrix size: %u x %u\n", numRows, numRows);
    if (rank == 0) printf("MPI ranks: %d\n", worldSize);
    if (rank == 0) printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    if (rank == 0) printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    if (rank == 0) printf("Iterations: %u\n", iterations);
    if (rank == 0) printf("Max value: %.2f\n", maxVal);
    if (rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val;                          // Non-zero values (rank 0)
    std::vector<index_t> h_cols;                        // Column indices (rank 0)
    std::vector<index_t> h_rowDelimiters;               // Row delimiters (rank 0)
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector (rank 0 when needed)

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Contiguous row ownership balances work while preserving CSR locality.
    const index_t rowBegin = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / worldSize);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / worldSize);
    const index_t localRows = rowEnd - rowBegin;
    index_t localNnz = 0;
    if (rank == 0) {
        localNnz = h_rowDelimiters[rowEnd] - h_rowDelimiters[rowBegin];
    }
    std::vector<index_t> localRowDelimiters(localRows + 1);
    std::vector<double> localVal;
    std::vector<index_t> localCols;
    if (rank == 0) {
        for (int p = 0; p < worldSize; ++p) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(numRows) * p) / worldSize);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(numRows) * (p + 1)) / worldSize);
            const index_t nnzBegin = h_rowDelimiters[begin];
            const index_t nnzCount = h_rowDelimiters[end] - nnzBegin;
            if (p == 0) {
                localNnz = nnzCount;
                for (index_t i = 0; i <= localRows; ++i) localRowDelimiters[i] = h_rowDelimiters[i] - nnzBegin;
                localVal.assign(h_val.begin() + nnzBegin, h_val.begin() + nnzBegin + nnzCount);
                localCols.assign(h_cols.begin() + nnzBegin, h_cols.begin() + nnzBegin + nnzCount);
            } else {
                MPI_Send(&nnzCount, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
                std::vector<index_t> delimiters(end - begin + 1);
                for (index_t i = 0; i <= end - begin; ++i) delimiters[i] = h_rowDelimiters[begin + i] - nnzBegin;
                MPI_Send(delimiters.data(), static_cast<int>(delimiters.size()), MPI_UINT32_T, p, 1, MPI_COMM_WORLD);
                MPI_Send(h_val.data() + nnzBegin, static_cast<int>(nnzCount), MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
                MPI_Send(h_cols.data() + nnzBegin, static_cast<int>(nnzCount), MPI_UINT32_T, p, 3, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(&localNnz, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localRowDelimiters.data(), static_cast<int>(localRowDelimiters.size()), MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localVal.resize(localNnz);
        localCols.resize(localNnz);
        MPI_Recv(localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }
    std::vector<double> localOut(localRows);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), h_vec.data(), localRows, localOut.data());
    }

    const double localDuration = MPI_Wtime() - start;
    double durationSeconds = 0.0;
    MPI_Reduce(&localDuration, &durationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        std::vector<int> counts(worldSize), displacements(worldSize);
        for (int p = 0; p < worldSize; ++p) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(numRows) * p) / worldSize);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(numRows) * (p + 1)) / worldSize);
            counts[p] = static_cast<int>(end - begin);
            displacements[p] = static_cast<int>(begin);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
    const double durationMs = durationSeconds * 1000.0;
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = durationSeconds > 0.0 ? (2.0 * nItems * iterations) / durationSeconds / 1e9 : 0.0;
    const double avgTime = iterations ? durationMs / static_cast<double>(iterations) : 0.0;
    
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
            MPI_Finalize();
            return 0;
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
