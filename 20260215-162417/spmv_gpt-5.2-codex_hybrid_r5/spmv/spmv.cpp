#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
#pragma omp parallel for schedule(static)
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

struct RowPartition {
    index_t start;
    index_t count;
};

RowPartition getRowPartition(const index_t totalRows, const int rank, const int size) {
    const index_t base = totalRows / static_cast<index_t>(size);
    const index_t rem = totalRows % static_cast<index_t>(size);
    const int remInt = static_cast<int>(rem);
    const index_t start = static_cast<index_t>(rank) * base + static_cast<index_t>(std::min(rank, remInt));
    const index_t count = base + (rank < remInt ? 1 : 0);
    return {start, count};
}

inline void mpiCheck(const int result, const char* call, const char* file, const int line) {
    if (result != MPI_SUCCESS) {
        char errStr[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(result, errStr, &len);
        fprintf(stderr, "MPI error at %s:%d code=%d (%s) in call %s\n",
                file, line, result, errStr, call);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define MPI_CHECK(call) mpiCheck((call), #call, __FILE__, __LINE__)

inline void cudaCheck(const cudaError_t result, const char* call, const char* file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) in call %s\n",
                file, line, static_cast<int>(result), cudaGetErrorString(result), call);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, const index_t numRows, double* out) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < numRows) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int earlyExit = 0;
    int exitCode = 0;

    // Parse command line arguments
    if (rank == 0) {
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
                printUsage(argv[0]);
                earlyExit = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExit = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (earlyExit) {
        MPI_CHECK(MPI_Finalize());
        return exitCode;
    }

    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&numRows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&sparsity, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&iterations, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD));
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

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
    std::vector<double> h_val;                  // Non-zero values
    std::vector<index_t> h_cols;                // Column indices
    std::vector<index_t> h_rowDelimiters;       // Row delimiters
    std::vector<double> h_vec(numRows);         // Dense vector
    std::vector<double> h_out;                  // Output vector

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_CHECK(MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    const RowPartition part = getRowPartition(numRows, rank, size);
    const index_t localRows = part.count;

    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    std::vector<int> rowDelimCounts;
    std::vector<int> rowDelimDispls;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;

    if (rank == 0) {
        rowCounts.resize(size);
        rowDispls.resize(size);
        rowDelimCounts.resize(size);
        rowDelimDispls.resize(size);
        nnzCounts.resize(size);
        nnzDispls.resize(size);

        for (int r = 0; r < size; ++r) {
            const RowPartition rp = getRowPartition(numRows, r, size);
            rowCounts[r] = static_cast<int>(rp.count);
            rowDispls[r] = static_cast<int>(rp.start);
            rowDelimCounts[r] = static_cast<int>(rp.count + 1);
            rowDelimDispls[r] = static_cast<int>(rp.start);
            const index_t nnzStart = h_rowDelimiters[rp.start];
            const index_t nnzEnd = h_rowDelimiters[rp.start + rp.count];
            nnzCounts[r] = static_cast<int>(nnzEnd - nnzStart);
            nnzDispls[r] = static_cast<int>(nnzStart);
        }
    }

    int localNnz = 0;
    MPI_CHECK(MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                          &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<index_t> localRowDelimiters(localRows + 1);
    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));

    MPI_CHECK(MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                           rank == 0 ? rowDelimCounts.data() : nullptr,
                           rank == 0 ? rowDelimDispls.data() : nullptr,
                           MPI_UNSIGNED,
                           localRowDelimiters.data(),
                           static_cast<int>(localRows + 1),
                           MPI_UNSIGNED, 0, MPI_COMM_WORLD));

    MPI_CHECK(MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                           rank == 0 ? nnzCounts.data() : nullptr,
                           rank == 0 ? nnzDispls.data() : nullptr,
                           MPI_DOUBLE,
                           localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                           rank == 0 ? nnzCounts.data() : nullptr,
                           rank == 0 ? nnzDispls.data() : nullptr,
                           MPI_UNSIGNED,
                           localCols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD));

    const index_t baseNnz = localRowDelimiters[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localRows + 1; ++i) {
        localRowDelimiters[i] -= baseNnz;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, static_cast<size_t>(localNnz) * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, static_cast<size_t>(localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(localRows) * sizeof(double)));
    }

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(),
                              static_cast<size_t>(localNnz) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(),
                              static_cast<size_t>(localNnz) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                          static_cast<size_t>(localRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(),
                          static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    const int blockSize = 256;
    const int gridSize = localRows > 0 ? static_cast<int>((localRows + blockSize - 1) / blockSize) : 0;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelimiters,
                                                d_vec, localRows, d_out);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double maxMs = 0.0;
    MPI_CHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out,
                              static_cast<size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    MPI_CHECK(MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                          rank == 0 ? h_out.data() : nullptr,
                          rank == 0 ? rowCounts.data() : nullptr,
                          rank == 0 ? rowDispls.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (d_out) {
        CUDA_CHECK(cudaFree(d_out));
    }
    if (d_vec) {
        CUDA_CHECK(cudaFree(d_vec));
    }
    if (d_rowDelimiters) {
        CUDA_CHECK(cudaFree(d_rowDelimiters));
    }
    if (d_cols) {
        CUDA_CHECK(cudaFree(d_cols));
    }
    if (d_val) {
        CUDA_CHECK(cudaFree(d_val));
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(maxMs));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int finalStatus = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
            finalStatus = 0;
        } else {
            printf("Validation: FAILED\n");
            finalStatus = 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&finalStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return finalStatus;
}
