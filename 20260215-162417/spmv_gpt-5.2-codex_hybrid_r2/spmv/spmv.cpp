#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

#ifndef MPI_UINT32_T
#define MPI_UINT32_T MPI_UNSIGNED
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t cudaStatus = (call);                               \
        if (cudaStatus != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(cudaStatus));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

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

static inline void checkMpiCount(const size_t value, const char* name, const int rank) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "MPI count overflow for %s: %zu exceeds INT_MAX\n", name, value);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline void computeRowRange(const index_t totalRows, const int worldSize, const int rank,
                                   index_t* startRow, index_t* endRow) {
    const index_t base = totalRows / static_cast<index_t>(worldSize);
    const int remainder = static_cast<int>(totalRows % static_cast<index_t>(worldSize));
    const index_t extra = static_cast<index_t>(rank < remainder ? 1 : 0);
    const index_t offset = static_cast<index_t>(std::min(rank, remainder));
    *startRow = base * static_cast<index_t>(rank) + offset;
    *endRow = *startRow + base + extra;
}

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, const index_t rows, double* out) {
    index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const index_t stride = static_cast<index_t>(gridDim.x) * blockDim.x;
    for (; row < rows; row += stride) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
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
    int parseStatus = 0;
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
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
        printf("MPI ranks: %d\n", worldSize);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;

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

    checkMpiCount(static_cast<size_t>(numRows), "vector size", rank);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    std::vector<int> outCounts;
    std::vector<int> outDispls;

    if (rank == 0) {
        nnzCounts.resize(worldSize);
        nnzDispls.resize(worldSize);
        rowCounts.resize(worldSize);
        rowDispls.resize(worldSize);
        outCounts.resize(worldSize);
        outDispls.resize(worldSize);

        for (int r = 0; r < worldSize; ++r) {
            index_t startRow = 0;
            index_t endRow = 0;
            computeRowRange(numRows, worldSize, r, &startRow, &endRow);
            const index_t localRows = endRow - startRow;
            const index_t startIdx = h_rowDelimiters[startRow];
            const index_t endIdx = h_rowDelimiters[endRow];
            const index_t localNnz = endIdx - startIdx;

            checkMpiCount(static_cast<size_t>(localNnz), "nnz count", rank);
            checkMpiCount(static_cast<size_t>(startIdx), "nnz displacement", rank);
            checkMpiCount(static_cast<size_t>(localRows + 1), "row delimiter count", rank);
            checkMpiCount(static_cast<size_t>(startRow), "row delimiter displacement", rank);

            nnzCounts[r] = static_cast<int>(localNnz);
            nnzDispls[r] = static_cast<int>(startIdx);
            rowCounts[r] = static_cast<int>(localRows + 1);
            rowDispls[r] = static_cast<int>(startRow);
            outCounts[r] = static_cast<int>(localRows);
            outDispls[r] = static_cast<int>(startRow);
        }
    }

    index_t localRowStart = 0;
    index_t localRowEnd = 0;
    computeRowRange(numRows, worldSize, rank, &localRowStart, &localRowEnd);
    const index_t localRows = localRowEnd - localRowStart;

    int localNnzCount = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT, &localNnzCount, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    const index_t localNnz = static_cast<index_t>(localNnzCount);
    std::vector<double> h_localVal(localNnz);
    std::vector<index_t> h_localCols(localNnz);
    std::vector<index_t> h_localRowDelims(localRows + 1);
    std::vector<double> h_localOut(localRows);

    const int* nnzCountsPtr = rank == 0 ? nnzCounts.data() : nullptr;
    const int* nnzDisplsPtr = rank == 0 ? nnzDispls.data() : nullptr;
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCountsPtr, nnzDisplsPtr, MPI_DOUBLE,
                 h_localVal.data(), localNnzCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCountsPtr, nnzDisplsPtr, MPI_UINT32_T,
                 h_localCols.data(), localNnzCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const int* rowCountsPtr = rank == 0 ? rowCounts.data() : nullptr;
    const int* rowDisplsPtr = rank == 0 ? rowDispls.data() : nullptr;
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, rowCountsPtr, rowDisplsPtr,
                 MPI_UINT32_T, h_localRowDelims.data(), static_cast<int>(localRows + 1),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t base = h_localRowDelims[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localRows + 1; ++i) {
        h_localRowDelims[i] -= base;
    }

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelims = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    const size_t valBytes = static_cast<size_t>(localNnz) * sizeof(double);
    const size_t colBytes = static_cast<size_t>(localNnz) * sizeof(index_t);
    const size_t rowBytes = static_cast<size_t>(localRows + 1) * sizeof(index_t);
    const size_t vecBytes = static_cast<size_t>(numRows) * sizeof(double);
    const size_t outBytes = static_cast<size_t>(localRows) * sizeof(double);

    if (valBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, valBytes));
        CUDA_CHECK(cudaMemcpy(d_val, h_localVal.data(), valBytes, cudaMemcpyHostToDevice));
    }
    if (colBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_cols, colBytes));
        CUDA_CHECK(cudaMemcpy(d_cols, h_localCols.data(), colBytes, cudaMemcpyHostToDevice));
    }
    if (rowBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_rowDelims, rowBytes));
        CUDA_CHECK(cudaMemcpy(d_rowDelims, h_localRowDelims.data(), rowBytes, cudaMemcpyHostToDevice));
    }
    if (vecBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_vec, vecBytes));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), vecBytes, cudaMemcpyHostToDevice));
    }
    if (outBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, outBytes));
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    int blocks = 0;
    if (localRows > 0) {
        blocks = static_cast<int>((localRows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        blocks = std::min(blocks, 65535);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvKernel<<<blocks, CUDA_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelims, d_vec, localRows, d_out);
        }
    }
    if (localRows > 0) {
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localElapsed = MPI_Wtime() - startTime;

    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (outBytes > 0) {
        CUDA_CHECK(cudaMemcpy(h_localOut.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    }

    const int* outCountsPtr = rank == 0 ? outCounts.data() : nullptr;
    const int* outDisplsPtr = rank == 0 ? outDispls.data() : nullptr;
    MPI_Gatherv(h_localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, outCountsPtr, outDisplsPtr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double avgTime = iterations > 0 ? (maxElapsed * 1000.0) / static_cast<double>(iterations) : 0.0;
        const double gflops = maxElapsed > 0.0
                                  ? (2.0 * static_cast<double>(nItems) *
                                     static_cast<double>(iterations)) /
                                        maxElapsed / 1e9
                                  : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(h_out, "OutputVector");
    }
 
    // Validation
    int exitCode = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
 
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d_val) {
        CUDA_CHECK(cudaFree(d_val));
    }
    if (d_cols) {
        CUDA_CHECK(cudaFree(d_cols));
    }
    if (d_rowDelims) {
        CUDA_CHECK(cudaFree(d_rowDelims));
    }
    if (d_vec) {
        CUDA_CHECK(cudaFree(d_vec));
    }
    if (d_out) {
        CUDA_CHECK(cudaFree(d_out));
    }

    MPI_Finalize();
    return exitCode;
}
