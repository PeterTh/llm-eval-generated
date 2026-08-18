#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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

__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           index_t rows, double* __restrict__ out) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    double sum = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j)
        sum += val[j] * vec[cols[j]];
    out[row] = sum;
}

inline void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, worldSize = 1;
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

    if (sparsity == 0 || iterations == 0 || numRows == 0) {
        if (rank == 0) printf("-n, -s, and -i must all be non-zero\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate number of non-zero elements
    const uint64_t matrixElements = static_cast<uint64_t>(numRows) * numRows;
    const index_t nItems = static_cast<index_t>(matrixElements / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / matrixElements));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n",
               worldSize, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / worldSize);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / worldSize);
    const index_t localRows = lastRow - firstRow;
    const index_t firstNnz = h_rowDelimiters[firstRow];
    const index_t lastNnz = h_rowDelimiters[lastRow];
    const index_t localNnz = lastNnz - firstNnz;
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowsPtr(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t r = 0; r <= localRows; ++r)
        localRowsPtr[r] = h_rowDelimiters[firstRow + r] - firstNnz;
#pragma omp parallel for schedule(static)
    for (index_t j = 0; j < localNnz; ++j) {
        localVal[j] = h_val[firstNnz + j];
        localCols[j] = h_cols[firstNnz + j];
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rows = nullptr;
    cudaCheck(cudaMalloc(&d_val, std::max<index_t>(1, localNnz) * sizeof(double)), "cudaMalloc(val)");
    cudaCheck(cudaMalloc(&d_cols, std::max<index_t>(1, localNnz) * sizeof(index_t)), "cudaMalloc(cols)");
    cudaCheck(cudaMalloc(&d_rows, std::max<index_t>(1, localRows + 1) * sizeof(index_t)), "cudaMalloc(rows)");
    cudaCheck(cudaMalloc(&d_vec, numRows * sizeof(double)), "cudaMalloc(vec)");
    cudaCheck(cudaMalloc(&d_out, std::max<index_t>(1, localRows) * sizeof(double)), "cudaMalloc(out)");
    cudaCheck(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice), "copy val");
    cudaCheck(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice), "copy cols");
    cudaCheck(cudaMemcpy(d_rows, localRowsPtr.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "copy rows");
    cudaCheck(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice), "copy vec");

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV on CUDA devices...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows != 0) {
            spmvKernel<<<(localRows + 255) / 256, 256>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out);
            cudaCheck(cudaGetLastError(), "spmvKernel");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(h_out.data() + firstRow, d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost), "copy output");
    std::vector<int> gatherCounts(worldSize), gatherDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        gatherCounts[r] = static_cast<int>((static_cast<uint64_t>(r + 1) * numRows) / worldSize -
                                           (static_cast<uint64_t>(r) * numRows) / worldSize);
        gatherDisplacements[r] = static_cast<int>((static_cast<uint64_t>(r) * numRows) / worldSize);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : h_out.data() + firstRow, localRows, MPI_DOUBLE,
                h_out.data(), gatherCounts.data(), gatherDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = elapsedSeconds * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsedSeconds / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);
    
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
        if (printResults) print_results(h_out, "OutputVector");

    // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vec); cudaFree(d_out);
            MPI_Comm_free(&localComm);
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vec); cudaFree(d_out);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
