#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>

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

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// One warp cooperates on each row, coalescing accesses to the CSR arrays.
__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* rows, const double* vec,
                           index_t nrows, double* out) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= nrows) return;
    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) out[row] = sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Comm_free(&localComm); MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const uint64_t totalEntries = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || totalEntries > UINT32_MAX || totalEntries / sparsity > INT_MAX || numRows > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid dimensions or matrix too large for CSR/MPI counts\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const index_t nItems = static_cast<index_t>(totalEntries / sparsity);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA devices available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    std::vector<int> rowCounts(size), rowOffsets(size), nnzCounts(size), nnzOffsets(size);
    for (int p = 0; p < size; ++p) {
        rowOffsets[p] = static_cast<int>(uint64_t(numRows) * p / size);
        rowCounts[p] = static_cast<int>(uint64_t(numRows) * (p + 1) / size) - rowOffsets[p];
    }
    const int localRows = rowCounts[rank];
    std::vector<double> vec(numRows), vals, output;
    std::vector<index_t> cols, rows;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / totalEntries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        vals.resize(nItems); cols.resize(nItems); rows.resize(size_t(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(vals.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        for (int p = 0; p < size; ++p) {
            nnzOffsets[p] = rows[rowOffsets[p]];
            nnzCounts[p] = rows[rowOffsets[p] + rowCounts[p]] - nnzOffsets[p];
        }
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localNnz = 0, base = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzOffsets.data(), 1, MPI_INT, &base, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> localVals(localNnz), localOut(localRows, 0.0);
    std::vector<index_t> localCols(localNnz), localRow(size_t(localRows) + 1);
    MPI_Scatterv(vals.data(), nnzCounts.data(), nnzOffsets.data(), MPI_DOUBLE, localVals.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(cols.data(), nnzCounts.data(), nnzOffsets.data(), MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // Scatter row starts; supply the final delimiter separately.
    MPI_Scatterv(rows.data(), rowCounts.data(), rowOffsets.data(), MPI_UINT32_T, localRow.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (int i = 0; i < localRows; ++i) localRow[i] -= base;
    localRow[localRows] = localNnz;
    if (rank == 0) { std::vector<double>().swap(vals); std::vector<index_t>().swap(cols); std::vector<index_t>().swap(rows); }

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(localVals.data(), localCols.data(), localRow.data(), vec.data(), localRows, reference.data());
    }
    double *dVals, *dVec, *dOut;
    index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVals, std::max(size_t(1), localVals.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max(size_t(1), localCols.size()) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, localRow.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, vec.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max(size_t(1), localOut.size()) * sizeof(double)));
    if (localNnz) {
        CUDA_CHECK(cudaMemcpy(dVals, localVals.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRows, localRow.data(), localRow.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, std::max(size_t(1), localOut.size()) * sizeof(double)));
    if (iterations && localRows) {
        spmvKernel<<<(localRows + 7) / 8, 256>>>(dVals, dCols, dRows, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows) spmvKernel<<<(localRows + 7) / 8, 256>>>(dVals, dCols, dRows, dVec, localRows, dOut);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
    }
    if (printResults) {
        if (rank == 0) output.resize(numRows);
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, output.data(), rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(output, "OutputVector");
    }
    int valid = 1, allValid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = verifyResults(reference.data(), localOut.data(), localRows) ? 1 : 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(dVals)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return allValid ? 0 : 1;
}
