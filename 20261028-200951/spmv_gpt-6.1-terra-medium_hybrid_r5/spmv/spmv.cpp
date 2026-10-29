#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

// One warp cooperates on a row, keeping accesses to CSR data coalesced.
__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* rows, const double* vec,
                           index_t nrows, double* out) {
    unsigned lane = threadIdx.x & 31;
    index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= nrows) return;
    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) out[row] = sum;
}

#define CUDA_CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(error)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
        else if (strcmp(argv[i], "-h") == 0) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    uint64_t cells = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || cells / sparsity > uint64_t(std::numeric_limits<int>::max()) || numRows > uint32_t(std::numeric_limits<int>::max() - 1)) {
        if (!rank) fprintf(stderr, "Invalid dimensions or matrix exceeds MPI count limits.\n");
        MPI_Finalize(); return 1;
    }
    index_t nItems = static_cast<index_t>(cells / sparsity);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\n", sparsity, nItems, 100.0 * (1.0 - double(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }
    std::vector<double> vec(numRows), val, out, reference;
    std::vector<index_t> cols, rows;
    if (!rank) {
        val.resize(nItems); cols.resize(nItems); rows.resize(size_t(numRows) + 1); out.resize(numRows);
        fill(vec.data(), numRows, maxVal); fill(val.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        if (validate) { printf("Computing reference solution...\n"); reference.resize(numRows); spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data()); }
    }
    MPI_Bcast(vec.data(), int(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<int> rowCounts(size), rowDispls(size), nnzCounts(size), nnzDispls(size);
    for (int p = 0; p < size; ++p) {
        rowDispls[p] = int(uint64_t(numRows) * p / size);
        rowCounts[p] = int(uint64_t(numRows) * (p + 1) / size) - rowDispls[p];
        if (!rank) { nnzDispls[p] = int(rows[rowDispls[p]]); nnzCounts[p] = int(rows[rowDispls[p] + rowCounts[p]] - rows[rowDispls[p]]); }
    }
    MPI_Bcast(nnzCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    int localRows = rowCounts[rank], localNnz = nnzCounts[rank];
    std::vector<index_t> localRow(size_t(localRows) + 1), localCol(localNnz);
    std::vector<double> localVal(localNnz), localOut(localRows, 0.0);
    // Scatter row starts, then append the end delimiter locally.
    MPI_Scatterv(rows.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T, localRow.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localRows; ++i) localRow[i] -= nnzDispls[rank];
    localRow[localRows] = localNnz;
    MPI_Scatterv(cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T, localCol.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    double *dVal, *dVec, *dOut;
    index_t *dCol, *dRow;
    CUDA_CHECK(cudaMalloc(&dVal, std::max(size_t(1), size_t(localNnz)) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCol, std::max(size_t(1), size_t(localNnz)) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRow, (size_t(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, size_t(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max(size_t(1), size_t(localRows)) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), size_t(localNnz) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCol, localCol.data(), size_t(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRow, localRow.data(), (size_t(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, size_t(localRows) * sizeof(double)));
    if (localRows && iterations) {
        spmvKernel<<<(localRows + 7) / 8, 256>>>(dVal, dCol, dRow, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter)
        if (localRows) spmvKernel<<<(localRows + 7) / 8, 256>>>(dVal, dCol, dRow, dVec, localRows, dOut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int status = 0;
    if (!rank) {
        printf("Computation time: %ld ms\n", long(maxElapsed * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
        if (printResults) print_results(out, "OutputVector");
        if (validate) { printf("Validating result...\n"); bool valid = verifyResults(reference.data(), out.data(), numRows); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); status = valid ? 0 : 1; }
    }
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCol)); CUDA_CHECK(cudaFree(dRow)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Comm_free(&localComm);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
