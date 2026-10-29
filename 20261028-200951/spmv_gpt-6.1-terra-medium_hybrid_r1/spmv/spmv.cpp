#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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


#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// A thread owns a row, preserving the original left-to-right summation order.
__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rows,
                           const double* vec, index_t n, double* out) {
    for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    uint64_t total = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || total / sparsity > INT_MAX || numRows > INT_MAX) {
        if (!rank) fprintf(stderr, "Invalid dimensions or matrix exceeds MPI count limits\n");
        MPI_Finalize(); return 1;
    }
    index_t nItems = static_cast<index_t>(total / sparsity);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> rowCounts(ranks), rowDispls(ranks), nnzCounts(ranks), nnzDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        rowDispls[r] = static_cast<int>((uint64_t(numRows) * r) / ranks);
        rowCounts[r] = static_cast<int>((uint64_t(numRows) * (r + 1)) / ranks) - rowDispls[r];
    }
    std::vector<double> vec(numRows), values;
    std::vector<index_t> cols, rows;
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / total));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        values.resize(nItems); cols.resize(nItems); rows.resize(size_t(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        for (int r = 0; r < ranks; ++r) {
            nnzDispls[r] = rows[rowDispls[r]];
            nnzCounts[r] = rows[rowDispls[r] + rowCounts[r]] - nnzDispls[r];
        }
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localNnz, nnzStart;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzDispls.data(), 1, MPI_INT, &nnzStart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int localRows = rowCounts[rank];
    std::vector<index_t> localRow(size_t(localRows) + 1), localCol(localNnz);
    std::vector<double> localVal(localNnz), localOut(localRows, 0.0);
    MPI_Scatterv(rows.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 localRow.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCol.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(values.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localRows; ++i) localRow[i] -= nnzStart;
    localRow[localRows] = localNnz;
    std::vector<double> reference;
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(localVal.data(), localCol.data(), localRow.data(), vec.data(), localRows, reference.data());
    }
    double *dVal, *dVec, *dOut;
    index_t *dCol, *dRow;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCol, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRow, (size_t(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, size_t(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)));
    if (localNnz) {
        CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCol, localCol.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRow, localRow.data(), (size_t(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows && iterations) {
        int blocks = std::min((localRows + 255LL) / 256, 65535LL);
        spmvKernel<<<blocks, 256>>>(dVal, dCol, dRow, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (localRows && iterations) {
        int blocks = std::min((localRows + 255LL) / 256, 65535LL);
        for (index_t i = 0; i < iterations; ++i)
            spmvKernel<<<blocks, 256>>>(dVal, dCol, dRow, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows && iterations)
        CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        std::vector<double> out(rank == 0 ? numRows : 0);
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(out, "OutputVector");
    }
    int valid = !validate || verifyResults(reference.data(), localOut.data(), localRows);
    int allValid;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate && !rank) printf("Validating result...\nValidation: %s\n", allValid ? "PASSED" : "FAILED");
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCol)); CUDA_CHECK(cudaFree(dRow));
    CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return allValid ? 0 : 1;
}
