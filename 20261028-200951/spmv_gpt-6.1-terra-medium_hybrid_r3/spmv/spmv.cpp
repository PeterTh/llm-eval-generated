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


#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// One warp cooperates on each row; adjacent lanes access adjacent CSR entries.
__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* offsets,
                           const double* vec, index_t rows, double* out) {
    unsigned long long thread = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned lane = thread & 31;
    unsigned long long row = thread >> 5;
    if (row >= rows) return;
    double sum = 0.0;
    for (unsigned long long j = static_cast<unsigned long long>(offsets[row]) + lane;
         j < offsets[row + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) out[row] = sum;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI thread support is insufficient\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
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
    uint64_t square = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || square > UINT32_MAX || numRows > INT_MAX) {
        if (!rank) fprintf(stderr, "Invalid matrix size or sparsity (CSR indices must fit in 32 bits).\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(square / sparsity);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\n", sparsity, nItems, 100.0 * (1.0 - double(nItems) / square));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\nInitializing data structures...\n", iterations, maxVal, validate ? "enabled" : "disabled");
    }
    std::vector<double> vec(numRows), vals, output, reference;
    std::vector<index_t> cols, offsets;
    if (!rank) {
        vals.resize(nItems); cols.resize(nItems); offsets.resize(size_t(numRows) + 1);
        fill(vec.data(), numRows, maxVal); fill(vals.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), offsets.data(), nItems, numRows);
        if (validate) { printf("Computing reference solution...\n"); reference.resize(numRows); spmvCpu(vals.data(), cols.data(), offsets.data(), vec.data(), numRows, reference.data()); }
        output.resize(numRows);
    }
    MPI_Bcast(vec.data(), int(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<int> rowCounts(ranks), rowDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        rowDispls[r] = int(uint64_t(numRows) * r / ranks);
        rowCounts[r] = int(uint64_t(numRows) * (r + 1) / ranks) - rowDispls[r];
    }
    int localRows = rowCounts[rank];
    std::vector<index_t> localOffsets(size_t(localRows) + 1);
    // The extra delimiter is sent separately because neighboring partitions overlap.
    MPI_Scatterv(offsets.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 localOffsets.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    index_t endOffset = 0;
    if (!rank) {
        for (int r = 1; r < ranks; ++r) {
            index_t end = offsets[rowDispls[r] + rowCounts[r]];
            MPI_Send(&end, 1, MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
        }
        endOffset = offsets[rowCounts[0]];
    } else MPI_Recv(&endOffset, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    index_t beginOffset = localRows ? localOffsets[0] : endOffset;
    index_t localNnz = endOffset - beginOffset;
    localOffsets[localRows] = endOffset;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= localRows; ++i) localOffsets[i] -= beginOffset;
    std::vector<double> localVals(localNnz);
    std::vector<index_t> localCols(localNnz);
    // Chunk transfers so that MPI's int count does not limit the CSR size.
    const uint64_t chunk = INT_MAX;
    if (!rank) {
        for (int r = 0; r < ranks; ++r) {
            uint64_t begin = offsets[rowDispls[r]], end = offsets[rowDispls[r] + rowCounts[r]];
            if (!r) {
                std::copy(vals.begin() + begin, vals.begin() + end, localVals.begin());
                std::copy(cols.begin() + begin, cols.begin() + end, localCols.begin());
            } else for (uint64_t pos = begin; pos < end; pos += chunk) {
                int count = int(std::min(chunk, end - pos));
                MPI_Send(vals.data() + pos, count, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
                MPI_Send(cols.data() + pos, count, MPI_UINT32_T, r, 2, MPI_COMM_WORLD);
            }
        }
        std::vector<double>().swap(vals); std::vector<index_t>().swap(cols); std::vector<index_t>().swap(offsets);
    } else for (uint64_t pos = 0; pos < localNnz; pos += chunk) {
        int count = int(std::min(chunk, uint64_t(localNnz) - pos));
        MPI_Recv(localVals.data() + pos, count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localCols.data() + pos, count, MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);
    double *dVals, *dVec, *dOut;
    index_t *dCols, *dOffsets;
    CUDA_CHECK(cudaMalloc(&dVals, std::max(size_t(1), size_t(localNnz)) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max(size_t(1), size_t(localNnz)) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dOffsets, localOffsets.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, size_t(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max(1, localRows) * sizeof(double)));
    if (localNnz) { CUDA_CHECK(cudaMemcpy(dVals, localVals.data(), size_t(localNnz) * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), size_t(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaMemcpy(dOffsets, localOffsets.data(), localOffsets.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, std::max(1, localRows) * sizeof(double)));
    if (localRows && iterations) {
        spmvKernel<<<(uint64_t(localRows) * 32 + 255) / 256, 256>>>(dVals, dCols, dOffsets, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows) spmvKernel<<<(uint64_t(localRows) * 32 + 255) / 256, 256>>>(dVals, dCols, dOffsets, dVec, localRows, dOut);
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> localOut(localRows);
    if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, output.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dVals)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dOffsets)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    int status = 0;
    if (!rank) {
        printf("Computation time: %ld ms\n", long(duration * 1000));
        printf("Average time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", iterations ? duration * 1000 / iterations : 0.0, duration > 0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
        if (printResults) print_results(output, "OutputVector");
        if (validate) { printf("Validating result...\n"); bool valid = verifyResults(reference.data(), output.data(), numRows); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); status = valid ? 0 : 1; }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return status;
}
