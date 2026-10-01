#include <algorithm>
#include <climits>
#include <cstdint>
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
#pragma omp parallel for schedule(static) if(dim > 4096)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// A warp owns a row. Its lanes stream through that row's nonzeros, then
// reduce the partial sums without shared memory or inter-warp synchronization.
__global__ void spmvGpu(const double* __restrict__ val,
                        const index_t* __restrict__ cols,
                        const index_t* __restrict__ rowOffsets,
                        const double* __restrict__ vec,
                        index_t rows, double* __restrict__ out) {
    const unsigned lane = threadIdx.x & 31;
    const index_t firstRow = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const index_t rowStride = gridDim.x * (blockDim.x / 32);
    for (index_t row = firstRow; row < rows; row += rowStride) {
        double sum = 0.0;
        for (index_t j = rowOffsets[row] + lane; j < rowOffsets[row + 1]; j += 32)
            sum += val[j] * vec[cols[j]];
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    int rank, ranks;
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

    // The original CSR format uses 32-bit offsets; MPI_Scatterv uses int counts.
    const uint64_t totalEntries = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = sparsity ? totalEntries / sparsity : 0;
    if (numRows == 0 || sparsity == 0 || nonzeros > INT_MAX || numRows > INT_MAX ||
        totalEntries > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid size, sparsity, or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - double(nItems) / double(totalEntries)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", ranks, omp_get_max_threads());
    }

    std::vector<double> h_val, h_vec(numRows), h_out;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<int> rowCounts(ranks), rowDispls(ranks), nzCounts(ranks), nzDispls(ranks);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Pick row boundaries near equal nonzero counts. Work remains row-owned.
        std::vector<index_t> boundaries(ranks + 1);
        boundaries[ranks] = numRows;
        for (int p = 1; p < ranks; ++p) {
            const index_t target = uint64_t(nItems) * p / ranks;
            boundaries[p] = static_cast<index_t>(std::lower_bound(
                h_rowDelimiters.begin(), h_rowDelimiters.end(), target) - h_rowDelimiters.begin());
        }
        for (int p = 0; p < ranks; ++p) {
            rowDispls[p] = boundaries[p];
            rowCounts[p] = boundaries[p + 1] - boundaries[p];
            nzDispls[p] = h_rowDelimiters[boundaries[p]];
            nzCounts[p] = h_rowDelimiters[boundaries[p + 1]] - nzDispls[p];
        }
    }

    int localRows, localNnz, firstNnz;
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nzDispls.data(), 1, MPI_INT, &firstNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<index_t> localOffsets(localRows + 1), localCols(localNnz);
    std::vector<double> localVal(localNnz), localOut(localRows);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_rowDelimiters.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 localOffsets.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nzCounts.data(), nzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_val.data(), nzCounts.data(), nzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int row = 0; row < localRows; ++row) localOffsets[row] -= firstNnz;
    localOffsets[localRows] = localNnz;

    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount;
    checkCuda(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "set device");
    MPI_Comm_free(&localComm);

    double *dVal, *dVec, *dOut;
    index_t *dCols, *dOffsets;
    checkCuda(cudaMalloc(&dVal, std::max(1, localNnz) * sizeof(double)), "allocate values");
    checkCuda(cudaMalloc(&dCols, std::max(1, localNnz) * sizeof(index_t)), "allocate columns");
    checkCuda(cudaMalloc(&dOffsets, (localRows + 1) * sizeof(index_t)), "allocate offsets");
    checkCuda(cudaMalloc(&dVec, numRows * sizeof(double)), "allocate vector");
    checkCuda(cudaMalloc(&dOut, std::max(1, localRows) * sizeof(double)), "allocate output");
    checkCuda(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice), "copy values");
    checkCuda(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice), "copy columns");
    checkCuda(cudaMemcpy(dOffsets, localOffsets.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "copy offsets");
    checkCuda(cudaMemcpy(dVec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice), "copy vector");

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        const int blocks = std::min(65535, (localRows + 127) / 128);
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvGpu<<<blocks, 128>>>(dVal, dCols, dOffsets, dVec, localRows, dOut);
        checkCuda(cudaGetLastError(), "launch SpMV kernel");
    }
    checkCuda(cudaDeviceSynchronize(), "synchronize SpMV");
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (iterations)
        checkCuda(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost), "copy output");
    if (rank == 0) h_out.resize(numRows);
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, h_out.data(),
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    checkCuda(cudaFree(dVal), "free values");
    checkCuda(cudaFree(dCols), "free columns");
    checkCuda(cudaFree(dOffsets), "free offsets");
    checkCuda(cudaFree(dVec), "free vector");
    checkCuda(cudaFree(dOut), "free output");

    int result = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
        if (printResults) print_results(h_out, "OutputVector");
    }
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        result = valid ? 0 : 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
