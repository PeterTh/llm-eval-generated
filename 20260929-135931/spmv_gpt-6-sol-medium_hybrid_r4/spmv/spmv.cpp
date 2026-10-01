#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                      (static_cast<uint64_t>(i) * dim + j);
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

// One warp handles one CSR row. Lanes accumulate independent entries and
// reduce within the warp; no atomics or inter-block synchronization are needed.
__global__ void spmvGpu(const double* val, const index_t* cols,
                        const index_t* rowDelimiters, const double* vec,
                        index_t rows, double* out) {
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x & (warpSize - 1);
    double sum = 0.0;
    for (index_t j = rowDelimiters[row] + lane;
         j < rowDelimiters[row + 1]; j += warpSize) {
        sum += val[j] * vec[cols[j]];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) out[row] = sum;
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI %s failed\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    mpiCheck(MPI_Init(&argc, &argv), "Init");
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "Comm_size");

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

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

    const uint64_t items64 = static_cast<uint64_t>(numRows) * numRows / (sparsity ? sparsity : 1);
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        numRows > INT_MAX || items64 > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size, sparsity, or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(items64);

    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm), "Comm_split_type");
    int localRank;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "local Comm_rank");
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "GetDeviceCount");
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA GPU available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "SetDevice");
    mpiCheck(MPI_Comm_free(&localComm), "Comm_free");

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                 (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute contiguous row ranges. Root retains the original random
    // generation order, so each rank sees the same matrix as the serial code.
    std::vector<int> rowCounts(ranks), rowOffsets(ranks), nnzCounts(ranks), nnzOffsets(ranks);
    for (int p = 0; p < ranks; ++p) {
        rowOffsets[p] = static_cast<int>(static_cast<uint64_t>(numRows) * p / ranks);
        const int end = static_cast<int>(static_cast<uint64_t>(numRows) * (p + 1) / ranks);
        rowCounts[p] = end - rowOffsets[p];
    }
    const int localRows = rowCounts[rank];
    std::vector<double> h_val, h_vec(numRows), h_out;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_reference;
    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        for (int p = 0; p < ranks; ++p) {
            nnzOffsets[p] = h_rowDelimiters[rowOffsets[p]];
            nnzCounts[p] = h_rowDelimiters[rowOffsets[p] + rowCounts[p]] - nnzOffsets[p];
        }
        h_out.resize(numRows);
    }
    mpiCheck(MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
                       MPI_COMM_WORLD), "Bcast vector");
    int localNnz = 0;
    mpiCheck(MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT,
                         0, MPI_COMM_WORLD), "Scatter nnz counts");
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz), localRowDelimiters(localRows + 1);
    mpiCheck(MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzOffsets.data(), MPI_DOUBLE,
                          localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "Scatterv values");
    mpiCheck(MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzOffsets.data(), MPI_UINT32_T,
                          localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD),
             "Scatterv columns");
    std::vector<int> delimiterCounts(ranks);
    if (rank == 0) {
        for (int p = 0; p < ranks; ++p) delimiterCounts[p] = rowCounts[p] + 1;
    }
    mpiCheck(MPI_Scatterv(h_rowDelimiters.data(), delimiterCounts.data(), rowOffsets.data(),
                          MPI_UINT32_T, localRowDelimiters.data(), localRows + 1,
                          MPI_UINT32_T, 0, MPI_COMM_WORLD), "Scatterv row delimiters");
    const index_t firstNnz = localRowDelimiters[0];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= localRows; ++i) localRowDelimiters[i] -= firstNnz;

    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rows;
    cudaCheck(cudaMalloc(&d_val, std::max(1, localNnz) * sizeof(double)), "Malloc values");
    cudaCheck(cudaMalloc(&d_cols, std::max(1, localNnz) * sizeof(index_t)), "Malloc columns");
    cudaCheck(cudaMalloc(&d_rows, (localRows + 1) * sizeof(index_t)), "Malloc row delimiters");
    cudaCheck(cudaMalloc(&d_vec, numRows * sizeof(double)), "Malloc vector");
    cudaCheck(cudaMalloc(&d_out, std::max(1, localRows) * sizeof(double)), "Malloc output");
    cudaCheck(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice), "Copy values");
    cudaCheck(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice), "Copy columns");
    cudaCheck(cudaMemcpy(d_rows, localRowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "Copy row delimiters");
    cudaCheck(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice), "Copy vector");

    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    const double start = MPI_Wtime();
    constexpr int threads = 256;
    constexpr int warpsPerBlock = threads / 32;
    const int blocks = (localRows + warpsPerBlock - 1) / warpsPerBlock;
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (blocks) spmvGpu<<<blocks, threads>>>(d_val, d_cols, d_rows, d_vec,
                                                  localRows, d_out);
    }
    cudaCheck(cudaGetLastError(), "SpMV launch");
    cudaCheck(cudaDeviceSynchronize(), "SpMV synchronize");
    std::vector<double> localOut(localRows);
    cudaCheck(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double),
                         cudaMemcpyDeviceToHost), "Copy output");
    mpiCheck(MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, h_out.data(),
                         rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0,
                         MPI_COMM_WORLD), "Gather output");
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD), "Reduce timing");
    cudaCheck(cudaFree(d_val), "Free values");
    cudaCheck(cudaFree(d_cols), "Free columns");
    cudaCheck(cudaFree(d_rows), "Free row delimiters");
    cudaCheck(cudaFree(d_vec), "Free vector");
    cudaCheck(cudaFree(d_out), "Free output");

    int result = 0;
    if (rank == 0) {
        const double milliseconds = maxElapsed * 1000.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", milliseconds / iterations);
        printf("Performance: %.3f GFLOPS\n",
               (2.0 * nItems * iterations) / maxElapsed / 1e9);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            result = verifyResults(h_reference.data(), h_out.data(), numRows) ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast result");
    mpiCheck(MPI_Finalize(), "Finalize");
    return result;
}
