#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
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
    for (int64_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// A warp handles several rows when rows are short, and one row when they are long.
// The matrix and input vector remain resident on the GPU for all iterations.
template <int THREADS_PER_ROW>
__global__ void spmvCuda(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rowDelimiters,
                         const double* __restrict__ vec, index_t rows,
                         double* __restrict__ out) {
    constexpr int BLOCK_THREADS = 256;
    const index_t row = blockIdx.x * (BLOCK_THREADS / THREADS_PER_ROW) +
                        threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    if (row >= rows) return;
    const unsigned active = __activemask();
    double sum = 0.0;
    for (index_t j = rowDelimiters[row] + lane;
         j < rowDelimiters[row + 1]; j += THREADS_PER_ROW) {
        sum += val[j] * vec[cols[j]];
    }
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(active, sum, offset, THREADS_PER_ROW);
    }
    if (lane == 0) out[row] = sum;
}

void checkCuda(cudaError_t error, int rank, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "MPI rank %d: %s: %s\n", rank, operation,
                cudaGetErrorString(error));
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
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
    const uint64_t items64 = sparsity ? uint64_t(numRows) * numRows / sparsity : 0;
    if (!sparsity || numRows > INT_MAX || items64 > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Matrix dimensions exceed supported MPI counts, or sparsity is zero.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(items64);

    // Assign each rank a local GPU, including when several GPUs share a node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "MPI rank %d: no CUDA GPU available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "cudaSetDevice");
    MPI_Comm_free(&nodeComm);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               numRows ? 100.0 * (1.0 - double(nItems) / (uint64_t(numRows) * numRows)) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        printf("Initializing data structures...\n");
    }

    std::vector<double> h_val, h_vec(numRows), h_out;
    std::vector<index_t> h_cols, h_rowDelimiters;
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Partition at CSR row boundaries near equal nonzero counts.
    std::vector<int> rowCounts(ranks), rowOffsets(ranks), valueCounts(ranks), valueOffsets(ranks);
    if (rank == 0) {
        std::vector<index_t> cuts(size_t(ranks) + 1);
        cuts[0] = 0;
        for (int r = 1; r < ranks; ++r) {
            const index_t target = static_cast<index_t>(items64 * r / ranks);
            cuts[r] = static_cast<index_t>(std::lower_bound(
                h_rowDelimiters.begin() + cuts[r - 1], h_rowDelimiters.end(), target) -
                h_rowDelimiters.begin());
            if (cuts[r] > numRows) cuts[r] = numRows;
        }
        cuts[ranks] = numRows;
        for (int r = 0; r < ranks; ++r) {
            rowOffsets[r] = static_cast<int>(cuts[r]);
            rowCounts[r] = static_cast<int>(cuts[r + 1] - cuts[r]);
            valueOffsets[r] = static_cast<int>(h_rowDelimiters[cuts[r]]);
            valueCounts[r] = static_cast<int>(h_rowDelimiters[cuts[r + 1]] - h_rowDelimiters[cuts[r]]);
        }
    }
    int localRows = 0, localItems = 0;
    MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(valueCounts.data(), 1, MPI_INT, &localItems, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> localVal(localItems), localOut(localRows);
    std::vector<index_t> localCols(localItems), localRow(size_t(localRows) + 1);
    MPI_Scatterv(h_val.data(), valueCounts.data(), valueOffsets.data(), MPI_DOUBLE,
                 localVal.data(), localItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), valueCounts.data(), valueOffsets.data(), MPI_UINT32_T,
                 localCols.data(), localItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_rowDelimiters.data(), rowCounts.data(), rowOffsets.data(), MPI_UINT32_T,
                 localRow.data(), localRows, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // The first received delimiter is the global value offset for this rank.
    const index_t localBase = localRows ? localRow[0] : 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localRows; ++i) localRow[i] -= localBase;
    localRow[localRows] = static_cast<index_t>(localItems);

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_row = nullptr;
    checkCuda(cudaMalloc(&d_val, size_t(std::max(localItems, 1)) * sizeof(double)), rank, "cudaMalloc values");
    checkCuda(cudaMalloc(&d_cols, size_t(std::max(localItems, 1)) * sizeof(index_t)), rank, "cudaMalloc columns");
    checkCuda(cudaMalloc(&d_row, (size_t(localRows) + 1) * sizeof(index_t)), rank, "cudaMalloc row offsets");
    checkCuda(cudaMalloc(&d_vec, size_t(std::max<index_t>(numRows, 1)) * sizeof(double)), rank, "cudaMalloc vector");
    checkCuda(cudaMalloc(&d_out, size_t(std::max(localRows, 1)) * sizeof(double)), rank, "cudaMalloc output");
    if (localItems) {
        checkCuda(cudaMemcpy(d_val, localVal.data(), size_t(localItems) * sizeof(double), cudaMemcpyHostToDevice), rank, "copy values");
        checkCuda(cudaMemcpy(d_cols, localCols.data(), size_t(localItems) * sizeof(index_t), cudaMemcpyHostToDevice), rank, "copy columns");
    }
    checkCuda(cudaMemcpy(d_row, localRow.data(), (size_t(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice), rank, "copy row offsets");
    if (numRows) checkCuda(cudaMemcpy(d_vec, h_vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice), rank, "copy vector");

    if (rank == 0) printf("Computing SpMV...\n");
    const int averageRowNnz = numRows ? static_cast<int>(items64 / numRows) : 0;
    const int threadsPerRow = averageRowNnz < 8 ? 4 : averageRowNnz < 16 ? 8 : averageRowNnz < 32 ? 16 : 32;
    const int blocks = (localRows + 256 / threadsPerRow - 1) / (256 / threadsPerRow);
    auto launch = [&]() {
        switch (threadsPerRow) {
            case 4: spmvCuda<4><<<blocks, 256>>>(d_val, d_cols, d_row, d_vec, localRows, d_out); break;
            case 8: spmvCuda<8><<<blocks, 256>>>(d_val, d_cols, d_row, d_vec, localRows, d_out); break;
            case 16: spmvCuda<16><<<blocks, 256>>>(d_val, d_cols, d_row, d_vec, localRows, d_out); break;
            default: spmvCuda<32><<<blocks, 256>>>(d_val, d_cols, d_row, d_vec, localRows, d_out); break;
        }
    };
    // Initialize the CUDA context and load the kernel before measuring iterations.
    if (iterations && localRows) {
        launch();
        checkCuda(cudaGetLastError(), rank, "SpMV warmup launch");
        checkCuda(cudaDeviceSynchronize(), rank, "SpMV warmup");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (iterations && localRows) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            launch();
        }
        checkCuda(cudaGetLastError(), rank, "SpMV kernel launch");
        checkCuda(cudaDeviceSynchronize(), rank, "SpMV kernel");
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (iterations && localRows) {
        checkCuda(cudaMemcpy(localOut.data(), d_out, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost), rank, "copy output");
    }
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, h_out.data(), rowCounts.data(),
                rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_row); cudaFree(d_vec); cudaFree(d_out);
    int status = 0;
    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", iterations ? milliseconds / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", elapsed > 0 ? 2.0 * nItems * iterations / elapsed / 1e9 : 0.0);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
