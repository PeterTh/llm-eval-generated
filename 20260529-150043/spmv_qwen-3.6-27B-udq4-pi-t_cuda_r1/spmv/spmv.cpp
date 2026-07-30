#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int BLOCK_SIZE = 256;

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
// CUDA Kernel: spmvGpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on GPU.
//   Each thread processes one row of the matrix. For rows with many elements,
//   a warp-level reduction using shared memory is used for performance.
//
// Arguments:
//   val:           array holding the non-zero values for the matrix
//   cols:          array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows
//   vec:           dense vector of size dim
//   dim:           number of rows/columns in the matrix
//   out:           output - result from the spmv calculation
//
// ****************************************************************************
__global__ void spmvGpu(const double* val, const index_t* cols,
                        const index_t* rowDelimiters, const double* vec,
                        const index_t dim, double* out) {
    // Each thread handles one row
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim) {
        double sum = 0.0;
        index_t rowStart = rowDelimiters[row];
        index_t rowEnd = rowDelimiters[row + 1];

        // Dot product for this row
        for (index_t j = rowStart; j < rowEnd; ++j) {
            sum += val[j] * vec[cols[j]];
        }

        out[row] = sum;
    }
}

// ****************************************************************************
// CUDA Kernel: spmvGpuCooperative
//
// Purpose:
//   Optimized SpMV kernel using shared memory and cooperative thread
//   reduction for rows with many non-zeros. Threads in a block cooperate
//   to process rows with high non-zero counts, distributing the work
//   across all threads in the warp/block.
//
// Arguments:
//   val:           array holding the non-zero values for the matrix
//   cols:          array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows
//   vec:           dense vector of size dim
//   dim:           number of rows/columns in the matrix
//   out:           output - result from the spmv calculation
//
// ****************************************************************************
__global__ void spmvGpuCooperative(const double* val, const index_t* cols,
                                   const index_t* rowDelimiters, const double* vec,
                                   const index_t dim, double* out) {
    extern __shared__ double sharedSum[];

    // Each block processes one row cooperatively
    index_t row = blockIdx.x;
    if (row >= dim) return;

    index_t rowStart = rowDelimiters[row];
    index_t rowEnd = rowDelimiters[row + 1];
    index_t rowLen = rowEnd - rowStart;
    index_t tid = threadIdx.x;

    // Distribute elements across threads in the block
    double partialSum = 0.0;
    for (index_t j = tid; j < rowLen; j += blockDim.x) {
        partialSum += val[rowStart + j] * vec[cols[rowStart + j]];
    }

    // Store partial sum in shared memory
    sharedSum[tid] = partialSum;
    __syncthreads();

    // Warp-level reduction using tree-based approach
    // For BLOCK_SIZE=256, we need 8 steps of reduction
    for (index_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
    }

    // Thread 0 writes the final result
    if (tid == 0) {
        out[row] = sharedSum[0];
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (kept for reference/validation only)
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

// ****************************************************************************
// Function: checkCudaError
//
// Purpose:
//   Helper to check CUDA errors and print diagnostic info
//
// ****************************************************************************
static inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s (%s)\n", msg,
                cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n",
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate host data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution on CPU
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate GPU memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    checkCudaError(cudaMalloc(&d_val, nItems * sizeof(double)), "d_val");
    checkCudaError(cudaMalloc(&d_cols, nItems * sizeof(index_t)), "d_cols");
    checkCudaError(cudaMalloc(&d_rowDelimiters, (numRows + 1) * sizeof(index_t)), "d_rowDelimiters");
    checkCudaError(cudaMalloc(&d_vec, numRows * sizeof(double)), "d_vec");
    checkCudaError(cudaMalloc(&d_out, numRows * sizeof(double)), "d_out");

    // Copy data to GPU
    checkCudaError(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double),
                               cudaMemcpyHostToDevice), "copy val");
    checkCudaError(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t),
                               cudaMemcpyHostToDevice), "copy cols");
    checkCudaError(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                               (numRows + 1) * sizeof(index_t),
                               cudaMemcpyHostToDevice), "copy rowDelimiters");
    checkCudaError(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                               cudaMemcpyHostToDevice), "copy vec");

    // Determine kernel configuration based on matrix characteristics
    // Compute max non-zeros per row for kernel selection
    index_t maxNnzPerRow = 0;
    for (index_t i = 0; i < numRows; ++i) {
        index_t nnz = h_rowDelimiters[i + 1] - h_rowDelimiters[i];
        if (nnz > maxNnzPerRow) maxNnzPerRow = nnz;
    }

    // Choose kernel: cooperative kernel is better for dense rows
    // Standard kernel is better for sparse rows (low nnz per row)
    bool useCooperative = (maxNnzPerRow > 64);

    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim;

    if (useCooperative) {
        // Cooperative kernel: one block per row
        gridDim.x = numRows;
    } else {
        // Standard kernel: one thread per row
        gridDim.x = (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // Create CUDA stream for async execution
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "stream create");

    // Perform SpMV computation on GPU
    printf("Computing SpMV (CUDA)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (useCooperative) {
            size_t sharedMemSize = BLOCK_SIZE * sizeof(double);
            spmvGpuCooperative<<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else {
            spmvGpu<<<gridDim, blockDim, 0, stream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        }
        checkCudaError(cudaGetLastError(), "kernel launch");
    }

    // Synchronize to get accurate timing
    checkCudaError(cudaStreamSynchronize(stream), "stream sync");
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double),
                               cudaMemcpyDeviceToHost), "copy result");

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);

    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        // Cleanup GPU resources
        checkCudaError(cudaStreamDestroy(stream), "stream destroy");
        checkCudaError(cudaFree(d_val), "free d_val");
        checkCudaError(cudaFree(d_cols), "free d_cols");
        checkCudaError(cudaFree(d_rowDelimiters), "free d_rowDelimiters");
        checkCudaError(cudaFree(d_vec), "free d_vec");
        checkCudaError(cudaFree(d_out), "free d_out");

        return valid ? 0 : 1;
    }

    // Cleanup GPU resources
    checkCudaError(cudaStreamDestroy(stream), "stream destroy");
    checkCudaError(cudaFree(d_val), "free d_val");
    checkCudaError(cudaFree(d_cols), "free d_cols");
    checkCudaError(cudaFree(d_rowDelimiters), "free d_rowDelimiters");
    checkCudaError(cudaFree(d_vec), "free d_vec");
    checkCudaError(cudaFree(d_out), "free d_out");

    return 0;
}
