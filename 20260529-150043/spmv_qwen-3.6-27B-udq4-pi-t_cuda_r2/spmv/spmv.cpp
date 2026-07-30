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

// CUDA error-checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

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
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;

    double prob =
        static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);

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
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format (CPU
//   reference for validation)
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
void spmvCpu(const double* val, const index_t* cols,
             const index_t* rowDelimiters, const double* vec, const index_t dim,
             double* out) {
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
// CUDA Kernel: spmvKernel
//
// Purpose:
//   Parallel sparse matrix-vector multiplication using CSR format.
//   Each thread block handles one row of the matrix. Threads within
//   the block compute partial dot products and reduce to a single
//   result using a tree-based reduction in shared memory.
//
// Arguments:
//   val:           device pointer to non-zero values
//   cols:          device pointer to column indices
//   rowDelimiters: device pointer to row delimiters (size dim+1)
//   vec:           device pointer to input dense vector
//   dim:           matrix dimension (rows = columns)
//   out:           device pointer to output vector
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t dim,
                           double* __restrict__ out) {
    extern __shared__ double sdata[];

    index_t row = blockIdx.x;
    if (row >= dim) return;

    double sum = 0.0;
    index_t start = rowDelimiters[row];
    index_t end = rowDelimiters[row + 1];
    index_t nnz = end - start;

    // Each thread accumulates partial dot-product contributions
    for (index_t j = threadIdx.x; j < nnz; j += blockDim.x) {
        sum += val[start + j] * vec[cols[start + j]];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (index_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this row
    if (threadIdx.x == 0) {
        out[row] = sdata[0];
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
bool verifyResults(const double* reference, const double* result,
                   const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got "
                       "%.10e\n",
                       i, ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got "
                       "%.10e (rel error: %.10e)\n",
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
    printf("  -n <num>     Number of rows/columns in the matrix (default: "
           "1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: "
           "10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements "
           "(default: 1.0)\n");
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

    // Initialize CUDA
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Using CUDA GPU: %s\n", prop.name);
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n",
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures on host
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows);

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // Allocate device memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, nItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, nItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (numRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, numRows * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          (numRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // For validation, compute reference solution on CPU
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Configure kernel launch parameters
    const int gridDim = numRows;
    const int blockDim = BLOCK_SIZE;
    const size_t sharedMemSize = blockDim * sizeof(double);

    // Warmup iteration (avoids cold-start effects)
    printf("Warming up GPU...\n");
    spmvKernel<<<gridDim, blockDim, sharedMemSize>>>(
        d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations using CUDA events
    printf("Computing SpMV...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<gridDim, blockDim, sharedMemSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Print timing results
    printf("Computation time: %.3f ms\n", milliseconds);
    const double gflops =
        (2.0 * nItems * iterations) / (milliseconds / 1000.0) / 1e9;
    const double avgTime = milliseconds / static_cast<double>(iterations);
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid =
            verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        // Cleanup CUDA events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Cleanup device memory
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
        CUDA_CHECK(cudaFree(d_rowDelimiters));
        CUDA_CHECK(cudaFree(d_vec));
        CUDA_CHECK(cudaFree(d_out));

        return valid ? 0 : 1;
    }

    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}