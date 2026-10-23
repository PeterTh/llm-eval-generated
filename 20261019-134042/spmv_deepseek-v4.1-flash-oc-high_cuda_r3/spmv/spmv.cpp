#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
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
// Function: spmvRowKernel
//
// Purpose:
//   GPU SpMV kernel for short rows: one thread per matrix row.
//
// ****************************************************************************
__global__ void spmvRowKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t dim,
                              double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) {
        return;
    }

    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t j = start; j < end; ++j) {
        sum += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }
    out[row] = sum;
}

// ****************************************************************************
// Function: spmvWarpKernel
//
// Purpose:
//   GPU SpMV kernel for long rows: one warp per matrix row. The warp walks
//   the row with coalesced loads and finishes with a warp reduction.
//
// ****************************************************************************
__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec, const index_t dim,
                               double* __restrict__ out) {
    const unsigned int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warpId = globalThread >> 5;
    if (warpId >= dim) {
        return;
    }

    const unsigned int lane = threadIdx.x & 31u;
    const index_t start = rowDelimiters[warpId];
    const index_t end = rowDelimiters[warpId + 1];

    double sum = 0.0;
    for (index_t j = start + lane; j < end; j += 32) {
        sum += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        out[warpId] = sum;
    }
}

// ****************************************************************************
// Function: launchSpMV
//
// Purpose:
//   Dispatches the GPU SpMV kernel best suited to the average row length.
//
// ****************************************************************************
void launchSpMV(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, const index_t dim, double* out, const double avgNnzPerRow) {
    if (dim == 0) {
        return;
    }

    if (avgNnzPerRow < 8.0) {
        constexpr unsigned int blockSize = 256;
        const unsigned int gridSize = (dim + blockSize - 1) / blockSize;
        spmvRowKernel<<<gridSize, blockSize>>>(val, cols, rowDelimiters, vec, dim, out);
    } else {
        constexpr unsigned int warpsPerBlock = 8;
        constexpr unsigned int blockSize = warpsPerBlock * 32;
        const unsigned int gridSize = (dim + warpsPerBlock - 1) / warpsPerBlock;
        spmvWarpKernel<<<gridSize, blockSize>>>(val, cols, rowDelimiters, vec, dim, out);
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

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Move the sparse matrix and dense vector to the GPU once
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

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    const double avgNnzPerRow =
        (numRows > 0) ? static_cast<double>(nItems) / static_cast<double>(numRows) : 0.0;

    // Perform SpMV computation on the GPU
    printf("Computing SpMV...\n");

    // Warm up the device (module load, context setup) before timing
    launchSpMV(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out, avgNnzPerRow);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));
    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpMV(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out, avgNnzPerRow);
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    const double durationMs = static_cast<double>(elapsedMs);
    const long computationTimeMs = static_cast<long>(std::llround(durationMs));

    printf("Computation time: %ld ms\n", computationTimeMs);

    // Calculate performance metrics
    const double seconds = durationMs / 1000.0;
    const double gflops = (seconds > 0.0)
                              ? (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
                                    seconds / 1e9
                              : 0.0;
    const double avgTime = (iterations > 0) ? durationMs / static_cast<double>(iterations) : 0.0;

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
