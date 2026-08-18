#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// A warp processes one CSR row.  This gives each row an independent unit of
// work while keeping the value and column-index loads coalesced for the usual
// case where a row contains more than one non-zero.
constexpr unsigned int CUDA_THREADS_PER_BLOCK = 256;
constexpr unsigned int CUDA_WARPS_PER_BLOCK = CUDA_THREADS_PER_BLOCK / 32;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double warpReduceSum(double value) {
    constexpr unsigned int fullWarpMask = 0xffffffffu;
    value += __shfl_down_sync(fullWarpMask, value, 16);
    value += __shfl_down_sync(fullWarpMask, value, 8);
    value += __shfl_down_sync(fullWarpMask, value, 4);
    value += __shfl_down_sync(fullWarpMask, value, 2);
    value += __shfl_down_sync(fullWarpMask, value, 1);
    return value;
}

__global__ void spmvWarpPerRow(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t dim,
                               const index_t iterations,
                               double* __restrict__ out) {
    const unsigned int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = globalThread / 32;
    const unsigned int lane = threadIdx.x & 31u;

    if (row >= dim) {
        return;
    }

    // Only one lane fetches the row bounds.  Broadcasting them avoids
    // redundant global loads while retaining a fully uniform reduction.
    index_t rowStart = 0;
    index_t rowEnd = 0;
    if (lane == 0) {
        rowStart = rowDelimiters[row];
        rowEnd = rowDelimiters[row + 1];
    }
    rowStart = __shfl_sync(0xffffffffu, rowStart, 0);
    rowEnd = __shfl_sync(0xffffffffu, rowEnd, 0);

    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        double sum = 0.0;
        for (index_t j = rowStart + lane; j < rowEnd; j += 32) {
            const index_t col = __ldg(cols + j);
            sum += __ldg(val + j) * __ldg(vec + col);
        }

        sum = warpReduceSum(sum);
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

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

    // Allocate device storage and transfer the immutable CSR inputs once.
    // The computation below is always performed by the CUDA kernel; there is
    // intentionally no CPU fallback path.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    const size_t deviceNnz = std::max<size_t>(nItems, 1);
    const size_t deviceDim = std::max<size_t>(numRows, 1);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_val), deviceNnz * sizeof(double)),
              "allocating matrix values");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cols), deviceNnz * sizeof(index_t)),
              "allocating column indices");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                         (static_cast<size_t>(numRows) + 1) * sizeof(index_t)),
              "allocating row delimiters");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_vec), deviceDim * sizeof(double)),
              "allocating input vector");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_out), deviceDim * sizeof(double)),
              "allocating output vector");

    if (nItems != 0) {
        checkCuda(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice),
                  "copying matrix values");
        checkCuda(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice),
                  "copying column indices");
    }
    checkCuda(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                         (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                         cudaMemcpyHostToDevice),
              "copying row delimiters");
    if (numRows != 0) {
        checkCuda(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "copying input vector");
    }
    // This also preserves the original zero-initialized output when -i 0 is
    // supplied.  For normal runs every row is overwritten by the kernel.
    checkCuda(cudaMemset(d_out, 0, deviceDim * sizeof(double)),
              "clearing output vector");

    const dim3 block(CUDA_THREADS_PER_BLOCK);
    const dim3 grid((static_cast<unsigned int>(numRows) + CUDA_WARPS_PER_BLOCK - 1) /
                    CUDA_WARPS_PER_BLOCK);

    // Pay CUDA context and first-launch setup costs before starting the timed
    // region.  The warm-up uses the same kernel and data as the benchmark.
    if (iterations != 0 && numRows != 0) {
        spmvWarpPerRow<<<grid, block>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, 1, d_out);
        checkCuda(cudaGetLastError(), "launching CUDA warm-up");
        checkCuda(cudaDeviceSynchronize(), "synchronizing CUDA warm-up");
    }

    // Perform SpMV computation
    printf("Computing SpMV on CUDA...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (numRows != 0) {
        spmvWarpPerRow<<<grid, block>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows,
                                        iterations, d_out);
    }
    checkCuda(cudaGetLastError(), "launching CUDA SpMV");
    checkCuda(cudaDeviceSynchronize(), "synchronizing CUDA SpMV");

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;

    if (numRows != 0) {
        checkCuda(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(numRows) * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "copying output vector");
    }
    checkCuda(cudaFree(d_val), "freeing matrix values");
    checkCuda(cudaFree(d_cols), "freeing column indices");
    checkCuda(cudaFree(d_rowDelimiters), "freeing row delimiters");
    checkCuda(cudaFree(d_vec), "freeing input vector");
    checkCuda(cudaFree(d_out), "freeing output vector");

    printf("Computation time: %.3f ms\n", duration.count());
    
    // Calculate performance metrics
    const double elapsedSeconds = duration.count() / 1000.0;
    const double gflops = (elapsedSeconds > 0.0)
                              ? (2.0 * nItems * iterations) / elapsedSeconds / 1e9
                              : 0.0;
    const double avgTime = (iterations != 0)
                               ? duration.count() / static_cast<double>(iterations)
                               : 0.0;
    
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
