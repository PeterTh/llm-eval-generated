#include <algorithm>
#include <cstdint>
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

// Fail immediately on CUDA errors.  In particular, this also turns a missing
// CUDA-capable device into a clear error instead of silently running on the CPU.
inline void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Assign one warp to each row.  Adjacent lanes read adjacent entries in the
// CSR arrays, and the shuffle reduction avoids shared memory and barriers.
__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               index_t dim, double* __restrict__ out) {
    constexpr unsigned WARP_SIZE = 32;
    const unsigned lane = threadIdx.x & (WARP_SIZE - 1);
    const index_t firstWarp =
        static_cast<index_t>((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE);
    const index_t warpStride =
        static_cast<index_t>((gridDim.x * blockDim.x) / WARP_SIZE);

    for (index_t row = firstWarp; row < dim; row += warpStride) {
        double sum = 0.0;
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        for (index_t item = begin + lane; item < end; item += WARP_SIZE) {
            sum += val[item] * vec[cols[item]];
        }
#pragma unroll
        for (unsigned offset = WARP_SIZE / 2; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

// A full warp is wasteful for matrices averaging only a handful of entries per
// row.  This remains fully GPU-parallel, assigning one CUDA thread per row.
__global__ void spmvScalarKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 index_t dim, double* __restrict__ out) {
    const index_t firstRow = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t rowStride = static_cast<index_t>(gridDim.x * blockDim.x);
    for (index_t row = firstRow; row < dim; row += rowStride) {
        double sum = 0.0;
        for (index_t item = rowDelimiters[row]; item < rowDelimiters[row + 1]; ++item) {
            sum += val[item] * vec[cols[item]];
        }
        out[row] = sum;
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
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                            (static_cast<uint64_t>(i) * dim + j);
            const uint64_t needToAssign = n - nnzAssigned;
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

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        fprintf(stderr, "Error: matrix size, sparsity, and iteration count must be non-zero.\n");
        return 1;
    }

    // Calculate number of non-zero elements
    const uint64_t matrixItems = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = matrixItems / sparsity;
    if (nItems64 > UINT32_MAX) {
        fprintf(stderr, "Error: the requested matrix has too many non-zero elements.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixItems));
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

    // Allocate device storage once and keep all benchmark data resident on the
    // GPU across iterations.  Host/device transfers are intentionally excluded
    // from the SpMV timing, just as initialization was excluded previously.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    if (nItems != 0) {
        checkCuda(cudaMalloc(&d_val, static_cast<size_t>(nItems) * sizeof(double)), "allocating values");
        checkCuda(cudaMalloc(&d_cols, static_cast<size_t>(nItems) * sizeof(index_t)), "allocating columns");
        checkCuda(cudaMemcpy(d_val, h_val.data(), static_cast<size_t>(nItems) * sizeof(double),
                             cudaMemcpyHostToDevice), "copying values");
        checkCuda(cudaMemcpy(d_cols, h_cols.data(), static_cast<size_t>(nItems) * sizeof(index_t),
                             cudaMemcpyHostToDevice), "copying columns");
    }
    checkCuda(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(numRows) + 1) * sizeof(index_t)),
              "allocating row delimiters");
    checkCuda(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)), "allocating vector");
    checkCuda(cudaMalloc(&d_out, static_cast<size_t>(numRows) * sizeof(double)), "allocating output");
    checkCuda(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                         (static_cast<size_t>(numRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice),
              "copying row delimiters");
    checkCuda(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                         cudaMemcpyHostToDevice), "copying vector");

    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, 0), "querying GPU properties");
    constexpr unsigned THREADS = 256;
    const bool useWarpKernel = nItems / numRows > 4;
    const uint64_t workUnits = useWarpKernel ? numRows : (static_cast<uint64_t>(numRows) + 31) / 32;
    const uint64_t wantedBlocks = (workUnits * 32 + THREADS - 1) / THREADS;
    const uint64_t residentBlocks = static_cast<uint64_t>(deviceProperties.multiProcessorCount) * 32;
    const unsigned blocks = static_cast<unsigned>(std::max<uint64_t>(1, std::min(wantedBlocks, residentBlocks)));

    // Warm up the selected kernel so context creation and JIT setup do not
    // contaminate the benchmark measurement.
    if (useWarpKernel) {
        spmvWarpKernel<<<blocks, THREADS>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    } else {
        spmvScalarKernel<<<blocks, THREADS>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    }
    checkCuda(cudaGetLastError(), "launching warm-up kernel");
    checkCuda(cudaDeviceSynchronize(), "running warm-up kernel");

    printf("Computing SpMV...\n");
    cudaEvent_t start;
    cudaEvent_t end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (useWarpKernel) {
            spmvWarpKernel<<<blocks, THREADS>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else {
            spmvScalarKernel<<<blocks, THREADS>>>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        }
    }
    checkCuda(cudaGetLastError(), "launching SpMV kernel");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "waiting for SpMV kernels");
    float durationMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&durationMs, start, end), "measuring SpMV time");
    checkCuda(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(numRows) * sizeof(double),
                         cudaMemcpyDeviceToHost), "copying output");

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(d_val), "freeing values");
    checkCuda(cudaFree(d_cols), "freeing columns");
    checkCuda(cudaFree(d_rowDelimiters), "freeing row delimiters");
    checkCuda(cudaFree(d_vec), "freeing vector");
    checkCuda(cudaFree(d_out), "freeing output");

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (durationMs * 1.0e6);
    const double avgTime = durationMs / iterations;
    
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
