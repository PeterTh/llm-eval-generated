#include <algorithm>
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

// CUDA execution configuration
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
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
// Kernel: spmvCsrVector
//
// Purpose:
//   CUDA implementation of CSR sparse matrix-vector multiplication. A group
//   ("vector") of VECTOR_SIZE consecutive threads cooperates on one matrix
//   row: the non-zeros of the row are read with a stride of VECTOR_SIZE so
//   that the accesses of the group coalesce, and the partial products are
//   combined with a warp-shuffle reduction. VECTOR_SIZE is chosen at runtime
//   to roughly match the average number of non-zeros per row, which keeps
//   both short and long rows efficient.
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
template <int VECTOR_SIZE>
__global__ void spmvCsrVector(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t dim,
                              double* __restrict__ out) {
    constexpr int VECTORS_PER_BLOCK = BLOCK_SIZE / VECTOR_SIZE;

    const int lane = threadIdx.x & (VECTOR_SIZE - 1);
    const int vectorInBlock = threadIdx.x / VECTOR_SIZE;
    const index_t firstRow = blockIdx.x * VECTORS_PER_BLOCK + vectorInBlock;
    const index_t rowStride = gridDim.x * VECTORS_PER_BLOCK;

    for (index_t row = firstRow; row < dim; row += rowStride) {
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];

        double t = 0.0;
        for (index_t j = rowStart + lane; j < rowEnd; j += VECTOR_SIZE) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }

        // Reduce the partial sums held by the threads of this vector
#pragma unroll
        for (int offset = VECTOR_SIZE / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, VECTOR_SIZE);
        }

        if (lane == 0) {
            out[row] = t;
        }
    }
}

// ****************************************************************************
// Function: launchSpmv
//
// Purpose:
//   Launches the SpMV kernel with the vector width that best matches the
//   average number of non-zeros per row.
//
// ****************************************************************************
void launchSpmv(const int vectorSize, const int numBlocks, const double* d_val,
                const index_t* d_cols, const index_t* d_rowDelimiters, const double* d_vec,
                const index_t dim, double* d_out) {
    switch (vectorSize) {
        case 2:
            spmvCsrVector<2><<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
            break;
        case 4:
            spmvCsrVector<4><<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
            break;
        case 8:
            spmvCsrVector<8><<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
            break;
        case 16:
            spmvCsrVector<16><<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
            break;
        default:
            spmvCsrVector<32><<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
            break;
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

    // Set up the GPU: allocate device buffers and upload the matrix and vector
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    printf("Device: %s (%d SMs)\n", props.name, props.multiProcessorCount);

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * nItems));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * nItems));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, sizeof(index_t) * (static_cast<size_t>(numRows) + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * numRows));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * numRows));

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          sizeof(index_t) * (static_cast<size_t>(numRows) + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    // Pick the number of threads per row from the average row length, rounded
    // to the next power of two in [2, 32]
    const double nnzPerRow = numRows > 0 ? static_cast<double>(nItems) / numRows : 0.0;
    int vectorSize = 2;
    while (vectorSize < WARP_SIZE && vectorSize < nnzPerRow) {
        vectorSize *= 2;
    }
    const int vectorsPerBlock = BLOCK_SIZE / vectorSize;
    const size_t neededBlocks = (static_cast<size_t>(numRows) + vectorsPerBlock - 1) / vectorsPerBlock;
    // Cap the grid so that each block does a fair amount of work (grid-stride loop)
    const size_t maxBlocks = static_cast<size_t>(props.multiProcessorCount) * 32;
    const int numBlocks = static_cast<int>(std::max<size_t>(1, std::min(neededBlocks, maxBlocks)));

    printf("Threads per row: %d, blocks: %d\n", vectorSize, numBlocks);

    // Warm up (kernel load, clock ramp up) - not part of the measurement
    launchSpmv(vectorSize, numBlocks, d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(vectorSize, numBlocks, d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaGetLastError());

    // Retrieve the result of the last iteration
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * numRows, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics; use the full timer resolution here since
    // the GPU can finish all iterations in well under a millisecond
    const double elapsedMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
    const double avgTime = elapsedMs / static_cast<double>(iterations);

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
