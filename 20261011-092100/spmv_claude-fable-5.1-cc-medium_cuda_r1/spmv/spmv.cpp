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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                           \
            exit(1);                                                                      \
        }                                                                                 \
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
//   Computes sparse matrix-vector multiplication using CSR format on the
//   host. Used only to produce the reference solution for validation.
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
// CUDA kernels
//
// Two CSR SpMV kernels are provided and selected at runtime based on the
// average number of non-zeros per row:
//
//   spmvCsrScalarKernel: one thread per row. Best for very short rows, where
//                        a warp per row would leave most lanes idle.
//
//   spmvCsrVectorKernel: one warp (32 lanes) per row. Lanes stride across the
//                        row's non-zeros so loads of val/cols are coalesced,
//                        then the partial sums are reduced with warp shuffles.
//                        Best for rows with >= ~32 non-zeros.
//
// The dense input vector is read through the read-only (texture) cache via
// __ldg since its accesses are irregular (gathers).
// ****************************************************************************
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

__global__ void spmvCsrScalarKernel(const double* __restrict__ val,
                                    const index_t* __restrict__ cols,
                                    const index_t* __restrict__ rowDelimiters,
                                    const double* __restrict__ vec,
                                    const index_t dim,
                                    double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double t = 0.0;
    for (index_t j = rowStart; j < rowEnd; ++j) {
        t += val[j] * __ldg(&vec[cols[j]]);
    }
    out[row] = t;
}

__global__ void __launch_bounds__(BLOCK_SIZE)
spmvCsrVectorKernel(const double* __restrict__ val,
                    const index_t* __restrict__ cols,
                    const index_t* __restrict__ rowDelimiters,
                    const double* __restrict__ vec,
                    const index_t dim,
                    double* __restrict__ out) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const index_t warpId = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    const index_t row = warpId;
    if (row >= dim) return;

    // Both delimiters are fetched by every lane; the loads hit the same
    // cache line so this costs no extra memory traffic.
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double t = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
        t += val[j] * __ldg(&vec[cols[j]]);
    }

    // Warp-level reduction of the per-lane partial sums
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, offset);
    }

    if (lane == 0) {
        out[row] = t;
    }
}

// ****************************************************************************
// Function: spmvGpu
//
// Purpose:
//   Launches the appropriate CSR SpMV kernel on device-resident data.
//
// Arguments:
//   d_val, d_cols, d_rowDelimiters, d_vec: device pointers to the CSR matrix
//                                          and dense vector
//   dim: number of rows/columns in the matrix
//   nnz: number of non-zero elements
//   d_out: device pointer to the output vector
//   stream: CUDA stream to launch on
//
// ****************************************************************************
void spmvGpu(const double* d_val, const index_t* d_cols, const index_t* d_rowDelimiters,
             const double* d_vec, const index_t dim, const index_t nnz, double* d_out,
             cudaStream_t stream) {
    if (dim == 0) return;
    const double avgNnzPerRow = static_cast<double>(nnz) / static_cast<double>(dim);
    if (avgNnzPerRow >= 8.0) {
        const unsigned grid = (dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        spmvCsrVectorKernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                             d_vec, dim, d_out);
    } else {
        const unsigned grid = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmvCsrScalarKernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                             d_vec, dim, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
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

    // Allocate device memory and upload the CSR matrix and dense vector
    printf("Uploading data to GPU...\n");
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    // cudaMalloc of zero bytes is legal but keep sizes non-zero-safe
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * (nItems ? nItems : 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * (nItems ? nItems : 1)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, sizeof(index_t) * (numRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * (numRows ? numRows : 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (numRows ? numRows : 1)));

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), sizeof(index_t) * (numRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double) * numRows));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up launch (not timed): absorbs module load / first-launch overhead
    spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %ld ms\n", static_cast<long>(durationMs));
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
    const double avgTime = durationMs / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * numRows, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    
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
