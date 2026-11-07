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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
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
//   (host reference implementation used for validation)
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
// Kernel: spmvCsrVectorKernel
//
// Purpose:
//   CSR "vector" SpMV: a group of THREADS_PER_ROW consecutive lanes of a warp
//   cooperates on one row. Lanes stride over the row's non-zeros so that
//   accesses to val/cols are coalesced, then the partial sums are combined
//   with warp shuffles. THREADS_PER_ROW is chosen on the host from the
//   average number of non-zeros per row so that short rows do not waste
//   lanes and long rows get full warps.
//
// ****************************************************************************
template <int THREADS_PER_ROW>
__global__ void __launch_bounds__(256)
spmvCsrVectorKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                    const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                    const index_t dim, double* __restrict__ out) {
    static_assert(THREADS_PER_ROW >= 1 && THREADS_PER_ROW <= 32 &&
                      (THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0,
                  "THREADS_PER_ROW must be a power of two <= warp size");

    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = tid & (THREADS_PER_ROW - 1);
    const unsigned row = tid / THREADS_PER_ROW;
    // Number of rows handled by the whole grid per "step" (grid-stride loop)
    const unsigned rowStride = (gridDim.x * blockDim.x) / THREADS_PER_ROW;

    for (unsigned r = row; r < dim; r += rowStride) {
        const index_t rowStart = __ldg(&rowDelimiters[r]);
        const index_t rowEnd = __ldg(&rowDelimiters[r + 1]);

        double t = 0.0;
        for (index_t j = rowStart + lane; j < rowEnd; j += THREADS_PER_ROW) {
            t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
        }

        // Reduce partial sums across the THREADS_PER_ROW lanes of this group
#pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, THREADS_PER_ROW);
        }

        if (lane == 0) {
            out[r] = t;
        }
    }
}

// ****************************************************************************
// Function: spmvGpu
//
// Purpose:
//   Launches the CSR SpMV kernel on device memory. Dispatches on the
//   number of lanes per row derived from the average row length.
//
// ****************************************************************************
void spmvGpu(const double* d_val, const index_t* d_cols, const index_t* d_rowDelimiters,
             const double* d_vec, const index_t dim, const index_t nnz, double* d_out,
             const int numSMs) {
    constexpr int BLOCK = 256;
    const double avgNnzPerRow = dim > 0 ? static_cast<double>(nnz) / dim : 0.0;

    // Pick lanes per row: smallest power of two >= average row length, capped at 32
    int threadsPerRow;
    if (avgNnzPerRow <= 2.0) threadsPerRow = 2;
    else if (avgNnzPerRow <= 4.0) threadsPerRow = 4;
    else if (avgNnzPerRow <= 8.0) threadsPerRow = 8;
    else if (avgNnzPerRow <= 16.0) threadsPerRow = 16;
    else threadsPerRow = 32;

    const unsigned long long totalThreads = static_cast<unsigned long long>(dim) * threadsPerRow;
    unsigned grid = static_cast<unsigned>((totalThreads + BLOCK - 1) / BLOCK);
    // Cap the grid so the grid-stride loop handles the tail; keep enough blocks to fill the GPU
    const unsigned maxGrid = static_cast<unsigned>(numSMs) * 32u;
    if (grid > maxGrid) grid = maxGrid;
    if (grid == 0) grid = 1;

    switch (threadsPerRow) {
        case 2:  spmvCsrVectorKernel<2><<<grid, BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out); break;
        case 4:  spmvCsrVectorKernel<4><<<grid, BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out); break;
        case 8:  spmvCsrVectorKernel<8><<<grid, BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out); break;
        case 16: spmvCsrVectorKernel<16><<<grid, BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out); break;
        default: spmvCsrVectorKernel<32><<<grid, BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out); break;
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

    // Set up the GPU and upload the matrix and vector once
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

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
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), (numRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Warm-up launch (module load, kernel JIT if needed) so the timed loop measures only SpMV
    spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out, prop.multiProcessorCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out,
                prop.multiProcessorCount);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Fetch the result from the device
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double), cudaMemcpyDeviceToHost));

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
