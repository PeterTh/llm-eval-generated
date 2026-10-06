#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

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
// Function: spmvKernel
//
// Purpose:
//   GPU CSR SpMV ("vector" variant). Each group of VEC consecutive threads
//   (VEC <= 32, power of two, within one warp) cooperatively processes one
//   row; partial sums are combined with warp shuffles. Rows are distributed
//   with a grid-stride loop.
//
// ****************************************************************************
template <int VEC>
__global__ void __launch_bounds__(256)
spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
           const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
           const index_t dim, double* __restrict__ out) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & (VEC - 1);
    const unsigned groupsPerGrid = (gridDim.x * blockDim.x) / VEC;
    for (unsigned row = tid / VEC; row < dim; row += groupsPerGrid) {
        const index_t rowStart = __ldg(&rowDelimiters[row]);
        const index_t rowEnd = __ldg(&rowDelimiters[row + 1]);
        double t0 = 0.0, t1 = 0.0;
        index_t j = rowStart + lane;
        for (; j + VEC < rowEnd; j += 2 * VEC) {
            t0 += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
            t1 += __ldg(&val[j + VEC]) * __ldg(&vec[__ldg(&cols[j + VEC])]);
        }
        if (j < rowEnd) {
            t0 += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
        }
        double t = t0 + t1;
        // All threads of a group share the same row, so the group-local mask is exact
        if constexpr (VEC > 1) {
            const unsigned mask = (VEC == 32) ? 0xffffffffu
                                              : (((1u << VEC) - 1u) << (threadIdx.x & 31u & ~(VEC - 1u)));
#pragma unroll
            for (int off = VEC / 2; off > 0; off >>= 1) {
                t += __shfl_down_sync(mask, t, off, VEC);
            }
        }
        if (lane == 0) {
            out[row] = t;
        }
    }
}

template <int VEC>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                       const double* vec, const index_t dim, double* out, int maxBlocks) {
    constexpr int threads = 256;
    const unsigned long long totalThreads = static_cast<unsigned long long>(dim) * VEC;
    unsigned long long blocks = (totalThreads + threads - 1) / threads;
    if (blocks > static_cast<unsigned long long>(maxBlocks)) blocks = maxBlocks;
    if (blocks == 0) return;
    spmvKernel<VEC><<<static_cast<unsigned>(blocks), threads>>>(val, cols, rowDelimiters, vec, dim, out);
}

// Selects threads-per-row based on the average row length and launches the kernel.
static void spmvGpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
                    const double* vec, const index_t dim, double* out, const double avgRowLen,
                    int maxBlocks) {
    if (avgRowLen <= 2.0)
        launchSpmv<1>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
    else if (avgRowLen <= 4.0)
        launchSpmv<2>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
    else if (avgRowLen <= 8.0)
        launchSpmv<4>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
    else if (avgRowLen <= 16.0)
        launchSpmv<8>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
    else if (avgRowLen <= 32.0)
        launchSpmv<16>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
    else
        launchSpmv<32>(val, cols, rowDelimiters, vec, dim, out, maxBlocks);
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

    // Allocate device memory and transfer inputs (setup, not timed)
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(1, sizeof(double) * nItems)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(1, sizeof(index_t) * nItems)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, sizeof(index_t) * (static_cast<size_t>(numRows) + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(1, sizeof(double) * numRows)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(1, sizeof(double) * numRows)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * nItems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          sizeof(index_t) * (static_cast<size_t>(numRows) + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    int device = 0, numSMs = 0, maxThreadsPerSM = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const int maxBlocks = std::max(1, numSMs * (maxThreadsPerSM / 256) * 8);
    const double avgRowLen = numRows > 0 ? static_cast<double>(nItems) / numRows : 0.0;

    // Warm-up launch (module load / clocks), not timed
    spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out, avgRowLen, maxBlocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out, avgRowLen, maxBlocks);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * numRows, cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
