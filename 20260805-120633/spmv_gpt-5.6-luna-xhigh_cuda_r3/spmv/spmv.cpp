#include <algorithm>
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

namespace {

void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

constexpr unsigned int CUDA_WARP_SIZE = 32;
constexpr unsigned int CUDA_THREADS_PER_BLOCK = 256;

// One warp owns one row.  Each lane walks a disjoint, strided portion of the
// row and the result is reduced with shuffle instructions.  This avoids global
// atomics, gives every row independent parallelism, and handles rows with very
// different lengths without a preprocessing pass.
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           const index_t dim,
                           double* __restrict__ out) {
    const unsigned int lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
    const unsigned int warpInGrid =
        (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_SIZE;
    const unsigned int warpsInGrid =
        (gridDim.x * blockDim.x) / CUDA_WARP_SIZE;

    // Use a grid-stride loop so the launch can be capped at a small multiple
    // of the SM count for large matrices without reducing occupancy.
    for (uint64_t row = warpInGrid; row < dim; row += warpsInGrid) {
        const index_t rowIndex = static_cast<index_t>(row);
        const index_t begin = rowDelimiters[rowIndex];
        const index_t end = rowDelimiters[rowIndex + 1];
        double sum = 0.0;

        for (index_t j = begin + lane; j < end; j += CUDA_WARP_SIZE) {
            sum += val[j] * vec[cols[j]];
        }

        // All 32 lanes participate, including lanes assigned no entries.
        #pragma unroll
        for (unsigned int offset = CUDA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            out[rowIndex] = sum;
        }
    }
}

template <typename T>
T* allocateDevice(const size_t count) {
    T* pointer = nullptr;
    if (count != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    }
    return pointer;
}

template <typename T>
void copyToDevice(T* destination, const std::vector<T>& source) {
    if (!source.empty()) {
        CUDA_CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
}

template <typename T>
void copyToHost(std::vector<T>& destination, const T* source) {
    if (!destination.empty()) {
        CUDA_CHECK(cudaMemcpy(destination.data(), source, destination.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
}

}  // namespace

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

    // Copy the immutable CSR input and dense vector to the GPU once.  The
    // benchmark loop then measures only repeated CUDA SpMV launches.
    double* d_val = allocateDevice<double>(h_val.size());
    index_t* d_cols = allocateDevice<index_t>(h_cols.size());
    index_t* d_rowDelimiters = allocateDevice<index_t>(h_rowDelimiters.size());
    double* d_vec = allocateDevice<double>(h_vec.size());
    double* d_out = allocateDevice<double>(h_out.size());

    copyToDevice(d_val, h_val);
    copyToDevice(d_cols, h_cols);
    copyToDevice(d_rowDelimiters, h_rowDelimiters);
    copyToDevice(d_vec, h_vec);
    if (!h_out.empty()) {
        // This also preserves the original zero-initialized output when
        // iterations is zero.
        CUDA_CHECK(cudaMemset(d_out, 0, h_out.size() * sizeof(double)));
    }

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    constexpr unsigned int warpsPerBlock = CUDA_THREADS_PER_BLOCK / CUDA_WARP_SIZE;
    const uint64_t blocksNeeded =
        (static_cast<uint64_t>(numRows) + warpsPerBlock - 1) / warpsPerBlock;
    const uint64_t residentBlockLimit =
        static_cast<uint64_t>(deviceProperties.multiProcessorCount) * 8;
    const unsigned int blockCount = static_cast<unsigned int>(
        std::max<uint64_t>(1, std::min(blocksNeeded, residentBlockLimit)));

    // Perform SpMV computation on the GPU.
    printf("Computing SpMV on CUDA GPU...\n");
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<blockCount, CUDA_THREADS_PER_BLOCK>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    copyToHost(h_out, d_out);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    const double durationMilliseconds = static_cast<double>(elapsedMilliseconds);
    printf("Computation time: %.3f ms\n", durationMilliseconds);

    // Calculate performance metrics.
    const double elapsedSeconds = durationMilliseconds / 1000.0;
    const double gflops = elapsedSeconds > 0.0
                              ? (2.0 * nItems * iterations) / elapsedSeconds / 1e9
                              : 0.0;
    const double avgTime = iterations != 0
                               ? durationMilliseconds / static_cast<double>(iterations)
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
