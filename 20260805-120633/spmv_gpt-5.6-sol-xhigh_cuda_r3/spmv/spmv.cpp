#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

void checkCuda(const cudaError_t error, const char* const operation,
               const char* const file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                     operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

template <typename T>
T* allocateDevice(const size_t count) {
    T* pointer = nullptr;
    // cudaMalloc does not accept a zero-byte allocation. Empty matrices still
    // need a valid (but never dereferenced) pointer for kernel arguments.
    const size_t allocationCount = count == 0 ? 1 : count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                          allocationCount * sizeof(T)));
    return pointer;
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
            const uint64_t numEntriesLeft =
                static_cast<uint64_t>(dim) * dim -
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

// A logical group of 1--32 adjacent threads cooperatively processes one CSR
// row. Adjacent threads read adjacent values and column indices, while the
// read-only vector uses the GPU's read-only cache. Choosing the group width at
// run time avoids wasting a full warp on very short rows without sacrificing
// coalescing or scalability for long rows.
template <unsigned int GROUP_SIZE>
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void spmvCsrKernel(const double* __restrict__ val,
                   const index_t* __restrict__ cols,
                   const index_t* __restrict__ rowDelimiters,
                   const double* __restrict__ vec, const index_t dim,
                   double* __restrict__ out) {
    static_assert(GROUP_SIZE >= 1 && GROUP_SIZE <= 32 &&
                  (GROUP_SIZE & (GROUP_SIZE - 1)) == 0);

    constexpr unsigned int groupsPerBlock = CUDA_BLOCK_SIZE / GROUP_SIZE;
    const unsigned int lane = threadIdx.x & (GROUP_SIZE - 1);
    const index_t firstRow =
        blockIdx.x * groupsPerBlock + threadIdx.x / GROUP_SIZE;
    const index_t rowStride = gridDim.x * groupsPerBlock;

    for (index_t row = firstRow; row < dim; row += rowStride) {
        const index_t rowBegin = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        double sum = 0.0;

        for (index_t item = rowBegin + lane; item < rowEnd;
             item += GROUP_SIZE) {
            const index_t column = cols[item];
            sum = fma(val[item], __ldg(vec + column), sum);
        }

        const unsigned int active = __activemask();
#pragma unroll
        for (unsigned int offset = GROUP_SIZE / 2; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(active, sum, offset, GROUP_SIZE);
        }

        if (lane == 0) {
            out[row] = sum;
        }
    }
}

template <unsigned int GROUP_SIZE>
int kernelGridSize(const index_t rows, const int multiprocessors) {
    constexpr int groupsPerBlock = CUDA_BLOCK_SIZE / GROUP_SIZE;
    const uint64_t blocksForRows =
        (static_cast<uint64_t>(rows) + groupsPerBlock - 1) / groupsPerBlock;

    int activeBlocksPerMultiprocessor = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerMultiprocessor, spmvCsrKernel<GROUP_SIZE>,
        CUDA_BLOCK_SIZE, 0));
    const uint64_t residentBlocks =
        static_cast<uint64_t>(activeBlocksPerMultiprocessor) * multiprocessors;
    const uint64_t grid = blocksForRows < residentBlocks
                              ? blocksForRows
                              : residentBlocks;
    return static_cast<int>(grid == 0 ? 1 : grid);
}

template <unsigned int GROUP_SIZE>
void launchSpmvKernel(const double* val, const index_t* cols,
                      const index_t* rowDelimiters, const double* vec,
                      const index_t rows, double* out, const int blocks) {
    spmvCsrKernel<GROUP_SIZE><<<blocks, CUDA_BLOCK_SIZE>>>(
        val, cols, rowDelimiters, vec, rows, out);
}

struct SpmvLaunchConfig {
    unsigned int groupSize;
    int blocks;
};

SpmvLaunchConfig makeSpmvLaunchConfig(const index_t rows,
                                      const index_t nonzeros,
                                      const int multiprocessors) {
    const uint64_t averageItemsPerRow =
        (static_cast<uint64_t>(nonzeros) + rows - 1) / rows;

    if (averageItemsPerRow <= 1) {
        return {1, kernelGridSize<1>(rows, multiprocessors)};
    }
    if (averageItemsPerRow <= 2) {
        return {2, kernelGridSize<2>(rows, multiprocessors)};
    }
    if (averageItemsPerRow <= 4) {
        return {4, kernelGridSize<4>(rows, multiprocessors)};
    }
    // Slightly overlap the nominal width ranges: for moderately ragged rows,
    // a smaller group wastes fewer lanes and outperforms rounding directly up
    // to the next power of two.
    if (averageItemsPerRow <= 13) {
        return {8, kernelGridSize<8>(rows, multiprocessors)};
    }
    if (averageItemsPerRow <= 52) {
        return {16, kernelGridSize<16>(rows, multiprocessors)};
    }
    return {32, kernelGridSize<32>(rows, multiprocessors)};
}

void launchSpmv(const double* val, const index_t* cols,
                const index_t* rowDelimiters, const double* vec,
                const index_t rows, double* out,
                const SpmvLaunchConfig config) {
    switch (config.groupSize) {
        case 1:
            launchSpmvKernel<1>(val, cols, rowDelimiters, vec, rows, out,
                                config.blocks);
            break;
        case 2:
            launchSpmvKernel<2>(val, cols, rowDelimiters, vec, rows, out,
                                config.blocks);
            break;
        case 4:
            launchSpmvKernel<4>(val, cols, rowDelimiters, vec, rows, out,
                                config.blocks);
            break;
        case 8:
            launchSpmvKernel<8>(val, cols, rowDelimiters, vec, rows, out,
                                config.blocks);
            break;
        case 16:
            launchSpmvKernel<16>(val, cols, rowDelimiters, vec, rows, out,
                                 config.blocks);
            break;
        default:
            launchSpmvKernel<32>(val, cols, rowDelimiters, vec, rows, out,
                                 config.blocks);
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

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        std::fprintf(stderr, "-n, -s, and -i must all be greater than zero\n");
        return 1;
    }

    // Calculate the number of non-zero elements without overflowing the CSR
    // index type. The original matrix generator stores offsets as uint32_t.
    const uint64_t matrixItems = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = matrixItems / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        std::fprintf(stderr,
                     "Matrix has too many non-zero elements for 32-bit CSR indices\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                      static_cast<double>(matrixItems)));
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

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    std::printf("CUDA device: %s\n", deviceProperties.name);

    double* d_val = allocateDevice<double>(nItems);
    index_t* d_cols = allocateDevice<index_t>(nItems);
    index_t* d_rowDelimiters = allocateDevice<index_t>(numRows + 1);
    double* d_vec = allocateDevice<double>(numRows);
    double* d_out = allocateDevice<double>(numRows);

    if (nItems != 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(),
                              static_cast<size_t>(nItems) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(),
                              static_cast<size_t>(nItems) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          static_cast<size_t>(numRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(),
                          static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));

    const SpmvLaunchConfig launchConfig = makeSpmvLaunchConfig(
        numRows, nItems, deviceProperties.multiProcessorCount);

    // Perform one untimed launch so context setup and lazy module loading do
    // not pollute the benchmark. All benchmark iterations execute on CUDA.
    printf("Computing SpMV...\n");
    launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out,
               launchConfig);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out,
                   launchConfig);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float durationMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&durationMs, start, end));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
    // Calculate performance metrics
    const double gflops =
        (2.0 * nItems * iterations) / (static_cast<double>(durationMs) * 1e6);
    const double avgTime = durationMs / static_cast<double>(iterations);
    
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
