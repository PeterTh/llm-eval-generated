#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

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

// CUDA failures are fatal for this benchmark: silently falling back to the CPU
// would make both its timing and its advertised execution mode misleading.
void checkCuda(cudaError_t status, const char* operation, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s at %s:%d: %s\n", operation, file, line,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

// A CSR row is handled by one logical CUDA vector.  Selecting a vector width
// close to the average row length preserves occupancy for very sparse inputs,
// while a full warp gives long rows coalesced reads and an efficient reduction.
template <unsigned THREADS_PER_ROW>
__global__ void spmvCudaKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               index_t numRows,
                               double* __restrict__ out) {
    constexpr unsigned THREADS_PER_BLOCK = 256;
    constexpr unsigned ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;

    const unsigned lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const unsigned subgroup = (threadIdx.x & 31u) / THREADS_PER_ROW;
    constexpr unsigned SUBGROUP_MASK = THREADS_PER_ROW == 32
                                           ? 0xffffffffu
                                           : ((1u << THREADS_PER_ROW) - 1u);
    const unsigned mask = SUBGROUP_MASK << (subgroup * THREADS_PER_ROW);
    index_t row = static_cast<index_t>(blockIdx.x * ROWS_PER_BLOCK +
                                       threadIdx.x / THREADS_PER_ROW);
    const index_t rowStride = static_cast<index_t>(gridDim.x * ROWS_PER_BLOCK);

    while (row < numRows) {
        double sum = 0.0;
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t item = rowDelimiters[row] + lane; item < rowEnd;
             item += THREADS_PER_ROW) {
            sum += val[item] * vec[cols[item]];
        }

#pragma unroll
        for (unsigned offset = THREADS_PER_ROW / 2; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset, THREADS_PER_ROW);
        }
        if (lane == 0) {
            out[row] = sum;
        }
        row += rowStride;
    }
}

template <unsigned THREADS_PER_ROW>
void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, index_t numRows, double* out, cudaStream_t stream = nullptr) {
    constexpr unsigned THREADS_PER_BLOCK = 256;
    constexpr unsigned ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
    const unsigned blocks = static_cast<unsigned>(
        (static_cast<uint64_t>(numRows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    spmvCudaKernel<THREADS_PER_ROW><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        val, cols, rowDelimiters, vec, numRows, out);
}

void launchSpmvAdaptive(const double* val, const index_t* cols,
                        const index_t* rowDelimiters, const double* vec,
                        index_t numRows, index_t nItems, double* out) {
    const uint64_t averageItemsPerRow =
        (static_cast<uint64_t>(nItems) + numRows - 1) / numRows;
    if (averageItemsPerRow <= 2) {
        launchSpmv<2>(val, cols, rowDelimiters, vec, numRows, out);
    } else if (averageItemsPerRow <= 4) {
        launchSpmv<4>(val, cols, rowDelimiters, vec, numRows, out);
    } else if (averageItemsPerRow <= 8) {
        launchSpmv<8>(val, cols, rowDelimiters, vec, numRows, out);
    } else if (averageItemsPerRow <= 16) {
        launchSpmv<16>(val, cols, rowDelimiters, vec, numRows, out);
    } else {
        launchSpmv<32>(val, cols, rowDelimiters, vec, numRows, out);
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
        std::fprintf(stderr, "Error: -n, -s, and -i must all be greater than zero.\n");
        return 1;
    }

    // Calculate number of non-zero elements
    const uint64_t matrixItems = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = matrixItems / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        std::fprintf(stderr, "Error: the requested matrix has too many non-zero elements.\n");
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

    // Allocate device data once and keep all transfers outside the measured region.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    const size_t storedItems = nItems == 0 ? 1 : nItems;
    CUDA_CHECK(cudaMalloc(&d_val, storedItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, storedItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(numRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(numRows) * sizeof(double)));
    if (nItems != 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Warm up the selected specialization before timing (also completes lazy CUDA setup/JIT).
    launchSpmvAdaptive(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation on the GPU. CUDA events measure device work rather
    // than host launch overhead and provide sub-millisecond resolution.
    printf("Computing SpMV...\n");
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmvAdaptive(d_val, d_cols, d_rowDelimiters, d_vec, numRows, nItems, d_out);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float durationMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&durationMs, start, end));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
    const double avgTime = durationMs / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(h_out, "OutputVector");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

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
