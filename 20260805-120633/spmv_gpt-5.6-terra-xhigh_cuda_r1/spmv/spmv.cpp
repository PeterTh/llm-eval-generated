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
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = 8;

// CUDA errors are not recoverable for this benchmark: continuing would make a
// performance result or validation result meaningless.
[[noreturn]] void cudaFail(const cudaError_t error, const char* expression,
                           const char* file, const int line) {
    fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
            file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(const cudaError_t error, const char* expression,
               const char* file, const int line) {
    if (error != cudaSuccess) {
        cudaFail(error, expression, file, line);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    size_t count_;
    T* data_ = nullptr;
};

// One warp computes one CSR row.  Consecutive lanes load consecutive CSR
// entries, so values and column indices are coalesced.  The vector accesses
// are read-only and use the GPU read-only cache; a shuffle reduction avoids
// shared-memory traffic and synchronization.
__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec,
                              const index_t dim,
                              double* __restrict__ out) {
    const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    index_t row = static_cast<index_t>(blockIdx.x * WARPS_PER_BLOCK + warp);
    const index_t rowStride = static_cast<index_t>(gridDim.x * WARPS_PER_BLOCK);

    for (; row < dim; row += rowStride) {
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        double sum = 0.0;

        for (index_t entry = rowStart + lane; entry < rowEnd; entry += WARP_SIZE) {
            const index_t col = cols[entry];
            sum += val[entry] * __ldg(&vec[col]);
        }

        for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        if (lane == 0) {
            out[row] = sum;
        }
    }
}

unsigned int spmvGridSize(const index_t numRows) {
    if (numRows == 0) {
        return 0;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    const unsigned int rowsPerBlock = WARPS_PER_BLOCK;
    const unsigned int requiredBlocks =
        (numRows + rowsPerBlock - 1) / rowsPerBlock;
    // A bounded, oversubscribed grid preserves load balancing through the
    // grid-stride loop without creating needless blocks for very large inputs.
    const unsigned int residentTarget =
        static_cast<unsigned int>(properties.multiProcessorCount) * 32;
    return requiredBlocks < residentTarget ? requiredBlocks : residentTarget;
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

    // Copy the CSR matrix and dense vector to the GPU once.  The measured
    // interval contains only repeated SpMV kernels, matching the original
    // benchmark's compute-only timing.
    DeviceBuffer<double> d_val(nItems);
    DeviceBuffer<index_t> d_cols(nItems);
    DeviceBuffer<index_t> d_rowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> d_vec(numRows);
    DeviceBuffer<double> d_out(numRows);

    if (nItems != 0) {
        CUDA_CHECK(cudaMemcpy(d_val.get(), h_val.data(), nItems * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols.get(), h_cols.data(), nItems * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    if (numRows != 0) {
        CUDA_CHECK(cudaMemcpy(d_vec.get(), h_vec.data(), numRows * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters.get(), h_rowDelimiters.data(),
                          (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));

    // Perform SpMV computation on the GPU.
    printf("Computing SpMV...\n");
    const unsigned int gridSize = spmvGridSize(numRows);
    const dim3 blockSize(WARP_SIZE * WARPS_PER_BLOCK);

    // Exclude first-launch module loading and GPU clock ramp from the timed
    // region.  It writes the same output as a measured iteration.
    if (numRows != 0 && iterations != 0) {
        spmvCsrKernel<<<gridSize, blockSize>>>(d_val.get(), d_cols.get(),
                                               d_rowDelimiters.get(), d_vec.get(),
                                               numRows, d_out.get());
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    if (numRows != 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCsrKernel<<<gridSize, blockSize>>>(d_val.get(), d_cols.get(),
                                                   d_rowDelimiters.get(), d_vec.get(),
                                                   numRows, d_out.get());
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float durationMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&durationMs, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    if (numRows != 0 && iterations != 0) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), numRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / durationMs / 1e6;
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
