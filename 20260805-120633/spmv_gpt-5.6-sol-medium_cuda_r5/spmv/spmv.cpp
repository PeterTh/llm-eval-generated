#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

void cudaCheck(cudaError_t status, const char* call, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line, call,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

// One thread per row minimizes scheduling/reduction overhead for very short rows.
__global__ void spmvScalarKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 index_t dim, double* __restrict__ out) {
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < dim; row += blockDim.x * gridDim.x) {
        double sum = 0.0;
        const index_t end = rowDelimiters[row + 1];
        for (index_t j = rowDelimiters[row]; j < end; ++j)
            sum += val[j] * vec[cols[j]];
        out[row] = sum;
    }
}

// A power-of-two lane group cooperatively processes each row. Selecting the
// group width from the mean row length retains coalescing without wasting most
// lanes on sparse rows. Shuffle reduction needs no shared memory or barriers.
template <unsigned VECTOR_SIZE>
__global__ void spmvVectorKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 index_t dim, double* __restrict__ out) {
    const index_t lane = threadIdx.x & (VECTOR_SIZE - 1);
    const index_t vectorInBlock = threadIdx.x / VECTOR_SIZE;
    const index_t vectorsPerBlock = blockDim.x / VECTOR_SIZE;
    const unsigned groupBase = (threadIdx.x & 31u) & ~(VECTOR_SIZE - 1u);
    const unsigned groupMask = (0xffffffffu >> (32u - VECTOR_SIZE)) << groupBase;
    index_t row = blockIdx.x * vectorsPerBlock + vectorInBlock;
    const index_t rowStride = gridDim.x * vectorsPerBlock;

    for (; row < dim; row += rowStride) {
        double sum = 0.0;
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        for (index_t j = begin + lane; j < end; j += VECTOR_SIZE)
            sum += val[j] * vec[cols[j]];
        for (unsigned offset = VECTOR_SIZE / 2; offset != 0; offset /= 2)
            sum += __shfl_down_sync(groupMask, sum, offset, VECTOR_SIZE);
        if (lane == 0) out[row] = sum;
    }
}

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
        fprintf(stderr, "Error: matrix size, sparsity, and iteration count must be positive.\n");
        return 1;
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

    // Transfer the immutable CSR matrix and vector once. They remain resident on
    // the device for the entire benchmark, so the timing measures only SpMV.
    DeviceBuffer<double> d_val(nItems);
    DeviceBuffer<index_t> d_cols(nItems);
    DeviceBuffer<index_t> d_rowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> d_vec(numRows);
    DeviceBuffer<double> d_out(numRows);
    CUDA_CHECK(cudaMemcpy(d_val.get(), h_val.data(), nItems * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols.get(), h_cols.data(), nItems * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters.get(), h_rowDelimiters.data(),
                          (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec.get(), h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProperties{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    constexpr unsigned THREADS = 256;
    const unsigned maxBlocks = static_cast<unsigned>(deviceProperties.multiProcessorCount) * 8u;
    const double entriesPerRow = static_cast<double>(nItems) / numRows;
    const bool useScalarKernel = entriesPerRow <= 4.0;
    const unsigned vectorSize = entriesPerRow <= 8.0 ? 8u
                              : entriesPerRow <= 16.0 ? 16u : 32u;
    const unsigned wantedBlocks = useScalarKernel
        ? (numRows + THREADS - 1) / THREADS
        : (numRows + (THREADS / vectorSize) - 1) / (THREADS / vectorSize);
    const unsigned blocks = std::max(1u, std::min(wantedBlocks, maxBlocks));

    auto launchSpmv = [&]() {
        if (useScalarKernel) {
            spmvScalarKernel<<<blocks, THREADS>>>(d_val.get(), d_cols.get(),
                d_rowDelimiters.get(), d_vec.get(), numRows, d_out.get());
        } else if (vectorSize == 8) {
            spmvVectorKernel<8><<<blocks, THREADS>>>(d_val.get(), d_cols.get(),
                d_rowDelimiters.get(), d_vec.get(), numRows, d_out.get());
        } else if (vectorSize == 16) {
            spmvVectorKernel<16><<<blocks, THREADS>>>(d_val.get(), d_cols.get(),
                d_rowDelimiters.get(), d_vec.get(), numRows, d_out.get());
        } else {
            spmvVectorKernel<32><<<blocks, THREADS>>>(d_val.get(), d_cols.get(),
                d_rowDelimiters.get(), d_vec.get(), numRows, d_out.get());
        }
    };

    // Warm up CUDA context creation and caches outside the measured region.
    launchSpmv();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Perform SpMV computation.
    printf("Computing SpMV...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float durationMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&durationMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), numRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (durationMs * 1.0e6);
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
