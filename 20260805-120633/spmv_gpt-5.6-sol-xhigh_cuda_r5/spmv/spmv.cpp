#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

// Report CUDA failures at their point of origin instead of silently producing
// invalid benchmark results.  There is deliberately no CPU fallback: the
// benchmark always executes its measured work on a CUDA device.
void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const size_t count = 0) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)) {}
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

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

// Each cooperative group owns one CSR row.  Selecting the group width from
// the matrix's average row length keeps short rows from wasting a full warp,
// while retaining coalesced matrix loads for longer rows.
template <int THREADS_PER_ROW>
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void spmvCsrVectorKernel(const double* __restrict__ val,
                        const index_t* __restrict__ cols,
                        const index_t* __restrict__ rowDelimiters,
                        const double* __restrict__ vec,
                        const index_t dim,
                        double* __restrict__ out) {
    static_assert(THREADS_PER_ROW == 1 || THREADS_PER_ROW == 2 ||
                  THREADS_PER_ROW == 4 || THREADS_PER_ROW == 8 ||
                  THREADS_PER_ROW == 16 || THREADS_PER_ROW == 32);

    const unsigned int lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const size_t globalThread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t rowNumber = globalThread / THREADS_PER_ROW;
    const bool validRow = rowNumber < dim;
    double sum = 0.0;

    if (validRow) {
        const index_t row = static_cast<index_t>(rowNumber);
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t element = rowStart + lane; element < rowEnd;
             element += THREADS_PER_ROW) {
            sum += val[element] * __ldg(vec + cols[element]);
        }
    }

#pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
    }

    if (validRow && lane == 0) {
        out[rowNumber] = sum;
    }
}

// Very long rows benefit from more than one warp.  This kernel assigns an
// entire block to a row, then performs a two-level warp/shared-memory
// reduction.  The matrix generated by this benchmark has balanced rows, so
// no atomics or load-balancing queues are needed.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void spmvCsrBlockKernel(const double* __restrict__ val,
                       const index_t* __restrict__ cols,
                       const index_t* __restrict__ rowDelimiters,
                       const double* __restrict__ vec,
                       const index_t dim,
                       double* __restrict__ out) {
    constexpr int WARPS_PER_BLOCK = CUDA_BLOCK_SIZE / 32;
    __shared__ double warpSums[WARPS_PER_BLOCK];

    const index_t row = blockIdx.x;
    if (row >= dim) {
        return;
    }

    double sum = 0.0;
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];
    for (index_t element = rowStart + threadIdx.x; element < rowEnd;
         element += blockDim.x) {
        sum += val[element] * __ldg(vec + cols[element]);
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        warpSums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < WARPS_PER_BLOCK ? warpSums[lane] : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

enum class SpmvKernel {
    Vector1,
    Vector2,
    Vector4,
    Vector8,
    Vector16,
    Vector32,
    Block
};

SpmvKernel chooseSpmvKernel(const index_t nonzeros, const index_t rows) {
    if (rows == 0) {
        return SpmvKernel::Vector1;
    }

    const double averageRowLength = static_cast<double>(nonzeros) / rows;
    if (averageRowLength <= 1.0) return SpmvKernel::Vector1;
    if (averageRowLength <= 2.0) return SpmvKernel::Vector2;
    if (averageRowLength <= 4.0) return SpmvKernel::Vector4;
    if (averageRowLength <= 8.0) return SpmvKernel::Vector8;
    if (averageRowLength <= 16.0) return SpmvKernel::Vector16;
    if (averageRowLength <= 2048.0) return SpmvKernel::Vector32;
    return SpmvKernel::Block;
}

template <int THREADS_PER_ROW>
void launchSpmvVector(const double* val, const index_t* cols,
                      const index_t* rowDelimiters, const double* vec,
                      const index_t dim, double* out) {
    constexpr unsigned int ROWS_PER_BLOCK = CUDA_BLOCK_SIZE / THREADS_PER_ROW;
    const unsigned int blocks = (dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    spmvCsrVectorKernel<THREADS_PER_ROW><<<blocks, CUDA_BLOCK_SIZE>>>(
        val, cols, rowDelimiters, vec, dim, out);
}

void launchSpmv(const SpmvKernel kernel, const double* val, const index_t* cols,
                const index_t* rowDelimiters, const double* vec,
                const index_t dim, double* out) {
    switch (kernel) {
        case SpmvKernel::Vector1:
            launchSpmvVector<1>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Vector2:
            launchSpmvVector<2>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Vector4:
            launchSpmvVector<4>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Vector8:
            launchSpmvVector<8>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Vector16:
            launchSpmvVector<16>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Vector32:
            launchSpmvVector<32>(val, cols, rowDelimiters, vec, dim, out);
            break;
        case SpmvKernel::Block:
            spmvCsrBlockKernel<<<dim, CUDA_BLOCK_SIZE>>>(
                val, cols, rowDelimiters, vec, dim, out);
            break;
    }
}

SpmvKernel tuneLongRowKernel(const SpmvKernel initialKernel,
                             const index_t nonzeros,
                             const double* val, const index_t* cols,
                             const index_t* rowDelimiters, const double* vec,
                             const index_t dim, double* out) {
    // The crossover between one warp and one block per row depends on both
    // row count and GPU generation.  A tiny one-time device-side calibration
    // avoids hard-coding a machine-specific threshold.  Short rows use the
    // deterministic group-width heuristic above because a block cannot win
    // enough work per row to offset its synchronization cost.
    if (dim == 0 || static_cast<double>(nonzeros) / dim < 256.0) {
        return initialKernel;
    }

    constexpr SpmvKernel CANDIDATES[] = {SpmvKernel::Vector32, SpmvKernel::Block};
    constexpr int CALIBRATION_ITERATIONS = 5;

    // Give both code paths the same cache/clock warm-up before comparing them.
    for (const SpmvKernel candidate : CANDIDATES) {
        launchSpmv(candidate, val, cols, rowDelimiters, vec, dim, out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent end;
    float bestTime = std::numeric_limits<float>::max();
    SpmvKernel bestKernel = initialKernel;

    for (const SpmvKernel candidate : CANDIDATES) {
        CUDA_CHECK(cudaEventRecord(start.get()));
        for (int iteration = 0; iteration < CALIBRATION_ITERATIONS; ++iteration) {
            launchSpmv(candidate, val, cols, rowDelimiters, vec, dim, out);
        }
        CUDA_CHECK(cudaEventRecord(end.get()));
        CUDA_CHECK(cudaEventSynchronize(end.get()));

        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), end.get()));
        if (elapsed < bestTime) {
            bestTime = elapsed;
            bestKernel = candidate;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return bestKernel;
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

    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero.\n");
        return 1;
    }

    // CSR uses 32-bit indices, so reject matrices whose nonzero array cannot
    // be represented instead of allowing integer wraparound.
    const uint64_t matrixElements = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = matrixElements / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The requested matrix has too many nonzero elements for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    const double sparsePercent = matrixElements == 0
                                     ? 100.0
                                     : 100.0 * (1.0 - static_cast<double>(nItems) /
                                                          static_cast<double>(matrixElements));
    printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, sparsePercent);
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(numRows) + 1);  // Row delimiters
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

    // Transfer invariant CSR data once.  All allocation, initialization, and
    // PCIe traffic remain outside the measured region, matching the original
    // benchmark's focus on SpMV compute throughput.
    DeviceBuffer<double> d_val(nItems);
    DeviceBuffer<index_t> d_cols(nItems);
    DeviceBuffer<index_t> d_rowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> d_vec(numRows);
    DeviceBuffer<double> d_out(numRows);

    if (nItems != 0) {
        CUDA_CHECK(cudaMemcpy(d_val.data(), h_val.data(),
                              static_cast<size_t>(nItems) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols.data(), h_cols.data(),
                              static_cast<size_t>(nItems) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters.data(), h_rowDelimiters.data(),
                          (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    if (numRows != 0) {
        CUDA_CHECK(cudaMemcpy(d_vec.data(), h_vec.data(),
                              static_cast<size_t>(numRows) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out.data(), 0,
                              static_cast<size_t>(numRows) * sizeof(double)));
    }

    int device = 0;
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    printf("CUDA device: %s\n", deviceProperties.name);

    SpmvKernel kernel = chooseSpmvKernel(nItems, numRows);
    if (iterations != 0) {
        kernel = tuneLongRowKernel(kernel, nItems, d_val.data(), d_cols.data(),
                                   d_rowDelimiters.data(), d_vec.data(),
                                   numRows, d_out.data());
    }

    // Warm the CUDA context and instruction/data caches before timing.  SpMV
    // is deterministic here, so this does not change the observable result.
    if (numRows != 0 && iterations != 0) {
        launchSpmv(kernel, d_val.data(), d_cols.data(), d_rowDelimiters.data(),
                   d_vec.data(), numRows, d_out.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    CudaEvent start;
    CudaEvent end;
    CUDA_CHECK(cudaEventRecord(start.get()));

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (numRows != 0) {
            launchSpmv(kernel, d_val.data(), d_cols.data(), d_rowDelimiters.data(),
                       d_vec.data(), numRows, d_out.data());
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end.get()));
    CUDA_CHECK(cudaEventSynchronize(end.get()));

    float durationMsFloat = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&durationMsFloat, start.get(), end.get()));
    const double durationMs = durationMsFloat;

    if (numRows != 0) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.data(),
                              static_cast<size_t>(numRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = durationMs > 0.0
                              ? (2.0 * nItems * iterations) / (durationMs * 1.0e6)
                              : 0.0;
    const double avgTime = iterations != 0
                               ? durationMs / static_cast<double>(iterations)
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
