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

// Keep all CUDA error handling in one place so asynchronous launch failures are
// reported at the operation which observes them instead of producing bad data.
void checkCuda(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t sizeBytes() const { return count_ * sizeof(T); }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
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

class CudaStream {
  public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(stream_); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_{};
};

class CudaGraph {
  public:
    explicit CudaGraph(cudaGraph_t graph) : graph_(graph) {
        CUDA_CHECK(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
    }

    ~CudaGraph() {
        cudaGraphExecDestroy(executable_);
        cudaGraphDestroy(graph_);
    }

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    void launch(const cudaStream_t stream) const {
        CUDA_CHECK(cudaGraphLaunch(executable_, stream));
    }

  private:
    cudaGraph_t graph_{};
    cudaGraphExec_t executable_{};
};

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

// Power-of-two cooperative groups avoid wasting a full warp on short rows.
// All groups in a hardware warp execute the same shuffle instructions, so a
// full active mask is valid while the width argument keeps reductions separate.
template <int THREADS_PER_ROW>
__global__ void spmvVectorKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 const index_t dim,
                                 double* __restrict__ out) {
    const int lane = threadIdx.x & (THREADS_PER_ROW - 1);
    index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_ROW;
    const index_t rowStride = (gridDim.x * blockDim.x) / THREADS_PER_ROW;

    for (; row < dim; row += rowStride) {
        double sum = 0.0;
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t element = rowStart + lane; element < rowEnd; element += THREADS_PER_ROW) {
            sum += val[element] * vec[cols[element]];
        }

#pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

// Very long rows need more parallelism than a single warp can expose. This
// kernel uses a full block per row and only eight shared-memory values during
// its two-level reduction.
__global__ void spmvBlockKernel(const double* __restrict__ val,
                                const index_t* __restrict__ cols,
                                const index_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vec,
                                const index_t dim,
                                double* __restrict__ out) {
    __shared__ double warpSums[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;

    for (index_t row = blockIdx.x; row < dim; row += gridDim.x) {
        double sum = 0.0;
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t element = rowDelimiters[row] + threadIdx.x;
             element < rowEnd;
             element += blockDim.x) {
            sum += val[element] * vec[cols[element]];
        }

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum;
        }
        __syncthreads();

        if (warp == 0) {
            sum = lane < WARPS_PER_BLOCK ? warpSums[lane] : 0.0;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) {
                out[row] = sum;
            }
        }
        __syncthreads();
    }
}

enum class SpmvKernel { Vector2, Vector4, Vector8, Vector16, Vector32, Block };

SpmvKernel chooseKernel(const index_t dim, const index_t nonzeros) {
    const double elementsPerRow = dim == 0 ? 0.0 : static_cast<double>(nonzeros) / dim;
    if (elementsPerRow <= 2.0) {
        return SpmvKernel::Vector2;
    }
    if (elementsPerRow <= 4.0) {
        return SpmvKernel::Vector4;
    }
    if (elementsPerRow <= 8.0) {
        return SpmvKernel::Vector8;
    }
    if (elementsPerRow <= 16.0) {
        return SpmvKernel::Vector16;
    }
    if (elementsPerRow <= 4096.0) {
        return SpmvKernel::Vector32;
    }
    return SpmvKernel::Block;
}

void launchSpmv(const SpmvKernel kernel,
                const double* const val,
                const index_t* const cols,
                const index_t* const rowDelimiters,
                const double* const vec,
                const index_t dim,
                double* const out,
                const int maxBlocks,
                const cudaStream_t stream) {
    if (dim == 0) {
        return;
    }

    switch (kernel) {
#define LAUNCH_VECTOR_KERNEL(vectorSize)                                                            \
    do {                                                                                             \
        constexpr int rowsPerBlock = THREADS_PER_BLOCK / (vectorSize);                              \
        const int rowBlocks = static_cast<int>((static_cast<uint64_t>(dim) + rowsPerBlock - 1) /    \
                                               rowsPerBlock);                                        \
        spmvVectorKernel<vectorSize>                                                                 \
            <<<rowBlocks < maxBlocks ? rowBlocks : maxBlocks, THREADS_PER_BLOCK, 0, stream>>>(       \
                val, cols, rowDelimiters, vec, dim, out);                                            \
    } while (false)
        case SpmvKernel::Vector2:
            LAUNCH_VECTOR_KERNEL(2);
            break;
        case SpmvKernel::Vector4:
            LAUNCH_VECTOR_KERNEL(4);
            break;
        case SpmvKernel::Vector8:
            LAUNCH_VECTOR_KERNEL(8);
            break;
        case SpmvKernel::Vector16:
            LAUNCH_VECTOR_KERNEL(16);
            break;
        case SpmvKernel::Vector32:
            LAUNCH_VECTOR_KERNEL(32);
            break;
#undef LAUNCH_VECTOR_KERNEL
        case SpmvKernel::Block: {
            const int rowBlocks = dim < static_cast<index_t>(maxBlocks) ? static_cast<int>(dim) : maxBlocks;
            spmvBlockKernel<<<rowBlocks, THREADS_PER_BLOCK, 0, stream>>>(
                val, cols, rowDelimiters, vec, dim, out);
            break;
        }
    }
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
        fprintf(stderr, "Matrix size, sparsity, and iteration count must all be greater than zero.\n");
        return 1;
    }

    // Compute this in 64 bits: the CSR representation still uses 32-bit
    // offsets, but the matrix dimension may exceed sqrt(UINT32_MAX).
    const uint64_t nItems64 = (static_cast<uint64_t>(numRows) * numRows) / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The requested matrix has too many non-zero elements for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                     (static_cast<double>(numRows) * numRows)));
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

    // Allocate device storage once and upload all read-only inputs before the
    // timed section. No host/device transfers occur between benchmark iterations.
    DeviceBuffer<double> d_val(nItems);
    DeviceBuffer<index_t> d_cols(nItems);
    DeviceBuffer<index_t> d_rowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> d_vec(numRows);
    DeviceBuffer<double> d_out(numRows);

    if (nItems != 0) {
        CUDA_CHECK(cudaMemcpy(d_val.data(), h_val.data(), d_val.sizeBytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols.data(), h_cols.data(), d_cols.sizeBytes(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters.data(), h_rowDelimiters.data(),
                          d_rowDelimiters.sizeBytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec.data(), h_vec.data(), d_vec.sizeBytes(), cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProperties{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    const int maxBlocks = deviceProperties.multiProcessorCount * 8;
    const SpmvKernel kernel = chooseKernel(numRows, nItems);
    CudaStream computeStream;

    // Warm up the CUDA context and selected kernel so one-time startup work is
    // not charged to the SpMV iterations.
    launchSpmv(kernel, d_val.data(), d_cols.data(), d_rowDelimiters.data(),
               d_vec.data(), numRows, d_out.data(), maxBlocks, computeStream.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(computeStream.get()));

    // Capture the complete iteration sequence once. Replaying it as a CUDA
    // Graph eliminates host launch gaps without changing the work performed.
    CUDA_CHECK(cudaStreamBeginCapture(computeStream.get(), cudaStreamCaptureModeGlobal));
    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(kernel, d_val.data(), d_cols.data(), d_rowDelimiters.data(),
                   d_vec.data(), numRows, d_out.data(), maxBlocks, computeStream.get());
    }
    cudaGraph_t capturedGraph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(computeStream.get(), &capturedGraph));
    CUDA_CHECK(cudaGetLastError());
    const CudaGraph iterationGraph(capturedGraph);

    // The first graph launch performs lazy driver setup; keep it out of the
    // measurement just like context and kernel warm-up.
    iterationGraph.launch(computeStream.get());
    CUDA_CHECK(cudaStreamSynchronize(computeStream.get()));

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    CudaEvent start;
    CudaEvent end;
    CUDA_CHECK(cudaEventRecord(start.get(), computeStream.get()));
    iterationGraph.launch(computeStream.get());
    CUDA_CHECK(cudaEventRecord(end.get(), computeStream.get()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(end.get()));
    float durationMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&durationMs, start.get(), end.get()));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.data(), d_out.sizeBytes(), cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
    // Calculate performance metrics
    const double durationSeconds = static_cast<double>(durationMs) / 1000.0;
    const double gflops = durationSeconds > 0.0
                              ? (2.0 * nItems * iterations) / durationSeconds / 1e9
                              : 0.0;
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
