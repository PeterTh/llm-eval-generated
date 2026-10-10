#include <algorithm>
#include <cmath>
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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        }
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

// Only warp sums enter shared memory; each output is written once, without atomics.
template <unsigned BlockSize>
__device__ void spmvBlockRow(const double* __restrict__ val,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rows,
                            const double* __restrict__ vec,
                            size_t row, double* __restrict__ out, double* warpSums) {
    const size_t begin = rows[row];
    const size_t end = rows[row + 1];
    double sum = 0.0;
    for (size_t j = begin + threadIdx.x; j < end; j += BlockSize) {
        sum += val[j] * __ldg(vec + cols[j]);
    }
    #pragma unroll
    for (unsigned offset = 16; offset != 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x % 32 == 0) warpSums[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < BlockSize / 32 ? warpSums[threadIdx.x] : 0.0;
        #pragma unroll
        for (unsigned offset = 16; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (threadIdx.x == 0) out[row] = sum;
    }
}

// Short rows share a warp; coalesced loads and cached vector reads avoid wasted
// bandwidth. Exceptionally long rows get a block in the same kernel launch, so
// they cannot serialize a small group or require another launch or atomic sums.
template <unsigned ThreadsPerRow>
__global__ void spmvRows(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rows,
                         const double* __restrict__ vec,
                         const index_t* __restrict__ longRows, unsigned numLongRows,
                         size_t dim, double* __restrict__ out) {
    __shared__ double warpSums[8];
    if (blockIdx.x < numLongRows) {
        spmvBlockRow<256>(val, cols, rows, vec, longRows[blockIdx.x], out, warpSums);
        return;
    }

    constexpr unsigned rowsPerBlock = 256 / ThreadsPerRow;
    const unsigned lane = threadIdx.x % ThreadsPerRow;
    const size_t stride = static_cast<size_t>(gridDim.x - numLongRows) * rowsPerBlock;
    for (size_t base = static_cast<size_t>(blockIdx.x - numLongRows) * rowsPerBlock;
         base < dim; base += stride) {
        const size_t row = base + threadIdx.x / ThreadsPerRow;
        size_t begin = 0, end = 0;
        bool shortRow = false;
        if (row < dim) {
            begin = rows[row];
            end = rows[row + 1];
            shortRow = end - begin <= ThreadsPerRow * 32;
        }
        // Every lane participates in the ballot, including padded/long rows.
        // This keeps complete groups in the shuffle mask on GPUs with
        // independent thread scheduling.
        const unsigned mask = __ballot_sync(0xffffffffu, shortRow);
        if (shortRow) {
            double sum = 0.0;
            for (size_t j = begin + lane; j < end; j += ThreadsPerRow) {
                sum += val[j] * __ldg(vec + cols[j]);
            }
            #pragma unroll
            for (unsigned offset = ThreadsPerRow / 2; offset != 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset, ThreadsPerRow);
            }
            if (lane == 0) out[row] = sum;
        }
    }
}

// Uniformly long rows use a block per row to expose more parallelism.
template <unsigned BlockSize>
__global__ void spmvLongRows(const double* __restrict__ val,
                             const index_t* __restrict__ cols,
                             const index_t* __restrict__ rows,
                             const double* __restrict__ vec,
                             size_t dim, double* __restrict__ out) {
    __shared__ double warpSums[BlockSize / 32];
    for (size_t row = blockIdx.x; row < dim; row += gridDim.x) {
        spmvBlockRow<BlockSize>(val, cols, rows, vec, row, out, warpSums);
        // Necessary when the grid-stride loop reuses shared memory.
        __syncthreads();
    }
}

class CudaSpmv {
public:
    CudaSpmv(const std::vector<double>& val, const std::vector<index_t>& cols,
             const std::vector<index_t>& rows, const std::vector<double>& vec)
        : val_(val.size()), cols_(cols.size()), rows_(rows.size()),
          vec_(vec.size()), out_(vec.size()), longRows_(vec.size()), dim_(vec.size()) {
        int device;
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));

        // Transfers and allocations occur once, outside the timed iterations.
        if (!val.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(val_.data(), val.data(), val.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(cols_.data(), cols.data(), cols.size() * sizeof(index_t),
                                       cudaMemcpyHostToDevice, stream_));
        }
        CUDA_CHECK(cudaMemcpyAsync(rows_.data(), rows.data(), rows.size() * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream_));
        if (dim_ != 0) {
            CUDA_CHECK(cudaMemcpyAsync(vec_.data(), vec.data(), vec.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemsetAsync(out_.data(), 0, dim_ * sizeof(double), stream_));
        }

        const double averageRowLength = dim_ != 0 ? static_cast<double>(val.size()) / dim_ : 0.0;
        threadsPerRow_ = 1;
        while (threadsPerRow_ < 16 && threadsPerRow_ * 2 <= averageRowLength) {
            threadsPerRow_ *= 2;
        }
        if (averageRowLength >= 128) threadsPerRow_ = 32;
        if (averageRowLength >= 512) threadsPerRow_ = 64;
        if (averageRowLength >= 1024) threadsPerRow_ = 128;
        if (averageRowLength >= 2048) threadsPerRow_ = 256;

        std::vector<index_t> longRows;
        if (threadsPerRow_ <= 32) {
            for (size_t row = 0; row < dim_; ++row) {
                if (rows[row + 1] - rows[row] > threadsPerRow_ * 32) {
                    longRows.push_back(static_cast<index_t>(row));
                }
            }
        }
        numLongRows_ = static_cast<unsigned>(longRows.size());
        if (numLongRows_ != 0) {
            CUDA_CHECK(cudaMemcpyAsync(longRows_.data(), longRows.data(), longRows.size() * sizeof(index_t),
                                       cudaMemcpyHostToDevice, stream_));
        }
        const size_t rowsPerBlock = threadsPerRow_ <= 32 ? 256 / threadsPerRow_ : 1;
        blocks_ = static_cast<unsigned>(std::min((dim_ + rowsPerBlock - 1) / rowsPerBlock,
                         static_cast<size_t>(properties.maxGridSize[0]) - numLongRows_)) + numLongRows_;
        // Complete uploads before the temporary host list is destroyed.
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~CudaSpmv() {
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
        cudaStreamDestroy(stream_);
    }
    CudaSpmv(const CudaSpmv&) = delete;
    CudaSpmv& operator=(const CudaSpmv&) = delete;

    double run(index_t iterations, std::vector<double>& out) {
        if (dim_ == 0) return 0.0;
        if (iterations == 0) {
            CUDA_CHECK(cudaMemcpyAsync(out.data(), out_.data(), dim_ * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            return 0.0;
        }

        // Warm up before timing, including any lazy loading of the CUDA kernels.
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // A bounded graph amortizes host launch overhead for repeated small SpMVs
        // without allocating a graph node for every benchmark iteration.
        const index_t batch = std::min<index_t>(iterations, 64);
        if (batch > 1) {
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            for (index_t i = 0; i < batch; ++i) launch();
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(graphExec_, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        CUDA_CHECK(cudaEventRecord(start_, stream_));
        if (graphExec_) {
            for (index_t i = 0; i < iterations / batch; ++i) {
                CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
            }
            for (index_t i = 0; i < iterations % batch; ++i) launch();
        } else {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));

        CUDA_CHECK(cudaMemcpyAsync(out.data(), out_.data(), dim_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return milliseconds;
    }

private:
    void launch() {
        #define LAUNCH_ROWS(width) \
            spmvRows<width><<<blocks_, 256, 0, stream_>>>( \
                val_.data(), cols_.data(), rows_.data(), vec_.data(), \
                longRows_.data(), numLongRows_, dim_, out_.data())
        switch (threadsPerRow_) {
            case 1: LAUNCH_ROWS(1); break;
            case 2: LAUNCH_ROWS(2); break;
            case 4: LAUNCH_ROWS(4); break;
            case 8: LAUNCH_ROWS(8); break;
            case 16: LAUNCH_ROWS(16); break;
            case 32: LAUNCH_ROWS(32); break;
            case 64:
                spmvLongRows<64><<<blocks_, 64, 0, stream_>>>(
                    val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data());
                break;
            case 128:
                spmvLongRows<128><<<blocks_, 128, 0, stream_>>>(
                    val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data());
                break;
            case 256:
                spmvLongRows<256><<<blocks_, 256, 0, stream_>>>(
                    val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data());
                break;
        }
        #undef LAUNCH_ROWS
    }

    DeviceBuffer<double> val_;
    DeviceBuffer<index_t> cols_;
    DeviceBuffer<index_t> rows_;
    DeviceBuffer<double> vec_;
    DeviceBuffer<double> out_;
    DeviceBuffer<index_t> longRows_;
    size_t dim_;
    unsigned numLongRows_;
    unsigned threadsPerRow_;
    unsigned blocks_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
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

    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be positive.\n");
        return 1;
    }
    // Compute sizes without overflowing the 32-bit CSR index type.
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros = matrixEntries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The matrix has too many nonzeros for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
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

    // All benchmark iterations execute on the GPU. The CPU routine above is
    // used only to generate an optional, independent validation reference.
    CudaSpmv gpu(h_val, h_cols, h_rowDelimiters, h_vec);
    printf("Computing SpMV...\n");
    const double milliseconds = gpu.run(iterations, h_out);

    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double gflops = milliseconds > 0.0 ? (2.0 * nItems * iterations) / milliseconds / 1e6 : 0.0;
    const double avgTime = iterations != 0 ? milliseconds / iterations : 0.0;
    
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
