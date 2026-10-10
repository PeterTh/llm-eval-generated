#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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
            const uint64_t numEntriesLeft = uint64_t{dim} * dim - (uint64_t{i} * dim + j);
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

// Each subgroup owns a row. Adjacent lanes read adjacent CSR entries, and
// shuffles reduce the double-precision partial sums without atomics.
template <unsigned ThreadsPerRow>
__global__ void spmvRowsKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t dim, double* __restrict__ out) {
    const unsigned lane = threadIdx.x & (ThreadsPerRow - 1);
    constexpr unsigned subgroupMask = 0xffffffffu >> (32 - ThreadsPerRow);
    const unsigned mask = subgroupMask << ((threadIdx.x & 31) & ~(ThreadsPerRow - 1));
    const size_t firstRow = size_t{blockIdx.x} * (blockDim.x / ThreadsPerRow)
                          + threadIdx.x / ThreadsPerRow;
    const size_t rowStride = size_t{gridDim.x} * (blockDim.x / ThreadsPerRow);
    for (size_t row = firstRow; row < dim; row += rowStride) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        double sum = 0.0;
        for (size_t j = size_t{begin} + lane; j < end; j += ThreadsPerRow) {
            sum += val[j] * vec[cols[j]];
        }
        #pragma unroll
        for (unsigned offset = ThreadsPerRow / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset, ThreadsPerRow);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

// Long rows use an entire block to expose enough parallel work even when the
// matrix has relatively few rows. The last reduction uses just one warp.
template <unsigned BlockSize>
__global__ void spmvLongRowsKernel(const double* __restrict__ val,
                                   const index_t* __restrict__ cols,
                                   const index_t* __restrict__ rowDelimiters,
                                   const double* __restrict__ vec,
                                   const index_t dim, double* __restrict__ out) {
    __shared__ double warpSums[BlockSize / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    for (size_t row = blockIdx.x; row < dim; row += gridDim.x) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        double sum = 0.0;
        for (size_t j = size_t{begin} + threadIdx.x; j < end; j += BlockSize) {
            sum += val[j] * vec[cols[j]];
        }
        #pragma unroll
        for (unsigned offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            sum = lane < BlockSize / 32 ? warpSums[lane] : 0.0;
            #pragma unroll
            for (unsigned offset = BlockSize / 64; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) {
                out[row] = sum;
            }
        }
        // Protect shared partial sums before this block advances to another row.
        if (row + gridDim.x < dim) {
            __syncthreads();
        }
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : bytes_(count * sizeof(T)) {
        if (bytes_ != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes_), "cudaMalloc");
        }
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }
    void upload(const T* source, cudaStream_t stream) {
        if (bytes_ != 0) {
            checkCuda(cudaMemcpyAsync(data_, source, bytes_, cudaMemcpyHostToDevice, stream),
                      "copy input to GPU");
        }
    }
    void download(T* destination, cudaStream_t stream) {
        if (bytes_ != 0) {
            checkCuda(cudaMemcpyAsync(destination, data_, bytes_, cudaMemcpyDeviceToHost, stream),
                      "copy output from GPU");
        }
    }
private:
    T* data_ = nullptr;
    size_t bytes_;
};

struct CudaStream {
    cudaStream_t handle = nullptr;
    CudaStream() { checkCuda(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking),
                            "create CUDA stream"); }
    ~CudaStream() { cudaStreamDestroy(handle); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

struct CudaEvent {
    cudaEvent_t handle = nullptr;
    CudaEvent() { checkCuda(cudaEventCreate(&handle), "create CUDA event"); }
    ~CudaEvent() { cudaEventDestroy(handle); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

struct CudaGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CudaGraph() = default;
    ~CudaGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
};

double spmvGpu(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& rowDelimiters, const std::vector<double>& vec,
               const index_t iterations, std::vector<double>& out) {
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "get CUDA device");
    checkCuda(cudaGetDeviceProperties(&properties, device), "query CUDA device");

    CudaStream stream;
    DeviceBuffer<double> d_val(val.size()), d_vec(vec.size()), d_out(out.size());
    DeviceBuffer<index_t> d_cols(cols.size()), d_rows(rowDelimiters.size());
    d_val.upload(val.data(), stream.handle);
    d_cols.upload(cols.data(), stream.handle);
    d_rows.upload(rowDelimiters.data(), stream.handle);
    d_vec.upload(vec.data(), stream.handle);

    const index_t rows = static_cast<index_t>(vec.size());
    const double averageLength = rows != 0 ? static_cast<double>(val.size()) / rows : 0.0;
    unsigned groupSize = averageLength <= 1 ? 1 : averageLength <= 2 ? 2
                       : averageLength <= 4 ? 4 : averageLength <= 8 ? 8
                       : averageLength <= 16 ? 16 : 32;
    if (averageLength >= 256 && rows <= unsigned(properties.multiProcessorCount) * 16) {
        groupSize = 128;
    }
    if (averageLength >= 2048) {
        groupSize = 256;
    }
    const unsigned blockSize = groupSize > 32 ? groupSize : 256;
    const size_t neededBlocks = (size_t{rows} + blockSize / groupSize - 1)
                             / (blockSize / groupSize);
    const unsigned blocks = static_cast<unsigned>(std::min(
        neededBlocks, size_t{static_cast<unsigned>(properties.multiProcessorCount)}
                    * (groupSize > 32 ? 16 : 8)));

    const auto launch = [&]() {
        if (rows == 0) return;
        switch (groupSize) {
        #define LAUNCH_ROWS(N) \
            case N: spmvRowsKernel<N><<<blocks, blockSize, 0, stream.handle>>>( \
                d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(), rows, d_out.data()); break
            LAUNCH_ROWS(1);
            LAUNCH_ROWS(2);
            LAUNCH_ROWS(4);
            LAUNCH_ROWS(8);
            LAUNCH_ROWS(16);
            LAUNCH_ROWS(32);
        #undef LAUNCH_ROWS
            case 128:
                spmvLongRowsKernel<128><<<blocks, 128, 0, stream.handle>>>(
                    d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(), rows, d_out.data());
                break;
            case 256:
                spmvLongRowsKernel<256><<<blocks, 256, 0, stream.handle>>>(
                    d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(), rows, d_out.data());
                break;
        }
        checkCuda(cudaGetLastError(), "launch SpMV kernel");
    };

    if (iterations != 0) {
        launch(); // Warm up before measuring; input transfers are also outside the timer.
    } else if (rows != 0) {
        checkCuda(cudaMemsetAsync(d_out.data(), 0, out.size() * sizeof(double), stream.handle),
                  "initialize output for zero iterations");
    }
    checkCuda(cudaStreamSynchronize(stream.handle), "prepare GPU inputs");

    // Replay a bounded chain of kernels to reduce launch overhead for short
    // matrices. Each requested iteration still performs a complete SpMV.
    CudaGraph graph;
    const index_t batch = std::min<index_t>(iterations, 128);
    if (rows != 0 && batch > 1) {
        checkCuda(cudaStreamBeginCapture(stream.handle, cudaStreamCaptureModeThreadLocal),
                  "begin CUDA graph capture");
        for (index_t i = 0; i < batch; ++i) launch();
        checkCuda(cudaStreamEndCapture(stream.handle, &graph.graph), "finish CUDA graph capture");
        checkCuda(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0),
                  "instantiate CUDA graph");
        checkCuda(cudaGraphUpload(graph.executable, stream.handle), "upload CUDA graph");
        checkCuda(cudaStreamSynchronize(stream.handle), "prepare CUDA graph");
    }

    CudaEvent start, stop;
    checkCuda(cudaEventRecord(start.handle, stream.handle), "record start time");
    index_t remaining = iterations;
    if (graph.executable) {
        while (remaining >= batch) {
            checkCuda(cudaGraphLaunch(graph.executable, stream.handle), "launch CUDA graph");
            remaining -= batch;
        }
    }
    if (rows != 0) {
        for (index_t i = 0; i < remaining; ++i) launch();
    }
    checkCuda(cudaEventRecord(stop.handle, stream.handle), "record stop time");
    checkCuda(cudaEventSynchronize(stop.handle), "wait for SpMV completion");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.handle, stop.handle), "measure SpMV time");
    d_out.download(out.data(), stream.handle);
    checkCuda(cudaStreamSynchronize(stream.handle), "wait for output transfer");
    return milliseconds;
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
    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero.\n");
        return 1;
    }
    const uint64_t totalEntries = uint64_t{numRows} * numRows;
    const uint64_t nonzeros = totalEntries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix has too many non-zero elements for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, totalEntries != 0 ? 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries) : 100.0);
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(size_t{numRows} + 1);  // Row delimiters
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

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    double milliseconds = 0.0;
    try {
        milliseconds = spmvGpu(h_val, h_cols, h_rowDelimiters, h_vec, iterations, h_out);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA SpMV failed: %s\n", error.what());
        return 1;
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double gflops = milliseconds > 0.0 ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
    const double avgTime = iterations != 0 ? milliseconds / iterations : 0.0;
    
    printf("Average time per iteration: %.6f ms\n", avgTime);
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
