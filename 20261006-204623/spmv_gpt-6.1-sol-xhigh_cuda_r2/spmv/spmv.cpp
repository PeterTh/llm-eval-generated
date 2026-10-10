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

void checkCuda(cudaError_t status, const char* operation) {
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

// Cooperating lanes access consecutive CSR entries. Short rows use subwarps
// so that sparse matrices do not waste an entire warp on each row.
template <unsigned int Width>
__global__ void spmvRows(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rowDelimiters,
                         const double* __restrict__ vec, index_t dim,
                         double* __restrict__ out) {
    const unsigned int lane = threadIdx.x & (Width - 1);
    const size_t firstRow = (static_cast<size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x) / Width;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x / Width;
    for (size_t row = firstRow; row < dim; row += stride) {
        const unsigned int mask = __activemask();
        const size_t begin = rowDelimiters[row];
        const size_t end = rowDelimiters[row + 1];
        // Independent accumulators hide load and double-precision latency.
        double sum0 = 0.0;
        double sum1 = 0.0;
        for (size_t j = begin + lane; j < end; j += 2 * Width) {
            sum0 += val[j] * vec[cols[j]];
            if (j + Width < end) {
                sum1 += val[j + Width] * vec[cols[j + Width]];
            }
        }
        double sum = sum0 + sum1;
        for (unsigned int offset = Width / 2; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset, Width);
        }
        if (lane == 0) out[row] = sum;
    }
}

// Long rows distribute their work over a block and reduce without atomics.
template <unsigned int Threads>
__global__ void spmvLongRows(const double* __restrict__ val,
                             const index_t* __restrict__ cols,
                             const index_t* __restrict__ rowDelimiters,
                             const double* __restrict__ vec, index_t dim,
                             double* __restrict__ out) {
    __shared__ double warpSums[Threads / 32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    for (size_t row = blockIdx.x; row < dim; row += gridDim.x) {
        const size_t end = rowDelimiters[row + 1];
        double sum0 = 0.0;
        double sum1 = 0.0;
        for (size_t j = static_cast<size_t>(rowDelimiters[row]) + threadIdx.x;
             j < end; j += 2 * Threads) {
            sum0 += val[j] * vec[cols[j]];
            if (j + Threads < end) {
                sum1 += val[j + Threads] * vec[cols[j + Threads]];
            }
        }
        double sum = sum0 + sum1;
        for (unsigned int offset = 16; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) warpSums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < Threads / 32 ? warpSums[lane] : 0.0;
            for (unsigned int offset = 16; offset != 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) out[row] = sum;
        }
        __syncthreads();
    }
}

class CudaSpmv {
public:
    CudaSpmv(const std::vector<double>& val, const std::vector<index_t>& cols,
             const std::vector<index_t>& rows, const std::vector<double>& vec)
        : val_(val.size()), cols_(cols.size()), rows_(rows.size()),
          vec_(vec.size()), out_(vec.size()), dim_(static_cast<index_t>(vec.size())) {
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));

        const size_t averageRow = dim_ ? val.size() / dim_ : 0;
        width_ = averageRow <= 1 ? 1 : averageRow <= 2 ? 2 : averageRow <= 4 ? 4 :
                 averageRow <= 8 ? 8 : averageRow <= 16 ? 16 : 32;
        longRows_ = averageRow >= 512;
        const size_t blocks = longRows_ ? dim_ :
            (static_cast<size_t>(dim_) * width_ + threads_ - 1) / threads_;
        blocks_ = static_cast<unsigned int>(std::min(
            blocks, static_cast<size_t>(properties.multiProcessorCount) * 12));

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
            CUDA_CHECK(cudaMemsetAsync(out_.data(), 0, vec.size() * sizeof(double), stream_));
        }
    }

    ~CudaSpmv() {
        if (graph_) cudaGraphExecDestroy(graph_);
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
        cudaStreamDestroy(stream_);
    }
    CudaSpmv(const CudaSpmv&) = delete;
    CudaSpmv& operator=(const CudaSpmv&) = delete;

    double run(index_t iterations, std::vector<double>& result) {
        const index_t batch = std::min<index_t>(iterations, 256);
        if (dim_ != 0 && iterations != 0) {
            // Warm up outside the timed region, including all input transfers.
            launch();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (batch > 1) {
                cudaGraph_t captured = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                for (index_t i = 0; i < batch; ++i) launch();
                CUDA_CHECK(cudaStreamEndCapture(stream_, &captured));
                CUDA_CHECK(cudaGraphInstantiate(&graph_, captured, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(captured));
                CUDA_CHECK(cudaGraphUpload(graph_, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            }
        }

        CUDA_CHECK(cudaEventRecord(start_, stream_));
        if (dim_ != 0 && iterations != 0) {
            if (graph_) {
                for (index_t i = 0; i < iterations / batch; ++i) {
                    CUDA_CHECK(cudaGraphLaunch(graph_, stream_));
                }
                for (index_t i = 0; i < iterations % batch; ++i) launch();
            } else {
                launch();
            }
        }
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        if (dim_ != 0) {
            CUDA_CHECK(cudaMemcpyAsync(result.data(), out_.data(), result.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        return milliseconds;
    }

private:
    void launch() {
        if (longRows_) {
            spmvLongRows<threads_><<<blocks_, threads_, 0, stream_>>>(
                val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data());
        } else {
            switch (width_) {
#define LAUNCH_ROWS(width) \
    case width: spmvRows<width><<<blocks_, threads_, 0, stream_>>>( \
        val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data()); break
                LAUNCH_ROWS(1);
                LAUNCH_ROWS(2);
                LAUNCH_ROWS(4);
                LAUNCH_ROWS(8);
                LAUNCH_ROWS(16);
                LAUNCH_ROWS(32);
#undef LAUNCH_ROWS
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static constexpr unsigned int threads_ = 128;
    DeviceBuffer<double> val_;
    DeviceBuffer<index_t> cols_;
    DeviceBuffer<index_t> rows_;
    DeviceBuffer<double> vec_;
    DeviceBuffer<double> out_;
    index_t dim_;
    unsigned int width_ = 32;
    unsigned int blocks_ = 0;
    bool longRows_ = false;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    cudaGraphExec_t graph_ = nullptr;
};

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
    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros = matrixSize / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix has too many non-zero elements for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, matrixSize ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize) : 100.0);
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

    // Perform SpMV computation
    CudaSpmv gpu(h_val, h_cols, h_rowDelimiters, h_vec);
    printf("Computing SpMV...\n");
    const double milliseconds = gpu.run(iterations, h_out);

    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double gflops = milliseconds > 0.0 ?
        (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
    const double avgTime = iterations ? milliseconds / iterations : 0.0;
    
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
