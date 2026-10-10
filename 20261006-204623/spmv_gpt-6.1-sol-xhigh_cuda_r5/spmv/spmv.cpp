#include <algorithm>
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
            const uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

void checkCuda(cudaError_t status, const char* operation, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                file, line, operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        }
    }

    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

// A subgroup processes each CSR row with coalesced matrix loads. Short rows use
// subwarps; long rows use multiple warps to expose more memory parallelism.
template <int ThreadsPerRow>
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           index_t dim, double* __restrict__ out) {
    constexpr int BlockSize = 256;
    const size_t row = (size_t(blockIdx.x) * BlockSize + threadIdx.x) / ThreadsPerRow;
    const unsigned lane = threadIdx.x % ThreadsPerRow;
    double sum = 0.0;
    if (row < dim) {
        const size_t begin = rowDelimiters[row];
        const size_t end = rowDelimiters[row + 1];
        for (size_t entry = begin + lane; entry < end; entry += ThreadsPerRow) {
            sum += val[entry] * __ldg(vec + cols[entry]);
        }
    }

    if constexpr (ThreadsPerRow <= 32) {
        if constexpr (ThreadsPerRow > 1) {
            #pragma unroll
            for (int offset = ThreadsPerRow / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset, ThreadsPerRow);
            }
        }
        if (lane == 0 && row < dim) out[row] = sum;
    } else {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        __shared__ double warpSums[BlockSize / 32];
        if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x / 32] = sum;
        // Invalid rows in the last block also participate in this barrier.
        __syncthreads();
        if (lane == 0 && row < dim) {
            sum = 0.0;
            #pragma unroll
            for (int warp = 0; warp < ThreadsPerRow / 32; ++warp) {
                sum += warpSums[threadIdx.x / 32 + warp];
            }
            out[row] = sum;
        }
    }
}

void launchSpmv(const double* val, const index_t* cols, const index_t* rows,
                const double* vec, index_t dim, double* out,
                int threadsPerRow, cudaStream_t stream) {
    constexpr unsigned BlockSize = 256;
    const unsigned blocks = static_cast<unsigned>(
        (size_t(dim) * threadsPerRow + BlockSize - 1) / BlockSize);
    switch (threadsPerRow) {
        case 1: spmvKernel<1><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 2: spmvKernel<2><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 4: spmvKernel<4><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 8: spmvKernel<8><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 16: spmvKernel<16><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 32: spmvKernel<32><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 128: spmvKernel<128><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
        case 256: spmvKernel<256><<<blocks, BlockSize, 0, stream>>>(val, cols, rows, vec, dim, out); break;
    }
}

double spmvGpu(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& rows, const std::vector<double>& vec,
               index_t iterations, std::vector<double>& out) {
    const index_t dim = static_cast<index_t>(vec.size());
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceBuffer<double> d_val(val.size()), d_vec(vec.size()), d_out(out.size());
    DeviceBuffer<index_t> d_cols(cols.size()), d_rows(rows.size());

    if (!val.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(d_val.data(), val.data(), val.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cols.data(), cols.data(), cols.size() * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_rows.data(), rows.data(), rows.size() * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    if (dim != 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_vec.data(), vec.data(), vec.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(d_out.data(), 0, out.size() * sizeof(double), stream));
    }

    // Match subwarps to the row length, increasing the group size when needed
    // to give every multiprocessor a block without wasting lanes on short rows.
    const double averageRowLength = dim != 0 ? double(val.size()) / dim : 0.0;
    int threadsPerRow = 1;
    while (threadsPerRow < 32 && 2 * threadsPerRow <= averageRowLength) threadsPerRow *= 2;
    if (dim != 0 && averageRowLength >= 1.0) {
        const size_t threadsForOccupancy =
            (size_t(properties.multiProcessorCount) * 256 + dim - 1) / dim;
        while (threadsPerRow < 32 && size_t(threadsPerRow) < threadsForOccupancy &&
               threadsPerRow < averageRowLength) {
            threadsPerRow *= 2;
        }
    }
    // Larger groups improve bandwidth on long rows when the matrix is streamed
    // from device memory. Smaller matrices favor warp-only reductions.
    const size_t matrixBytes = val.size() * (sizeof(double) + sizeof(index_t));
    if (averageRowLength >= 512.0 && matrixBytes > 8 * size_t(properties.l2CacheSize)) {
        threadsPerRow = averageRowLength >= 2048.0 ? 256 : 128;
    }

    const auto launch = [&]() {
        launchSpmv(d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(),
                   dim, d_out.data(), threadsPerRow, stream);
    };

    // Warm up the selected kernel and complete input transfers before timing.
    if (dim != 0 && iterations != 0) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr, tailGraph = nullptr;
    cudaGraphExec_t executable = nullptr, tailExecutable = nullptr;
    constexpr index_t MaxGraphIterations = 64;
    const index_t batch = std::min(iterations, MaxGraphIterations);
    const index_t tail = batch != 0 ? iterations % batch : 0;
    const auto capture = [&](index_t count, cudaGraph_t& captured, cudaGraphExec_t& exec) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < count; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(exec, stream));
    };
    if (dim != 0 && iterations > 1) {
        capture(batch, graph, executable);
        if (tail != 0) capture(tail, tailGraph, tailExecutable);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (dim != 0 && iterations != 0) {
        if (executable != nullptr) {
            for (index_t i = 0; i < iterations / batch; ++i) {
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
            }
            if (tailExecutable != nullptr) CUDA_CHECK(cudaGraphLaunch(tailExecutable, stream));
        } else {
            launch();
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (dim != 0) {
        CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out.data(), out.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    if (tailExecutable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(tailExecutable));
    if (tailGraph != nullptr) CUDA_CHECK(cudaGraphDestroy(tailGraph));
    if (executable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
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
    const uint64_t matrixElements = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = matrixElements / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The matrix exceeds the 32-bit CSR index capacity.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, matrixElements != 0 ? 100.0 * (1.0 - double(nItems) / matrixElements) : 100.0);
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(size_t(numRows) + 1);  // Row delimiters
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
    const double milliseconds = spmvGpu(h_val, h_cols, h_rowDelimiters, h_vec, iterations, h_out);

    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double gflops = iterations != 0 && milliseconds > 0.0
        ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
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
