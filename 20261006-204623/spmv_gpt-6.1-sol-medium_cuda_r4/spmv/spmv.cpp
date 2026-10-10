#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

// All benchmark multiplications run on the GPU; the CPU routine above is only
// used to construct the optional reference result.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Short rows share a warp. Each subgroup owns one row, so there are no atomics
// or inter-block synchronization, and adjacent lanes read adjacent CSR entries.
template <unsigned Width>
__global__ void spmvRows(const double* __restrict__ values,
                         const index_t* __restrict__ columns,
                         const index_t* __restrict__ offsets,
                         const double* __restrict__ vector,
                         index_t rows, double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    const unsigned warpBase = ((blockIdx.x * blockDim.x + threadIdx.x) / 32) * (32 / Width);
    const unsigned rowInWarp = (threadIdx.x % 32) / Width;
    const unsigned stride = gridDim.x * blockDim.x / Width;
    // All lanes in a warp reach the ballot, including those beyond the final
    // row. The mask is fixed before the variable-length accumulation loops.
    for (uint64_t base = warpBase; base < rows; base += stride) {
        const uint64_t row = base + rowInWarp;
        const unsigned mask = __ballot_sync(0xffffffffu, row < rows);
        if (row >= rows) continue;
        const index_t begin = offsets[row];
        const index_t end = offsets[row + 1];
        double sum = 0.0;
        // A 64-bit loop counter also handles CSR offsets near UINT32_MAX.
        for (uint64_t j = uint64_t(begin) + lane; j < end; j += Width) {
            sum += values[j] * __ldg(vector + columns[j]);
        }
        for (unsigned delta = Width / 2; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(mask, sum, delta, Width);
        }
        if (lane == 0) output[row] = sum;
    }
}

// Very long rows use a whole block to expose more parallelism even when the
// matrix has relatively few rows. Warp partial sums are combined in shared memory.
__global__ void spmvLongRows(const double* __restrict__ values,
                             const index_t* __restrict__ columns,
                             const index_t* __restrict__ offsets,
                             const double* __restrict__ vector,
                             index_t rows, double* __restrict__ output) {
    __shared__ double partial[4];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    for (index_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0;
        const index_t end = offsets[row + 1];
        for (uint64_t j = uint64_t(offsets[row]) + threadIdx.x; j < end; j += 128) {
            sum += values[j] * __ldg(vector + columns[j]);
        }
        for (unsigned delta = 16; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        }
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 4 ? partial[lane] : 0.0;
            for (unsigned delta = 2; delta > 0; delta /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, delta);
            }
            if (lane == 0) output[row] = sum;
        }
        __syncthreads();
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(T)), "allocation");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
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

    if (numRows == 0 || sparsity == 0) {
        fprintf(stderr, "Matrix size and sparsity must be positive.\n");
        return 1;
    }
    // CSR offsets are 32-bit, but the square can exceed that range.
    const uint64_t matrixEntries = uint64_t(numRows) * numRows;
    if (matrixEntries / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Number of non-zero elements exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixEntries / sparsity);

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

    // Allocate and upload once. Inputs and intermediate results remain resident
    // for all iterations; only the final output is downloaded.
    checkCuda(cudaSetDevice(0), "device selection");
    cudaDeviceProp properties;
    checkCuda(cudaGetDeviceProperties(&properties, 0), "device properties");
    DeviceBuffer<double> d_val(nItems), d_vec(numRows), d_out(numRows);
    DeviceBuffer<index_t> d_cols(nItems), d_offsets(size_t(numRows) + 1);
    checkCuda(cudaMemcpy(d_val.data, h_val.data(), size_t(nItems) * sizeof(double),
                         cudaMemcpyHostToDevice), "matrix upload");
    checkCuda(cudaMemcpy(d_cols.data, h_cols.data(), size_t(nItems) * sizeof(index_t),
                         cudaMemcpyHostToDevice), "columns upload");
    checkCuda(cudaMemcpy(d_offsets.data, h_rowDelimiters.data(),
                         (size_t(numRows) + 1) * sizeof(index_t),
                         cudaMemcpyHostToDevice), "row offsets upload");
    checkCuda(cudaMemcpy(d_vec.data, h_vec.data(), size_t(numRows) * sizeof(double),
                         cudaMemcpyHostToDevice), "vector upload");

    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation");
    const double entriesPerRow = double(nItems) / numRows;
    const unsigned width = entriesPerRow <= 2 ? 2 : entriesPerRow <= 4 ? 4 :
                     entriesPerRow <= 8 ? 8 : entriesPerRow <= 16 ? 16 : 32;
    const bool longRows = entriesPerRow >= 2048;
    const unsigned groupsPerBlock = 128 / width;
    const unsigned blocks = std::min<unsigned>(
        longRows ? numRows : (uint64_t(numRows) + groupsPerBlock - 1) / groupsPerBlock,
        properties.multiProcessorCount * 16);
    auto launch = [&]() {
        if (longRows) {
            spmvLongRows<<<blocks, 128, 0, stream>>>(d_val.data, d_cols.data, d_offsets.data,
                                                   d_vec.data, numRows, d_out.data);
        } else {
            switch (width) {
#define LAUNCH_ROWS(W) \
    case W: spmvRows<W><<<blocks, 128, 0, stream>>>(d_val.data, d_cols.data, d_offsets.data, \
                                                  d_vec.data, numRows, d_out.data); break
                LAUNCH_ROWS(2);
                LAUNCH_ROWS(4);
                LAUNCH_ROWS(8);
                LAUNCH_ROWS(16);
                LAUNCH_ROWS(32);
#undef LAUNCH_ROWS
            }
        }
    };

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    // Batched graph replay removes most CPU launch overhead for repeated small
    // SpMVs without constructing a graph proportional to the iteration count.
    const index_t batchSize = std::min<index_t>(iterations, 256);
    if (iterations >= 32) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "graph capture");
        for (index_t i = 0; i < batchSize; ++i) launch();
        checkCuda(cudaStreamEndCapture(stream, &graph), "graph capture completion");
        checkCuda(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0),
                  "graph instantiation");
        checkCuda(cudaGraphUpload(executableGraph, stream), "graph upload");
    }
    if (iterations != 0) {
        launch(); // Warm up kernel and caches before measuring.
        checkCuda(cudaGetLastError(), "warmup launch");
    } else {
        // Match the original zero-iteration, zero-initialized output.
        checkCuda(cudaMemsetAsync(d_out.data, 0, size_t(numRows) * sizeof(double), stream),
                  "output initialization");
    }
    checkCuda(cudaStreamSynchronize(stream), "warmup completion");

    printf("Computing SpMV...\n");
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "start event creation");
    checkCuda(cudaEventCreate(&end), "end event creation");
    checkCuda(cudaEventRecord(start, stream), "start event recording");
    index_t remaining = iterations;
    if (executableGraph) {
        while (remaining >= batchSize) {
            checkCuda(cudaGraphLaunch(executableGraph, stream), "graph launch");
            remaining -= batchSize;
        }
    }
    for (index_t i = 0; i < remaining; ++i) launch();
    checkCuda(cudaGetLastError(), "SpMV launch");
    checkCuda(cudaEventRecord(end, stream), "end event recording");
    checkCuda(cudaEventSynchronize(end), "SpMV completion");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "elapsed time");
    checkCuda(cudaMemcpy(h_out.data(), d_out.data, size_t(numRows) * sizeof(double),
                         cudaMemcpyDeviceToHost), "result download");

    checkCuda(cudaEventDestroy(start), "start event destruction");
    checkCuda(cudaEventDestroy(end), "end event destruction");
    if (executableGraph) checkCuda(cudaGraphExecDestroy(executableGraph), "graph executable destruction");
    if (graph) checkCuda(cudaGraphDestroy(graph), "graph destruction");
    checkCuda(cudaStreamDestroy(stream), "stream destruction");

    printf("Computation time: %.3f ms\n", double(elapsedMs));
    const double gflops = elapsedMs > 0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0.0;
    const double avgTime = iterations ? double(elapsedMs) / iterations : 0.0;
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
