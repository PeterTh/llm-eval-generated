#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned CUDA_BLOCK_SIZE = 256;

// Fail immediately on CUDA errors: this benchmark is intentionally GPU-only
// and must never silently fall back to a serial implementation.
void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                file, line, expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

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

    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t bytes() const { return count_ * sizeof(T); }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
void copyToDevice(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    if (!source.empty()) {
        CUDA_CHECK(cudaMemcpy(destination.get(), source.data(), destination.bytes(),
                              cudaMemcpyHostToDevice));
    }
}

// Each logical vector cooperatively processes one CSR row.  Template
// specialization lets short rows share a warp without wasting lanes, while
// long rows use a full warp for coalesced value/index loads.  The generated
// random matrix has tightly clustered row lengths, making this inexpensive
// average-row-length dispatch especially effective.
template <unsigned THREADS_PER_ROW>
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void spmvCsrKernel(const double* __restrict__ val,
                   const index_t* __restrict__ cols,
                   const index_t* __restrict__ rowDelimiters,
                   const double* __restrict__ vec,
                   const index_t numRows,
                   double* __restrict__ out) {
    static_assert(THREADS_PER_ROW >= 1 && THREADS_PER_ROW <= 32 &&
                  (THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0);

    constexpr unsigned ROWS_PER_BLOCK = CUDA_BLOCK_SIZE / THREADS_PER_ROW;
    const unsigned rowInBlock = threadIdx.x / THREADS_PER_ROW;
    const unsigned lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const index_t row = static_cast<index_t>(blockIdx.x * ROWS_PER_BLOCK + rowInBlock);

    double sum = 0.0;
    if (row < numRows) {
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t element = rowStart + lane; element < rowEnd;
             element += THREADS_PER_ROW) {
            sum = fma(val[element], vec[cols[element]], sum);
        }
    }

    #pragma unroll
    for (unsigned offset = THREADS_PER_ROW / 2; offset != 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
    }

    if (row < numRows && lane == 0) {
        out[row] = sum;
    }
}

template <unsigned THREADS_PER_ROW>
void launchSpmvKernel(const double* val, const index_t* cols,
                      const index_t* rowDelimiters, const double* vec,
                      const index_t numRows, double* out, cudaStream_t stream) {
    constexpr unsigned ROWS_PER_BLOCK = CUDA_BLOCK_SIZE / THREADS_PER_ROW;
    const unsigned blocks = static_cast<unsigned>(
        (static_cast<uint64_t>(numRows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    spmvCsrKernel<THREADS_PER_ROW><<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
        val, cols, rowDelimiters, vec, numRows, out);
}

unsigned selectThreadsPerRow(const index_t numRows, const index_t nItems) {
    const double average = static_cast<double>(nItems) / numRows;

    // Start with the width that minimizes idle lanes for this row length.
    unsigned width = 32;
    if (average <= 1.0) {
        width = 1;
    } else if (average <= 2.0) {
        width = 4;
    } else if (average <= 8.0) {
        width = 8;
    } else if (average <= 32.0) {
        width = 16;
    }

    // A width selected solely from row length can expose too few blocks for
    // small matrices. Ensure roughly 1.5 blocks per SM so all processors get
    // work, without forcing a full warp on every very short row.
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const uint64_t targetThreads = static_cast<uint64_t>(properties.multiProcessorCount) *
                                   CUDA_BLOCK_SIZE * 3 / 2;
    while (width < 32 && static_cast<uint64_t>(numRows) * width < targetThreads) {
        width *= 2;
    }
    return width;
}

void launchSpmv(const unsigned threadsPerRow,
                const double* val, const index_t* cols,
                const index_t* rowDelimiters, const double* vec,
                const index_t numRows, double* out, cudaStream_t stream) {
    switch (threadsPerRow) {
        case 1:  launchSpmvKernel<1>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
        case 2:  launchSpmvKernel<2>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
        case 4:  launchSpmvKernel<4>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
        case 8:  launchSpmvKernel<8>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
        case 16: launchSpmvKernel<16>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
        default: launchSpmvKernel<32>(val, cols, rowDelimiters, vec, numRows, out, stream); break;
    }
}

cudaGraphExec_t createSpmvGraph(const index_t launchCount,
                                const unsigned threadsPerRow,
                                const double* val, const index_t* cols,
                                const index_t* rowDelimiters, const double* vec,
                                const index_t numRows, double* out,
                                cudaStream_t stream) {
    if (launchCount == 0) {
        return nullptr;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (index_t launch = 0; launch < launchCount; ++launch) {
        launchSpmv(threadsPerRow, val, cols, rowDelimiters, vec, numRows, out, stream);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return executable;
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
    const uint64_t matrixEntries = static_cast<uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / static_cast<double>(matrixEntries);

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = matrixEntries -
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

    if (numRows == 0 || sparsity == 0) {
        fprintf(stderr, "Matrix size and sparsity must both be greater than zero.\n");
        return 1;
    }

    // Calculate number of non-zero elements
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t itemCount = matrixEntries / sparsity;
    if (itemCount > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The requested matrix has too many non-zero elements for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(itemCount);

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

    // Allocate device storage and transfer all invariant inputs before the
    // timed region.  The output remains resident across every iteration.
    DeviceBuffer<double> d_val(nItems);
    DeviceBuffer<index_t> d_cols(nItems);
    DeviceBuffer<index_t> d_rowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> d_vec(numRows);
    DeviceBuffer<double> d_out(numRows);

    copyToDevice(d_val, h_val);
    copyToDevice(d_cols, h_cols);
    copyToDevice(d_rowDelimiters, h_rowDelimiters);
    copyToDevice(d_vec, h_vec);
    CUDA_CHECK(cudaMemset(d_out.get(), 0, d_out.bytes()));

    const unsigned threadsPerRow = selectThreadsPerRow(numRows, nItems);
    cudaStream_t computeStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

    // Warm up the selected specialization so context setup and lazy module
    // loading cannot contaminate the measured iterations.
    if (iterations != 0) {
        launchSpmv(threadsPerRow, d_val.get(), d_cols.get(),
                   d_rowDelimiters.get(), d_vec.get(), numRows, d_out.get(), computeStream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    // Batched CUDA graphs retain one real kernel per requested iteration but
    // remove repeated CPU launch latency and improve back-to-back dispatch.
    constexpr index_t MAX_GRAPH_BATCH = 256;
    const index_t graphBatch = iterations < MAX_GRAPH_BATCH ? iterations : MAX_GRAPH_BATCH;
    const index_t fullGraphLaunches = graphBatch != 0 ? iterations / graphBatch : 0;
    const index_t graphRemainder = graphBatch != 0 ? iterations % graphBatch : 0;
    cudaGraphExec_t batchGraph = createSpmvGraph(
        graphBatch, threadsPerRow, d_val.get(), d_cols.get(), d_rowDelimiters.get(),
        d_vec.get(), numRows, d_out.get(), computeStream);
    cudaGraphExec_t remainderGraph = createSpmvGraph(
        graphRemainder, threadsPerRow, d_val.get(), d_cols.get(), d_rowDelimiters.get(),
        d_vec.get(), numRows, d_out.get(), computeStream);

    // Perform SpMV computation. CUDA events measure actual GPU execution and
    // preserve the original contract that one independent SpMV is performed
    // for each requested iteration.
    printf("Computing SpMV...\n");
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, computeStream));

    for (index_t launch = 0; launch < fullGraphLaunches; ++launch) {
        CUDA_CHECK(cudaGraphLaunch(batchGraph, computeStream));
    }
    if (remainderGraph != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(remainderGraph, computeStream));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, computeStream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float durationMsFloat = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&durationMsFloat, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    if (batchGraph != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(batchGraph));
    }
    if (remainderGraph != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(remainderGraph));
    }
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    const double durationMs = durationMsFloat;

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = durationMs > 0.0
        ? (2.0 * nItems * iterations) / (durationMs * 1.0e6)
        : 0.0;
    const double avgTime = iterations > 0
        ? durationMs / static_cast<double>(iterations)
        : 0.0;
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy back only the final result, after timing, for validation/output.
    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), d_out.bytes(),
                              cudaMemcpyDeviceToHost));
    }
    
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
