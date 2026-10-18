#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Own device storage for the entire benchmark; transfers are outside the loop.
template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(count, size_t{1}) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Each power-of-two subgroup owns a row. Adjacent lanes access adjacent CSR
// entries, and shuffle reductions avoid atomics and shared-memory barriers.
template <int Width>
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vec, index_t dim,
                           double* __restrict__ out) {
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const unsigned lane = threadIdx.x % Width;
    double sum = 0.0;
    if (row < dim) {
        const index_t begin = rows[row];
        const index_t end = rows[row + 1];
        for (uint64_t j = static_cast<uint64_t>(begin) + lane; j < end; j += Width)
            sum += val[j] * __ldg(vec + cols[j]);
    }
    // All lanes participate, including padding groups in the final block.
    for (int offset = Width / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, Width);
    if (row < dim && lane == 0)
        out[row] = sum;
}

// Long rows use a full block, exposing more parallelism even for small dim.
__global__ void spmvLongRowsKernel(const double* __restrict__ val,
                                   const index_t* __restrict__ cols,
                                   const index_t* __restrict__ rows,
                                   const double* __restrict__ vec,
                                   double* __restrict__ out) {
    __shared__ double partial[8];
    const index_t row = blockIdx.x;
    const index_t end = rows[row + 1];
    double sum = 0.0;
    for (uint64_t j = static_cast<uint64_t>(rows[row]) + threadIdx.x;
         j < end; j += blockDim.x)
        sum += val[j] * __ldg(vec + cols[j]);
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (threadIdx.x % 32 == 0)
        partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (threadIdx.x == 0)
            out[row] = sum;
    }
}

template <int Width>
void launchSpmv(const double* val, const index_t* cols, const index_t* rows,
                const double* vec, index_t dim, double* out, cudaStream_t stream) {
    constexpr unsigned threads = 128;
    const unsigned blocks = (static_cast<uint64_t>(dim) + threads / Width - 1)
                            / (threads / Width);
    spmvKernel<Width><<<blocks, threads, 0, stream>>>(val, cols, rows, vec, dim, out);
}

void spmvGpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, index_t nnz, double* out,
             cudaStream_t stream = nullptr) {
    const double average = static_cast<double>(nnz) / dim;
    if (average >= 512)
        spmvLongRowsKernel<<<dim, 256, 0, stream>>>(val, cols, rows, vec, out);
    else if (average > 16)
        launchSpmv<32>(val, cols, rows, vec, dim, out, stream);
    else if (average > 8)
        launchSpmv<16>(val, cols, rows, vec, dim, out, stream);
    else if (average > 4)
        launchSpmv<8>(val, cols, rows, vec, dim, out, stream);
    else if (average > 2)
        launchSpmv<4>(val, cols, rows, vec, dim, out, stream);
    else if (average > 1)
        launchSpmv<2>(val, cols, rows, vec, dim, out, stream);
    else
        launchSpmv<1>(val, cols, rows, vec, dim, out, stream);
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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim
                                      - (static_cast<uint64_t>(i) * dim + j);
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
        if (!std::isfinite(res) || !std::isfinite(ref)) {
            printf("Validation failed at index %u: non-finite result\n", i);
            return false;
        }
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
    if (numRows == 0 || sparsity == 0) {
        fprintf(stderr, "Matrix dimension and sparsity must be positive.\n");
        return 1;
    }
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    if (entries / sparsity > UINT32_MAX) {
        fprintf(stderr, "Matrix has too many nonzeros for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nItems = entries / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
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

    DeviceBuffer<double> d_val(nItems), d_vec(numRows), d_out(numRows);
    DeviceBuffer<index_t> d_cols(nItems), d_rows(static_cast<size_t>(numRows) + 1);
    if (nItems != 0) {
        cudaCheck(cudaMemcpy(d_val.data, h_val.data(), h_val.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_cols.data, h_cols.data(), h_cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    cudaCheck(cudaMemcpy(d_rows.data, h_rowDelimiters.data(), h_rowDelimiters.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_vec.data, h_vec.data(), h_vec.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out.data, 0, h_out.size() * sizeof(double)));

    // Warm up the selected kernel before recording GPU execution time.
    if (iterations != 0) {
        spmvGpu(d_val.data, d_cols.data, d_rows.data, d_vec.data, numRows, nItems, d_out.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
    }
    // Bound graph size for large iteration counts while amortizing host launch
    // overhead. Each graph node still performs one complete SpMV iteration.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const index_t batch = std::min(iterations, index_t{128});
    if (batch > 1) {
        // Capture on a dedicated stream so every node uses the same ordering.
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (index_t iter = 0; iter < batch; ++iter)
            spmvGpu(d_val.data, d_cols.data, d_rows.data, d_vec.data,
                    numRows, nItems, d_out.data, stream);
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(graphExec, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    printf("Computing SpMV...\n");
    cudaCheck(cudaEventRecord(start, stream));

    index_t remaining = iterations;
    if (graphExec != nullptr) {
        for (; remaining >= batch; remaining -= batch)
            cudaCheck(cudaGraphLaunch(graphExec, stream));
    }
    for (index_t iter = 0; iter < remaining; ++iter) {
        spmvGpu(d_val.data, d_cols.data, d_rows.data, d_vec.data,
                numRows, nItems, d_out.data, stream);
    }

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(end, stream));
    cudaCheck(cudaEventSynchronize(end));
    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    if (graphExec != nullptr) {
        cudaCheck(cudaGraphExecDestroy(graphExec));
        cudaCheck(cudaGraphDestroy(graph));
    }
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaMemcpy(h_out.data(), d_out.data, h_out.size() * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));
    
    // Calculate performance metrics
    const double gflops = elapsedMs > 0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0.0;
    const double avgTime = iterations != 0 ? elapsedMs / static_cast<double>(iterations) : 0.0;
    
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
