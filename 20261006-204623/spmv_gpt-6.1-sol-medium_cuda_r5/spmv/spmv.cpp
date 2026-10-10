#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

// Each group handles one CSR row. Neighboring lanes load neighboring nonzeros,
// and short rows use subwarps so multiple rows share a warp.
template <int ThreadsPerRow>
__global__ void spmvRows(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rows,
                         const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    const size_t thread = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const size_t row = thread / ThreadsPerRow;
    const unsigned lane = threadIdx.x % ThreadsPerRow;
    // All lanes participate in the shuffle, including groups beyond the last row.
    index_t begin = 0, end = 0;
    if (row < dim && lane == 0) {
        begin = rows[row];
        end = rows[row + 1];
    }
    begin = __shfl_sync(0xffffffff, begin, 0, ThreadsPerRow);
    end = __shfl_sync(0xffffffff, end, 0, ThreadsPerRow);
    double sum = 0.0;
    for (size_t j = size_t{begin} + lane; j < end; j += ThreadsPerRow) {
        sum += val[j] * __ldg(vec + cols[j]);
    }
    for (int offset = ThreadsPerRow / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, ThreadsPerRow);
    }
    if (row < dim && lane == 0) out[row] = sum;
}

// Long rows get an entire block, exposing more parallelism even when the
// matrix has few rows. No atomics are required in either kernel.
__global__ void spmvLongRows(const double* __restrict__ val,
                             const index_t* __restrict__ cols,
                             const index_t* __restrict__ rows,
                             const double* __restrict__ vec,
                             double* __restrict__ out) {
    constexpr unsigned blockSize = 128;
    __shared__ double warpSums[blockSize / 32];
    const index_t row = blockIdx.x;
    double sum = 0.0;
    for (size_t j = size_t{rows[row]} + threadIdx.x; j < rows[row + 1]; j += blockSize) {
        sum += val[j] * __ldg(vec + cols[j]);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    const unsigned lane = threadIdx.x % 32;
    if (lane == 0) warpSums[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < blockSize / 32 ? warpSums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Own all CUDA resources, including during an error in allocation or execution.
class CudaSpmv {
    double *val = nullptr, *vec = nullptr, *out = nullptr;
    index_t *cols = nullptr, *rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    index_t dim = 0;
    unsigned threadsPerRow = 32;

    template <typename T>
    void upload(T*& device, const std::vector<T>& host) {
        // cudaMalloc does not accept a zero-size allocation on every runtime.
        const size_t bytes = host.size() * sizeof(T);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), std::max(bytes, sizeof(T))),
                  "cudaMalloc");
        if (bytes != 0)
            cudaCheck(cudaMemcpyAsync(device, host.data(), bytes, cudaMemcpyHostToDevice, stream),
                      "upload data");
    }

    template <int Width>
    void launchRows() {
        constexpr unsigned blockSize = 128;
        const unsigned blocks = static_cast<unsigned>(
            (size_t{dim} * Width + blockSize - 1) / blockSize);
        spmvRows<Width><<<blocks, blockSize, 0, stream>>>(val, cols, rows, vec, dim, out);
    }

    void launch() {
        switch (threadsPerRow) {
            case 2: launchRows<2>(); break;
            case 4: launchRows<4>(); break;
            case 8: launchRows<8>(); break;
            case 16: launchRows<16>(); break;
            case 32: launchRows<32>(); break;
            default: spmvLongRows<<<dim, 128, 0, stream>>>(val, cols, rows, vec, out); break;
        }
    }

public:
    CudaSpmv() = default;
    CudaSpmv(const CudaSpmv&) = delete;
    CudaSpmv& operator=(const CudaSpmv&) = delete;
    ~CudaSpmv() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(out);
        cudaFree(rows);
        cudaFree(cols);
        cudaFree(vec);
        cudaFree(val);
        if (stream) cudaStreamDestroy(stream);
    }

    void initialize(const std::vector<double>& hostVal, const std::vector<index_t>& hostCols,
                    const std::vector<index_t>& hostRows, const std::vector<double>& hostVec) {
        dim = static_cast<index_t>(hostVec.size());
        const size_t average = hostVal.size() / dim;
        threadsPerRow = average <= 2 ? 2 : average <= 4 ? 4 : average <= 8 ? 8 :
                        average <= 16 ? 16 : average < 256 ? 32 : 128;
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        cudaCheck(cudaEventCreate(&start), "create start event");
        cudaCheck(cudaEventCreate(&stop), "create stop event");
        upload(val, hostVal);
        upload(cols, hostCols);
        upload(rows, hostRows);
        upload(vec, hostVec);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&out), size_t{dim} * sizeof(double)),
                  "allocate output");
        cudaCheck(cudaMemsetAsync(out, 0, size_t{dim} * sizeof(double), stream), "clear output");
        cudaCheck(cudaStreamSynchronize(stream), "finish upload");
    }

    float run(index_t iterations, std::vector<double>& hostOut) {
        const index_t batch = std::min(iterations, index_t{128});
        if (iterations > 0) {
            // Warm up outside the measurement; the input remains unchanged.
            launch();
            cudaCheck(cudaGetLastError(), "launch warmup");
            cudaCheck(cudaStreamSynchronize(stream), "finish warmup");
            // Bound graph size for arbitrarily large iteration counts. Replaying
            // a batch avoids CPU launch latency between small SpMV kernels.
            cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                      "begin graph capture");
            for (index_t i = 0; i < batch; ++i) launch();
            cudaCheck(cudaGetLastError(), "capture SpMV kernels");
            cudaCheck(cudaStreamEndCapture(stream, &graph), "end graph capture");
            cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                      "instantiate graph");
            cudaCheck(cudaGraphUpload(executable, stream), "upload graph");
            cudaCheck(cudaStreamSynchronize(stream), "finish graph upload");
        }
        cudaCheck(cudaEventRecord(start, stream), "record start");
        if (batch > 0) {
            for (index_t i = 0; i < iterations / batch; ++i)
                cudaCheck(cudaGraphLaunch(executable, stream), "launch SpMV graph");
            for (index_t i = 0; i < iterations % batch; ++i) launch();
            cudaCheck(cudaGetLastError(), "launch remaining SpMV kernels");
        }
        cudaCheck(cudaEventRecord(stop, stream), "record stop");
        cudaCheck(cudaEventSynchronize(stop), "finish SpMV");
        float milliseconds = 0.0f;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop), "measure SpMV");
        cudaCheck(cudaMemcpyAsync(hostOut.data(), out, size_t{dim} * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "download result");
        cudaCheck(cudaStreamSynchronize(stream), "finish download");
        return iterations == 0 ? 0.0f : milliseconds;
    }
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

int main(int argc, char** argv) try {
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
    const uint64_t matrixSize = uint64_t{numRows} * numRows;
    if (matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Nonzero count exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize));
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

    CudaSpmv gpu;
    gpu.initialize(h_val, h_cols, h_rowDelimiters, h_vec);

    printf("Computing SpMV...\n");
    const double milliseconds = gpu.run(iterations, h_out);
    printf("Computation time: %.3f ms\n", milliseconds);

    const double gflops = milliseconds > 0.0
        ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
    const double avgTime = iterations > 0 ? milliseconds / iterations : 0.0;

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
} catch (const std::exception& error) {
    fprintf(stderr, "SpMV failed: %s\n", error.what());
    return 1;
}
