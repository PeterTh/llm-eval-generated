#include <algorithm>
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

// Every benchmark multiplication runs on CUDA; failures are fatal rather than
// silently switching to the CPU implementation used for validation.
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* source, size_t count) {
        if (count) cudaCheck(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// A subgroup handles each short row. Neighboring lanes read neighboring CSR
// entries, and shuffle reductions need neither atomics nor shared memory.
template <unsigned ThreadsPerRow>
__global__ void spmvRows(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rows,
                         const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    constexpr unsigned RowsPerBlock = 128 / ThreadsPerRow;
    const unsigned lane = threadIdx.x % ThreadsPerRow;
    const size_t row = size_t(blockIdx.x) * RowsPerBlock + threadIdx.x / ThreadsPerRow;
    double sum = 0.0;
    if (row < dim) {
        const index_t begin = rows[row], end = rows[row + 1];
        for (size_t j = size_t(begin) + lane; j < end; j += ThreadsPerRow)
            sum += val[j] * __ldg(vec + cols[j]);
    }
    // All lanes participate, including lanes belonging to out-of-range rows.
    for (unsigned offset = ThreadsPerRow / 2; offset; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, ThreadsPerRow);
    if (lane == 0 && row < dim) out[row] = sum;
}

// Long rows use a whole block so even modest matrix dimensions expose enough
// work to occupy the GPU. Only four partial sums go through shared memory.
__global__ void spmvLongRows(const double* __restrict__ val,
                             const index_t* __restrict__ cols,
                             const index_t* __restrict__ rows,
                             const double* __restrict__ vec,
                             index_t dim, double* __restrict__ out) {
    __shared__ double partial[4];
    for (size_t row = blockIdx.x; row < dim; row += gridDim.x) {
        double sum = 0.0;
        const index_t begin = rows[row], end = rows[row + 1];
        for (size_t j = size_t(begin) + threadIdx.x; j < end; j += 128)
            sum += val[j] * __ldg(vec + cols[j]);
        for (unsigned offset = 16; offset; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        const unsigned lane = threadIdx.x % 32;
        if (lane == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = lane < 4 ? partial[lane] : 0.0;
            for (unsigned offset = 2; offset; offset /= 2)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0) out[row] = sum;
        }
        __syncthreads();
    }
}

double spmvGpu(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& rows, const std::vector<double>& vec,
               index_t iterations, std::vector<double>& out) {
    DeviceBuffer<double> d_val(val.size()), d_vec(vec.size()), d_out(out.size());
    DeviceBuffer<index_t> d_cols(cols.size()), d_rows(rows.size());
    d_val.upload(val.data(), val.size());
    d_cols.upload(cols.data(), cols.size());
    d_rows.upload(rows.data(), rows.size());
    d_vec.upload(vec.data(), vec.size());
    if (!iterations) return 0.0; // Preserve the original zero-iteration output.

    const index_t dim = static_cast<index_t>(vec.size());
    const double entriesPerRow = static_cast<double>(val.size()) / dim;
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&]() {
        if (entriesPerRow >= 256) {
            spmvLongRows<<<std::min(dim, index_t(65535)), 128, 0, stream>>>(
                d_val.data, d_cols.data, d_rows.data, d_vec.data, dim, d_out.data);
        } else {
            const unsigned width = entriesPerRow <= 2 ? 1 : entriesPerRow <= 4 ? 2 :
                                   entriesPerRow <= 8 ? 4 : entriesPerRow <= 16 ? 8 :
                                   entriesPerRow <= 32 ? 16 : 32;
            const unsigned blocks = static_cast<unsigned>((size_t(dim) * width + 127) / 128);
            switch (width) {
#define LAUNCH_ROWS(W) case W: spmvRows<W><<<blocks, 128, 0, stream>>>(d_val.data, d_cols.data, d_rows.data, d_vec.data, dim, d_out.data); break
                LAUNCH_ROWS(1);
                LAUNCH_ROWS(2);
                LAUNCH_ROWS(4);
                LAUNCH_ROWS(8);
                LAUNCH_ROWS(16);
                LAUNCH_ROWS(32);
#undef LAUNCH_ROWS
            }
        }
    };
    // Warm up outside the timed region, then amortize host launch overhead
    // across up to 64 repetitions without changing the iteration count.
    launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    const index_t batch = std::min(iterations, index_t(64));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (index_t i = 0; i < batch; ++i) launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamEndCapture(stream, &graph));
    cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    cudaCheck(cudaGraphUpload(executable, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, stream));
    for (index_t i = 0; i < iterations / batch; ++i)
        cudaCheck(cudaGraphLaunch(executable, stream));
    for (index_t i = 0; i < iterations % batch; ++i) launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(stop, stream));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, stop));
    cudaCheck(cudaMemcpy(out.data(), d_out.data, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaGraphExecDestroy(executable));
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    return elapsedMs;
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
    if (!numRows || !sparsity) {
        fprintf(stderr, "Matrix dimension and sparsity must be positive.\n");
        return 1;
    }
    const uint64_t matrixEntries = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = matrixEntries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix exceeds the CSR 32-bit index capacity.\n");
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
    const double elapsedMs = spmvGpu(h_val, h_cols, h_rowDelimiters, h_vec,
                                     iterations, h_out);
    printf("Computation time: %.3f ms\n", elapsedMs);

    // CUDA events retain sub-millisecond precision for short GPU kernels.
    const double gflops = elapsedMs > 0.0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0.0;
    const double avgTime = iterations ? elapsedMs / iterations : 0.0;

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
