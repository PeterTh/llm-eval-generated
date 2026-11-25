#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim - i) * dim - j;
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

// Adjacent lanes gather a consecutive group of CSR entries. Broadcast their
// products in column order to retain the serial floating-point sum exactly.
template <unsigned Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ offsets,
                           const double* __restrict__ vector,
                           index_t rows, double* __restrict__ output) {
    const index_t row = blockIdx.x * (blockDim.x / Width) + threadIdx.x / Width;
    if (row >= rows) return;
    const unsigned lane = threadIdx.x % Width;
    const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x % 32) - lane);
    index_t pos = offsets[row];
    const index_t end = offsets[row + 1ull];
    double sum = 0.0;
    // Issue independent gathers ahead of the ordered reduction. This also
    // keeps a single unusually long row from serializing all its memory reads.
    constexpr unsigned batch = 8;
    while (end - pos >= batch * Width) {
        double products[batch];
        #pragma unroll
        for (unsigned b = 0; b < batch; ++b) {
            const index_t j = pos + b * Width + lane;
            products[b] = __dmul_rn(values[j], vector[columns[j]]);
        }
        #pragma unroll
        for (unsigned b = 0; b < batch; ++b) {
            #pragma unroll
            for (unsigned k = 0; k < Width; ++k)
                sum += __shfl_sync(mask, products[b], k, Width);
        }
        pos += batch * Width;
    }
    while (end - pos >= Width) {
        const index_t j = pos + lane;
        const double product = __dmul_rn(values[j], vector[columns[j]]);
        #pragma unroll
        for (unsigned k = 0; k < Width; ++k)
            sum += __shfl_sync(mask, product, k, Width);
        pos += Width;
    }
    if (pos < end) {
        const unsigned count = end - pos;
        double product = 0.0;
        if (lane < count) {
            const index_t j = pos + lane;
            product = __dmul_rn(values[j], vector[columns[j]]);
        }
        for (unsigned k = 0; k < count; ++k)
            sum += __shfl_sync(mask, product, k, Width);
    }
    if (lane == 0) output[row] = sum;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        if (!host.empty())
            checkCuda(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "upload SpMV input");
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* ptr = nullptr;
};

template <unsigned Width>
void runIterations(const double* values, const index_t* columns, const index_t* offsets,
                   const double* vector, index_t rows, double* output, index_t iterations, cudaEvent_t start) {
    constexpr unsigned threads = 128;
    constexpr unsigned rowsPerBlock = threads / Width;
    const unsigned blocks = rows ? 1 + (rows - 1) / rowsPerBlock : 1;
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, spmvKernel<Width>), "load SpMV kernel");
    checkCuda(cudaEventRecord(start), "record start event");
    for (index_t i = 0; i < iterations; ++i)
        spmvKernel<Width><<<blocks, threads>>>(values, columns, offsets, vector, rows, output);
    checkCuda(cudaGetLastError(), "launch SpMV kernel");
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
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    if (!sparsity || entries / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix requires positive sparsity and 32-bit CSR offsets.\n");
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

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    DeviceBuffer<double> values(h_val), vector(h_vec), output(numRows);
    DeviceBuffer<index_t> columns(h_cols), offsets(h_rowDelimiters);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    const unsigned mean = numRows ? nItems / numRows : 0;
    auto run = mean <= 2 ? runIterations<1> : mean <= 16 ? runIterations<4>
                          : mean <= 1024 ? runIterations<8> : runIterations<32>;
    run(values.ptr, columns.ptr, offsets.ptr, vector.ptr, numRows, output.ptr, iterations, start);
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "complete SpMV");
    float elapsedMs;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measure SpMV time");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");

    if (numRows && iterations)
        checkCuda(cudaMemcpy(h_out.data(), output.ptr, numRows * sizeof(double),
                             cudaMemcpyDeviceToHost), "download SpMV result");

    printf("Computation time: %.3f ms\n", elapsedMs);
    const double gflops = elapsedMs > 0 ? 2.0 * nItems * iterations / (elapsedMs * 1e6) : 0;
    const double avgTime = iterations ? elapsedMs / static_cast<double>(iterations) : 0;

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
