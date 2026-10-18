#include <cuda_runtime.h>

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

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Allocate once; all iterations reuse the same device-resident CSR matrix.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        // cudaMalloc does not accept a zero-sized allocation on every runtime.
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), (count ? count : 1) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }
private:
    T* data_ = nullptr;
};

// Each power-of-two subgroup cooperates on one row. Adjacent lanes load
// adjacent CSR entries, and short rows share a warp without shared memory.
template <int Width>
__global__ void spmvRows(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rows,
                         const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & (Width - 1);
    const size_t row = thread / Width;
    double sum = 0.0;
    if (row < dim) {
        const size_t end = rows[row + 1];
        for (size_t j = size_t(rows[row]) + lane; j < end; j += Width) {
            sum += val[j] * __ldg(vec + cols[j]);
        }
    }
    // All lanes, including padding rows, participate in the shuffle.
    for (int offset = Width / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset, Width);
    }
    if (row < dim && lane == 0) out[row] = sum;
}

// Long rows use a whole block, exposing more parallelism even for matrices
// with few rows. Only the eight warp partials need shared memory.
__global__ void spmvLongRows(const double* __restrict__ val,
                             const index_t* __restrict__ cols,
                             const index_t* __restrict__ rows,
                             const double* __restrict__ vec,
                             double* __restrict__ out) {
    const index_t row = blockIdx.x;
    double sum = 0.0;
    const size_t end = rows[row + 1];
    for (size_t j = size_t(rows[row]) + threadIdx.x; j < end; j += blockDim.x) {
        sum += val[j] * __ldg(vec + cols[j]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    __shared__ double partials[8];
    if ((threadIdx.x & 31) == 0) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partials[threadIdx.x] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (threadIdx.x == 0) out[row] = sum;
    }
}

void launchSpmv(const double* val, const index_t* cols, const index_t* rows,
                const double* vec, index_t dim, double* out, int width) {
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((size_t(dim) * width + threads - 1) / threads);
    switch (width) {
        case 2: spmvRows<2><<<blocks, threads>>>(val, cols, rows, vec, dim, out); break;
        case 4: spmvRows<4><<<blocks, threads>>>(val, cols, rows, vec, dim, out); break;
        case 8: spmvRows<8><<<blocks, threads>>>(val, cols, rows, vec, dim, out); break;
        case 16: spmvRows<16><<<blocks, threads>>>(val, cols, rows, vec, dim, out); break;
        case 32: spmvRows<32><<<blocks, threads>>>(val, cols, rows, vec, dim, out); break;
        default: spmvLongRows<<<dim, threads>>>(val, cols, rows, vec, out); break;
    }
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
    if (numRows == 0 || sparsity == 0 ||
        uint64_t(numRows) * numRows / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix dimensions and sparsity must be positive and fit 32-bit CSR indices.\n");
        return 1;
    }
    const uint64_t matrixEntries = uint64_t(numRows) * numRows;
    const index_t nItems = matrixEntries / sparsity;

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

    DeviceBuffer<double> d_val(nItems), d_vec(numRows), d_out(numRows);
    DeviceBuffer<index_t> d_cols(nItems), d_rows(size_t(numRows) + 1);
    cudaCheck(cudaMemcpy(d_val.data(), h_val.data(), size_t(nItems) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_cols.data(), h_cols.data(), size_t(nItems) * sizeof(index_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_rows.data(), h_rowDelimiters.data(), (size_t(numRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_vec.data(), h_vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_out.data(), 0, size_t(numRows) * sizeof(double)));

    const double averageRowLength = static_cast<double>(nItems) / numRows;
    const int width = averageRowLength <= 2 ? 2 : averageRowLength <= 4 ? 4 :
                      averageRowLength <= 8 ? 8 : averageRowLength <= 16 ? 16 :
                      averageRowLength <= 512 ? 32 : 256;
    // Warm up before timing, without altering zero-iteration output semantics.
    if (iterations != 0) {
        launchSpmv(d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(), numRows, d_out.data(), width);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
    }

    // CUDA events measure completed GPU work, excluding setup and transfers.
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    printf("Computing SpMV...\n");
    cudaCheck(cudaEventRecord(start));

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(d_val.data(), d_cols.data(), d_rows.data(), d_vec.data(), numRows, d_out.data(), width);
    }

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end));
    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, end));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaMemcpy(h_out.data(), d_out.data(), size_t(numRows) * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));
    
    // Calculate performance metrics
    const double gflops = elapsedMs > 0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0.0;
    const double avgTime = iterations ? elapsedMs / static_cast<double>(iterations) : 0.0;
    
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
