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
            uint64_t numEntriesLeft = uint64_t(dim - i) * dim - j;
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
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<typename T>
T* upload(const std::vector<T>& host) {
    if (host.empty()) return nullptr;
    T* device;
    const size_t bytes = host.size() * sizeof(T);
    checkCuda(cudaMalloc(&device, bytes), "allocate input");
    checkCuda(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "upload input");
    return device;
}

// A contiguous group of lanes walks each CSR row in parallel. Short rows
// share a warp, while long rows use the full warp for coalesced loads.
template<unsigned Width>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ offsets, const double* __restrict__ vec,
                           index_t rows, double* __restrict__ out) {
    const unsigned lane = threadIdx.x % Width;
    const index_t row = blockIdx.x * (blockDim.x / Width) + threadIdx.x / Width;
    if (row >= rows) return;

    double sum = 0.0;
    const size_t end = offsets[row + 1];
    for (size_t j = size_t(offsets[row]) + lane; j < end; j += Width)
        sum += val[j] * vec[cols[j]];

    const unsigned mask = 0xffffffffu;
    for (unsigned delta = Width / 2; delta; delta /= 2)
        sum += __shfl_down_sync(mask, sum, delta, Width);
    if (lane == 0) out[row] = sum;
}

template<unsigned Width>
float benchmark(const double* val, const index_t* cols, const index_t* offsets,
                const double* vec, index_t rows, double* out, index_t iterations) {
    constexpr unsigned threads = 128;
    constexpr unsigned rowsPerBlock = threads / Width;
    const unsigned blocks = rows / rowsPerBlock + (rows % rowsPerBlock != 0);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, spmvKernel<Width>), "load SpMV kernel");
    checkCuda(cudaEventRecord(start), "record start");
    if (blocks) {
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<Width><<<blocks, threads>>>(val, cols, offsets, vec, rows, out);
        checkCuda(cudaGetLastError(), "launch SpMV");
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "finish SpMV");
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "measure SpMV");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    return ms;
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
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (!sparsity || entries / sparsity > UINT32_MAX) {
        fprintf(stderr, "Invalid sparsity or too many nonzeros for CSR indices\n");
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
    const auto d_val = upload(h_val);
    const auto d_cols = upload(h_cols);
    const auto d_offsets = upload(h_rowDelimiters);
    const auto d_vec = upload(h_vec);
    double* d_out = nullptr;
    if (numRows)
        checkCuda(cudaMalloc(&d_out, h_out.size() * sizeof(double)), "allocate output");

    const unsigned meanLength = numRows ? nItems / numRows : 0;
    auto run = meanLength <= 2 ? benchmark<1> : meanLength <= 8 ? benchmark<4> :
               meanLength <= 16 ? benchmark<8> : meanLength <= 32 ? benchmark<16> : benchmark<32>;
    const double ms = run(d_val, d_cols, d_offsets, d_vec, numRows, d_out, iterations);
    if (numRows && iterations && (validate || printResults))
        checkCuda(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "download output");
    checkCuda(cudaFree(d_out), "free output");
    checkCuda(cudaFree(d_vec), "free vector");
    checkCuda(cudaFree(d_offsets), "free offsets");
    checkCuda(cudaFree(d_cols), "free columns");
    checkCuda(cudaFree(d_val), "free values");

    printf("Computation time: %.3f ms\n", ms);
    const double gflops = ms ? 2.0 * nItems * iterations / (ms * 1e6) : 0.0;
    const double avgTime = iterations ? ms / iterations : 0.0;

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
