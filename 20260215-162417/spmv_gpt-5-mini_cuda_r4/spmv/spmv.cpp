#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Simple CUDA error checking helper
static inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

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
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
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
// Function: spmvGpu
//
// Purpose:
//   CUDA implementation of sparse matrix-vector multiplication (CSR)
//   Each CUDA thread computes one output row.
//
// ****************************************************************************
__global__ void spmvKernel(const double* val, const uint32_t* cols, const uint32_t* rowDelimiters,
                           const double* vec, const uint32_t dim, double* out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    double sum = 0.0;
    const uint32_t start = rowDelimiters[row];
    const uint32_t end = rowDelimiters[row + 1];
    for (uint32_t j = start; j < end; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    out[row] = sum;
}

void spmvGpu(const double* h_val, const index_t* h_cols, const index_t* h_rowDelimiters,
             const double* h_vec, const index_t dim, double* h_out, const index_t iterations) {
    // Device pointers
    double* d_val = nullptr;
    uint32_t* d_cols = nullptr;
    uint32_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    const size_t valBytes = static_cast<size_t>(h_rowDelimiters[dim]) * sizeof(double);
    const size_t colsBytes = static_cast<size_t>(h_rowDelimiters[dim]) * sizeof(uint32_t);
    const size_t rowDelimBytes = static_cast<size_t>(dim + 1) * sizeof(uint32_t);
    const size_t vecBytes = static_cast<size_t>(dim) * sizeof(double);
    const size_t outBytes = static_cast<size_t>(dim) * sizeof(double);

    // Allocate device memory
    cudaCheck(cudaMalloc(&d_val, valBytes), "cudaMalloc d_val");
    cudaCheck(cudaMalloc(&d_cols, colsBytes), "cudaMalloc d_cols");
    cudaCheck(cudaMalloc(&d_rowDelimiters, rowDelimBytes), "cudaMalloc d_rowDelimiters");
    cudaCheck(cudaMalloc(&d_vec, vecBytes), "cudaMalloc d_vec");
    cudaCheck(cudaMalloc(&d_out, outBytes), "cudaMalloc d_out");

    // Copy to device
    cudaCheck(cudaMemcpy(d_val, h_val, valBytes, cudaMemcpyHostToDevice), "cudaMemcpy val");
    // Cast index arrays to uint32_t for device
    std::vector<uint32_t> tmpCols(h_rowDelimiters[dim]);
    for (size_t i = 0; i < tmpCols.size(); ++i) tmpCols[i] = static_cast<uint32_t>(h_cols[i]);
    cudaCheck(cudaMemcpy(d_cols, tmpCols.data(), colsBytes, cudaMemcpyHostToDevice), "cudaMemcpy cols");
    std::vector<uint32_t> tmpRowDelim(dim + 1);
    for (size_t i = 0; i < tmpRowDelim.size(); ++i) tmpRowDelim[i] = static_cast<uint32_t>(h_rowDelimiters[i]);
    cudaCheck(cudaMemcpy(d_rowDelimiters, tmpRowDelim.data(), rowDelimBytes, cudaMemcpyHostToDevice), "cudaMemcpy rowDelims");
    cudaCheck(cudaMemcpy(d_vec, h_vec, vecBytes, cudaMemcpyHostToDevice), "cudaMemcpy vec");

    // Launch kernel
    const uint32_t threadsPerBlock = 256u;
    const uint32_t blocks = static_cast<uint32_t>((dim + threadsPerBlock - 1) / threadsPerBlock);

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<blocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
        cudaCheck(cudaGetLastError(), "spmvKernel launch");
    }

    // Copy result back
    cudaCheck(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost), "cudaMemcpy out");

    // Free device memory
    cudaCheck(cudaFree(d_val), "cudaFree d_val");
    cudaCheck(cudaFree(d_cols), "cudaFree d_cols");
    cudaCheck(cudaFree(d_rowDelimiters), "cudaFree d_rowDelimiters");
    cudaCheck(cudaFree(d_vec), "cudaFree d_vec");
    cudaCheck(cudaFree(d_out), "cudaFree d_out");
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
    const index_t nItems = (numRows * numRows) / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
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

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    spmvGpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_out.data(), iterations);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
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
