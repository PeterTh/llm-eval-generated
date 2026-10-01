#include <algorithm>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A group of adjacent lanes handles one CSR row. Groups of fewer than 32
// lanes keep the GPU busy when rows contain only a few nonzero entries.
template <int LANES_PER_ROW>
__global__ void spmvCuda(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rowDelimiters,
                         const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    constexpr int BLOCK_SIZE = 256;
    const int lane = threadIdx.x % LANES_PER_ROW;
    const index_t row = blockIdx.x * (BLOCK_SIZE / LANES_PER_ROW)
                      + threadIdx.x / LANES_PER_ROW;
    if (row >= dim) return;

    double sum = 0.0;
    const index_t end = rowDelimiters[row + 1];
    for (index_t j = rowDelimiters[row] + lane; j < end; j += LANES_PER_ROW) {
        sum += val[j] * vec[cols[j]];
    }
    for (int offset = LANES_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, LANES_PER_ROW);
    }
    if (lane == 0) out[row] = sum;
}

template <int LANES_PER_ROW>
void launchSpmv(const double* val, const index_t* cols,
                const index_t* rowDelimiters, const double* vec,
                index_t dim, double* out) {
    constexpr int BLOCK_SIZE = 256;
    const unsigned int rowsPerBlock = BLOCK_SIZE / LANES_PER_ROW;
    const unsigned int blocks = (dim + rowsPerBlock - 1) / rowsPerBlock;
    spmvCuda<LANES_PER_ROW><<<blocks, BLOCK_SIZE>>>(
        val, cols, rowDelimiters, vec, dim, out);
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

    // Keep inputs resident on the GPU across all benchmark iterations.
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_val),
                         std::max<size_t>(1, h_val.size() * sizeof(double))), "allocating values");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                         std::max<size_t>(1, h_cols.size() * sizeof(index_t))), "allocating columns");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                         h_rowDelimiters.size() * sizeof(index_t)), "allocating row offsets");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                         h_vec.size() * sizeof(double)), "allocating vector");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_out),
                         h_out.size() * sizeof(double)), "allocating output");
    checkCuda(cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "copying values");
    checkCuda(cudaMemcpy(d_cols, h_cols.data(), h_cols.size() * sizeof(index_t),
                         cudaMemcpyHostToDevice), "copying columns");
    checkCuda(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                         h_rowDelimiters.size() * sizeof(index_t),
                         cudaMemcpyHostToDevice), "copying row offsets");
    checkCuda(cudaMemcpy(d_vec, h_vec.data(), h_vec.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "copying vector");

    const index_t entriesPerRow = numRows ? nItems / numRows : 0;
    cudaEvent_t startEvent, endEvent;
    checkCuda(cudaEventCreate(&startEvent), "creating start event");
    checkCuda(cudaEventCreate(&endEvent), "creating end event");

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    checkCuda(cudaEventRecord(startEvent), "recording start event");

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (entriesPerRow <= 2) {
            launchSpmv<2>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else if (entriesPerRow <= 4) {
            launchSpmv<4>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else if (entriesPerRow <= 8) {
            launchSpmv<8>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else if (entriesPerRow <= 16) {
            launchSpmv<16>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        } else {
            launchSpmv<32>(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
        }
    }
    checkCuda(cudaGetLastError(), "launching SpMV kernel");
    checkCuda(cudaEventRecord(endEvent), "recording end event");
    checkCuda(cudaEventSynchronize(endEvent), "running SpMV kernel");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent), "measuring SpMV");
    if (iterations > 0) {
        checkCuda(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "copying output");
    }
    checkCuda(cudaEventDestroy(startEvent), "destroying start event");
    checkCuda(cudaEventDestroy(endEvent), "destroying end event");
    checkCuda(cudaFree(d_val), "freeing values");
    checkCuda(cudaFree(d_cols), "freeing columns");
    checkCuda(cudaFree(d_rowDelimiters), "freeing row offsets");
    checkCuda(cudaFree(d_vec), "freeing vector");
    checkCuda(cudaFree(d_out), "freeing output");

    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
    const double avgTime = elapsedMs / static_cast<double>(iterations);
    
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
