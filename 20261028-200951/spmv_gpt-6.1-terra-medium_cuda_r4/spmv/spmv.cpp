#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim - (static_cast<uint64_t>(i) * dim + j);
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

// Each subgroup computes one CSR row; small groups avoid wasting lanes on
// sparse rows, while larger groups provide coalesced access for longer rows.
template <int GROUP>
__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* rows, const double* vec,
                           index_t dim, double* out) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = thread & (GROUP - 1);
    const index_t row = thread / GROUP;
    if (row >= dim) return;
    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += GROUP)
        sum += val[j] * vec[cols[j]];
    const unsigned int active = __activemask();
    for (int offset = GROUP / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(active, sum, offset, GROUP);
    if (lane == 0) out[row] = sum;
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        exit(1);
    }
}

template <int GROUP>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rows,
                       const double* vec, index_t dim, double* out) {
    constexpr unsigned int block = 256;
    const unsigned int grid = (static_cast<unsigned long long>(dim) * GROUP + block - 1) / block;
    if (grid) spmvKernel<GROUP><<<grid, block>>>(val, cols, rows, vec, dim, out);
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
    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero\n");
        return 1;
    }
    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    if (matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Number of non-zero elements exceeds CSR index capacity\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, matrixSize ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize) : 100.0);
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

    // Transfer inputs once, then time only the repeated device computations.
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rows = nullptr;
    const size_t valueBytes = static_cast<size_t>(nItems) * sizeof(double);
    const size_t colBytes = static_cast<size_t>(nItems) * sizeof(index_t);
    const size_t vectorBytes = static_cast<size_t>(numRows) * sizeof(double);
    const size_t rowBytes = (static_cast<size_t>(numRows) + 1) * sizeof(index_t);
    if (valueBytes) checkCuda(cudaMalloc(&d_val, valueBytes), "allocating values");
    if (colBytes) checkCuda(cudaMalloc(&d_cols, colBytes), "allocating columns");
    if (vectorBytes) {
        checkCuda(cudaMalloc(&d_vec, vectorBytes), "allocating vector");
        checkCuda(cudaMalloc(&d_out, vectorBytes), "allocating output");
        checkCuda(cudaMemcpy(d_vec, h_vec.data(), vectorBytes, cudaMemcpyHostToDevice), "copying vector");
        checkCuda(cudaMemset(d_out, 0, vectorBytes), "initializing output");
    }
    checkCuda(cudaMalloc(&d_rows, rowBytes), "allocating row offsets");
    if (valueBytes) checkCuda(cudaMemcpy(d_val, h_val.data(), valueBytes, cudaMemcpyHostToDevice), "copying values");
    if (colBytes) checkCuda(cudaMemcpy(d_cols, h_cols.data(), colBytes, cudaMemcpyHostToDevice), "copying columns");
    checkCuda(cudaMemcpy(d_rows, h_rowDelimiters.data(), rowBytes, cudaMemcpyHostToDevice), "copying row offsets");

    printf("Computing SpMV...\n");
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    const double averageRow = numRows ? static_cast<double>(nItems) / numRows : 0;
    auto launch = [&]() {
        if (averageRow <= 2) launchSpmv<2>(d_val, d_cols, d_rows, d_vec, numRows, d_out);
        else if (averageRow <= 4) launchSpmv<4>(d_val, d_cols, d_rows, d_vec, numRows, d_out);
        else if (averageRow <= 8) launchSpmv<8>(d_val, d_cols, d_rows, d_vec, numRows, d_out);
        else if (averageRow <= 16) launchSpmv<16>(d_val, d_cols, d_rows, d_vec, numRows, d_out);
        else launchSpmv<32>(d_val, d_cols, d_rows, d_vec, numRows, d_out);
    };
    if (iterations && numRows) {
        launch();
        checkCuda(cudaGetLastError(), "launching warmup");
        checkCuda(cudaDeviceSynchronize(), "executing warmup");
    }
    checkCuda(cudaEventRecord(start), "recording start event");
    for (index_t iter = 0; iter < iterations; ++iter) launch();
    checkCuda(cudaGetLastError(), "launching SpMV");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "executing SpMV");
    float elapsedMs = 0;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "measuring SpMV");
    if (vectorBytes) checkCuda(cudaMemcpy(h_out.data(), d_out, vectorBytes, cudaMemcpyDeviceToHost), "copying output");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(d_val), "freeing values");
    checkCuda(cudaFree(d_cols), "freeing columns");
    checkCuda(cudaFree(d_rows), "freeing row offsets");
    checkCuda(cudaFree(d_vec), "freeing vector");
    checkCuda(cudaFree(d_out), "freeing output");

    printf("Computation time: %.3f ms\n", elapsedMs);
    const double gflops = elapsedMs > 0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0;
    const double avgTime = iterations ? elapsedMs / iterations : 0;
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
